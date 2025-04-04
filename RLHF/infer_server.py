# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json

import os
from dataclasses import dataclass, field
from typing import Optional, List, Literal
import logging

import torch
import transformers
import argparse
from transformers import set_seed

from transformers import AutoTokenizer

from lora_utils import (
    SavePeftModelCallback,
    print_trainable_parameters,
    get_last_checkpoint,
    DEFAULT_PAD_TOKEN,
)
from data_utils.data_utils_rm import make_binary_reward_modeling_data_module
from models.reward_model import (
    RewardConfig,
    RewardModel,
    RewardModelTrainer as Trainer,
    compute_reward_modeling_metrics,
)

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token
from llava.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)

from llava.train.train import smart_tokenizer_and_embedding_resize
from data_utils.common_utils import preprocess

torch.backends.cuda.matmul.allow_tf32 = True

logger = logging.getLogger(__name__)


class DisableLogger:
    def __enter__(self):
        logging.disable(logging.CRITICAL)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        logging.disable(logging.NOTSET)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-12b")
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."
        },
    )
    # from llava
    version: Optional[str] = field(default="v1")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(
        default=-1
    )  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    dataset_path: str = field(default="tatsu-lab/alpaca_farm")
    dataset_name: str = field(default=None, metadata={"help": "Dataset name"})
    eval_dataset_path: str = field(default="tatsu-lab/alpaca_farm")
    eval_dataset_name: str = field(default="alpaca_human_preference")
    eval_size: int = field(
        default=500,
        metadata={
            "help": "Number of examples to split out from training to use for evaluation."
        },
    )
    # from llava
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = field(default=None)
    reward_prompt_file: Optional[str] = field(default=None)
    image_to_caption_file: Optional[str] = field(default=None)


@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    # from llava
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    # From AlpacaFarm
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be left padded to this length always during training."
        },
    )
    query_len: int = field(default=None, metadata={"help": "Length of the query."})
    response_len: int = field(
        default=None, metadata={"help": "Length of the response."}
    )
    label_names: List[str] = field(
        default_factory=lambda: ["index_0", "index_1", "choice"],
        metadata={
            "help": "Names of the labels in the dataset. "
            "This is needed to get transformers.Trainer to not throw those tensors away before `compute_loss`."
            "By default, the trainer throws away columns it doesn't recognize when creating the "
            "`train_dataloader` (see `_remove_unused_columns`). "
        },
    )
    padding: Literal["max_length", "longest"] = field(
        default="longest",
        metadata={
            "help": "Padding strategy. If 'max_length', pads to `model_max_length` always; this might lead to some "
            "redundant compute. If 'longest', pads to the longest sequence in the batch, capped by `model_max_length`."
        },
    )
    # From QLoRA
    full_finetune: bool = field(
        default=False, metadata={"help": "Finetune the entire model without adapters."}
    )
    adam8bit: bool = field(default=False, metadata={"help": "Use 8-bit adam."})
    double_quant: bool = field(
        default=True,
        metadata={
            "help": "Compress the quantization statistics through double quantization."
        },
    )
    quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type to use. Should be one of `fp4` or `nf4`."
        },
    )
    bits: int = field(default=4, metadata={"help": "How many bits to use."})
    lora_modules: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "Which modules to use LoRA on. If None, will use all linear layers."
        },
    )
    lora_r: int = field(default=64, metadata={"help": "Lora R dimension."})
    lora_alpha: float = field(default=16, metadata={"help": " Lora alpha."})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout."})
    report_to: str = field(
        default="none",
        metadata={"help": "To use wandb or something else for reporting."},
    )
    resume_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory containing the checkpoint to resume."},
    )
    output_dir: str = field(
        default="./output", metadata={"help": "The output dir for logs and checkpoints"}
    )
    optim: str = field(
        default="paged_adamw_32bit", metadata={"help": "The optimizer to be used"}
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={
            "help": "The training batch size per GPU. Increase for better speed."
        },
    )
    gradient_accumulation_steps: int = field(
        default=16,
        metadata={
            "help": "How many gradients to accumulate before to perform an optimizer step"
        },
    )
    weight_decay: float = field(
        default=0.0, metadata={"help": "The L2 weight decay rate of AdamW"}
    )  # use lora dropout instead for regularization if needed
    learning_rate: float = field(default=0.0002, metadata={"help": "The learnign rate"})
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "Removed unused columns. Needed to make this codebase work."},
    )
    max_grad_norm: float = field(
        default=0.3,
        metadata={
            "help": "Gradient clipping max norm. This is tuned and works well for all models tested."
        },
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Use gradient checkpointing. You want to use this."},
    )
    do_train: bool = field(
        default=True,
        metadata={"help": "To train or not to train, that is the question?"},
    )
    lr_scheduler_type: str = field(
        default="constant",
        metadata={
            "help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"
        },
    )
    warmup_ratio: float = field(
        default=0.03, metadata={"help": "Fraction of steps to do a warmup for"}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "The frequency of update steps after which to log the loss"},
    )
    group_by_length: bool = field(
        default=True,
        metadata={
            "help": "Group sequences into batches with same length. Saves memory and speeds up training considerably."
        },
    )
    save_strategy: str = field(
        default="steps", metadata={"help": "When to save checkpoints"}
    )
    save_steps: int = field(default=250, metadata={"help": "How often to save a model"})
    save_total_limit: int = field(
        default=40,
        metadata={
            "help": "How many checkpoints to save before the oldest is overwritten"
        },
    )
    resume_from_training: bool = field(
        default=False, metadata={"help": "Resume from training"}
    )


def rank0_print(*args):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print(*args)

def bridge_process(image, resize_size=256):
    """
    Takes in environment and observation and returns resized image as numpy array.

    NOTE (Moo Jin): To make input images in distribution with respect to the inputs seen at training time, we follow
                    the same resizing scheme used in the Octo dataloader, which OpenVLA uses for training.
    """
    import tensorflow as tf
    # Preprocess the image the exact same way that the Berkeley Bridge folks did it
    # to minimize distribution shift.
    # NOTE (Moo Jin): Yes, we resize down to 256x256 first even though the image may end up being
    # resized up to a different resolution by some models. This is just so that we're in-distribution
    # w.r.t. the original preprocessing at train time.
    IMAGE_BASE_PREPROCESS_SIZE = 128
    # Resize to image size expected by model
    image = tf.image.encode_jpeg(image)  # Encode as JPEG, as done in RLDS dataset builder
    image = tf.io.decode_image(image, expand_animations=False, dtype=tf.uint8)  # Immediately decode back
    image = tf.image.resize(
        image, (IMAGE_BASE_PREPROCESS_SIZE, IMAGE_BASE_PREPROCESS_SIZE), method="lanczos3", antialias=True
    )
    image = tf.image.resize(image, (resize_size, resize_size), method="lanczos3", antialias=True)
    image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)
    return image.numpy()


class RobotRewardModel:
    def __init__(self):
        hfparser = transformers.HfArgumentParser(
            (ModelArguments, DataArguments, TrainingArguments)
        )
        (
            model_args,
            data_args,
            training_args,
            extra_args,
        ) = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
        args = argparse.Namespace(
            **vars(model_args), **vars(data_args), **vars(training_args)
        )

        tokenizer_model_name = args.model_name_or_path
        TokenizerClass = AutoTokenizer

        # Tokenizer
        tokenizer = TokenizerClass.from_pretrained(
            tokenizer_model_name,
            cache_dir=args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="left",
            truncation_side="right",
            use_fast=False,
        )

        if model_args.version == "v0":
            if tokenizer.pad_token is None:
                smart_tokenizer_and_embedding_resize(
                    special_tokens_dict=dict(pad_token="[PAD]"),
                    tokenizer=tokenizer,
                    model=model,
                )
        elif model_args.version == "v0.5":
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.unk_token
            if model_args.version in conversation_lib.conv_templates:
                conversation_lib.default_conversation = conversation_lib.conv_templates[
                    model_args.version
                ]
            else:
                conversation_lib.default_conversation = conversation_lib.conv_templates[
                    "vicuna_v1"
                ]

        if model_args.vision_tower is not None:
            config = RewardConfig(backbone_model_name_or_path=model_args.model_name_or_path)

            with DisableLogger():
                model = RewardModel(
                    args=args,
                    config=config,
                    qlora=True,
                    checkpoint_dir="/root/LLaVA-RLHF/model_dir/checkpoint",
                    tokenizer=tokenizer,
                ).to(torch.bfloat16)

            model.backbone_model.config.use_cache = True
            print_trainable_parameters(args, model)
            print("loaded model")

            with DisableLogger():
                model_temp = model.backbone_model

            vision_tower = model_temp.get_vision_tower()
            if not vision_tower.is_loaded:
                vision_tower.load_model()

            data_args.image_processor = vision_tower.image_processor
            data_args.is_multimodal = True
            model_temp.config.mm_use_im_start_end = (
                data_args.mm_use_im_start_end
            ) = model_args.mm_use_im_start_end
            training_args.use_im_start_end = model_args.mm_use_im_start_end

            # print(model)
            set_seed(args.seed)
            self.tokenizer = tokenizer
            self.model = model
            self.model.eval()
            self.data_args = data_args

    def get_rewards(self, instruction, image_path, actions):
        batch_size = len(actions)
        #input id works -----------------------------------------------------------
        from action_processing import ActionTokenizer
        action_tokenizer = ActionTokenizer(self.tokenizer)
        from llava.conversation import conv_templates, SeparatorStyle
        conv_mode = "vicuna_v1"
        conv_template = conv_templates[conv_mode].copy()

        action_in_ids= []
        instruction = instruction.lower().rstrip('.')

        for action in actions:
            # Prepare conversation
            action_id = np.array(action)
            if type(action_id[0]) == float:
                action_id = action_tokenizer(action)
            # print(action_id)
            # if isinstance(action, list) and all(isinstance(x, int) for x in action):
            #     # print(type(action))
            #     action_id = action
            # else:
            #     action_id = action_tokenizer(action)
            holder = "hello hello hello hello hello hello hello" 
            inp = (f"shows the current observation from the robot's wrist-mounted camera. "
                    f"The robot manipulation arm is attempting to {instruction}. "
                    f"What action should the robot take to effectively accomplish the task? "
                    f"ASSISTANT: The robot should take the action: {holder} </s> "
                    f"USER: Please evaluate the quality of the robot action. "
                    f"A good robot action should consider different factors, "
                    f"especially interactions with surrounding objects and human preferences.\n"
                    f"ASSISTANT: Based on how humans would control the robot arm and the "
                    f"awareness of the situation, the quality score of the robot action is")

            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp

            conv = conv_template.copy()
            conv.append_message(conv.roles[0], inp)
            prompt = conv.get_prompt()

            # prompt = "<s> " + prompt
            prompt = prompt.replace("<image>", " holder ")

            # print("original prompt: ", prompt)

            in_ids = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding="longest",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
            ).input_ids

            # print("id: ", in_ids)

            repeated_indices = (in_ids == 22172).nonzero()
            start_idx = repeated_indices[0][1].item()  # Get the first occurrence
            end_idx = repeated_indices[-1][1].item() + 1  # Get the last occurrence + 1
            in_ids[0, start_idx:end_idx] = torch.tensor(action_id-1000)

            first_19464_idx = (in_ids == 19464).nonzero()
            start_idx = first_19464_idx[0][1].item()  # Get the first occurrence
            in_ids[0, start_idx: start_idx+1] = -200

            in_ids = in_ids[:, :-1]

            # Find first occurrence of 25
            # first_25_idx = (in_ids == 25).nonzero()
            # start_idx = first_25_idx[0][1].item()  # Get the first occurrence
            # in_ids[0, start_idx+1] = -220
            # in_ids = torch.cat([in_ids[:, :start_idx+2], in_ids[:, start_idx+3:]], dim=1)

            # Find first occurrence of 256
            # first_256_idx = (in_ids == 256).nonzero()
            # start_idx = first_256_idx[0][1].item()  # Get the first occurrence
            # in_ids[0, start_idx] = 220

            in_ids = torch.tensor(in_ids, dtype=torch.long).squeeze(0)

            # print("updated id: ", in_ids)
            # print(in_ids.shape)
            action_in_ids.append(in_ids)

        # ex_input_ids = torch.tensor(action_in_ids, dtype=torch.long)
        from typing import Sequence
        import einops
        def pad_sequence_from_left(
            sequences: Sequence[torch.Tensor],
            batch_first: bool = False,
            padding_value: float = 0.0,
        ):
            """Mirror of `torch.nn.utils.rnn.pad_sequence`, but pad from left."""
            sequences = tuple(sequence.flip(0) for sequence in sequences)
            padded_sequence = torch._C._nn.pad_sequence(
                sequences, batch_first, padding_value
            )  # noqa
            padded_sequence = padded_sequence.flip(int(batch_first))
            return padded_sequence

        def _left_pad_helper(ex_input_ids, batch_size):
            # TODO(lxuechen): Potentially replace with `transformers.PretrainedTokenizerBase.prepare_for_model`.
            # `instances` is a list of dicts, each dict has key whose value is a list of tensors, possibly of unequal length.
            # input_ids = [seq for instance in instances for seq in instance[key]]  # Flatten.
            input_ids = [seq for seq in ex_input_ids]
            input_ids = pad_sequence_from_left(
                input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id,
            )
            input_ids = einops.rearrange(
                input_ids,
                "(bsz num_candidates) max_seq_len -> bsz num_candidates max_seq_len",
                num_candidates=batch_size,
            )
            return input_ids
        input_ids = _left_pad_helper(action_in_ids, batch_size).squeeze(0)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()

        # print("input_ids: ", input_ids.shape)
        # print("pad id: ", in_ids)
        # print("attn_mask: ", attention_mask.shape)
        #input id works -----------------------------------------------------------

        #image loading works
        from PIL import Image
        processor = self.data_args.image_processor
        
        image = Image.open(image_path).convert("RGB")
        # image = bridge_process(image)
        # scaled_image = np.clip((image + 1) * 127.5, 0, 255).astype(np.uint8)
        # image = Image.fromarray(scaled_image).convert("RGB")

        if self.data_args.image_aspect_ratio == "pad":
            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(
                        pil_img.mode, (width, width), background_color
                    )
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(
                        pil_img.mode, (height, height), background_color
                    )
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result
            # def resize_to_256(pil_img):
            #     width, height = pil_img.size
            #     if width == 256 and height == 256:
            #         return pil_img
            #     else:
            #         return pil_img.resize((256, 256), Image.Resampling.LANCZOS)

            # image = resize_to_256(image)
            image = expand2square(
                image, tuple(int(x * 255) for x in processor.image_mean)
            )
            image = processor.preprocess(image, return_tensors="pt")[
                "pixel_values"
            ][0]

        images = image.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        model_inputs = {
            "input_ids": input_ids.cuda(0).to(torch.int64),
            "attention_mask": attention_mask.cuda(0).to(torch.int64),
            "images": images.cuda(0).to(torch.bfloat16)
        }
        # with torch.no_grad():
        scores = self.model.forward(**model_inputs)
        return scores.rewards.detach().cpu().tolist()

if __name__ == "__main__":
    # rm = RobotRewardModel()

    # instruction = "move the yellow knife to the right of the pan"
    # image_path = "images/robot.jpg"
    # actions = [
    #     [-0.0006071124225854874, -0.001102231559343636, -0.002975916489958763, -0.0037233866751194, 0.009374408982694149, 0.00042649864917621017, 1.003713607788086], #action0
    #     [0.0007309613865800202, -0.00033146265195682645, 8.855393389239907e-05, 0.0023672617971897125, -0.00297730159945786, 0.0071182833053171635, 1.0025840997695923],
    #             [-0.0006071124225854874, -0.001102231559343636, -0.002975916489958763, -0.0037233866751194, 0.009374408982694149, 0.00042649864917621017, 1.003713607788086], #action0
    #     [0.0007309613865800202, -0.00033146265195682645, 8.855393389239907e-05, 0.0023672617971897125, -0.00297730159945786, 0.0071182833053171635, 1.0025840997695923],
    # ]

    # scores = rm.get_reward(instruction, image_path, actions)
    # print(scores)
    from fastapi import FastAPI, HTTPException, Request
    from pydantic import BaseModel
    import uvicorn
    import json_numpy as json
    import numpy as np

    app = FastAPI()

    reward_model = RobotRewardModel()

    class InputData(BaseModel):
        instruction: str
        image_path: str

    @app.get("/")
    async def read_root():
        return {"message": "RM server up"}

    @app.post("/process")
    async def process_data(request: Request):
        body = await request.body()
        data = json.loads(body)

        instruction = data.get("instruction")
        image_path = data.get("image_path")
        action = data.get("action")

        if not isinstance(instruction, str):
            raise HTTPException(status_code=400, detail="Instruction must be a string")
        if not isinstance(image_path, str):
            raise HTTPException(status_code=400, detail="Image path must be a string")

        action_array = np.array(action)

        if action_array.ndim != 2:
            raise HTTPException(status_code=400, detail="Action must be a 2D array")

        import time
        start_time = time.time()

        rewards = reward_model.get_rewards(instruction, image_path, action_array)
        
        execution_time = time.time() - start_time
        print(execution_time)

        return {"rewards": rewards}

    uvicorn.run(app, host="0.0.0.0", port=3100)
