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

from moellava import conversation as conversation_lib
from moellava.model import *
from moellava.mm_utils import tokenizer_image_token
from moellava.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)

from moellava.train.train import smart_tokenizer_and_embedding_resize
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
    # from moellava
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
    # from moellava
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
    # from moellava
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


def train():
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

    if args.resume_dir is not None:
        checkpoint_dir, completed_training = args.resume_dir, False
    else:
        checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)

    if completed_training:
        rank0_print("Detected that training was already completed!")

    if checkpoint_dir is None:
        rank0_print("Training from scratch.")
    else:
        rank0_print("Loading from checkpoint:", checkpoint_dir)
        if args.resume_from_training:
            rank0_print("Resuming from training not supported yet. Exiting.")
            exit(1)

    tokenizer_model_name = args.model_name_or_path
    TokenizerClass = AutoTokenizer

    # Tokenizer
    tokenizer = TokenizerClass.from_pretrained(
        "stabilityai/stablelm-2-1_6b",
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

        with DisableLogger():
            model = MoELLaVALlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
            )
        # print(model)
        vision_tower = model.get_image_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True
        model.config.mm_use_im_start_end = (
            data_args.mm_use_im_start_end
        ) = model_args.mm_use_im_start_end
        training_args.use_im_start_end = model_args.mm_use_im_start_end
    
    #get input ids from prompt-----------------------------------------------------------
    from action_processing import ActionTokenizer
    action_tokenizer = ActionTokenizer(tokenizer)
    actions = [
        [-0.0006071124225854874, -0.001102231559343636, -0.002975916489958763, -0.0037233866751194, 0.009374408982694149, 0.00042649864917621017, 1.003713607788086], #action0
        [0.0007309613865800202, -0.00033146265195682645, 8.855393389239907e-05, 0.0023672617971897125, -0.00297730159945786, 0.0071182833053171635, 1.0025840997695923]
    ]
    instruction = "place utensil in between towel and pot"
    from moellava.conversation import conv_templates, SeparatorStyle
    conv_mode = "vicuna_v1"
    conv_template = conv_templates[conv_mode].copy()

    for action in actions:
        # Prepare conversation
        action_str = action_tokenizer(action)
        inp = (f"shows the current observation from the robot's wrist-mounted camera. "
                f"The robot manipulation arm is attempting to {instruction}. "
                f"What action should the robot take to effectively accomplish the task? "
                f"ASSISTANT: The robot should take the action: {action_str} "
                f"USER: Please evaluate the quality of the robot action. "
                f"A good robot action should consider different factors, "
                f"especially interactions with surrounding objects and human preferences.\n"
                f"ASSISTANT: Based on how humans would control the robot arm and the "
                f"awareness of the situation, the quality score of the robot action is</s")

        if model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp

        conv = conv_template.copy()
        conv.append_message(conv.roles[0], inp)
        prompt = conv.get_prompt()
        prompt = prompt.replace("<image>", "<|endoftext|>")
        in_ids = tokenizer(prompt).input_ids
        in_ids.pop()
        in_ids[25]=-200
        print(torch.tensor(in_ids, dtype=torch.int64))

    # config = RewardConfig(backbone_model_name_or_path=model_args.model_name_or_path)

    # with DisableLogger():
    #     model = RewardModel(
    #         args=args,
    #         config=config,
    #         qlora=True,
    #         checkpoint_dir="/root/LLaVA-RLHF/model_dir/checkpoint-4800",
    #         tokenizer=tokenizer,
    #     )

    # model.backbone_model.config.use_cache = False
    # print_trainable_parameters(args, model)
    # print("loaded model")
    # # print(model)
    # set_seed(args.seed)
    
    #input id works -----------------------------------------------------------

    ex_input_ids_0 = [32,   6369,   1990,    264,   1217,    323,    459,  15592,  11376,
          44658,  18328,    430,  67349,    279,   9293,   5178,    315,    264,
          12585,  34786,   1887,     13,  14194,     25,    220,   -200,    198,
          60556,    279,   1510,  22695,    505,    279,  12585,    596,  33271,
          78830,   6382,     13,    578,  12585,  34786,   6916,    374,  19969,
            311,   2035,  82036,    321,    304,   1990,  43713,    323,   3419,
             13,   3639,   1957,   1288,    279,  12585,   1935,    311,  13750,
          22829,    279,   3465,     30,  36660,   3931,   2891,     25,    578,
          12585,   1288,   1935,    279,   1957,     25,    379,  12855, 100160,
         100166, 100164, 100147, 100155, 100035, 100257,  14194,     25,   5321,
          15806,    279,   4367,    315,    279,  12585,   1957,     13,    362,
           1695,  12585,   1957,   1288,   2980,   2204,   9547,     11,   5423,
          22639,    449,  14932,   6302,    323,   3823,  19882,    627,   5045,
           3931,   2891,     25,  20817,    389,   1268,  12966,   1053,   2585,
            279,  12585,   6916,    323,    279,  17985,    315,    279,   6671,
             11,    279,   4367,   5573,    315,    279,  12585,   1957,    374,
            524,  82]
    
    ex_input_ids_1 = [32,   6369,   1990,    264,   1217,    323,    459,  15592,  11376,
        44658,  18328,    430,  67349,    279,   9293,   5178,    315,    264,
        12585,  34786,   1887,     13,  14194,     25,    220,   -200,    198,
        60556,    279,   1510,  22695,    505,    279,  12585,    596,  33271,
        78830,   6382,     13,    578,  12585,  34786,   6916,    374,  19969,
        311,   2035,  82036,    321,    304,   1990,  43713,    323,   3419,
            13,   3639,   1957,   1288,    279,  12585,   1935,    311,  13750,
        22829,    279,   3465,     30,  36660,   3931,   2891,     25,    578,
        12585,   1288,   1935,    279,   1957,     25,    379,  12855, 100160,
        100166, 100164, 100147, 100155, 100035, 100257,  14194,     25,   5321,
        15806,    279,   4367,    315,    279,  12585,   1957,     13,    362,
        1695,  12585,   1957,   1288,   2980,   2204,   9547,     11,   5423,
        22639,    449,  14932,   6302,    323,   3823,  19882,    627,   5045,
        3931,   2891,     25,  20817,    389,   1268,  12966,   1053,   2585,
        279,  12585,   6916,    323,    279,  17985,    315,    279,   6671,
            11,    279,   4367,   5573,    315,    279,  12585,   1957,    374,
        524,  82]
    print("ex_input_ids_1: ", torch.tensor(ex_input_ids_1, dtype=torch.int64))
    input = [ex_input_ids_0, ex_input_ids_1]

    ex_input_ids = torch.tensor(input, dtype=torch.long)
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
            padding_value=tokenizer.pad_token_id,
        )
        input_ids = einops.rearrange(
            input_ids,
            "(bsz num_candidates) max_seq_len -> bsz num_candidates max_seq_len",
            num_candidates=batch_size,
        )
        return input_ids
    batch_size = 2
    input_ids = _left_pad_helper(ex_input_ids, batch_size).squeeze(0)
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long()

    # print("input_ids: ", input_ids.shape)
    # print("attn_mask: ", attention_mask.shape)
    #input id works -----------------------------------------------------------


    #image loading works
    from PIL import Image
    processor = data_args.image_processor
    image = Image.open("robot.jpg").convert("RGB")
    if data_args.image_aspect_ratio == "pad":
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

        image = expand2square(
            image, tuple(int(x * 255) for x in processor.image_mean)
        )
        image = processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

    images = image.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    # print(image)
    # print(image.shape)
    # image loading works

    model_inputs = {
        "input_ids": input_ids.cuda(0).to(torch.int64),
        "attention_mask": attention_mask.cuda(0).to(torch.int64),
        "images": images.cuda(0).to(torch.bfloat16)
    }
    score = model.forward(**model_inputs)
    print("score: ", score)

if __name__ == "__main__":
    train()
