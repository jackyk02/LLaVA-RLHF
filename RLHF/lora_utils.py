# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os
from os.path import exists, join, isdir
import shutil
import sys
from typing import Optional, Dict, Sequence, List

import torch
import transformers
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import pickle

from models.reward_model import RewardModel

DEFAULT_PAD_TOKEN = "[PAD]"

def save_state_dict(output_state_dict: Dict, output_dir: str, filename: str = "model_weights.pkl") -> None:
    """
    Safely serialize and save a model state dictionary using pickle.
    
    Args:
        output_state_dict: The model state dictionary to save
        output_dir: Directory to save the file in
        filename: Name of the output file (default: model_weights.pkl)
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    # Convert tensors to CPU before saving
    cpu_state_dict = {
        key: value.cpu() if isinstance(value, torch.Tensor) else value 
        for key, value in output_state_dict.items()
    }
    
    # Use highest protocol for better performance
    with open(output_path, 'wb') as f:
        pickle.dump(cpu_state_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Successfully saved state dict to {output_path}")

class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        print("Saving PEFT checkpoint...")

        global_rank = int(os.environ.get("RANK", 0))

        if global_rank == 0:
            #save adapter from ppo trainer:
            from peft.utils import WEIGHTS_NAME, get_peft_model_state_dict
            save_directory="/root/LLaVA-RLHF/save_dir"
            save_directory = os.path.join(
                save_directory, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            )
            os.makedirs(save_directory, exist_ok=True)
            model = kwargs["model"].backbone_model

            for adapter_name, peft_config in model.peft_config.items():
                # save only the trainable weights
                output_state_dict = get_peft_model_state_dict(
                    model,
                    state_dict=kwargs.get("state_dict", None),
                    adapter_name=adapter_name,
                )
                print(output_state_dict)
                output_dir = (
                    os.path.join(save_directory, adapter_name)
                    if adapter_name != "default"
                    else save_directory
                )
                os.makedirs(output_dir, exist_ok=True)

                # torch.save(output_state_dict, os.path.join(output_dir, WEIGHTS_NAME))
                save_state_dict(output_state_dict, output_dir)

                # save the config and change the inference mode to `True`
                if peft_config.base_model_name_or_path is None:
                    peft_config.base_model_name_or_path = (
                        model.base_model.model.__dict__.get("name_or_path", None)
                    )
                inference_mode = peft_config.inference_mode
                peft_config.inference_mode = True
                peft_config.save_pretrained(output_dir)
                peft_config.inference_mode = inference_mode

            # previous saving:
            print("Saving model checkpoint to %s" % args.output_dir)
            if state.best_model_checkpoint is not None:
                checkpoint_folder = state.best_model_checkpoint
            else:
                checkpoint_folder = os.path.join(
                    args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
                )

            peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
            reward_head_path = os.path.join(checkpoint_folder, "reward_head")

            if isinstance(kwargs["model"], RewardModel):
                kwargs["model"].backbone_model.save_pretrained(peft_model_path)
                torch.save(
                    kwargs["model"].reward_head.state_dict(),
                    reward_head_path,
                )
            else:
                kwargs["model"].save_pretrained(peft_model_path)

            pytorch_model_paths = glob.glob(
                os.path.join(checkpoint_folder, "pytorch_model*.bin")
            )
            for pytorch_model_path in pytorch_model_paths:
                if os.path.exists(pytorch_model_path):
                    os.remove(pytorch_model_path)

            optimizer_path = os.path.join(checkpoint_folder, "optimizer.pt")
            if os.path.exists(optimizer_path):
                os.remove(optimizer_path)

        else:
            print("Skipping PEFT checkpoint save on rank %d" % global_rank)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            global_rank = int(os.environ.get("RANK", 0))
            if global_rank == 0:
                with open(fname, "a"):
                    os.utime(fname, times)

        touch(join(args.output_dir, "completed"))
        self.save_model(args, state, kwargs)


def print_trainable_parameters(args, model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    if args.bits == 4:
        trainable_params /= 2
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )


def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, "completed"))
        if is_completed:
            return None, True  # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith(
                "checkpoint"
            ):
                max_step = max(max_step, int(filename.replace("checkpoint-", "")))
        if max_step == 0:
            return None, is_completed  # training started, but no checkpoint
        checkpoint_dir = join(checkpoint_dir, f"checkpoint-{max_step}")
        print(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed  # checkpoint found!
    return None, False  # first training
