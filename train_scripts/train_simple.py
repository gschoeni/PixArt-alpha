"""Simplified Fine-tuning script for Stable Diffusion for text2image with support for LoRA."""

import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path
from typing import List, Union
from PIL import Image
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from datasets import load_dataset
from peft import LoraConfig, get_peft_model_state_dict, get_peft_model
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, Transformer2DModel
from transformers import T5EncoderModel, T5Tokenizer
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.25.0.dev0")

logger = logging.getLogger(__name__)


def save_model_card(repo_id: str, images=None, base_model=str, dataset_name=str, repo_folder=None):
    img_str = ""
    if images:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            img_str += f"![img_{i}](./image_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- lora
inference: true
---
    """
    model_card = f"""
# LoRA text2image fine-tuning - {repo_id}
These are LoRA adaption weights for {base_model}. The weights were fine-tuned on the {dataset_name} dataset. You can find some example images in the following. \n
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="The name of the Dataset (from the HuggingFace hub) to train on.",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        required=True,
        help="The config of the Dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    args = parser.parse_args()
    return args


DATASET_NAME_MAPPING = {"lambdalabs/pokemon-blip-captions": ("image", "text"),}


def main():
    args = parse_args()

    # Set default parameters
    image_column = "image"
    caption_column = "action"
    seed = 42
    resolution = 512
    train_batch_size = 8
    num_train_epochs = 20
    gradient_accumulation_steps = 1
    learning_rate = 3e-04
    lr_scheduler = "cosine"
    lr_warmup_steps = 0
    dataloader_num_workers = 8
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_weight_decay = 0.03
    adam_epsilon = 1e-10
    max_grad_norm = 1.0
    proportion_empty_prompts = 0
    logging_dir = "logs"
    checkpointing_steps = 100
    checkpoints_total_limit = 5
    rank = 16
    trigger_prompt = ""

    logging_dir = Path(args.output_dir, logging_dir)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_warning()
    diffusers.utils.logging.set_verbosity_info()

    # Set the training seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Handle the repository creation
    os.makedirs(args.output_dir, exist_ok=True)

    # See Section 3.1. of the paper.
    max_length = 120

    # For mixed precision training we cast all non-trainable weights to half-precision
    weight_dtype = torch.bfloat16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler", torch_dtype=weight_dtype)
    tokenizer = T5Tokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", torch_dtype=weight_dtype)

    text_encoder = T5EncoderModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=weight_dtype)
    text_encoder.requires_grad_(False)
    text_encoder.to(device)

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", torch_dtype=weight_dtype)
    vae.requires_grad_(False)
    vae.to(device)

    transformer = Transformer2DModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="transformer", torch_dtype=weight_dtype)

    # freeze parameters of models to save more memory
    transformer.requires_grad_(False)

    # Freeze the transformer parameters before adding adapters
    for param in transformer.parameters():
        param.requires_grad_(False)

    lora_config = LoraConfig(
        r=rank,
        init_lora_weights="gaussian",
        target_modules=[
            "to_k",
            "to_q",
            "to_v",
            "to_out.0",
            "proj_in",
            "proj_out",
            "ff.net.0.proj",
            "ff.net.2",
            "proj",
            "linear",
            "linear_1",
            "linear_2",
        ]
    )

    # Move transformer to device
    transformer.to(device)

    def cast_training_params(model: Union[torch.nn.Module, List[torch.nn.Module]], dtype=torch.float32):
        if not isinstance(model, list):
            model = [model]
        for m in model:
            for param in m.parameters():
                # only upcast trainable parameters into fp32
                if param.requires_grad:
                    param.data = param.to(dtype)

    transformer = get_peft_model(transformer, lora_config)
    # only upcast trainable parameters (LoRA) into fp32
    cast_training_params(transformer, dtype=torch.float32)

    transformer.print_trainable_parameters()

    lora_layers = filter(lambda p: p.requires_grad, transformer.parameters())

    # Enable gradient checkpointing
    transformer.enable_gradient_checkpointing()

    # Initialize the optimizer
    optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        lora_layers,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    # Get the datasets
    # Downloading and loading a dataset from the hub.
    dataset = load_dataset(
        "parquet",
        data_files=args.dataset_config_name,
    )

    # Preprocessing the datasets.
    column_names = dataset["train"].column_names

    # Get the column names for input/target.
    dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
    if image_column not in column_names:
        raise ValueError(f"--image_column' value '{image_column}' needs to be one of: {', '.join(column_names)}")
    if caption_column not in column_names:
        raise ValueError(f"--caption_column' value '{caption_column}' needs to be one of: {', '.join(column_names)}")

    # Preprocessing the datasets.
    def tokenize_captions(examples, is_train=True, proportion_empty_prompts=0., max_length=120):
        captions = []
        for caption in examples[caption_column]:
            # Prepend the trigger prompt to the caption if it is set
            if trigger_prompt:
                caption = f"{trigger_prompt} {caption}"
            # Add an empty string to the beginning of the list if it's empty
            if random.random() < proportion_empty_prompts:
                captions.append("")
            elif isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(f"Caption column `{caption_column}` should contain either strings or lists of strings.")
        inputs = tokenizer(captions, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
        return inputs.input_ids, inputs.attention_mask

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        image_paths = [os.path.join(args.dataset_name, image_path) for image_path in examples[image_column]]
        images = [Image.open(path).convert("RGB") for path in image_paths]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"], examples['prompt_attention_mask'] = tokenize_captions(examples, proportion_empty_prompts=proportion_empty_prompts, max_length=max_length)
        return examples

    # Set the training transforms
    train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        prompt_attention_mask = torch.stack([example["prompt_attention_mask"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids, 'prompt_attention_mask': prompt_attention_mask}

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=train_batch_size,
        num_workers=dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    max_train_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps,
        num_training_steps=max_train_steps,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    max_train_steps = num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = train_batch_size * gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0
    initial_global_step = 0

    progress_bar = tqdm(range(0, max_train_steps), initial=initial_global_step, desc="Steps")

    # Initialize tensorboard writer
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=logging_dir)

    for epoch in range(first_epoch, num_train_epochs):
        transformer.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Convert images to latent space
            latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)

            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            prompt_embeds = text_encoder(batch["input_ids"], attention_mask=batch['prompt_attention_mask'])[0]
            prompt_attention_mask = batch['prompt_attention_mask']

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            # Prepare micro-conditions.
            added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
            if transformer.config.sample_size == 128:
                resolution_tensor = torch.tensor([resolution, resolution]).repeat(bsz, 1)
                aspect_ratio = torch.tensor([float(resolution / resolution)]).repeat(bsz, 1)
                resolution_tensor = resolution_tensor.to(dtype=weight_dtype, device=latents.device)
                aspect_ratio = aspect_ratio.to(dtype=weight_dtype, device=latents.device)
                added_cond_kwargs = {"resolution": resolution_tensor, "aspect_ratio": aspect_ratio}

            # Predict the noise residual and compute loss
            model_pred = transformer(noisy_latents,
                                     encoder_hidden_states=prompt_embeds,
                                     encoder_attention_mask=prompt_attention_mask,
                                     timestep=timesteps,
                                     added_cond_kwargs=added_cond_kwargs).sample.chunk(2, 1)[0]

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            # Accumulate gradients
            loss = loss / gradient_accumulation_steps
            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(lora_layers, max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                progress_bar.update(1)
                global_step += 1

                # Log training loss
                writer.add_scalar("train_loss", loss.item() * gradient_accumulation_steps, global_step)

                train_loss = 0.0

                if global_step % checkpointing_steps == 0:
                    # Save checkpoint
                    checkpoints = os.listdir(args.output_dir)
                    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                    if len(checkpoints) >= checkpoints_total_limit:
                        num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
                        removing_checkpoints = checkpoints[0:num_to_remove]

                        logger.info(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
                        logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                        for removing_checkpoint in removing_checkpoints:
                            removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                            shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(save_path, exist_ok=True)

                    # Save LoRA weights
                    transformer.save_pretrained(save_path)
                    transformer_lora_state_dict = get_peft_model_state_dict(transformer)
                    StableDiffusionPipeline.save_lora_weights(
                        save_directory=save_path,
                        unet_lora_layers=transformer_lora_state_dict,
                        safe_serialization=True,
                    )

                    logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

    # Save the final lora layers
    transformer.save_pretrained(args.output_dir)
    lora_state_dict = get_peft_model_state_dict(transformer)
    StableDiffusionPipeline.save_lora_weights(os.path.join(args.output_dir, "transformer_lora"), lora_state_dict)

    writer.close()

    logger.info("Training completed!")


if __name__ == "__main__":
    main()