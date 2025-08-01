#!/bin/bash

# This script provisions a Runpod environment for Kohya_SS.

# Exit on error
set -e

# Variables
KOHYA_SS_DIR="/workspace/kohya_ss"
KOHYA_SS_GIT_URL="https://github.com/bmaltais/kohya_ss.git"

# --- System Packages ---
APT_PACKAGES=(
    "git"
    "python3-pip"
    "python3-venv"
    "libgl1-mesa-glx"
    "libglib2.0-0"
)

# --- Python Packages ---
PIP_PACKAGES=(
    "torch==2.1.2"
    "torchvision==0.16.2"
    "torchaudio==2.1.2"
    "xformers==0.0.23.post1"
    "accelerate==0.25.0"
    "transformers==4.36.2"
    "diffusers==0.25.1"
    "safetensors==0.4.1"
    "peft==0.7.1"
    "prodigyopt==1.0"
    "bitsandbytes==0.42.0"
    "gradio==3.48.0"
)

# --- Model Downloads ---
CHECKPOINT_MODELS=(
    "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors"
    "https://huggingface.co/kingcashflow/modelcheckpoints/resolve/main/AIIM_Realism.safetensors"
)

# --- Helper Functions ---

provisioning_print_header() {
    echo "========================================="
    echo "   Kohya_SS Runpod Provisioning Script   "
    echo "========================================="
}

install_apt_packages() {
    echo "--> Installing system packages..."
    sudo apt-get update
    sudo apt-get install -y "${APT_PACKAGES[@]}"
}

install_pip_packages() {
    echo "--> Installing Python packages..."
    pip install --upgrade pip
    pip install "${PIP_PACKAGES[@]}"
}

clone_kohya_ss() {
    if [ ! -d "$KOHYA_SS_DIR" ]; then
        echo "--> Cloning Kohya_SS repository..."
        git clone "$KOHYA_SS_GIT_URL" "$KOHYA_SS_DIR"
    else
        echo "--> Kohya_SS repository already exists. Skipping clone."
    fi
    cd "$KOHYA_SS_DIR"
}

download_models() {
    echo "--> Downloading models..."
    local model_dir="$KOHYA_SS_DIR/models/sdxl"
    mkdir -p "$model_dir"

    for url in "${CHECKPOINT_MODELS[@]}"; do
        filename=$(basename "$url")
        target_file="$model_dir/$filename"
        if [ ! -f "$target_file" ]; then
            echo "    -> Downloading $filename..."
            wget -q -O "$target_file" "$url"
        else
            echo "    -> $filename already exists. Skipping."
        fi
    done
}

create_config_file() {
    echo "--> Creating Kohya_SS config file..."
    cat <<EOF > "$KOHYA_SS_DIR/config.json"
{
  "LoRA_type": "Standard",
  "LyCORIS_preset": "full",
  "adaptive_noise_scale": 0,
  "additional_parameters": "",
  "async_upload": false,
  "block_alphas": "",
  "block_dims": "",
  "block_lr_zero_threshold": "",
  "bucket_no_upscale": false,
  "bucket_reso_steps": 64,
  "bypass_mode": false,
  "cache_latents": true,
  "cache_latents_to_disk": true,
  "caption_dropout_every_n_epochs": 0,
  "caption_dropout_rate": 0.05,
  "caption_extension": ".txt",
  "clip_skip": 1,
  "color_aug": false,
  "constrain": 0,
  "conv_alpha": 1,
  "conv_block_alphas": "",
  "conv_block_dims": "",
  "conv_dim": 1,
  "dataset_config": "",
  "debiased_estimation_loss": false,
  "decompose_both": false,
  "dim_from_weights": false,
  "dora_wd": false,
  "down_lr_weight": "",
  "dynamo_backend": "no",
  "dynamo_mode": "default",
  "dynamo_use_dynamic": false,
  "dynamo_use_fullgraph": false,
  "enable_bucket": true,
  "epoch": 10,
  "extra_accelerate_launch_args": "",
  "factor": -1,
  "flip_aug": false,
  "fp8_base": false,
  "full_bf16": false,
  "full_fp16": false,
  "gpu_ids": "",
  "gradient_accumulation_steps": 1,
  "gradient_checkpointing": true,
  "huber_c": 0.1,
  "huber_schedule": "snr",
  "huggingface_path_in_repo": "",
  "huggingface_repo_id": "",
  "huggingface_repo_type": "",
  "huggingface_repo_visibility": "",
  "huggingface_token": "",
  "ip_noise_gamma": 0,
  "ip_noise_gamma_random_strength": false,
  "keep_tokens": 0,
  "learning_rate": 4e-05,
  "log_tracker_config": "",
  "log_tracker_name": "",
  "log_with": "",
  "loss_type": "l2",
  "lr_scheduler": "constant",
  "lr_scheduler_args": "",
  "lr_scheduler_num_cycles": 1,
  "lr_scheduler_power": 1,
  "lr_warmup": 0,
  "main_process_port": 0,
  "masked_loss": false,
  "max_bucket_reso": 3048,
  "max_data_loader_n_workers": 0,
  "max_grad_norm": 1,
  "max_resolution": "1024,1024",
  "max_timestep": 1000,
  "max_token_length": 75,
  "max_train_epochs": 0,
  "max_train_steps": 0,
  "mem_eff_attn": false,
  "metadata_author": "",
  "metadata_description": "",
  "metadata_license": "",
  "metadata_tags": "",
  "metadata_title": "",
  "mid_lr_weight": "",
  "min_bucket_reso": 256,
  "min_snr_gamma": 5,
  "min_timestep": 0,
  "mixed_precision": "fp16",
  "model_list": "custom",
  "module_dropout": 0,
  "multi_gpu": false,
  "multires_noise_discount": 0,
  "multires_noise_iterations": 0,
  "network_alpha": 1,
  "network_dim": 256,
  "network_dropout": 0,
  "network_weights": "",
  "noise_offset": 0,
  "noise_offset_random_strength": false,
  "noise_offset_type": "Original",
  "num_cpu_threads_per_process": 2,
  "num_machines": 1,
  "num_processes": 1,
  "optimizer": "Adafactor",
  "optimizer_args": "scale_parameter=False relative_step=False warmup_init=False",
  "persistent_data_loader_workers": false,
  "pretrained_model_name_or_path": "/workspace/kohya_ss/models/sd_xl_base_1.0.safetensors",
  "prior_loss_weight": 1,
  "random_crop": false,
  "rank_dropout": 0,
  "rank_dropout_scale": false,
  "reg_data_dir": "",
  "rescaled": false,
  "resume": "",
  "resume_from_huggingface": "",
  "sample_every_n_epochs": 0,
  "sample_every_n_steps": 0,
  "sample_prompts": "",
  "sample_sampler": "euler_a",
  "save_as_bool": false,
  "save_every_n_epochs": 1,
  "save_every_n_steps": 0,
  "save_last_n_steps": 0,
  "save_last_n_steps_state": 0,
  "save_model_as": "safetensors",
  "save_precision": "bf16",
  "save_state": false,
  "save_state_on_train_end": false,
  "save_state_to_huggingface": false,
  "scale_v_pred_loss_like_noise_pred": false,
  "scale_weight_norms": 0,
  "sdxl": true,
  "sdxl_cache_text_encoder_outputs": false,
  "sdxl_no_half_vae": true,
  "seed": 0,
  "shuffle_caption": false,
  "stop_text_encoder_training": 0,
  "text_encoder_lr": 4e-05,
  "train_batch_size": 1,
  "train_norm": false,
  "train_on_input": true,
  "training_comment": "",
  "unet_lr": 4e-05,
  "unit": 1,
  "up_lr_weight": "",
  "use_cp": false,
  "use_scalar": false,
  "use_tucker": false,
  "v2": false,
  "v_parameterization": false,
  "v_pred_like_loss": 0,
  "vae": "",
  "vae_batch_size": 0,
  "wandb_api_key": "",
  "wandb_run_name": "",
  "weighted_captions": false,
  "xformers": "xformers"
}
EOF
}

start_kohya_ss() {
    echo "--> Launching Kohya_SS GUI..."
    ./gui.sh --listen 0.0.0.0 --server_port 7860 --headless --config "$KOHYA_SS_DIR/config.json"
}

# --- Main Execution ---

provisioning_print_header

install_apt_packages
clone_kohya_ss
install_pip_packages
download_models
create_config_file
start_kohya_ss

echo "--> Provisioning complete. Kohya_SS is running."