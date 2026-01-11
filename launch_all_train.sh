#!/bin/bash
accelerate launch sdxl_train_network.py --dataset_config="dataset1.toml" --config_file="config1.toml"  --flow_model  --flow_use_ot  --flow_timestep_distribution logit_normal --flow_uniform_static_ratio 3 --vae_batch_size 6 --flow_logit_mean -0.2  --flow_logit_std 1.5 --use_zero_cond=False --ddp_timeout 3600 --network_train_unet_only
