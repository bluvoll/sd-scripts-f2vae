#!/bin/bash
accelerate launch sdxl_train.py --dataset_config="caching.toml" --config_file="config-example-LoCON.toml" --flow_model  --flow_use_ot  --flow_timestep_distribution logit_normal --flow_uniform_static_ratio 3 --vae_batch_size 6 --flow_logit_mean -0.2  --flow_logit_std 1.5 --fused_optimizer_groups 7 --use_zero_cond=True --freeze_unet_blocks 4,5,6,7,8,9,10,11,12,13,14,15,16,17,18
echo "All training jobs finished. Press any key to close..."
read -n 1 -s -r -p ""
