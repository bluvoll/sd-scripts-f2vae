#!/bin/bash
accelerate launch sdxl_train.py --dataset_config="dataset1.toml" --config_file="config1.toml" --flow_model  --flow_use_ot  --flow_timestep_distribution logit_normal --flow_uniform_static_ratio 2.5 --vae_batch_size 6 --flow_logit_mean -0.2  --flow_logit_std 1.5 --fused_optimizer_groups 7 --use_zero_cond=True  --skip_existing
echo "All training jobs finished. Press any key to close..."
read -n 1 -s -r -p ""
