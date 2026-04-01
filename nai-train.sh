#!/bin/bash
accelerate launch train_sd15.py --dataset_config="dataset_sd15.toml" --config_file="config_sd15.toml"  --flow_model  --flow_use_ot  --flow_timestep_distribution logit_normal --flow_uniform_static_ratio 2 --vae_batch_size 8 --prefetch_factor 2 --ddp_timeout 3600  --flow_logit_mean -0.2  --flow_logit_std 1.5 --fused_optimizer_groups 10 --skip_existing
echo "All training jobs finished. Press any key to close..." 
read -n 1 -s -r -p ""
