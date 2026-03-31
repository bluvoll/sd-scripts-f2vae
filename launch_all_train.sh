#!/bin/bash
accelerate launch sdxl_train.py --dataset_config="dataset1.toml" --config_file="config1.toml"  --skip_existing --flow_model  --flow_use_ot  --flow_timestep_distribution logit_normal --flow_uniform_static_ratio 3.5 --vae_batch_size 5 --skip_existing --prefetch_factor 2 --use_zero_cond=True --vae_type flux2 --latent_channels 32 --vae_custom_scale 0.6043 --vae_custom_shift 0.0760 --ddp_timeout 3600  --flow_logit_mean -0.2  --flow_logit_std 1.5 --deepspeed --fused_optimizer_groups 10
echo "All training jobs finished. Press any key to close..." 
read -n 1 -s -r -p ""
