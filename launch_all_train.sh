#!/bin/bash
accelerate launch --num_processes=1 sdxl_train.py --dataset_config="dataset.toml" --config_file="config1.toml"    --use_zero_cond=False --ddp_timeout 3600 --skip_existing --save_state --save_state_on_train_end --flow_model  --flow_use_ot  --flow_timestep_distribution uniform --flow_uniform_static_ratio 2 --vae_batch_size 24 --skip_existing --prefetch_factor 2 
echo "All training jobs finished. Press any key to close..." 
read -n 1 -s -r -p ""
