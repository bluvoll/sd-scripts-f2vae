

### What for?
A shitty experiment of Rectified Flow for SDXL you can find it here in [Civitai] (https://civitai.com/models/2071356/experimental-noobai-with-rectified-flow-eq-vae)

## Hard requirements
- Torch_linear_assigment or SciPy
- A working venv, you can take the one from LoRa_Easy_Training_Scripts/backend/sd-scripts/venv and install whatever is missing.

## Additional Commands

- --flow_model Required to train into Rectified Flow target 
- --flow_use_ot Not needed, want info? ask lodestone
- --flow_timestep_distribution = uniform or logit_normal, just run uniform
- --flow_uniform_static_ratio allows static values for shift, default 2.5
- --contrastive_flow_matching Magic thingie that makes training results a bit sharper
- --cfm_lambda needed by the one above, default 0.05, I prefer 0.02
- --flow_logit_mean needed by logit_normal timestep_distribution
- --flow_logit_std needed by logit_normal timestep_distribution
- --flow_uniform_base_pixels default 1048576 or 1024x1024 allows dynamic shifting based on resolutions useful for higher than 1024x1024 training.
- --flow_uniform_shift allows dynamic shifting
- --vae_custom_scale suggested for Anzhc's eq-vae put 0.1406
- --vae_custom_shift suggested for Anzhc's eq-vae put -0.4743
- --vae_reflection_padding suggested to use with Anzhc's eq-vae, my shitty experiment wasn't trained with this.
- --use_sga attaches Lodestone's stochastic accumulator for gradient accumulation, currently borked.
- --vae_type flux2 Self Explanatory, can work with Flux 1 too
- --latent_channels 32 Self Explanatory, use 16 for Flux 1 if you into that.
- --skip_existing  Skips latents check, imagine checking ++10 Million latents when it can take a couple hours and time is money.
- --vae_custom_scale 0.6043 Flux 2 VAE
- --vae_custom_shift 0.0760 Flux 2 VAE

You can use launch_all_train.sh to start training, or copy the command to an activated venv, works on windows or linux w/e the file itself has a working template

## Info on the shitty test

Trained with --flow_use_ot --flow_timestep_distribution uniform --flow_uniform_static_ratio 2.5 AdamW8bit Kahan Summation at 2e-5, caption and tag dropout of 0.1 with keep token separator based on NoobAI's own dataset, for one combined epoch of 486k unique images, scheduler was constant with warmup with 10% warmup.

## Dataset

Uses Deepghs' danbooru webp from 0000 to 0065 with the *exact same* tags as NoobAI's own run extra files not present in NoobAI's dataset were purged.

## Why tho?

Just sharing the codebase used to train the Rectified Flow model, so both people that know can scrutinize it, and people that *think* they do can shit on it.

## How to train?

Activate venv, install missing shit, edit config and dataset toml as per your needs run launch_all_training.sh, or run accelerate manually, simple as that.
