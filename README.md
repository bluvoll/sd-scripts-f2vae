### What for?

SDXL-based models ([NoobAI-XL](https://civitai.com/models/833294?modelVersionId=1190596)) converted to Rectified flow / RF and more:
- Mugen | RF + Flux 2 VAE conversion – [CivitAI](https://civitai.com/models/2237480) / [🤗-Link](https://huggingface.co/CabalResearch/Mugen)
  - no support on Machina's sd-scripts because no Flux2 VAE support (yet?) 
- ChenkinRF | RF only, default SDXL VAE – [CivitAI](https://civitai.com/models/2363696/chenkinnoob-xl-rectified-flow) / [🤗-Link](https://huggingface.co/ChenkinRF/ChenkinNoob-XL-v0.3-Rectified-Flow)
  - compatible with Machina's sd-scripts and other trainers supporting RF objective for SDXL
- Shitty experiment | RF + Anzhc's EQ-VAE – [CivitAI](https://civitai.com/models/2071356/experimental-noobai-with-rectified-flow-eq-vae)
  - compatible with Machina's sd-scripts and other trainers supporting RF objective for SDXL as well as VAE reflection (optional but will improve image border edges)

## Hard requirements
- torch_linear_assigment or SciPy
- A working venv, you can take the one from LoRa_Easy_Training_Scripts/backend/sd-scripts/venv and install whatever is missing.

## Additional Commands

- `--flow_model` Required to train into Rectified Flow target  
- `--flow_use_ot` ot = Optimal Transport; Not needed but use anyway, want info? ask lodestone  
- `--flow_timestep_distribution=uniform|logit_normal`
  - In any case either should work 
  - Mugen: logit_normal suggested
  - ChenkinRF: used a mix of logit_normal (early training) and uniform, logit_normal suggested
  - Shitty experiment: was trained with uniform.

- `--flow_logit_mean` needed by logit_normal timestep_distribution
  - Mugen (and others probably): **-0.2**
- `--flow_logit_std` needed by logit_normal timestep_distribution
  - Mugen (and others probably): **1.5**

- Shift, choose one
  - `--flow_uniform_static_ratio` allows static values for shift, default 2.5
    - rough suggested values:
      - **~6.0** – Mugen 
      - **2.5** – ChenkinRF
      - **~2.5** – Shitty Experiment
  - `--flow_uniform_base_pixels` default 1048576 (1024x1024), allows dynamic shifting for >1024 training  
  - `--flow_uniform_shift` allows dynamic shifting

---

- `--vae_type=flux2` Required for Mugen; Self explanatory, can work with Flux 1 too  
- `--latent_channels=32` Required for Mugen; Self explanatory, use 16 for Flux 1 if you into that  

- `--vae_custom_scale` Mugen / Flux 2 VAE: `0.6043` | Anzhc eq-vae: `0.1406`  
- `--vae_custom_shift` Mugen / Flux 2 VAE: `0.0760` | Anzhc eq-vae: `-0.4743`

- `--vae_reflection_padding` suggested for Anzhc's eq-vae, my shitty experiment wasn't trained with this   

#### Misc options
- `--contrastive_flow_matching` Magic thingie that makes training results a bit sharper | Mugen didn't use this  
  - `--cfm_lambda` needed by the one above, default 0.05, I prefer 0.02  

- `--skip_existing` skips latents check, avoids wasting hours checking ++10M latents
- ~~`--use_sga` attaches Lodestone's stochastic accumulator for gradient accumulation;~~ currently borked  

You can use launch_all_train.sh to start training, or copy the command to an activated venv, works on windows or linux w/e the file itself has a working template

## Info on the shitty test

(For some info on Mugen, see the readme in the HuggingFace repo)

Trained with --flow_use_ot --flow_timestep_distribution uniform --flow_uniform_static_ratio 2.5 AdamW8bit Kahan Summation at 2e-5, caption and tag dropout of 0.1 with keep token separator based on NoobAI's own dataset, for one combined epoch of 486k unique images, scheduler was constant with warmup with 10% warmup.

### Dataset

Uses Deepghs' danbooru webp from 0000 to 0065 with the *exact same* tags as NoobAI's own run extra files not present in NoobAI's dataset were purged.

### Why tho?

Just sharing the codebase used to train the Rectified Flow model, so both people that know can scrutinize it, and people that *think* they do can shit on it.

## How to train?

Activate venv, install missing shit, edit config and dataset toml as per your needs run launch_all_training.sh, or run accelerate manually, simple as that.
