import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from PIL import Image
import sys
import os


def cosine_schedule_zsnr(num_timesteps=1000):
    """Cosine schedule with zero terminal SNR (v-pred compatible)."""
    steps = np.arange(num_timesteps + 1, dtype=np.float64) / num_timesteps
    f_t = np.cos((steps + 0.008) / 1.008 * np.pi / 2) ** 2
    alphas_cumprod = f_t / f_t[0]
    alphas_cumprod[-1] = 0.0  # zero terminal SNR
    return alphas_cumprod[1:]  # indices 1..num_timesteps


def main():
    if len(sys.argv) < 2:
        print("Usage: python noise_visualizer.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        sys.exit(1)

    # Load and normalize image to [0, 1]
    img = Image.open(image_path).convert("RGB")
    img = img.resize((512, 512), Image.LANCZOS)
    x_0 = np.array(img, dtype=np.float32) / 255.0

    # Fixed noise for consistency
    np.random.seed(42)
    noise = np.random.randn(*x_0.shape).astype(np.float32)

    num_timesteps = 1000
    alphas_cumprod = cosine_schedule_zsnr(num_timesteps)

    # --- Noisy image generators ---
    def ddpm_noisy(t):
        a = alphas_cumprod[t]
        return np.clip(np.sqrt(a) * x_0 + np.sqrt(1 - a) * noise, 0, 1)

    def ddpm_vpred_target(t):
        a = alphas_cumprod[t]
        v = np.sqrt(a) * noise - np.sqrt(1 - a) * x_0
        return (v - v.min()) / (v.max() - v.min() + 1e-8)

    def flow_noisy(t):
        s = t / (num_timesteps - 1)
        return np.clip((1 - s) * x_0 + s * noise, 0, 1)

    # Flow velocity is constant: v = noise - x_0
    flow_vel = noise - x_0
    flow_vel_vis = (flow_vel - flow_vel.min()) / (flow_vel.max() - flow_vel.min() + 1e-8)

    # --- Build figure ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.subplots_adjust(bottom=0.14, top=0.90, hspace=0.25)

    # Row 0: original | DDPM x_t | Flow x_t
    axes[0, 0].imshow(x_0);           axes[0, 0].set_title("Original");           axes[0, 0].axis("off")
    im_ddpm = axes[0, 1].imshow(ddpm_noisy(0)); axes[0, 1].set_title("DDPM zSNR  x_t"); axes[0, 1].axis("off")
    im_flow = axes[0, 2].imshow(flow_noisy(0)); axes[0, 2].set_title("Rectified Flow  x_t"); axes[0, 2].axis("off")

    # Row 1: noise | DDPM v-pred target | Flow velocity target (constant)
    noise_vis = np.clip((noise - noise.min()) / (noise.max() - noise.min()), 0, 1)
    axes[1, 0].imshow(noise_vis);     axes[1, 0].set_title("Noise");              axes[1, 0].axis("off")
    im_vp = axes[1, 1].imshow(ddpm_vpred_target(0)); axes[1, 1].set_title("DDPM v-pred target"); axes[1, 1].axis("off")
    axes[1, 2].imshow(flow_vel_vis);  axes[1, 2].set_title("RF velocity (constant: noise - x\u2080)"); axes[1, 2].axis("off")

    info = fig.text(0.5, 0.93, "", ha="center", fontsize=11, family="monospace")

    # Slider
    ax_sl = fig.add_axes([0.15, 0.05, 0.70, 0.03])
    slider = Slider(ax_sl, "Timestep", 0, num_timesteps - 1, valinit=0, valstep=1)

    def update(val):
        t = int(slider.val)
        a = alphas_cumprod[t]
        s = t / (num_timesteps - 1)

        snr_ddpm = a / (1 - a) if a < 1 else float("inf")
        snr_flow = ((1 - s) / max(s, 1e-9)) ** 2

        im_ddpm.set_data(ddpm_noisy(t))
        im_flow.set_data(flow_noisy(t))
        im_vp.set_data(ddpm_vpred_target(t))

        axes[0, 1].set_title(f"DDPM zSNR  x_t  (t={t})")
        axes[0, 2].set_title(f"Rectified Flow  x_t  (t={t})")
        axes[1, 1].set_title(f"DDPM v-pred target  (t={t})")

        info.set_text(
            f"t={t:4d}/999   "
            f"DDPM: \u03B1\u0304={a:.4f}  SNR={snr_ddpm:8.2f}      "
            f"Flow: t={s:.4f}  SNR={snr_flow:8.2f}"
        )
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(0)

    fig.suptitle("DDPM zSNR v-pred  vs  Rectified Flow", fontsize=14)
    plt.show()


if __name__ == "__main__":
    main()
