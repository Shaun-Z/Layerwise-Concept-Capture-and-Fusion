from typing import List, Optional
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from open_clip import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

def _to_pil(image: torch.Tensor | Image.Image, target_wh: tuple[int, int]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.resize(target_wh)
    assert image.ndim == 3  # [channel, height, width]
    image_denormed = (image.detach().cpu() * torch.tensor(OPENAI_DATASET_STD)[:, None, None]) \
                    + torch.tensor(OPENAI_DATASET_MEAN)[:, None, None]
    arr = (image_denormed.permute(1, 2, 0).numpy() * 255).astype("uint8")
    return Image.fromarray(arr).resize(target_wh)

def visualize(
    images: torch.Tensor | Image.Image,
    heatmaps: torch.Tensor,
    alpha: float = 0.7,
    text_prompts: Optional[List[str]] = None,
    save_dir: Optional[Path] = None,
    title: Optional[str] = None,
):
    """
    Overlay heatmaps on the input image.
    Args:
        image: PIL or normalized tensor [C,H,W] or [1,C,H,W]
        heatmaps: [N, H, W] or [1,N,H,W] tensor in [0,1]
        alpha: overlay strength
        text_prompts: optional titles per heatmap
        save_dir: optional directory to save pngs
        title: optional title for the original image
    """
    assert heatmaps.ndim == 4   # [batch_size, num_concepts, H, W]
    H, W = heatmaps.shape[-2:]
    num_images = heatmaps.shape[0]
    num_concepts = heatmaps.shape[1]

    if text_prompts is None:
        text_prompts = [str(i) for i in range(num_concepts)]
    pil_imgs = [_to_pil(img, (H, W)) for img in images]
    img_cvs = [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) for img in pil_imgs]
    heatmaps_np = (heatmaps.detach().cpu().numpy() * 255).astype("uint8")   # [N, num_concepts, 1, H, W]
    # heat_maps = [cv2.applyColorMap(hm, cv2.COLORMAP_JET) for hm in heatmaps_np]
    # overlays = [(1 - alpha) * img_cv + alpha * hm for hm in heat_maps]
    fig, axes = plt.subplots(num_images,
                             1 + heatmaps_np.shape[1],
                             figsize=(4 * (1 + heatmaps_np.shape[1]), 4),
                             squeeze=False)
    axes = np.atleast_1d(axes)
    for i, (pil_img, img_cv) in enumerate(zip(pil_imgs, img_cvs)):
        axes[i, 0].imshow(pil_img)
        axes[i, 0].axis("off")
        # if title:
        #     axes[i, 0].set_title(title)

        for j, hm in enumerate(heatmaps_np[i]):
            hm = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
            overlay = (1 - alpha) * img_cv + alpha * hm
            ov_rgb = cv2.cvtColor(overlay.astype("uint8"), cv2.COLOR_BGR2RGB)
            axes[i, j + 1].imshow(ov_rgb)
            axes[i, j + 1].axis("off")
            if j < len(text_prompts):
                axes[i, j + 1].set_title(str(text_prompts[j]))
    plt.tight_layout()
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"heatmap_{title}.png"
        # Image.fromarray(ov_rgb.astype("uint8")).save(out_path)
        plt.savefig(out_path)
        plt.close()
    return fig