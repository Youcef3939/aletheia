import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
from config import OUTPUTS_DIR, VIZ_CONFIG
from utils.logger import get_logger

logger = get_logger("viz")


def normalize_tensor(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu().numpy() # type: ignore
    tensor = tensor - tensor.min()
    tensor = tensor / (tensor.max() + 1e-8)
    return tensor # type: ignore


def tensor_to_heatmap(attribution: torch.Tensor, colormap: str = VIZ_CONFIG["colormap"]) -> np.ndarray:
    norm_attr = normalize_tensor(attribution)
    cmap = cm.get_cmap(colormap)
    heatmap = cmap(norm_attr)[:, :, :3]  
    heatmap = (heatmap * 255).astype(np.uint8)
    return heatmap


def overlay_heatmap(
    original_image: Image.Image,
    attribution: torch.Tensor,
    colormap: str = VIZ_CONFIG["colormap"],
    alpha: float = VIZ_CONFIG["alpha"],
    save_path: str = None # type: ignore
) -> Image.Image:
    try:
        original = original_image.convert("RGB")
        heatmap_arr = tensor_to_heatmap(attribution, colormap)
        heatmap_img = Image.fromarray(heatmap_arr).resize(original.size)

        overlayed = Image.blend(original, heatmap_img, alpha=alpha)

        if save_path is None:
            save_path = os.path.join(OUTPUTS_DIR, "overlay.png")
        overlayed.save(save_path)
        logger.info(f"Heatmap overlay saved to {save_path}")

        return overlayed

    except Exception as e:
        logger.error(f"Error in overlay_heatmap: {e}")
        raise

if __name__ == "__main__":
    from PIL import Image