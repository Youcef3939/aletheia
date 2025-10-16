import os
from PIL import Image
import torch

from config import DEFAULT_MODEL, DEFAULT_EXPLAINER, DATA_DIR, OUTPUTS_DIR, DEVICE
from utils.logger import get_logger
from utils.viz import overlay_heatmap
from models.model_loader import load_model, infer
from explainers.captum_explainers import generate_explanation
from metrics.xai_metrics import faithfulness_deletion, sensitivity

logger = get_logger("pipeline")


def process_image(image_path: str, model, preprocess, explainer: str = DEFAULT_EXPLAINER):
    try:
        image = Image.open(image_path).convert("RGB")
        input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

        predicted_class, confidence = infer(model, preprocess, image)

        attr_map = generate_explanation(model, input_tensor, explainer=explainer, target=predicted_class) # type: ignore

        filename = os.path.basename(image_path)
        overlay_path = os.path.join(OUTPUTS_DIR, f"overlay_{filename}")
        overlayed_image = overlay_heatmap(image, attr_map, save_path=overlay_path)

        perturb = input_tensor + 0.01 * torch.randn_like(input_tensor)
        attr_map_perturbed = generate_explanation(model, perturb, explainer=explainer, target=predicted_class) # type: ignore

        metrics = {
            "faithfulness": faithfulness_deletion(model, input_tensor, attr_map, target=predicted_class),
            "sensitivity": sensitivity(attr_map, attr_map_perturbed)
        }

        result = {
            "image_path": image_path,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "attribution_map": attr_map,
            "overlay_path": overlay_path,
            "metrics": metrics
        }

        logger.info(f"Pipeline completed for {image_path}")
        return result

    except Exception as e:
        logger.error(f"Pipeline failed for {image_path}: {e}")
        raise


def run_pipeline_on_folder(folder_path: str, model_name: str = DEFAULT_MODEL, explainer: str = DEFAULT_EXPLAINER):
    model, preprocess, _ = load_model(model_name)
    results = []

    for file in os.listdir(folder_path):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(folder_path, file)
            res = process_image(image_path, model, preprocess, explainer)
            results.append(res)

    logger.info(f"pipeline completed for folder: {folder_path}")
    return results

if __name__ == "__main__":
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    results = run_pipeline_on_folder(DATA_DIR)
    for r in results:
        print(f"Image: {r['image_path']}, Predicted class: {r['predicted_class']}, Confidence: {r['confidence']:.4f}")
        print(f"Overlay saved at: {r['overlay_path']}")
        print(f"Metrics: {r['metrics']}")