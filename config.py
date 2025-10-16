import os
import torch # type: ignore

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data" , "samples")
SAMPLES_DIR = os.path.join(DATA_DIR, "samples")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
MODELS_DIR = os.path.join(BASE_DIR, "models", "weights")

for path in [OUTPUTS_DIR, LOGS_DIR, MODELS_DIR]:
    os.makedirs(path, exist_ok=True)

DEFAULT_MODEL = "resnet50"

AVAILABLE_MODELS = {
    "resnet50": {
        "weights": "IMAGENET1K_V2",
        "input_size": (224, 224),
        "num_classes": 1000,
    },
    # add more models here if you contribute
}



AVAILABLE_EXPLAINERS = ["gradcam", "integrated_gradients", "deeplift"]

DEFAULT_EXPLAINER = "gradcam"



DEFAULT_METRICS = ["faithfulness", "sensitivity"]

METRIC_CONFIG = {
    "faithfulness_steps": 10,
    "perturbation_strength": 0.2,
}



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

LOG_LEVEL = "INFO"
LOG_FILE = os.path.join(LOGS_DIR, "aletheia.log")



VIZ_CONFIG = {
    "colormap": "jet",
    "alpha": 0.5,
    "output_format": "png",
}



DASHBOARD_CONFIG = {
    "title": "Aletheia — Explainable AI Dashboard",
    "description": "Visualize and understand your model’s decisions",
    "default_model": DEFAULT_MODEL,
    "default_explainer": DEFAULT_EXPLAINER,
}



def summarize():
    print("aletheia configuration summary")
    print("---------------------------------")
    print(f"base directory: {BASE_DIR}")
    print(f"device: {DEVICE}")
    print(f"default model: {DEFAULT_MODEL}")
    print(f"default explainer: {DEFAULT_EXPLAINER}")
    print(f"metrics: {', '.join(DEFAULT_METRICS)}")
    print(f"log file: {LOG_FILE}")
    print(f"visualization: {VIZ_CONFIG['colormap']} (α={VIZ_CONFIG['alpha']})")
    print("---------------------------------")


if __name__ == "__main__":
    summarize()