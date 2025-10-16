from modulefinder import Module
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from config import DEVICE, AVAILABLE_MODELS, DEFAULT_MODEL
from utils.logger import get_logger

logger = get_logger("models")
def make_model_deeplift_compatible(model):
    for name, module in Module.named_modules(): # type: ignore
        if isinstance(module, nn.ReLU) and module.inplace:
            module.inplace = False
        return model

def load_model(model_name: str = DEFAULT_MODEL, pretrained: bool = True):
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model_name} not available. Choose from {list(AVAILABLE_MODELS.keys())}")

    model_info = AVAILABLE_MODELS[model_name]
    input_size = model_info.get("input_size", (224, 224))
    num_classes = model_info.get("num_classes", 1000)

    logger.info(f"Loading model {model_name} (pretrained={pretrained}) on {DEVICE}")

    if model_name.lower() == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    else:
        raise NotImplementedError(f"Model {model_name} is registered but not implemented in loader.")

    if num_classes != 1000:
        if hasattr(model, "fc"):
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        else:
            raise NotImplementedError("Custom classifier replacement not implemented for this architecture")

    model = model.to(DEVICE)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    logger.info(f"Model {model_name} loaded successfully with input size {input_size}")

    return model, preprocess, num_classes


def infer(model: torch.nn.Module, preprocess, image: Image.Image):
    try:
        input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probs, dim=1)
        confidence = confidence.item()
        predicted_class = predicted_class.item()

        logger.info(f"Inference done: class={predicted_class}, confidence={confidence:.4f}")
        return predicted_class, confidence

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise

if __name__ == "__main__":
    from PIL import Image
    test_img = Image.new("RGB", (224, 224), color="gray")
    model, preprocess, num_classes = load_model()
    cls, conf = infer(model, preprocess, test_img)
    print(f"Predicted class: {cls}, Confidence: {conf:.4f}")