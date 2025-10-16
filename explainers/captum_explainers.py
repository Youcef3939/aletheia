import torch
from captum.attr import IntegratedGradients, LayerGradCam, DeepLift
from config import DEFAULT_EXPLAINER
from utils.logger import get_logger

logger = get_logger("explainers")


def generate_explanation(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    explainer: str = DEFAULT_EXPLAINER,
    target: int = None, # type: ignore
    target_layer=None
) -> torch.Tensor:
    model.eval()
    input_tensor = input_tensor.clone().detach()
    input_tensor.requires_grad = True

    try:
        if target is None:
            with torch.no_grad():
                outputs = model(input_tensor)
                target = torch.argmax(outputs, dim=1).item()
            logger.info(f"no target provided. Using predicted class {target}")

        explainer = explainer.lower()
        if explainer == "integrated_gradients":
            ig = IntegratedGradients(model)
            attr = ig.attribute(input_tensor, target=target, n_steps=50)
        elif explainer == "deeplift":
            dl = DeepLift(model)
            attr = dl.attribute(input_tensor, target=target)
        elif explainer == "gradcam":
            if target_layer is None:
                for name, module in reversed(list(model.named_modules())):
                    if isinstance(module, torch.nn.Conv2d):
                        target_layer = module
                        break
            if target_layer is None:
                raise ValueError("GradCAM requires a target convolutional layer")
            lgc = LayerGradCam(model, target_layer)
            attr = lgc.attribute(input_tensor, target=target)
            attr = torch.nn.functional.interpolate(attr, size=input_tensor.shape[2:], mode='bilinear', align_corners=False) # type: ignore
        else:
            raise ValueError(f"Explainer '{explainer}' not supported. Choose from IntegratedGradients, DeepLift, GradCAM")

        if attr.ndim == 4:
            attr = attr.squeeze(0).sum(dim=0)  
        logger.info(f"Attribution generated using {explainer} for target class {target}")
        return attr.detach().cpu()

    except Exception as e:
        logger.error(f"Failed to generate explanation: {e}")
        raise


if __name__ == "__main__":
    import torchvision.models as models
    import torchvision.transforms as transforms
    from PIL import Image
    import torch

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.eval()
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img = Image.new("RGB", (224, 224), color="gray")
    input_tensor = preprocess(img).unsqueeze(0) # type: ignore

    attr = generate_explanation(model, input_tensor, explainer="integrated_gradients")
    print("Attribution map shape:", attr.shape)