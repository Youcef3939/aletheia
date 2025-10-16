import torch
from utils.logger import get_logger

logger = get_logger("metrics")


def faithfulness_deletion(model, input_tensor, attr_map, target=None, steps=10):
    model.eval()
    input_clone = input_tensor.clone().detach()
    h, w = attr_map.shape
    flattened = attr_map.flatten()
    sorted_idx = torch.argsort(flattened, descending=True)  
    drop_values = []

    with torch.no_grad():
        for i in range(1, steps + 1):
            mask = torch.ones(h * w, device=flattened.device)
            top_k = int(i * len(flattened) / steps)
            mask[sorted_idx[:top_k]] = 0
            mask = mask.view(1, 1, h, w)
            masked_input = input_clone * mask  
            outputs = model(masked_input)
            prob = torch.softmax(outputs, dim=1)[0, target if target is not None else outputs.argmax()]
            drop_values.append(prob.item())

    faithfulness_score = 1.0 - sum(drop_values) / len(drop_values)
    logger.info(f"Faithfulness score: {faithfulness_score:.4f}")
    return faithfulness_score


def sensitivity(attr_map_original, attr_map_perturbed):
    diff = torch.abs(attr_map_original - attr_map_perturbed).mean().item()
    logger.info(f"Sensitivity (mean L1 difference): {diff:.6f}")
    return diff

if __name__ == "__main__":
    import torch
    dummy_attr = torch.rand((224, 224))
    dummy_attr_perturbed = dummy_attr + 0.01 * torch.randn((224, 224))
    sens = sensitivity(dummy_attr, dummy_attr_perturbed)
    print("Sensitivity:", sens)