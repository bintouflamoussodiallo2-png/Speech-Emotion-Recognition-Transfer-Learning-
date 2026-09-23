import torch 
import torch.nn as nn
import timm 

SUPPORTED_MODELS = ["vgg16", "resnet18", "resnet34", "resnet50"]

def get_model(model_name: str, num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    """
    for flexibility
    freeze_backbone : if True, only the head is trained (transfer learning)
                        if False, all weights are updated (fine-tuning)
    """

    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown model; {model_name}. Choose from: {SUPPORTED_MODELS}")

    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=num_classes
    )

    if freeze_backbone:
        _freeze_backbone(model)

    return model

def _freeze_backbone(model: nn.Module) -> None:

    # common head layer name 
    heads = ["head", "classifier", "fc"]

    for name, params in model.named_parameters():
        is_head = any(keyword in name for keyword in heads)
        params.requires_grad = is_head

        # The parameters trainable for example compare freeze=false vs true
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())

        print(f"Trainable parameters: {trainable:,} / {total:,}"
              f"({100 * trainable / total:.1f}%)")
    