"""Models from timm, the start weights, and the trainable parts."""
import timm
import torch


def load_weights(model, path):
    """Load a local checkpoint. The head is skipped, because the class count differs.

    The DeiT-III release names the layer scales `gamma_1` and `gamma_2`. timm 1.x names
    them `ls1.gamma` and `ls2.gamma`.
    """
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd = sd.get("model", sd)
    sd = {k.replace(".gamma_1", ".ls1.gamma").replace(".gamma_2", ".ls2.gamma"): v
          for k, v in sd.items() if not k.startswith("head.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected or [k for k in missing if not k.startswith("head.")]:
        raise ValueError(f"weights do not fit: missing {missing}, unexpected {unexpected}")


def set_trainable(model, trainable):
    """`all`, or `attn_head_pos`: the attention layers, the head, and the position embedding,
    as `ViTDDLM` of the public ViT-DD code does for the driver branch."""
    if trainable == "all":
        return
    if trainable != "attn_head_pos":
        raise ValueError(f"unknown trainable parts {trainable!r}")
    for name, p in model.named_parameters():
        p.requires_grad = ".attn." in name or name.startswith("head.") or name == "pos_embed"


def create(cfg):
    model = timm.create_model(cfg["model"], pretrained=False, num_classes=10,
                              drop_rate=cfg.get("drop", 0.0),
                              drop_path_rate=cfg.get("drop_path", 0.0),
                              img_size=cfg.get("input_size", 224))
    if cfg.get("weights"):
        load_weights(model, cfg["weights"])
    set_trainable(model, cfg.get("trainable", "all"))
    return model
