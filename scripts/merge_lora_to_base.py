import os
import argparse
import torch
import safetensors.torch

import openpi.models.pi0_config as pi0_config
import openpi.models_pytorch.pi0_pytorch as pi0_pytorch

from peft import LoraConfig, TaskType, get_peft_model


def _get_submodule(root: torch.nn.Module, path: str) -> torch.nn.Module:
    m = root
    for p in path.split("."):
        m = getattr(m, p)
    return m


def _set_submodule(root: torch.nn.Module, path: str, new: torch.nn.Module) -> None:
    parts = path.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new)


def apply_lora_to_hf_backbone(pi_model: torch.nn.Module, r=16, alpha=32, dropout=0.05) -> str:
    backbone_path = None
    for name, mod in pi_model.named_modules():
        if name and hasattr(mod, "prepare_inputs_for_generation"):
            backbone_path = name
            break
    if backbone_path is None:
        for cand in ["paligemma", "language_model", "llm", "backbone", "transformer", "model"]:
            if hasattr(pi_model, cand):
                backbone_path = cand
                break
    if backbone_path is None:
        raise RuntimeError("Could not find HF-like backbone module to LoRA-wrap.")

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=target_modules,
    )
    backbone = _get_submodule(pi_model, backbone_path)
    backbone = get_peft_model(backbone, lora_cfg)
    _set_submodule(pi_model, backbone_path, backbone)
    return backbone_path


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_pytorch_dir", required=True)      # contains model.safetensors
    ap.add_argument("--lora_step_dir", required=True)         # contains lora_adapter/
    ap.add_argument("--out_dir", required=True)               # where to write merged model.safetensors
    ap.add_argument("--action_dim", type=int, default=32)
    ap.add_argument("--action_horizon", type=int, default=10)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = pi0_config.Pi0Config(pi05=True, action_dim=args.action_dim, action_horizon=args.action_horizon)
    model = pi0_pytorch.PI0Pytorch(cfg).to(device)

    # load base
    base_ckpt = os.path.join(args.base_pytorch_dir, "model.safetensors")
    safetensors.torch.load_model(model, base_ckpt, device=str(device))
    print("[info] loaded base:", base_ckpt)

    # apply lora + load adapter
    backbone_path = apply_lora_to_hf_backbone(model)
    backbone = _get_submodule(model, backbone_path)
    adapter_dir = os.path.join(args.lora_step_dir, "lora_adapter")
    backbone.load_adapter(adapter_dir, adapter_name="default")
    backbone.set_adapter("default")
    print("[info] loaded lora adapter:", adapter_dir)
    print("[info] backbone:", backbone_path)

    # merge LoRA into weights (PEFT method)
    merged_backbone = backbone.merge_and_unload()   # returns a plain HF module w/ merged weights
    _set_submodule(model, backbone_path, merged_backbone)
    print("[info] merged lora into base weights")

    # save merged full model weights
    out_path = os.path.join(args.out_dir, "model.safetensors")
    safetensors.torch.save_model(model, out_path)
    print("[saved]", out_path)

    # copy config.json if you want (optional)
    cfg_src = os.path.join(args.base_pytorch_dir, "config.json")
    if os.path.exists(cfg_src):
        import shutil
        shutil.copy2(cfg_src, os.path.join(args.out_dir, "config.json"))
        print("[saved]", os.path.join(args.out_dir, "config.json"))


if __name__ == "__main__":
    main()