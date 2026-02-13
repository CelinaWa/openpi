import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    """Ensure uint8 HWC."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:  # CHW -> HWC
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class DroneToTableInputs(transforms.DataTransformFn):
    """
    Drone policy input mapping (LIBERO-style keys).

    Expects keys (after repack):
      - "observation/state"        (3,)
      - "observation/image"        (H,W,3) or (3,H,W)  [room_view]
      - "observation/wrist_image"  (H,W,3) or (3,H,W)  [drone_front]
      - "prompt"                  (string)
      - optional: "actions"       (3,) during training
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # Only mask padding images for PI0_FAST; match LIBERO logic
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # training only
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)

        # prompt (language instruction)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class DroneToTableOutputs(transforms.DataTransformFn):
    """Inference output mapping: return only xyz (3 dims)."""

    action_dim: int = 3

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        return {"actions": actions[:, : self.action_dim]}
