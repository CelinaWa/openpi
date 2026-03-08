"""Train pi0/pi0.5 with STL preference pairs using a DPO-style objective.

Expected dataset layout:
  <data_root>/
    <stl_group_0>/
      stl.txt
      sat/*.npz
      unsat/*.npz
    <stl_group_1>/
      stl.txt
      sat/*.npz
      unsat/*.npz

Each .npz demo should contain at least:
  - image: (H, W, 3) or (3, H, W)
  - wrist_image: (H, W, 3) or (3, H, W)
  - state: (state_dim,)
  - actions: (action_horizon, action_dim_raw)
Optional:
  - prompt: str
  - stl_text: str (overrides group-level stl.txt)

This script runs:
  1) SFT warmup on satisfying demos only (optional)
  2) DPO updates on paired (sat > unsat) demos for same STL
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
import dataclasses
import logging
import pathlib
import random
from typing import Any

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.training.config as _config
import openpi.training.optimizer as _optimizer
import openpi.transforms as _transforms


def _init_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


@dataclasses.dataclass(frozen=True)
class DemoPair:
    stl_text: str
    sat_path: pathlib.Path
    unsat_path: pathlib.Path


def _load_npz_dict(path: pathlib.Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def _maybe_scalar_to_str(x: Any) -> str | None:
    if x is None:
        return None
    if isinstance(x, str):
        return x
    arr = np.asarray(x)
    if arr.shape == ():
        return str(arr.item())
    return None


def _discover_pairs(data_root: pathlib.Path) -> list[DemoPair]:
    pairs: list[DemoPair] = []
    for stl_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        sat_dir = stl_dir / "sat"
        unsat_dir = stl_dir / "unsat"
        if not sat_dir.exists() or not unsat_dir.exists():
            continue
        stl_text = ""
        stl_file = stl_dir / "stl.txt"
        if stl_file.exists():
            stl_text = stl_file.read_text().strip()
        sat_files = sorted([p for p in sat_dir.iterdir() if p.suffix == ".npz"])
        unsat_files = sorted([p for p in unsat_dir.iterdir() if p.suffix == ".npz"])
        if not sat_files or not unsat_files:
            continue
        n_pairs = max(len(sat_files), len(unsat_files))
        for i in range(n_pairs):
            pairs.append(
                DemoPair(
                    stl_text=stl_text,
                    sat_path=sat_files[i % len(sat_files)],
                    unsat_path=unsat_files[i % len(unsat_files)],
                )
            )
    if not pairs:
        raise ValueError(f"No sat/unsat STL pairs found under {data_root}")
    return pairs


def _split_pairs(
    pairs: list[DemoPair], *, val_split_ratio: float | None, seed: int
) -> tuple[list[DemoPair], list[DemoPair]]:
    if val_split_ratio is None:
        return pairs, []
    if not (0.0 < val_split_ratio < 1.0):
        raise ValueError(f"val_split_ratio must be in (0,1), got {val_split_ratio}")
    if len(pairs) < 2:
        raise ValueError("Need at least 2 pairs to split train/val.")
    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_split_ratio))
    if n_val >= len(shuffled):
        n_val = len(shuffled) - 1
    return shuffled[n_val:], shuffled[:n_val]


def _build_transform(config: _config.TrainConfig) -> _transforms.DataTransformFn:
    data_config = config.data.create(config.assets_dirs, config.model)
    norm_stats = {}
    if data_config.norm_stats is not None:
        norm_stats = data_config.norm_stats
    return _transforms.compose(
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ]
    )


def _raw_demo_to_input_dict(
    demo: dict[str, Any],
    *,
    stl_text: str,
    default_prompt: str,
) -> dict[str, Any]:
    prompt = _maybe_scalar_to_str(demo.get("prompt")) or default_prompt
    demo_stl = _maybe_scalar_to_str(demo.get("stl_text")) or stl_text
    if not demo_stl:
        raise ValueError("Missing STL text. Provide stl.txt in each STL group or stl_text in demo files.")
    return {
        "observation/image": np.asarray(demo["image"]),
        "observation/wrist_image": np.asarray(demo["wrist_image"]),
        "observation/state": np.asarray(demo["state"], dtype=np.float32),
        "actions": np.asarray(demo["actions"], dtype=np.float32),
        "prompt": prompt,
        "stl_text": demo_stl,
    }


def _stack_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *samples)


class PairBatchIterator:
    def __init__(
        self,
        *,
        pairs: list[DemoPair],
        batch_size: int,
        transform: _transforms.DataTransformFn,
        default_prompt: str,
        seed: int,
    ):
        self._pairs = pairs
        self._batch_size = batch_size
        self._transform = transform
        self._default_prompt = default_prompt
        self._rng = random.Random(seed)

    def __iter__(self) -> Iterator[tuple[_model.Observation[np.ndarray], _model.Actions, _model.Observation[np.ndarray], _model.Actions]]:
        while True:
            batch_pairs = [self._rng.choice(self._pairs) for _ in range(self._batch_size)]
            sat_samples: list[dict[str, Any]] = []
            unsat_samples: list[dict[str, Any]] = []
            for pair in batch_pairs:
                sat_demo = _load_npz_dict(pair.sat_path)
                unsat_demo = _load_npz_dict(pair.unsat_path)
                sat_raw = _raw_demo_to_input_dict(sat_demo, stl_text=pair.stl_text, default_prompt=self._default_prompt)
                unsat_raw = _raw_demo_to_input_dict(
                    unsat_demo, stl_text=pair.stl_text, default_prompt=self._default_prompt
                )
                sat_samples.append(self._transform(sat_raw))
                unsat_samples.append(self._transform(unsat_raw))
            sat_batch = _stack_samples(sat_samples)
            unsat_batch = _stack_samples(unsat_samples)
            sat_obs = _model.Observation.from_dict(sat_batch)
            unsat_obs = _model.Observation.from_dict(unsat_batch)
            yield sat_obs, sat_batch["actions"], unsat_obs, unsat_batch["actions"]


def _load_partial_params(weight_loader, params_shape: at.Params) -> at.Params:
    loaded_params = weight_loader.load(params_shape)
    flat_expected = traverse_util.flatten_dict(params_shape)
    flat_loaded = traverse_util.flatten_dict(loaded_params)
    overlap = flat_expected.keys() & flat_loaded.keys()
    expected_overlap = {k: flat_expected[k] for k in overlap}
    loaded_overlap = {k: flat_loaded[k] for k in overlap}
    at.check_pytree_equality(
        expected=traverse_util.unflatten_dict(expected_overlap),
        got=traverse_util.unflatten_dict(loaded_overlap),
        check_shapes=True,
        check_dtypes=True,
    )
    return traverse_util.unflatten_dict(
        {k: v for k, v in loaded_overlap.items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@dataclasses.dataclass
class SimpleTrainState:
    step: int
    params: nnx.State
    ref_params: nnx.State
    model_def: nnx.GraphDef[_model.BaseModel]
    tx: optax.GradientTransformation
    opt_state: optax.OptState


def _make_state(config: _config.TrainConfig, rng: at.KeyArrayLike) -> SimpleTrainState:
    model = config.model.create(rng)
    graphdef, state = nnx.split(model)
    partial = _load_partial_params(config.weight_loader, state.to_pure_dict())
    state.replace_by_pure_dict(partial)
    params = nnx.state(nnx.merge(graphdef, state))
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)
    opt_state = tx.init(params.filter(config.trainable_filter))
    ref_params = nnx.State.from_pure_dict(params.to_pure_dict())
    return SimpleTrainState(step=0, params=params, ref_params=ref_params, model_def=graphdef, tx=tx, opt_state=opt_state)


def _mean_loss(model: _model.BaseModel, rng: at.KeyArrayLike, obs: _model.Observation, actions: _model.Actions, *, train: bool):
    return jnp.mean(model.compute_loss(rng, obs, actions, train=train))


def _sft_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: SimpleTrainState,
    obs: _model.Observation,
    actions: _model.Actions,
) -> tuple[SimpleTrainState, dict[str, float]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()
    train_rng = jax.random.fold_in(rng, state.step)
    diff_state = nnx.DiffState(0, config.trainable_filter)

    def loss_fn(m):
        return _mean_loss(m, train_rng, obs, actions, train=True)

    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model)
    trainable = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, trainable)
    new_trainable = optax.apply_updates(trainable, updates)
    nnx.update(model, new_trainable)
    new_params = nnx.state(model)
    return (
        dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state),
        {"sft_loss": float(loss)},
    )


def _dpo_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: SimpleTrainState,
    sat_obs: _model.Observation,
    sat_actions: _model.Actions,
    unsat_obs: _model.Observation,
    unsat_actions: _model.Actions,
    *,
    beta: float,
) -> tuple[SimpleTrainState, dict[str, float]]:
    model = nnx.merge(state.model_def, state.params)
    ref_model = nnx.merge(state.model_def, state.ref_params)
    model.train()
    ref_model.eval()
    rng = jax.random.fold_in(rng, state.step)
    rng_c, rng_r, rng_ref_c, rng_ref_r = jax.random.split(rng, 4)
    diff_state = nnx.DiffState(0, config.trainable_filter)

    def dpo_loss_fn(m):
        pi_c = -_mean_loss(m, rng_c, sat_obs, sat_actions, train=True)
        pi_r = -_mean_loss(m, rng_r, unsat_obs, unsat_actions, train=True)
        ref_c = -_mean_loss(ref_model, rng_ref_c, sat_obs, sat_actions, train=False)
        ref_r = -_mean_loss(ref_model, rng_ref_r, unsat_obs, unsat_actions, train=False)
        logits = beta * ((pi_c - pi_r) - jax.lax.stop_gradient(ref_c - ref_r))
        loss = -jax.nn.log_sigmoid(logits)
        return loss, (pi_c, pi_r, ref_c, ref_r, logits)

    (loss, aux), grads = nnx.value_and_grad(dpo_loss_fn, argnums=diff_state, has_aux=True)(model)
    pi_c, pi_r, ref_c, ref_r, logits = aux
    trainable = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, trainable)
    new_trainable = optax.apply_updates(trainable, updates)
    nnx.update(model, new_trainable)
    new_params = nnx.state(model)
    info = {
        "dpo_loss": float(loss),
        "pi_c": float(pi_c),
        "pi_r": float(pi_r),
        "ref_c": float(ref_c),
        "ref_r": float(ref_r),
        "dpo_logits": float(logits),
    }
    return dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state), info


def _evaluate(
    *,
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: SimpleTrainState,
    val_pairs: list[DemoPair],
    transform: _transforms.DataTransformFn,
    default_prompt: str,
    batch_size: int,
    beta: float,
    num_batches: int,
) -> dict[str, float]:
    if not val_pairs:
        return {}
    val_iter = iter(
        PairBatchIterator(
            pairs=val_pairs,
            batch_size=batch_size,
            transform=transform,
            default_prompt=default_prompt,
            seed=0,
        )
    )
    dpo_losses = []
    sat_losses = []
    for _ in range(num_batches):
        sat_obs, sat_actions, unsat_obs, unsat_actions = next(val_iter)
        model = nnx.merge(state.model_def, state.params)
        ref_model = nnx.merge(state.model_def, state.ref_params)
        model.eval()
        ref_model.eval()
        rng, r1, r2, r3, r4, r5 = jax.random.split(rng, 6)
        sat_loss = _mean_loss(model, r1, sat_obs, sat_actions, train=False)
        pi_c = -_mean_loss(model, r2, sat_obs, sat_actions, train=False)
        pi_r = -_mean_loss(model, r3, unsat_obs, unsat_actions, train=False)
        ref_c = -_mean_loss(ref_model, r4, sat_obs, sat_actions, train=False)
        ref_r = -_mean_loss(ref_model, r5, unsat_obs, unsat_actions, train=False)
        logits = beta * ((pi_c - pi_r) - (ref_c - ref_r))
        dpo_loss = -jax.nn.log_sigmoid(logits)
        dpo_losses.append(dpo_loss)
        sat_losses.append(sat_loss)
    return {
        "val_dpo_loss": float(jnp.mean(jnp.stack(dpo_losses))),
        "val_sat_sft_loss": float(jnp.mean(jnp.stack(sat_losses))),
    }


def main() -> None:
    _init_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", type=str, required=True)
    parser.add_argument("--data-root", type=pathlib.Path, required=True)
    parser.add_argument("--num-steps", type=int, default=2000)
    parser.add_argument("--sft-warmup-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--default-prompt", type=str, default="follow the STL constraint")
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--val-split-ratio", type=float, default=None)
    parser.add_argument("--val-interval", type=int, default=None)
    parser.add_argument("--val-num-batches", type=int, default=None)
    args = parser.parse_args()

    config = _config.get_config(args.config_name)
    logging.info("Using config: %s", args.config_name)
    pairs = _discover_pairs(args.data_root)
    val_split_ratio = args.val_split_ratio if args.val_split_ratio is not None else config.val_split_ratio
    train_pairs, val_pairs = _split_pairs(pairs, val_split_ratio=val_split_ratio, seed=args.seed)
    logging.info("Discovered %d STL pairs (%d train / %d val).", len(pairs), len(train_pairs), len(val_pairs))
    transform = _build_transform(config)

    batch_iter = iter(
        PairBatchIterator(
            pairs=train_pairs,
            batch_size=args.batch_size,
            transform=transform,
            default_prompt=args.default_prompt,
            seed=args.seed,
        )
    )

    rng = jax.random.key(args.seed)
    rng, init_rng = jax.random.split(rng)
    state = _make_state(config, init_rng)
    logging.info("Initialized model and optimizer.")

    val_interval = args.val_interval if args.val_interval is not None else config.val_interval
    val_num_batches = args.val_num_batches if args.val_num_batches is not None else (config.val_num_batches or 20)

    for step in range(args.num_steps):
        sat_obs, sat_actions, unsat_obs, unsat_actions = next(batch_iter)
        if step < args.sft_warmup_steps:
            state, info = _sft_step(config, rng, state, sat_obs, sat_actions)
        else:
            state, info = _dpo_step(
                config,
                rng,
                state,
                sat_obs,
                sat_actions,
                unsat_obs,
                unsat_actions,
                beta=args.beta,
            )
        if step % args.log_interval == 0:
            logging.info("step=%d %s", step, " ".join(f"{k}={v:.4f}" for k, v in info.items()))
        if val_pairs and step % val_interval == 0 and step > 0:
            val_info = _evaluate(
                config=config,
                rng=rng,
                state=state,
                val_pairs=val_pairs,
                transform=transform,
                default_prompt=args.default_prompt,
                batch_size=args.batch_size,
                beta=args.beta,
                num_batches=val_num_batches,
            )
            logging.info("step=%d %s", step, " ".join(f"{k}={v:.4f}" for k, v in val_info.items()))
        rng = jax.random.fold_in(rng, step + 1)

    logging.info("Finished DPO training loop for %d steps.", args.num_steps)


if __name__ == "__main__":
    main()