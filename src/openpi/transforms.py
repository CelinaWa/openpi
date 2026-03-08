from collections.abc import Callable, Mapping, Sequence
import dataclasses
import hashlib
import re
from typing import Protocol, TypeAlias, TypeVar, runtime_checkable

import flax.traverse_util as traverse_util
import jax
import numpy as np
from openpi_client import image_tools

from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize

DataDict: TypeAlias = at.PyTree
NormStats: TypeAlias = _normalize.NormStats


T = TypeVar("T")
S = TypeVar("S")


@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """


@dataclasses.dataclass(frozen=True)
class Group:
    """A group of transforms."""

    # Transforms that are applied to the model input data.
    inputs: Sequence[DataTransformFn] = ()

    # Transforms that are applied to the model output data.
    outputs: Sequence[DataTransformFn] = ()

    def push(self, *, inputs: Sequence[DataTransformFn] = (), outputs: Sequence[DataTransformFn] = ()) -> "Group":
        """Append transforms to the group and return a new group.

        Args:
            inputs: Appended to the *end* of the current input transforms.
            outputs: Appended to the *beginning* of the current output transforms.

        Returns:
            A new group with the appended transforms.
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """Repacks an input dictionary into a new dictionary.

    Repacking is defined using a dictionary where the keys are the new keys and the values
    are the flattened paths to the old keys. We use '/' as the separator during flattening.

    Example:
    {
        "images": {
            "cam_high": "observation.images.top",
            "cam_low": "observation.images.bottom",
        },
        "state": "observation.state",
        "actions": "action",
    }
    """

    structure: at.PyTree[str]

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)
        return jax.tree.map(lambda k: flat_item[k], self.structure)


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
        return data


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        return apply_tree(
            data,
            self.norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )

    def _normalize(self, x, stats: NormStats):
        mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]
        return (x - mean) / (std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        # Make sure that all the keys in the norm stats are present in the data.
        return apply_tree(
            data,
            self.norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=True,
        )

    def _unnormalize(self, x, stats: NormStats):
        mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
        std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
        return x * (std + 1e-6) + mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01, stats.q99
        if (dim := q01.shape[-1]) < x.shape[-1]:
            return np.concatenate([(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]], axis=-1)
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    height: int
    width: int

    def __call__(self, data: DataDict) -> DataDict:
        data["image"] = {k: image_tools.resize_with_pad(v, self.height, self.width) for k, v in data["image"].items()}
        return data


@dataclasses.dataclass(frozen=True)
class SubsampleActions(DataTransformFn):
    stride: int

    def __call__(self, data: DataDict) -> DataDict:
        data["actions"] = data["actions"][:: self.stride]
        return data


@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """Repacks absolute actions into delta action space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteActions(DataTransformFn):
    """Repacks delta actions into absolute action space."""

    # Boolean mask for the action dimensions to be repacked into absolute action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] += np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        if not isinstance(prompt, str):
            prompt = prompt.item()

        tokens, token_masks = self.tokenizer.tokenize(prompt, state)
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}


@dataclasses.dataclass(frozen=True)
class TokenizeFASTInputs(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        state, actions = data["state"], data.get("actions")
        tokens, token_mask, ar_mask, loss_mask = self.tokenizer.tokenize(prompt, state, actions)
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
        }


@dataclasses.dataclass(frozen=True)
class _STLNode:
    kind: str
    children: list["_STLNode"]
    ap_text: str = ""
    ts: float = -1.0
    te: float = -1.0


class _STLParser:
    """Small recursive-descent parser for STL-like expressions."""

    def __init__(self, tokens: list[str]):
        self.tokens = tokens
        self.pos = 0

    def _peek(self) -> str | None:
        if self.pos >= len(self.tokens):
            return None
        return self.tokens[self.pos]

    def _consume(self) -> str:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def _maybe_interval(self) -> tuple[float, float]:
        if self._peek() != "[":
            return -1.0, -1.0
        self._consume()  # '['
        left = self._consume()
        sep = self._consume()
        right = self._consume()
        end = self._consume()
        if sep not in (":", ",") or end != "]":
            raise ValueError("Invalid interval format, expected [ts:te] or [ts,te].")
        return float(left), float(right)

    def parse(self) -> _STLNode:
        node = self._parse_or()
        if self._peek() is not None:
            raise ValueError("Unexpected trailing tokens in STL expression.")
        return node

    def _parse_or(self) -> _STLNode:
        node = self._parse_and()
        while (tok := self._peek()) is not None and tok.upper() in ("|", "||", "OR"):
            self._consume()
            rhs = self._parse_and()
            node = _STLNode(kind="OR", children=[node, rhs])
        return node

    def _parse_and(self) -> _STLNode:
        node = self._parse_until()
        while (tok := self._peek()) is not None and tok.upper() in ("&", "&&", "AND"):
            self._consume()
            rhs = self._parse_until()
            node = _STLNode(kind="AND", children=[node, rhs])
        return node

    def _parse_until(self) -> _STLNode:
        node = self._parse_unary()
        while (tok := self._peek()) is not None and tok.upper() in ("U", "UNTIL"):
            self._consume()
            ts, te = self._maybe_interval()
            rhs = self._parse_unary()
            node = _STLNode(kind="UNTIL", children=[node, rhs], ts=ts, te=te)
        return node

    def _parse_unary(self) -> _STLNode:
        tok = self._peek()
        if tok is None:
            raise ValueError("Unexpected end of STL expression.")
        tok_u = tok.upper()
        if tok_u in ("!", "NOT"):
            self._consume()
            return _STLNode(kind="NOT", children=[self._parse_unary()])
        if tok_u in ("F", "EVENTUALLY"):
            self._consume()
            ts, te = self._maybe_interval()
            return _STLNode(kind="EVENTUALLY", children=[self._parse_unary()], ts=ts, te=te)
        if tok_u in ("G", "ALWAYS"):
            self._consume()
            ts, te = self._maybe_interval()
            return _STLNode(kind="ALWAYS", children=[self._parse_unary()], ts=ts, te=te)
        if tok == "(":
            self._consume()
            node = self._parse_or()
            if self._peek() != ")":
                raise ValueError("Missing ')' in STL expression.")
            self._consume()
            return node
        return self._parse_ap()

    def _parse_ap(self) -> _STLNode:
        pieces: list[str] = []
        paren_depth = 0
        while (tok := self._peek()) is not None:
            tok_u = tok.upper()
            if tok == "(":
                paren_depth += 1
                pieces.append(self._consume())
                continue
            if tok == ")":
                if paren_depth == 0:
                    break
                paren_depth -= 1
                pieces.append(self._consume())
                continue
            if paren_depth == 0 and tok_u in ("&", "&&", "AND", "|", "||", "OR", "U", "UNTIL"):
                break
            pieces.append(self._consume())
        if not pieces:
            raise ValueError("Failed to parse AP node.")
        return _STLNode(kind="AP", children=[], ap_text="".join(pieces))


@dataclasses.dataclass(frozen=True)
class TokenizeSTLText(DataTransformFn):
    """Converts raw STL text into syntax-tree graph nodes (operators + APs)."""

    max_nodes: int = 32
    vocab_size: int = 4096
    input_key: str = "stl_text"
    # If true, parsing failures raise; otherwise falls back to simple token chain.
    strict_parse: bool = False
    semantic_tokenizer: _tokenizer.PaligemmaTokenizer | None = None
    ap_max_tokens: int = 8
    default_stl_text: str | None = None
    use_symbolic_ids: bool = False
    use_object_hash_in_8d: bool = False

    def _normalize_object_name(self, obj_name: str) -> str:
        obj_name = obj_name.strip().lower()
        obj_name = re.sub(r"\s+", " ", obj_name)
        return obj_name

    def _parse_ap_semantics(self, ap_text: str) -> tuple[float, float, str | None, float]:
        """Extract AP type, object id/text, and threshold from canonical APs.

        Supported forms:
          - reach(obj), avoid(obj) -> threshold defaults to 0
          - reach(obj, 5), avoid(obj, 5)
        """
        text = ap_text.strip().lower()

        m = re.match(
            r"^\s*(reach|avoid)\s*\(\s*([^)]+?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)\s*$",
            text,
        )
        threshold = 0.0
        if m is not None:
            ap_word = m.group(1)
            obj_name = self._normalize_object_name(m.group(2))
            threshold = float(m.group(3))
        else:
            m = re.match(r"^\s*(reach|avoid)\s*\(\s*([^)]+?)\s*\)\s*$", text)
            if m is None:
                return 0.0, -1.0, None, 0.0
            ap_word = m.group(1)
            obj_name = self._normalize_object_name(m.group(2))

        ap_type_id = 1.0 if ap_word == "reach" else 2.0
        obj_hash = int(hashlib.sha1(obj_name.encode("utf-8")).hexdigest(), 16)
        obj_id_norm = float(obj_hash % max(1, self.vocab_size)) / float(max(1, self.vocab_size - 1))
        return ap_type_id, obj_id_norm, obj_name, threshold

    def __call__(self, data: DataDict) -> DataDict:
        if "stl_node_mask" in data and "stl_adjacency" in data:
            return data

        stl_text = data.get(self.input_key)
        if stl_text is None:
            if self.default_stl_text is None:
                return data
            stl_text = self.default_stl_text
        if not isinstance(stl_text, str):
            stl_text = stl_text.item()

        token_ids = np.zeros((self.max_nodes,), dtype=np.int32)
        node_mask = np.zeros((self.max_nodes,), dtype=bool)
        adjacency = np.zeros((self.max_nodes, self.max_nodes), dtype=bool)
        node_features = np.zeros((self.max_nodes, 8), dtype=np.float32)
        text_token_ids = np.zeros((self.max_nodes, self.ap_max_tokens), dtype=np.int32)
        text_token_mask = np.zeros((self.max_nodes, self.ap_max_tokens), dtype=bool)

        op_ids = {
            "AND": 1,
            "OR": 2,
            "NOT": 3,
            "EVENTUALLY": 4,
            "ALWAYS": 5,
            "UNTIL": 6,
        }
        op_token_offset = 1
        ap_token_offset = 128

        try:
            tokens = re.findall(
                r"<=|>=|==|!=|\|\||&&|[()\[\],:]|[<>!&|]|[A-Za-z_][A-Za-z0-9_]*|-?\d+(?:\.\d+)?",
                stl_text,
            )
            root = _STLParser(tokens).parse()

            node_records: list[tuple[int, _STLNode, int, str | None, int]] = []
            edges: list[tuple[int, int]] = []

            def walk(node: _STLNode, depth: int, parent_op: str | None, child_idx: int) -> int:
                idx = len(node_records)
                node_records.append((idx, node, depth, parent_op, child_idx))
                for c_i, child in enumerate(node.children):
                    child_idx_ = walk(child, depth + 1, node.kind, c_i)
                    edges.append((idx, child_idx_))  # parent <- child
                return idx

            walk(root, depth=0, parent_op=None, child_idx=0)
            node_count = min(len(node_records), self.max_nodes)

            for i in range(node_count):
                _, node, depth, parent_op, child_idx = node_records[i]
                node_mask[i] = True
                semantic_phrase = None
                if node.kind == "AP":
                    ap_hash = int(hashlib.sha1(node.ap_text.lower().encode("utf-8")).hexdigest(), 16)
                    if self.use_symbolic_ids:
                        token_ids[i] = ap_token_offset + (ap_hash % max(1, self.vocab_size - ap_token_offset))
                    operator_id = 0.0
                    ap_type_id, obj_id_norm, obj_name, threshold = self._parse_ap_semantics(node.ap_text)
                    if not self.use_object_hash_in_8d:
                        obj_id_norm = -1.0
                    is_ap = 1.0
                    # Semantic token branch uses object identity only.
                    semantic_phrase = obj_name if obj_name is not None else None
                else:
                    operator_id = float(op_ids[node.kind])
                    if self.use_symbolic_ids:
                        token_ids[i] = op_token_offset + int(operator_id)
                    ap_type_id = 0.0
                    obj_id_norm = -1.0
                    is_ap = 0.0
                    threshold = 0.0
                is_left_until = float(parent_op == "UNTIL" and child_idx == 0)
                # Use the reserved channel for AP distance threshold when present.
                reserved = float(threshold)
                node_features[i] = np.asarray(
                    [
                        operator_id,
                        float(node.ts),
                        float(node.te),
                        ap_type_id,
                        obj_id_norm,
                        is_ap,
                        is_left_until,
                        reserved,
                    ],
                    dtype=np.float32,
                )
                if self.semantic_tokenizer is not None and semantic_phrase:
                    ap_tokens = self.semantic_tokenizer._tokenizer.encode(semantic_phrase, add_bos=False, add_eos=False)
                    ap_tokens = ap_tokens[: self.ap_max_tokens]
                    text_token_ids[i, : len(ap_tokens)] = np.asarray(ap_tokens, dtype=np.int32)
                    text_token_mask[i, : len(ap_tokens)] = True

            for parent, child in edges:
                if parent < self.max_nodes and child < self.max_nodes:
                    adjacency[parent, child] = True

        except Exception:
            if self.strict_parse:
                raise
            # Fallback: keep a chain graph using lexical tokens to avoid dropping data.
            raw_nodes = re.findall(r"[A-Za-z_]+|\d+(?:\.\d+)?|<=|>=|==|!=|&&|\|\||[()&|!<>+\-*/]", stl_text)
            node_count = min(len(raw_nodes), self.max_nodes)
            for i in range(node_count):
                token = raw_nodes[i].lower()
                token_hash = int(hashlib.sha1(token.encode("utf-8")).hexdigest(), 16)
                if self.use_symbolic_ids:
                    token_ids[i] = ap_token_offset + (token_hash % max(1, self.vocab_size - ap_token_offset))
                node_mask[i] = True
                node_features[i] = np.asarray(
                    [0.0, -1.0, -1.0, 0.0, -1.0, 1.0, 0.0, 0.0],
                    dtype=np.float32,
                )
                if i > 0:
                    adjacency[i, i - 1] = True

        output = {
            **data,
            "stl_node_mask": node_mask,
            "stl_adjacency": adjacency,
            "stl_node_text_embeddings": node_features,
        }
        if self.use_symbolic_ids:
            output["stl_node_token_ids"] = token_ids
        if self.semantic_tokenizer is not None:
            output["stl_node_text_token_ids"] = text_token_ids
            output["stl_node_text_token_mask"] = text_token_mask
        return output


@dataclasses.dataclass(frozen=True)
class ExtractFASTActions(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer
    action_horizon: int
    action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data
        # Model outputs are saved in "actions", but for FAST models they represent tokens.
        tokens = data.pop("actions")
        actions = self.tokenizer.extract_actions(tokens.astype(np.int32), self.action_horizon, self.action_dim)
        return {
            **data,
            "actions": actions,
        }


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask(DataTransformFn):
    """Extracts a prompt from the current LeRobot dataset task."""

    # Contains the LeRobot dataset tasks (dataset.meta.tasks).
    tasks: dict[int, str]

    def __call__(self, data: DataDict) -> DataDict:
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')

        task_index = int(data["task_index"])
        if (prompt := self.tasks.get(task_index)) is None:
            raise ValueError(f"{task_index=} not found in task mapping: {self.tasks}")

        return {**data, "prompt": prompt}


@dataclasses.dataclass(frozen=True)
class PadStatesAndActions(DataTransformFn):
    """Zero-pads states and actions to the model action dimension."""

    model_action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        data["state"] = pad_to_dim(data["state"], self.model_action_dim, axis=-1)
        if "actions" in data:
            data["actions"] = pad_to_dim(data["actions"], self.model_action_dim, axis=-1)
        return data


def flatten_dict(tree: at.PyTree) -> dict:
    """Flatten a nested dictionary. Uses '/' as the separator."""
    return traverse_util.flatten_dict(tree, sep="/")


def unflatten_dict(tree: dict) -> at.PyTree:
    """Unflatten a flattened dictionary. Assumes that '/' was used as a separator."""
    return traverse_util.unflatten_dict(tree, sep="/")


def transform_dict(patterns: Mapping[str, str | None], tree: at.PyTree) -> at.PyTree:
    """Transform the structure of a nested dictionary using a set of patterns.

    The transformation is defined using the `patterns` dictionary. The keys are the
    input keys that should be matched and the values are the new names inside the output
    dictionary. If the value is None, the input key is removed.

    Both keys and values should represent flattened paths using '/' as the separator.
    Keys can be regular expressions and values can include backreferences to the
    matched groups (see `re.sub` for more details). Note that the regular expression
    must match the entire key.

    The order inside the `patterns` dictionary is important. Only the first pattern that
    matches the input key will be used.

    See unit tests for more examples.

    Args:
        patterns: A mapping from old keys to new keys.
        tree: The nested dictionary to transform.

    Returns:
        The transformed nested dictionary.
    """
    data = flatten_dict(tree)

    # Compile the patterns.
    compiled = {re.compile(k): v for k, v in patterns.items()}

    output = {}
    for k in data:
        for pattern, repl in compiled.items():
            if pattern.fullmatch(k):
                new_k = pattern.sub(repl, k, count=1) if repl is not None else None
                break
        else:
            # Use the original key if no match is found.
            new_k = k

        if new_k is not None:
            if new_k in output:
                raise ValueError(f"Key '{new_k}' already exists in output")
            output[new_k] = data[k]

    # Validate the output structure to make sure that it can be unflattened.
    names = sorted(output)
    for i in range(len(names) - 1):
        name, next_name = names[i : i + 2]
        if next_name.startswith(name + "/"):
            raise ValueError(f"Leaf '{name}' aliases a node of '{next_name}'")

    return unflatten_dict(output)


def apply_tree(
    tree: at.PyTree[T], selector: at.PyTree[S], fn: Callable[[T, S], T], *, strict: bool = False
) -> at.PyTree[T]:
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: T) -> T:
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    return x


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


def _assert_quantile_stats(norm_stats: at.PyTree[NormStats]) -> None:
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99."
            )
