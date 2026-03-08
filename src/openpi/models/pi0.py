import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


class SemanticSTLEncoder(nnx.Module):
    """Lightweight child->parent message passing encoder for semantic STL graphs."""

    def __init__(self, width: int, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        self.width = width
        self.vocab_size = config.stl_vocab_size
        self.gnn_layers = config.stl_gnn_layers
        self.text_embedding_dim = config.stl_text_embedding_dim
        self.use_llm_token_embeddings = config.stl_use_llm_token_embeddings
        self.use_symbolic_ids = config.stl_use_symbolic_ids
        self.use_gcn_conv = config.stl_use_gcn_conv

        # Used when ids are custom STL vocabulary ids.
        if self.use_symbolic_ids and not self.use_llm_token_embeddings:
            self.vocab_proj = nnx.Linear(self.vocab_size, width, rngs=rngs)
        else:
            self.vocab_proj = None
        self.text_proj = nnx.Linear(self.text_embedding_dim, width, rngs=rngs)
        # Keep per-layer submodules in nnx.Dict so parameters are tracked in this NNX version.
        self.self_mlps = nnx.Dict(
            **{f"layer_{i}": nnx.Linear(width, width, rngs=rngs) for i in range(self.gnn_layers)}
        )
        self.child_mlps = nnx.Dict(
            **{f"layer_{i}": nnx.Linear(width, width, rngs=rngs) for i in range(self.gnn_layers)}
        )

    def _project_text_embeddings(self, text_embeddings: jax.Array) -> jax.Array:
        current_dim = text_embeddings.shape[-1]
        if current_dim == self.text_embedding_dim:
            return self.text_proj(text_embeddings)
        # Keep this robust to mismatched external text embedding dimensions.
        if current_dim < self.text_embedding_dim:
            pad = self.text_embedding_dim - current_dim
            text_embeddings = jnp.pad(text_embeddings, ((0, 0), (0, 0), (0, pad)))
        else:
            text_embeddings = text_embeddings[..., : self.text_embedding_dim]
        return self.text_proj(text_embeddings)

    @at.typecheck
    def __call__(
        self,
        *,
        token_embeddings: at.Float[at.Array, "b n emb"] | None,
        token_ids: at.Int[at.Array, "b n"] | None,
        text_embeddings: at.Float[at.Array, "b n d"] | None,
        node_mask: at.Bool[at.Array, "b n"],
        adjacency: at.Bool[at.Array, "b n n"],
    ) -> at.Float[at.Array, "b n emb"]:
        if token_embeddings is None and token_ids is None and text_embeddings is None:
            raise ValueError("STL encoder requires one of token_embeddings, token_ids, or text_embeddings.")

        h = None
        if token_embeddings is not None:
            h = token_embeddings

        if token_ids is not None:
            if self.use_llm_token_embeddings:
                raise ValueError("Expected LLM-provided token embeddings when stl_use_llm_token_embeddings=True.")
            if self.vocab_proj is None:
                raise ValueError("Received symbolic STL token ids but symbolic-id branch is disabled.")
            token_ids = jnp.clip(token_ids, 0, self.vocab_size - 1)
            token_one_hot = jax.nn.one_hot(token_ids, self.vocab_size, dtype=jnp.float32)
            token_h = self.vocab_proj(token_one_hot)
            h = token_h if h is None else h + token_h

        if text_embeddings is not None:
            text_h = self._project_text_embeddings(text_embeddings)
            h = text_h if h is None else h + text_h

        assert h is not None
        node_mask_f = node_mask.astype(h.dtype)
        h = h * node_mask_f[..., None]
        adjacency_f = adjacency.astype(h.dtype)
        node_valid_pair = node_mask_f[:, :, None] * node_mask_f[:, None, :]

        if self.use_gcn_conv:
            # GCNConv-style normalized propagation:
            # H' = D^{-1/2} (A + I) D^{-1/2} H W
            n = adjacency_f.shape[1]
            eye = jnp.eye(n, dtype=h.dtype)[None, :, :]
            a_hat = (adjacency_f + eye) * node_valid_pair
            degree = jnp.sum(a_hat, axis=-1)
            degree_inv_sqrt = jnp.where(degree > 0, jnp.power(degree, -0.5), 0.0)
            norm_adj = degree_inv_sqrt[:, :, None] * a_hat * degree_inv_sqrt[:, None, :]
        else:
            norm_adj = adjacency_f * node_valid_pair

        for i in range(self.gnn_layers):
            self_mlp = self.self_mlps[f"layer_{i}"]
            child_mlp = self.child_mlps[f"layer_{i}"]
            child_agg = jnp.einsum("bij,bjd->bid", norm_adj, h, precision=jax.lax.Precision.HIGHEST)
            h_next = nnx.swish(self_mlp(h) + child_mlp(child_agg))
            h = h + h_next
            h = h * node_mask_f[..., None]

        return h


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        self.use_stl = config.use_stl
        self.stl_use_llm_token_embeddings = config.stl_use_llm_token_embeddings
        if self.use_stl:
            self.stl_encoder = SemanticSTLEncoder(paligemma_config.width, config, rngs=rngs)
        else:
            self.stl_encoder = None
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]

        # add semantic STL graph tokens, if provided
        if self.stl_encoder is not None:
            if obs.stl_node_mask is not None and obs.stl_adjacency is not None:
                token_embeddings = None
                token_ids = None
                if obs.stl_node_token_ids is not None:
                    if self.stl_use_llm_token_embeddings:
                        token_embeddings = self.PaliGemma.llm(obs.stl_node_token_ids, method="embed")
                    else:
                        token_ids = obs.stl_node_token_ids

                # Optional semantic AP tokens per node (tokenized AP text/object phrases).
                if obs.stl_node_text_token_ids is not None and obs.stl_node_text_token_mask is not None:
                    bsz, n_nodes, n_ap_tokens = obs.stl_node_text_token_ids.shape
                    flat_ids = einops.rearrange(obs.stl_node_text_token_ids, "b n t -> (b n) t")
                    flat_mask = einops.rearrange(obs.stl_node_text_token_mask, "b n t -> (b n) t")
                    flat_emb = self.PaliGemma.llm(flat_ids, method="embed")
                    flat_mask_f = flat_mask.astype(flat_emb.dtype)
                    denom = jnp.maximum(jnp.sum(flat_mask_f, axis=-1, keepdims=True), 1.0)
                    pooled = jnp.sum(flat_emb * flat_mask_f[..., None], axis=1) / denom
                    semantic_token_embeddings = einops.rearrange(pooled, "(b n) e -> b n e", b=bsz, n=n_nodes)
                    token_embeddings = (
                        semantic_token_embeddings
                        if token_embeddings is None
                        else token_embeddings + semantic_token_embeddings
                    )
                if (
                    token_embeddings is not None
                    or token_ids is not None
                    or obs.stl_node_text_embeddings is not None
                ):
                    stl_tokens = self.stl_encoder(
                        token_embeddings=token_embeddings,
                        token_ids=token_ids,
                        text_embeddings=obs.stl_node_text_embeddings,
                        node_mask=obs.stl_node_mask,
                        adjacency=obs.stl_adjacency,
                    )
                    tokens.append(stl_tokens)
                    input_mask.append(obs.stl_node_mask)
                    ar_mask += [False] * stl_tokens.shape[1]
            else:
                logger.warning("STL encoder is enabled but stl_node_mask/stl_adjacency are missing; skipping STL tokens.")
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
