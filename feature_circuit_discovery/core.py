"""Core SAE circuit discovery: model loading, feature selection, and gradient computation.

This module provides the main computational primitives:
  - JumpReLUSAE: Sparse autoencoder with jump-ReLU activation
  - load_sae: Load SAE weights from HuggingFace (with LRU caching)
  - get_active_features: Find frequently-active SAE features across prompts
  - get_contrastive_features: Find differentially-active features between two groups
  - compute_gradient_matrix: Gradient of downstream features w.r.t. upstream features
  - compute_gradient_matrix_jacobian: Jacobian-based alternative (partial forward pass)
  - compute_gradient_matrices_batch: Batched gradients from one upstream to many downstream layers
"""

import gc
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_safetensors
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from feature_circuit_discovery.data import (
    SAE_HF_REPO,
    MAX_SAE_LAYER,
    canonical_sae_filenames,
)


class JumpReLUSAE(nn.Module):
    def __init__(self, d_model: int, d_sae: int) -> None:
        super().__init__()
        self.W_enc = nn.Parameter(torch.zeros(d_model, d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))
        self.threshold = nn.Parameter(torch.zeros(d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def encode(self, input_acts: torch.Tensor) -> torch.Tensor:
        pre_acts = input_acts @ self.W_enc + self.b_enc
        mask = (pre_acts > self.threshold).float()
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return acts @ self.W_dec + self.b_dec

    def forward(self, acts: torch.Tensor) -> torch.Tensor:
        acts = self.encode(acts)
        recon = self.decode(acts)
        return recon


# ---------------------------------------------------------------------------
# SAE caching with bounded LRU — avoids redundant disk I/O
# while keeping memory bounded (each SAE is ~300MB, so maxsize=3 ≈ 900MB)
# ---------------------------------------------------------------------------
_SAE_CACHE_MAXSIZE = 3
_sae_cache: OrderedDict[tuple[str, str], JumpReLUSAE] = OrderedDict()


def load_sae(filename: str, device: torch.device) -> JumpReLUSAE:
    cache_key = (filename, str(device))
    if cache_key in _sae_cache:
        _sae_cache.move_to_end(cache_key)
        return _sae_cache[cache_key]

    path_to_params = hf_hub_download(
        repo_id=SAE_HF_REPO,
        filename=filename,
        force_download=False,
    )

    if filename.endswith(".npz"):
        np_params = np.load(path_to_params)
        pt_params = {k: torch.from_numpy(np_params[k]).to(device) for k in np_params.files}
    else:
        tensors = load_safetensors(path_to_params)
        KEY_MAP = {"w_enc": "W_enc", "w_dec": "W_dec", "b_enc": "b_enc", "b_dec": "b_dec", "threshold": "threshold"}
        pt_params = {KEY_MAP.get(k, k): v for k, v in tensors.items()}
        pt_params = {k: v.to(device) for k, v in pt_params.items()}

    sae = JumpReLUSAE(pt_params["W_enc"].shape[0], pt_params["W_enc"].shape[1]).to(device)
    sae.load_state_dict(pt_params)

    if len(_sae_cache) >= _SAE_CACHE_MAXSIZE:
        _sae_cache.popitem(last=False)

    _sae_cache[cache_key] = sae
    return sae


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------


def find_frequent_nonzero_indices(
    tensor_list: list[torch.Tensor], threshold: float = 0.9
) -> list[torch.Tensor]:
    """Find SAE feature indices that are active in >= threshold fraction of prompts.

    Args:
        tensor_list: List of (batch_size, seq_len, d_sae) tensors, one per layer.
        threshold: Minimum fraction of prompts where a feature must be active.

    Returns:
        List of 1D tensors of active feature indices, one per layer.
    """
    result_indices: list[torch.Tensor] = []
    for tensor in tensor_list:
        # A feature is "active for a prompt" if it fires at any seq position
        any_active = (tensor != 0).any(dim=1)  # (batch_size, d_sae)
        freq = any_active.float().mean(dim=0)  # (d_sae,)
        valid_indices = torch.where(freq >= threshold)[0]
        result_indices.append(valid_indices)
    return result_indices


def get_active_features(
    prompts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: torch.device,
    threshold: float = 0.9,
) -> list[torch.Tensor]:
    """Find SAE features frequently active across prompts via a single forward pass.

    Args:
        prompts: List of input prompts.
        tokenizer: Model tokenizer.
        model: The language model.
        device: Torch device.
        threshold: Minimum fraction of prompts where a feature must be active.

    Returns:
        List of 1D tensors of active feature indices, one per layer.
    """
    tokenized_prompts = (
        tokenizer(prompts, return_tensors="pt", add_special_tokens=True, padding=True)
        .data["input_ids"]
        .to(device)
    )

    with torch.no_grad():
        outputs = model(tokenized_prompts, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        del outputs

        feature_acts = []
        num_layers = min(model.config.num_hidden_layers, MAX_SAE_LAYER + 1)
        for layer in tqdm(range(num_layers)):
            sae = load_sae(canonical_sae_filenames[layer], device)
            # hidden_states[0] = embedding output, hidden_states[N+1] = output of layer N
            sae_activations = sae.encode(hidden_states[layer + 1].float())
            feature_acts.append(sae_activations)

        del hidden_states
        gc.collect()

    activated_features = find_frequent_nonzero_indices(feature_acts, threshold=threshold)
    del feature_acts
    gc.collect()
    return activated_features


def get_contrastive_features(
    prompts_a: list[str],
    prompts_b: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: torch.device,
    n_features: int = 50,
) -> tuple[list[torch.Tensor], dict[int, dict[int, float]]]:
    """Find SAE features that are differentially active between two prompt groups.

    Ranks features by |mean_activation_a - mean_activation_b| at the last token
    position, selecting those most different between the two groups.

    Args:
        prompts_a: First group of prompts (e.g. "Yes" answers).
        prompts_b: Second group of prompts (e.g. "No" answers).
        tokenizer: Model tokenizer.
        model: The language model.
        device: Torch device.
        n_features: Number of top differentially active features to select per layer.

    Returns:
        Tuple of (activated_features, activation_frequencies).
        activated_features: List of 1D tensors of feature indices per layer.
        activation_frequencies: Dict mapping layer -> {feature_id: diff_score}.
    """
    def _encode_batch(prompts):
        tokenized = tokenizer(prompts, return_tensors="pt", add_special_tokens=True, padding=True)
        tokens = tokenized["input_ids"].to(device)
        attn_mask = tokenized["attention_mask"].to(device)
        with torch.no_grad():
            model_kwargs = dict(output_hidden_states=True)
            pad_token_id = 0  # Gemma pad token
            has_padding = (tokens == pad_token_id).any()
            if has_padding:
                model_kwargs["attention_mask"] = attn_mask
            outputs = model(tokens, **model_kwargs)
            hidden_states = outputs.hidden_states
            del outputs
            last_pos = tokens.shape[1] - 1
            layer_acts = []
            num_layers = min(model.config.num_hidden_layers, MAX_SAE_LAYER + 1)
            for layer in range(num_layers):
                sae = load_sae(canonical_sae_filenames[layer], device)
                acts = sae.encode(hidden_states[layer + 1].float())
                layer_acts.append(acts[:, last_pos, :])
            del hidden_states
        return layer_acts

    print("  Encoding group A...")
    acts_a = _encode_batch(prompts_a)
    print("  Encoding group B...")
    acts_b = _encode_batch(prompts_b)
    gc.collect()

    num_layers = min(model.config.num_hidden_layers, MAX_SAE_LAYER + 1)
    activated_features = []
    activation_frequencies: dict[int, dict[int, float]] = {}

    for layer in range(num_layers):
        mean_a = acts_a[layer].mean(dim=0)
        mean_b = acts_b[layer].mean(dim=0)
        diff = (mean_a - mean_b).abs()

        topk = torch.topk(diff, min(n_features, diff.shape[0]))
        feat_ids = topk.indices
        activated_features.append(feat_ids)

        activation_frequencies[layer] = {
            int(fid): round(float(diff[fid]), 6) for fid in feat_ids.cpu()
        }

    del acts_a, acts_b
    gc.collect()
    return activated_features, activation_frequencies


# ---------------------------------------------------------------------------
# Gradient computation
# ---------------------------------------------------------------------------


def compute_gradient_matrix(
    inputs: torch.Tensor,
    upstream_layer_idx: int,
    downstream_layer_idx: int,
    upstream_features: torch.Tensor,
    downstream_features: torch.Tensor,
    model: AutoModelForCausalLM,
    verbose: bool = False,
) -> torch.Tensor:
    """Compute gradient matrix of downstream SAE features w.r.t. upstream features.

    Hooks into the upstream layer to add a perturbation a · W_dec[features] at
    the last token position, then computes ∂f_downstream/∂a via autograd.

    Args:
        inputs: Tokenized input (1, seq_len).
        upstream_layer_idx: Layer to perturb.
        downstream_layer_idx: Layer to read downstream features from.
        upstream_features: Feature indices in the upstream SAE.
        downstream_features: Feature indices in the downstream SAE.
        model: The language model.
        verbose: Print progress.

    Returns:
        Gradient matrix of shape (n_downstream, n_upstream).
    """
    device = model.device
    if device is None:
        device = next(model.parameters()).device

    torch.cuda.empty_cache()
    gc.collect()

    with torch.enable_grad():
        if verbose:
            print(f"\nloading upstream sae (Layer {upstream_layer_idx})")
        sae_upstream = load_sae(canonical_sae_filenames[upstream_layer_idx], device)

        if verbose:
            print(f"loading downstream sae (Layer {downstream_layer_idx})")
        sae_downstream = load_sae(canonical_sae_filenames[downstream_layer_idx], device)

        m = len(upstream_features)
        n = len(downstream_features)
        d_model = sae_upstream.W_dec.size(1)

        a = torch.zeros(m, requires_grad=True, device=device)
        feature_vectors = sae_upstream.W_dec[upstream_features, :]  # (m, d_model)

        def modify_residual_activations(module, input, output):
            residual = output[0] if isinstance(output, tuple) else output
            residual_modified = residual.clone()

            weighted_feature_vectors = (
                a.view(m, 1, 1) * feature_vectors.view(m, 1, d_model)
            ).sum(dim=0)
            expanded_weighted_feature_vector = weighted_feature_vectors.expand(
                residual_modified.size(0), -1, -1
            )

            if residual_modified.dtype != expanded_weighted_feature_vector.dtype:
                expanded_weighted_feature_vector = expanded_weighted_feature_vector.to(
                    residual_modified.dtype
                )
            residual_modified[:, -1:, :] += expanded_weighted_feature_vector

            return (
                (residual_modified,) + output[1:]
                if isinstance(output, tuple)
                else residual_modified
            )

        if verbose:
            print("conducting forward pass")
        hook = model.model.layers[upstream_layer_idx].register_forward_hook(
            modify_residual_activations
        )
        outputs = model(inputs, output_hidden_states=True)
        hook.remove()

        hidden_states = outputs.hidden_states
        act_downstream = hidden_states[downstream_layer_idx + 1]

        sae_downstream_acts = sae_downstream.encode(act_downstream.float())

        batch_idx = 0
        last_pos = inputs.shape[1] - 1
        features_downstream = sae_downstream_acts[batch_idx, last_pos, downstream_features]

        gradient_matrix = torch.zeros(n, m, device=device)

        if verbose:
            print("computing gradients")
            iterator = tqdm(range(n))
        else:
            iterator = range(n)

        for i in iterator:
            (grad,) = torch.autograd.grad(
                features_downstream[i],
                a,
                retain_graph=(i < n - 1),
                allow_unused=True,
            )
            if grad is not None:
                gradient_matrix[i, :] = grad.detach()

    torch.cuda.empty_cache()
    gc.collect()
    if verbose:
        print("done")
    return gradient_matrix


def compute_gradient_matrix_jacobian(
    inputs: torch.Tensor,
    upstream_layer_idx: int,
    downstream_layer_idx: int,
    upstream_features: torch.Tensor,
    downstream_features: torch.Tensor,
    model: AutoModelForCausalLM,
    verbose: bool = False,
    vectorize: bool = True,
) -> torch.Tensor:
    """Same semantics as compute_gradient_matrix but uses torch.autograd.functional.jacobian.

    Only runs layers upstream..downstream (not the full model), and batches
    backward passes via vmap when vectorize=True.
    """
    device = model.device
    if device is None:
        device = next(model.parameters()).device

    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    if verbose:
        print(f"loading SAEs (layers {upstream_layer_idx}, {downstream_layer_idx})")
    sae_upstream = load_sae(canonical_sae_filenames[upstream_layer_idx], device)
    sae_downstream = load_sae(canonical_sae_filenames[downstream_layer_idx], device)

    m = len(upstream_features)
    d_model = sae_upstream.W_dec.size(1)

    feature_vectors = sae_upstream.W_dec[upstream_features, :]  # (m, d_model)

    batch_idx = 0
    last_pos = inputs.shape[1] - 1

    if verbose:
        print("running inference pass (no grad)")

    with torch.no_grad():
        outputs = model(inputs, output_hidden_states=True)
        cached_upstream_hidden = outputs.hidden_states[upstream_layer_idx + 1].detach()

        position_ids = torch.arange(inputs.shape[1], device=device).unsqueeze(0)
        rotary_emb = model.model.rotary_emb
        has_layer_types = isinstance(getattr(rotary_emb, 'rope_type', None), dict)

        cached_pos_embs = {}
        if has_layer_types:
            for layer_idx in range(upstream_layer_idx + 1, downstream_layer_idx + 1):
                attn_type = model.model.layers[layer_idx].attention_type
                if attn_type not in cached_pos_embs:
                    pe = rotary_emb(cached_upstream_hidden, position_ids, attn_type)
                    cached_pos_embs[attn_type] = (pe[0].detach(), pe[1].detach())
        else:
            pe = rotary_emb(cached_upstream_hidden, position_ids)
            default_pe = (pe[0].detach(), pe[1].detach())

        del outputs

    def f(a_val: torch.Tensor) -> torch.Tensor:
        h = cached_upstream_hidden.clone()

        weighted = (
            a_val.view(m, 1, 1) * feature_vectors.view(m, 1, d_model)
        ).sum(dim=0)
        expanded = weighted.expand(h.size(0), -1, -1)
        if h.dtype != expanded.dtype:
            expanded = expanded.to(h.dtype)
        h[:, -1:, :] += expanded

        for layer_idx in range(upstream_layer_idx + 1, downstream_layer_idx + 1):
            layer = model.model.layers[layer_idx]
            if has_layer_types:
                pe = cached_pos_embs[layer.attention_type]
            else:
                pe = default_pe
            layer_out = layer(h, position_embeddings=pe)
            h = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        sae_acts = sae_downstream.encode(h.float())
        return sae_acts[batch_idx, last_pos, downstream_features]

    if verbose:
        print(f"computing Jacobian over layers {upstream_layer_idx}→{downstream_layer_idx} "
              f"(vectorize={vectorize})")

    with torch.enable_grad():
        a_zero = torch.zeros(m, device=device)
        J = torch.autograd.functional.jacobian(f, a_zero, vectorize=vectorize)
        gradient_matrix = J.detach()

    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    if verbose:
        print("done")
    return gradient_matrix


def compute_gradient_matrices_batch(
    inputs: torch.Tensor,
    upstream_layer_idx: int,
    downstream_pairs: list[tuple[int, torch.Tensor]],
    upstream_features: torch.Tensor,
    model: AutoModelForCausalLM,
    verbose: bool = False,
    logit_token_ids: list[int] | None = None,
    logit_token_ids_per_prompt: torch.Tensor | None = None,
    prompt_signs: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
) -> tuple[list[torch.Tensor], torch.Tensor | None]:
    """Compute gradient matrices from one upstream layer to multiple downstream layers.

    Uses a single forward pass with a hook on the upstream layer, then computes
    gradients for each downstream layer from the shared computation graph.

    When prompt_signs is provided, computes label-aligned gradients:
        E_p[σ(p) * ∂f_downstream / ∂a]
    where σ(p) = +1 for group A, -1 for group B.

    Args:
        inputs: Tokenized input (batch_size, seq_len).
        upstream_layer_idx: Which layer to hook.
        downstream_pairs: List of (downstream_layer_idx, downstream_features_tensor).
        upstream_features: Feature indices in the upstream SAE.
        model: The language model.
        verbose: Print progress.
        logit_token_ids: Optional list of token IDs for logit gradients (same for all prompts).
        logit_token_ids_per_prompt: Optional (n_slots, batch_size) tensor of per-prompt token IDs.
            Mutually exclusive with logit_token_ids.
        prompt_signs: Optional (batch_size,) tensor of +1/-1 for label-aligned averaging.
        attention_mask: Optional (batch_size, seq_len) attention mask for padded inputs.

    Returns:
        Tuple of (list of gradient matrices, logit gradient matrix or None).
    """
    device = model.device
    if device is None:
        device = next(model.parameters()).device

    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    with torch.enable_grad():
        sae_upstream = load_sae(canonical_sae_filenames[upstream_layer_idx], device)

        m = len(upstream_features)
        d_model = sae_upstream.W_dec.size(1)
        a = torch.zeros(m, requires_grad=True, device=device)
        feature_vectors = sae_upstream.W_dec[upstream_features, :]  # (m, d_model)

        def modify_residual_activations(module, input, output):
            residual = output[0] if isinstance(output, tuple) else output
            residual_modified = residual.clone()
            weighted = (
                a.view(m, 1, 1) * feature_vectors.view(m, 1, d_model)
            ).sum(dim=0)
            expanded = weighted.expand(residual_modified.size(0), -1, -1)
            if residual_modified.dtype != expanded.dtype:
                expanded = expanded.to(residual_modified.dtype)
            residual_modified[:, -1:, :] += expanded
            return (
                (residual_modified,) + output[1:]
                if isinstance(output, tuple)
                else residual_modified
            )

        hook = model.model.layers[upstream_layer_idx].register_forward_hook(
            modify_residual_activations
        )
        model_kwargs = dict(output_hidden_states=True)
        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask
        outputs = model(inputs, **model_kwargs)
        hook.remove()

        hidden_states = outputs.hidden_states
        last_pos = inputs.shape[1] - 1

        if inputs.shape[0] > 1:
            pad_token_id = 0  # Gemma pad token
            has_padding = (inputs == pad_token_id).any()
            if has_padding and attention_mask is None:
                assert False, (
                    f"Padding detected in input but no attention_mask provided! "
                    f"The model will attend to pad tokens, corrupting hidden states. "
                    f"Pass attention_mask from the tokenizer output."
                )

        results = []
        if logit_token_ids_per_prompt is not None:
            n_logit_grads = logit_token_ids_per_prompt.shape[0]
        elif logit_token_ids:
            n_logit_grads = len(logit_token_ids)
        else:
            n_logit_grads = 0
        total_grads = sum(len(df) for _, df in downstream_pairs) + n_logit_grads
        grad_done = 0

        for pair_idx, (down_layer_idx, down_features) in enumerate(downstream_pairs):
            sae_downstream = load_sae(canonical_sae_filenames[down_layer_idx], device)
            act_downstream = hidden_states[down_layer_idx + 1]
            sae_acts = sae_downstream.encode(act_downstream.float())

            n = len(down_features)
            raw = sae_acts[:, last_pos, down_features]  # (batch, n)
            if prompt_signs is not None:
                raw = raw * prompt_signs.view(-1, 1)
            features_downstream = raw.mean(dim=0)  # (n,)

            gradient_matrix = torch.zeros(n, m, device=device)

            for i in range(n):
                grad_done += 1
                is_last = (grad_done == total_grads)
                (grad,) = torch.autograd.grad(
                    features_downstream[i],
                    a,
                    retain_graph=not is_last,
                    allow_unused=True,
                )
                if grad is not None:
                    gradient_matrix[i, :] = grad.detach()

            results.append(gradient_matrix)

            if verbose:
                print(f"    -> L{down_layer_idx}: max={gradient_matrix.abs().max():.4f}, "
                      f"nonzero={gradient_matrix.count_nonzero()}/{gradient_matrix.numel()}")

        logit_grad_matrix = None
        if logit_token_ids_per_prompt is not None:
            last_pos = inputs.shape[1] - 1
            raw_logits = outputs.logits[:, last_pos, :]
            batch_size = raw_logits.shape[0]
            batch_arange = torch.arange(batch_size, device=device)

            logit_grad_matrix = torch.zeros(n_logit_grads, m, device=device)
            for slot_idx in range(n_logit_grads):
                token_ids_for_slot = logit_token_ids_per_prompt[slot_idx]
                per_prompt_logits = raw_logits[batch_arange, token_ids_for_slot]
                if prompt_signs is not None:
                    per_prompt_logits = per_prompt_logits * prompt_signs
                scalar = per_prompt_logits.mean()

                grad_done += 1
                is_last = (grad_done == total_grads)
                (grad,) = torch.autograd.grad(
                    scalar,
                    a,
                    retain_graph=not is_last,
                    allow_unused=True,
                )
                if grad is not None:
                    logit_grad_matrix[slot_idx, :] = grad.detach()

            if verbose:
                for slot_idx in range(n_logit_grads):
                    row = logit_grad_matrix[slot_idx]
                    print(f"    -> Logit slot {slot_idx}: max={row.abs().max():.4f}, "
                          f"nonzero={row.count_nonzero()}/{row.numel()}")

        elif logit_token_ids:
            last_pos = inputs.shape[1] - 1
            raw_logits = outputs.logits[:, last_pos, :]
            if prompt_signs is not None:
                raw_logits = raw_logits * prompt_signs.view(-1, 1)
            logits = raw_logits.mean(dim=0)

            logit_grad_matrix = torch.zeros(n_logit_grads, m, device=device)
            for t_idx, token_id in enumerate(logit_token_ids):
                grad_done += 1
                is_last = (grad_done == total_grads)
                (grad,) = torch.autograd.grad(
                    logits[token_id],
                    a,
                    retain_graph=not is_last,
                    allow_unused=True,
                )
                if grad is not None:
                    logit_grad_matrix[t_idx, :] = grad.detach()

            if verbose:
                for t_idx, token_id in enumerate(logit_token_ids):
                    row = logit_grad_matrix[t_idx]
                    print(f"    -> Logit token {token_id}: max={row.abs().max():.4f}, "
                          f"nonzero={row.count_nonzero()}/{row.numel()}")

    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    return results, logit_grad_matrix
