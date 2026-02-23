# %%
"""
Optimized version of feature_grad_exp.py
========================================
Behavior-preserving optimizations applied:
  1. SAE caching — avoid redundant disk I/O + CPU→GPU transfers
  2. Single forward pass in get_active_features (output_hidden_states=True)
  3. torch.no_grad() for inference-only code paths
  4. torch.enable_grad() context manager instead of global set_grad_enabled
  5. Vectorized feature vector gathering (fancy indexing vs. Python loop)
  6. Fixed load_sae double device transfer (move to device before load_state_dict)
  7. Fixed threshold bug in draw_bipartite_graph (parameter was shadowed)
  8. torch.autograd.grad instead of .backward() + release graph on last iteration
"""

import gc
import random

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from matplotlib import cm
from safetensors.torch import load_file as load_safetensors
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from feature_circuit_discovery.data import (
    SAE_HF_REPO,
    MAX_SAE_LAYER,
    canonical_sae_filenames,
)

# ---------------------------------------------------------------------------
# SAE model (unchanged behavior)
# ---------------------------------------------------------------------------


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
# [OPT 1] SAE caching with bounded LRU — avoids redundant disk I/O
# while keeping memory bounded (each SAE is ~300MB, so maxsize=3 ≈ 900MB)
# ---------------------------------------------------------------------------
from collections import OrderedDict

_SAE_CACHE_MAXSIZE = 3
_sae_cache: OrderedDict[tuple[str, str], JumpReLUSAE] = OrderedDict()


def load_sae(filename: str, device: torch.device) -> JumpReLUSAE:
    cache_key = (filename, str(device))
    if cache_key in _sae_cache:
        _sae_cache.move_to_end(cache_key)  # mark as recently used
        return _sae_cache[cache_key]

    path_to_params = hf_hub_download(
        repo_id=SAE_HF_REPO,
        filename=filename,
        force_download=False,
    )
    tensors = load_safetensors(path_to_params)
    # Gemma Scope 2 uses lowercase keys (w_enc, b_dec); remap to JumpReLUSAE (W_enc, b_dec)
    KEY_MAP = {"w_enc": "W_enc", "w_dec": "W_dec", "b_enc": "b_enc", "b_dec": "b_dec", "threshold": "threshold"}
    pt_params = {KEY_MAP.get(k, k): v for k, v in tensors.items()}
    pt_params = {k: v.to(device) for k, v in pt_params.items()}

    # [OPT 6] Move SAE to device *before* load_state_dict to avoid
    # double transfer (CPU→GPU in pt_params, then GPU→CPU in load_state_dict,
    # then CPU→GPU again in .to(device))
    sae = JumpReLUSAE(pt_params["W_enc"].shape[0], pt_params["W_enc"].shape[1]).to(device)
    sae.load_state_dict(pt_params)

    # Evict oldest entry if cache is full
    if len(_sae_cache) >= _SAE_CACHE_MAXSIZE:
        _sae_cache.popitem(last=False)

    _sae_cache[cache_key] = sae
    return sae


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def find_frequent_nonzero_indices(
    tensor_list: list[torch.Tensor], threshold: float = 0.9
) -> list[torch.Tensor]:
    """Find SAE feature indices that are active in >= threshold fraction of prompts.

    BUG FIX: Original code did torch.where(2D_tensor)[0] which returns
    *sequence position* indices, not feature indices. Fixed by first
    aggregating over the sequence dimension: a feature counts as "active
    for a prompt" if it fires at any token position in that prompt.
    """
    result_indices: list[torch.Tensor] = []
    for tensor in tensor_list:
        # tensor shape: (batch_size, seq_len, d_sae)
        # A feature is "active for a prompt" if it fires at any seq position
        any_active = (tensor != 0).any(dim=1)  # (batch_size, d_sae)
        freq = any_active.float().mean(dim=0)  # (d_sae,) — fraction of prompts
        valid_indices = torch.where(freq >= threshold)[0]  # 1D → [0] is correct
        result_indices.append(valid_indices)
    return result_indices


# ---------------------------------------------------------------------------
# [OPT 2 + 3] get_active_features — single forward pass + no_grad
# ---------------------------------------------------------------------------


def get_active_features(
    prompts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: torch.device,
    threshold: float = 0.9,
) -> list[torch.Tensor]:
    tokenized_prompts = (
        tokenizer(prompts, return_tensors="pt", add_special_tokens=True, padding=True)
        .data["input_ids"]
        .to(device)
    )

    # [OPT 3] No gradients needed — this is pure inference
    # [OPT 2] Single forward pass to get all hidden states at once
    #          instead of 26 separate hook-based forward passes
    with torch.no_grad():
        outputs = model(tokenized_prompts, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple of (num_layers + 1) tensors
        del outputs  # free model output memory early

        feature_acts = []
        num_layers = min(model.config.num_hidden_layers, MAX_SAE_LAYER + 1)
        for layer in tqdm(range(num_layers)):
            sae = load_sae(canonical_sae_filenames[layer], device)
            # hidden_states[layer + 1] is the output of model.model.layers[layer]
            # (index 0 is the embedding output, so layer N output is at index N+1)
            sae_activations = sae.encode(hidden_states[layer + 1].float())
            feature_acts.append(sae_activations)

        del hidden_states
        gc.collect()

    activated_features = find_frequent_nonzero_indices(feature_acts, threshold=threshold)
    del feature_acts
    gc.collect()
    return activated_features


# ---------------------------------------------------------------------------
# Contrastive feature selection — find differentially active features
# ---------------------------------------------------------------------------


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
        tokens = (
            tokenizer(prompts, return_tensors="pt", add_special_tokens=True, padding=True)
            .data["input_ids"]
            .to(device)
        )
        with torch.no_grad():
            outputs = model(tokens, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            del outputs
            # Encode all layers through SAEs, extract last token position
            last_pos = tokens.shape[1] - 1
            layer_acts = []
            num_layers = min(model.config.num_hidden_layers, MAX_SAE_LAYER + 1)
            for layer in range(num_layers):
                sae = load_sae(canonical_sae_filenames[layer], device)
                acts = sae.encode(hidden_states[layer + 1].float())
                # Use last token position: (batch, d_sae)
                layer_acts.append(acts[:, last_pos, :])
            del hidden_states
        return layer_acts

    print("  Encoding group A...")
    acts_a = _encode_batch(prompts_a)  # list of (batch_a, d_sae) per layer
    print("  Encoding group B...")
    acts_b = _encode_batch(prompts_b)  # list of (batch_b, d_sae) per layer
    gc.collect()

    num_layers = min(model.config.num_hidden_layers, MAX_SAE_LAYER + 1)
    activated_features = []
    activation_frequencies: dict[int, dict[int, float]] = {}

    for layer in range(num_layers):
        mean_a = acts_a[layer].mean(dim=0)  # (d_sae,)
        mean_b = acts_b[layer].mean(dim=0)  # (d_sae,)
        diff = (mean_a - mean_b).abs()  # (d_sae,)

        # Select top n_features by |mean difference|
        topk = torch.topk(diff, min(n_features, diff.shape[0]))
        feat_ids = topk.indices
        activated_features.append(feat_ids)

        # Store diff scores for export
        activation_frequencies[layer] = {
            int(fid): round(float(diff[fid]), 6) for fid in feat_ids.cpu()
        }

    del acts_a, acts_b
    gc.collect()
    return activated_features, activation_frequencies


# ---------------------------------------------------------------------------
# [OPT 4 + 5 + 8] compute_gradient_matrix — vectorized + autograd.grad
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
    device = model.device
    if device is None:
        device = next(model.parameters()).device

    torch.cuda.empty_cache()
    gc.collect()

    # [OPT 4] Use context manager instead of global set_grad_enabled
    with torch.enable_grad():
        if verbose:
            print()
            print(f"loading upstream sae (Layer {upstream_layer_idx})")
        sae_upstream = load_sae(canonical_sae_filenames[upstream_layer_idx], device)

        if verbose:
            print("upstream sae loaded")
            print(f"loading downstream sae (Layer {downstream_layer_idx})")
        sae_downstream = load_sae(canonical_sae_filenames[downstream_layer_idx], device)

        if verbose:
            print("downstream sae loaded")

        m = len(upstream_features)
        n = len(downstream_features)
        d_model = sae_upstream.W_dec.size(1)

        a = torch.zeros(m, requires_grad=True, device=device)

        # [OPT 5] Vectorized feature vector gathering — single index op
        # instead of Python loop + torch.stack
        feature_vectors = sae_upstream.W_dec[upstream_features, :]  # shape: (m, d_model)

        def modify_residual_activations(module, input, output):
            residual = output[0] if isinstance(output, tuple) else output
            residual_modified = residual.clone()

            weighted_feature_vectors = (
                a.view(m, 1, 1) * feature_vectors.view(m, 1, d_model)
            ).sum(dim=0)
            expanded_weighted_feature_vector = weighted_feature_vectors.expand(
                residual_modified.size(0), -1, -1
            )

            # Ensure dtype match to avoid breaking autograd with in-place mixed-precision ops
            if residual_modified.dtype != expanded_weighted_feature_vector.dtype:
                expanded_weighted_feature_vector = expanded_weighted_feature_vector.to(
                    residual_modified.dtype
                )
            # Perturb at the last token position (prediction position)
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
        # hidden_states[0] = embedding output, hidden_states[N+1] = output of layer N
        act_downstream = hidden_states[downstream_layer_idx + 1]

        sae_downstream_acts = sae_downstream.encode(act_downstream.float())

        batch_idx = 0
        last_pos = inputs.shape[1] - 1  # last token — where the model produces its prediction
        features_downstream = sae_downstream_acts[batch_idx, last_pos, downstream_features]

        gradient_matrix = torch.zeros(n, m, device=device)

        # [OPT 8] Use torch.autograd.grad instead of .backward():
        # - Avoids manual .grad zeroing boilerplate
        # - Releases computation graph on the last iteration (retain_graph=False)
        if verbose:
            print("computing gradients")
            iterator = tqdm(range(n))
        else:
            iterator = range(n)

        for i in iterator:
            (grad,) = torch.autograd.grad(
                features_downstream[i],
                a,
                retain_graph=(i < n - 1),  # release graph on last iteration
                allow_unused=True,
            )
            if grad is not None:
                gradient_matrix[i, :] = grad.detach()
            # else: row stays zero (feature doesn't depend on upstream perturbation)

    # Clean up
    torch.cuda.empty_cache()
    gc.collect()
    if verbose:
        print("done")
    return gradient_matrix


# ---------------------------------------------------------------------------
# Batched gradient computation — one forward pass per upstream layer
# ---------------------------------------------------------------------------


def compute_gradient_matrices_batch(
    inputs: torch.Tensor,
    upstream_layer_idx: int,
    downstream_pairs: list[tuple[int, torch.Tensor]],
    upstream_features: torch.Tensor,
    model: AutoModelForCausalLM,
    verbose: bool = False,
    logit_token_ids: list[int] | None = None,
) -> tuple[list[torch.Tensor], torch.Tensor | None]:
    """Compute gradient matrices from one upstream layer to multiple downstream layers.

    Uses a single forward pass with a hook on the upstream layer, then computes
    gradients for each downstream layer from the shared computation graph.
    This is ~N times faster than N separate calls to compute_gradient_matrix.

    Args:
        inputs: Tokenized input (1, seq_len).
        upstream_layer_idx: Which layer to hook.
        downstream_pairs: List of (downstream_layer_idx, downstream_features_tensor).
        upstream_features: Feature indices in the upstream SAE.
        model: The language model.
        verbose: Print progress.
        logit_token_ids: Optional list of token IDs to compute logit gradients for
            (e.g. Yes/No tokens). Returns a (len(logit_token_ids), n_upstream) matrix.

    Returns:
        Tuple of (list of gradient matrices, logit gradient matrix or None).
        Each gradient matrix is (n_down, n_up). Logit matrix is (n_logit_tokens, n_up).
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
            # Perturb at the last token position (prediction position)
            residual_modified[:, -1:, :] += expanded
            return (
                (residual_modified,) + output[1:]
                if isinstance(output, tuple)
                else residual_modified
            )

        hook = model.model.layers[upstream_layer_idx].register_forward_hook(
            modify_residual_activations
        )
        outputs = model(inputs, output_hidden_states=True)
        hook.remove()

        hidden_states = outputs.hidden_states
        last_pos = inputs.shape[1] - 1  # last token position

        results = []
        n_logit_grads = len(logit_token_ids) if logit_token_ids else 0
        total_grads = sum(len(df) for _, df in downstream_pairs) + n_logit_grads
        grad_done = 0

        for pair_idx, (down_layer_idx, down_features) in enumerate(downstream_pairs):
            sae_downstream = load_sae(canonical_sae_filenames[down_layer_idx], device)
            act_downstream = hidden_states[down_layer_idx + 1]
            sae_acts = sae_downstream.encode(act_downstream.float())

            n = len(down_features)
            features_downstream = sae_acts[0, last_pos, down_features]  # last token

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

        # Compute logit gradients: how upstream features influence specific output logits
        logit_grad_matrix = None
        if logit_token_ids:
            # Use logits at the last token position (prediction position)
            last_pos = inputs.shape[1] - 1
            logits = outputs.logits[0, last_pos, :]  # (vocab_size,)

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


# ---------------------------------------------------------------------------
# [OPT 7] draw_bipartite_graph — threshold parameter actually respected
# ---------------------------------------------------------------------------


def draw_bipartite_graph(tensor: torch.Tensor, threshold: float = 0.05) -> None:
    tensor = tensor.cpu()
    # BUG FIX: removed `threshold = 0.05` that shadowed the parameter

    m, n = tensor.shape

    B = nx.Graph()

    left_nodes = [f"left_{i}" for i in range(m)]
    B.add_nodes_from(left_nodes, bipartite=0)

    right_nodes = [f"right_{j}" for j in range(n)]
    B.add_nodes_from(right_nodes, bipartite=1)

    for i in range(m):
        for j in range(n):
            if abs(tensor[i, j]) >= threshold:
                B.add_edge(left_nodes[i], right_nodes[j], weight=tensor[i, j])

    pos = nx.bipartite_layout(B, left_nodes)
    edges = B.edges(data=True)
    edge_weights = np.array([edge[2]["weight"] for edge in edges])

    if len(edge_weights) == 0:
        print(f"No edges above threshold={threshold}")
        return

    norm = plt.Normalize(vmin=edge_weights.min(), vmax=edge_weights.max())
    colors = cm.viridis(norm(edge_weights))

    nx.draw(B, pos, with_labels=True, node_color="lightblue", node_size=500)

    for (u, v, d), color in zip(edges, colors):
        nx.draw_networkx_edges(
            B,
            pos,
            edgelist=[(u, v)],
            width=d["weight"] * 5,
            edge_color=[color],
            alpha=0.8,
        )

    sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, label="Connection Strength")
    plt.savefig(f"experiments/circuit_layer_{getattr(draw_bipartite_graph, '_call_idx', 0)}.png",
                dpi=150, bbox_inches="tight")
    draw_bipartite_graph._call_idx = getattr(draw_bipartite_graph, '_call_idx', 0) + 1
    plt.close()


# ===========================================================================
# Main experiment (same logic as feature_grad_exp.py)
# ===========================================================================

# %%
if __name__ == "__main__":
    from feature_circuit_discovery.data import MODEL_ID

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.float32
        ).to(device)
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        # Use bfloat16 to halve model memory (~5GB vs ~10GB in float32)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16
        ).to(device)
        print("Using CPU with bfloat16")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # %%
    def generate_addition_prompt(n: int = 1) -> list[str]:
        """Generate n random 3-digit addition prompts like '304 + 583 = '."""
        prompts = []
        for _ in range(n):
            a = random.randint(100, 999)
            b = random.randint(100, 999)
            prompts.append(f"{a} + {b} = ")
        return prompts

    random.seed(42)
    prompt_data = generate_addition_prompt(n=10)
    print(f"Prompts: {prompt_data}")

    inputs = tokenizer.encode(
        prompt_data[0], return_tensors="pt", add_special_tokens=True
    ).to(device)

    matrices = []
    activated_features = get_active_features(prompt_data, tokenizer, model, device)
    for layer in tqdm(range(len(activated_features) - 1)):
        grad_matrix = compute_gradient_matrix(
            inputs,
            layer,
            layer + 1,
            activated_features[layer],
            activated_features[layer + 1],
            model,
            verbose=True,
        )
        matrices.append(grad_matrix.cpu())  # move to CPU, free compute graph refs

    # Free model + SAE cache before stats/graphs to avoid OOM
    del model, tokenizer, inputs, activated_features
    _sae_cache.clear()
    gc.collect()

    # %%
    import sys
    for i, mat in enumerate(matrices):
        print(f"Layer {i} -> {i+1}: grad matrix shape={mat.shape}, "
              f"max={mat.abs().max():.4f}, mean={mat.abs().mean():.4f}")
    sys.stdout.flush()

    for idx, mat in enumerate(matrices):
        # Skip graph drawing for large matrices — nx.bipartite_layout is O(n^2)
        # and will hang with thousands of nodes per side
        if max(mat.shape) > 500:
            print(f"  Skipping graph for layer {idx}->{idx+1} (too large: {mat.shape})")
            continue
        draw_bipartite_graph(mat, threshold=0.05)
