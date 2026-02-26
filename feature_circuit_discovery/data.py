# Multi-model SAE configuration registry
# ========================================
# Supports Gemma 3 1B (safetensors) and Gemma 2 2B (npz).
# Call set_model("gemma-2-2b") before importing other modules that read
# the module-level variables below.

from __future__ import annotations

_MODEL_CONFIGS: dict[str, dict] = {
    "gemma-3-1b": {
        "model_id": "google/gemma-3-1b-pt",
        "sae_hf_repo": "google/gemma-scope-2-1b-pt",
        "sae_width": "16k",
        "max_sae_layer": 20,
        "neuronpedia_model_id": "gemma-3-1b",
        "neuronpedia_sae_set": "gemmascope-2-res-16k",
        "sae_filenames": [
            f"resid_post_all/layer_{i}_width_16k_l0_big/params.safetensors"
            for i in range(21)
        ],
    },
    "gemma-2-2b": {
        "model_id": "google/gemma-2-2b",
        "sae_hf_repo": "google/gemma-scope-2b-pt-res",
        "sae_width": "16k",
        "max_sae_layer": 25,
        "neuronpedia_model_id": "gemma-2-2b",
        "neuronpedia_sae_set": "gemmascope-res-16k",
        # Each layer has a unique average_l0 value — hardcoded canonical paths
        "sae_filenames": [
            "layer_0/width_16k/average_l0_105/params.npz",
            "layer_1/width_16k/average_l0_102/params.npz",
            "layer_2/width_16k/average_l0_141/params.npz",
            "layer_3/width_16k/average_l0_59/params.npz",
            "layer_4/width_16k/average_l0_124/params.npz",
            "layer_5/width_16k/average_l0_68/params.npz",
            "layer_6/width_16k/average_l0_70/params.npz",
            "layer_7/width_16k/average_l0_69/params.npz",
            "layer_8/width_16k/average_l0_71/params.npz",
            "layer_9/width_16k/average_l0_73/params.npz",
            "layer_10/width_16k/average_l0_77/params.npz",
            "layer_11/width_16k/average_l0_80/params.npz",
            "layer_12/width_16k/average_l0_82/params.npz",
            "layer_13/width_16k/average_l0_84/params.npz",
            "layer_14/width_16k/average_l0_84/params.npz",
            "layer_15/width_16k/average_l0_78/params.npz",
            "layer_16/width_16k/average_l0_78/params.npz",
            "layer_17/width_16k/average_l0_77/params.npz",
            "layer_18/width_16k/average_l0_74/params.npz",
            "layer_19/width_16k/average_l0_73/params.npz",
            "layer_20/width_16k/average_l0_71/params.npz",
            "layer_21/width_16k/average_l0_70/params.npz",
            "layer_22/width_16k/average_l0_72/params.npz",
            "layer_23/width_16k/average_l0_75/params.npz",
            "layer_24/width_16k/average_l0_73/params.npz",
            "layer_25/width_16k/average_l0_116/params.npz",
        ],
    },
}

# ---------------------------------------------------------------------------
# Active configuration — module-level variables used by the rest of the codebase
# ---------------------------------------------------------------------------
_active = "gemma-3-1b"

MODEL_ID: str = _MODEL_CONFIGS[_active]["model_id"]
SAE_HF_REPO: str = _MODEL_CONFIGS[_active]["sae_hf_repo"]
SAE_WIDTH: str = _MODEL_CONFIGS[_active]["sae_width"]
MAX_SAE_LAYER: int = _MODEL_CONFIGS[_active]["max_sae_layer"]
NEURONPEDIA_MODEL_ID: str = _MODEL_CONFIGS[_active]["neuronpedia_model_id"]
NEURONPEDIA_SAE_SET: str = _MODEL_CONFIGS[_active]["neuronpedia_sae_set"]
canonical_sae_filenames: list[str] = list(_MODEL_CONFIGS[_active]["sae_filenames"])


def set_model(name: str) -> None:
    """Switch the active model config. Must be called before other modules read data.py vars.

    Args:
        name: One of "gemma-3-1b" or "gemma-2-2b".
    """
    global MODEL_ID, SAE_HF_REPO, SAE_WIDTH, MAX_SAE_LAYER
    global NEURONPEDIA_MODEL_ID, NEURONPEDIA_SAE_SET, canonical_sae_filenames
    global _active

    if name not in _MODEL_CONFIGS:
        raise ValueError(f"Unknown model {name!r}. Choose from: {list(_MODEL_CONFIGS)}")

    _active = name
    cfg = _MODEL_CONFIGS[name]
    MODEL_ID = cfg["model_id"]
    SAE_HF_REPO = cfg["sae_hf_repo"]
    SAE_WIDTH = cfg["sae_width"]
    MAX_SAE_LAYER = cfg["max_sae_layer"]
    NEURONPEDIA_MODEL_ID = cfg["neuronpedia_model_id"]
    NEURONPEDIA_SAE_SET = cfg["neuronpedia_sae_set"]
    canonical_sae_filenames[:] = cfg["sae_filenames"]
