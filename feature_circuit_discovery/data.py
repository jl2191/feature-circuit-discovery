# Gemma Scope 2 SAE configuration for Gemma 3 1B
SAE_HF_REPO = "google/gemma-scope-2-1b-pt"
SAE_L0_SIZE = "big"  # Options: "big", "small"
SAE_WIDTH = "16k"

# Model configuration
MODEL_ID = "google/gemma-3-1b-pt"
NEURONPEDIA_MODEL_ID = "gemma-3-1b"
NEURONPEDIA_SAE_SET = "gemmascope-2-res-16k"


def sae_filename(layer: int) -> str:
    """Return the HuggingFace Hub path for a given layer's SAE weights."""
    return f"resid_post_all/layer_{layer}_width_{SAE_WIDTH}_l0_{SAE_L0_SIZE}/params.safetensors"


# resid_post_all covers layers 0-20 for the 1B model
MAX_SAE_LAYER = 20
canonical_sae_filenames = [sae_filename(i) for i in range(MAX_SAE_LAYER + 1)]
