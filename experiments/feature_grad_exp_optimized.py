"""Re-exports from feature_circuit_discovery.core + standalone experiment script.

All core functions now live in feature_circuit_discovery.core.
This module re-exports them for backward compatibility with existing experiment scripts.
"""

# Re-export everything for backward compat
from feature_circuit_discovery.core import (  # noqa: F401
    JumpReLUSAE,
    _SAE_CACHE_MAXSIZE,
    _sae_cache,
    load_sae,
    find_frequent_nonzero_indices,
    get_active_features,
    get_contrastive_features,
    compute_gradient_matrix,
    compute_gradient_matrix_jacobian,
    compute_gradient_matrices_batch,
)


if __name__ == "__main__":
    import gc
    import random
    import sys

    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16
        ).to(device)
        print("Using CPU with bfloat16")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    def generate_addition_prompt(n: int = 1) -> list[str]:
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
        matrices.append(grad_matrix.cpu())

    del model, tokenizer, inputs, activated_features
    _sae_cache.clear()
    gc.collect()

    for i, mat in enumerate(matrices):
        print(f"Layer {i} -> {i+1}: grad matrix shape={mat.shape}, "
              f"max={mat.abs().max():.4f}, mean={mat.abs().mean():.4f}")
    sys.stdout.flush()
