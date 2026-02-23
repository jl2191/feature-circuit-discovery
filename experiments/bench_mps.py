"""Benchmark: MPS vs CPU for gradient computation."""
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from experiments.feature_grad_exp_optimized import compute_gradient_matrix, get_active_features

for device_name in ["cpu", "mps"]:
    print(f"\n{'='*50}")
    print(f"  Device: {device_name}")
    print(f"{'='*50}")

    device = torch.device(device_name)

    try:
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2-2b", dtype=torch.bfloat16 if device_name == "cpu" else torch.float32
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
        print(f"  Model loaded in {time.time()-t0:.1f}s")

        prompts = ["754 + 214 = ", "125 + 859 = "]
        inputs = tokenizer.encode(prompts[0], return_tensors="pt", add_special_tokens=True).to(device)

        t0 = time.time()
        activated_features = get_active_features(prompts, tokenizer, model, device)
        print(f"  get_active_features: {time.time()-t0:.1f}s")

        up_feats = activated_features[0][:10]
        down_feats = activated_features[1][:10]

        t0 = time.time()
        grad_matrix = compute_gradient_matrix(
            inputs, 0, 1, up_feats, down_feats, model, verbose=False,
        )
        elapsed = time.time() - t0
        per_grad = elapsed / len(down_feats)
        print(f"  10x10 gradient matrix: {elapsed:.1f}s ({per_grad:.2f}s/grad)")
        print(f"  max={grad_matrix.abs().max():.4f}, nonzero={grad_matrix.count_nonzero()}/{grad_matrix.numel()}")

        del model, tokenizer, inputs, activated_features, grad_matrix
        torch.mps.empty_cache() if device_name == "mps" else None
    except Exception as e:
        print(f"  FAILED: {e}")

    import gc
    gc.collect()
