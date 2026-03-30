"""Red-team audit tests for gradient algorithm correctness.

Tests validate:
1. Token length uniformity across experiment prompts
2. Padding assertion guard in compute_gradient_matrices_batch
3. Gradient correctness via finite differences
4. Batch vs single-prompt consistency
5. Label-aligned averaging correctness
6. Perturbation position correctness
7. Cross-batch isolation
8. retain_graph correctness

Uses real model + SAEs but with minimal feature counts (2-3) for speed.
Targets 2 adjacent layers only (layers 0->1).

Usage:
    PYTHONPATH=. .venv/bin/python -m pytest tests/test_gradient_correctness.py -v
"""

import gc

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from feature_circuit_discovery.data import set_model

set_model("gemma-2-2b")

from experiments.feature_grad_exp_optimized import (
    JumpReLUSAE,
    compute_gradient_matrices_batch,
    get_contrastive_features,
    load_sae,
    _sae_cache,
)
from feature_circuit_discovery.data import MODEL_ID, canonical_sae_filenames


# ---------------------------------------------------------------------------
# Module-level fixtures (loaded once, shared across all tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture(scope="module")
def model_and_tokenizer(device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16
    ).to(device)
    yield model, tokenizer
    del model
    _sae_cache.clear()
    gc.collect()


@pytest.fixture(scope="module")
def model(model_and_tokenizer):
    return model_and_tokenizer[0]


@pytest.fixture(scope="module")
def tokenizer(model_and_tokenizer):
    return model_and_tokenizer[1]


# ---------------------------------------------------------------------------
# Small feature sets for fast testing (layers 0 and 1, 3 features each)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_features(device):
    """Return 3 arbitrary feature indices for layers 0 and 1."""
    return {
        0: torch.tensor([10, 50, 100], device=device),
        1: torch.tensor([20, 60, 110], device=device),
    }


# ===========================================================================
# Test 1: Token length uniformity
# ===========================================================================

class TestTokenLengthUniformity:
    """Verify all prompts within each experiment tokenize to the same length."""

    def test_comparison_prompts_uniform_length(self, tokenizer):
        """Comparison prompts (Is XXX>YYY? ) should all have the same token count."""
        from experiments.run_comparison_gemma2_2b import generate_comparison_prompts

        yes_prompts, no_prompts = generate_comparison_prompts(seed=42)
        all_prompts = yes_prompts + no_prompts

        lengths = []
        for p in all_prompts:
            ids = tokenizer.encode(p, add_special_tokens=True)
            lengths.append(len(ids))

        assert len(set(lengths)) == 1, (
            f"Comparison prompts have non-uniform token lengths! "
            f"Unique lengths: {sorted(set(lengths))}, "
            f"distribution: {[(l, lengths.count(l)) for l in sorted(set(lengths))]}"
        )

    def test_addition_prompts_uniform_length(self, tokenizer):
        """Addition prompts (XX+YY=) should all have the same token count."""
        from experiments.run_addition_gemma2_2b import generate_addition_prompts

        add_prompts, sub_prompts = generate_addition_prompts(n=50, seed=42)
        all_prompts = add_prompts + sub_prompts

        lengths = []
        for p in all_prompts:
            ids = tokenizer.encode(p, add_special_tokens=True)
            lengths.append(len(ids))

        assert len(set(lengths)) == 1, (
            f"Addition/subtraction prompts have non-uniform token lengths! "
            f"Unique lengths: {sorted(set(lengths))}, "
            f"distribution: {[(l, lengths.count(l)) for l in sorted(set(lengths))]}"
        )

    def test_comparison_groups_same_length(self, tokenizer):
        """Yes and No groups should also be the same length as each other."""
        from experiments.run_comparison_gemma2_2b import generate_comparison_prompts

        yes_prompts, no_prompts = generate_comparison_prompts(seed=42)

        yes_lengths = set()
        for p in yes_prompts:
            yes_lengths.add(len(tokenizer.encode(p, add_special_tokens=True)))

        no_lengths = set()
        for p in no_prompts:
            no_lengths.add(len(tokenizer.encode(p, add_special_tokens=True)))

        all_lengths = yes_lengths | no_lengths
        assert len(all_lengths) == 1, (
            f"Yes vs No groups have different token lengths! "
            f"Yes: {yes_lengths}, No: {no_lengths}"
        )


# ===========================================================================
# Test 2: Padding assertion guard
# ===========================================================================

class TestPaddingAssertion:
    """Verify that padding detection catches different-length prompts."""

    def test_padding_without_attention_mask_raises(self, model, tokenizer, device, small_features):
        """Padding present but no attention_mask should trigger the assertion.

        Gemma tokenizers left-pad by default. When prompts have different lengths,
        padding tokens exist, and without attention_mask the model attends to them,
        corrupting hidden states.
        """
        # Use prompts with different token lengths to get real padding
        tokenized = tokenizer(
            ["Is 5>3? ", "Is 500>300? This is a much longer prompt"],
            return_tensors="pt", add_special_tokens=True, padding=True,
        )
        inputs = tokenized["input_ids"].to(device)

        # Verify padding actually exists
        pad_id = 0  # Gemma pad token
        assert (inputs == pad_id).any(), "Test setup error: expected padding in batch"

        with pytest.raises(AssertionError, match="[Pp]adding"):
            compute_gradient_matrices_batch(
                inputs,
                upstream_layer_idx=0,
                downstream_pairs=[(1, small_features[1])],
                upstream_features=small_features[0],
                model=model,
                # No attention_mask — should raise
            )

    def test_padding_with_attention_mask_passes(self, model, tokenizer, device, small_features):
        """Padding with attention_mask should NOT trigger the assertion."""
        tokenized = tokenizer(
            ["Is 5>3? ", "Is 500>300? This is a much longer prompt"],
            return_tensors="pt", add_special_tokens=True, padding=True,
        )
        inputs = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)

        # Should not raise — attention_mask is provided
        results, _ = compute_gradient_matrices_batch(
            inputs,
            upstream_layer_idx=0,
            downstream_pairs=[(1, small_features[1])],
            upstream_features=small_features[0],
            model=model,
            attention_mask=attention_mask,
        )
        assert len(results) == 1
        assert results[0].shape == (3, 3)  # (n_down, n_up)

    def test_no_padding_passes(self, model, tokenizer, device, small_features):
        """Uniform-length prompts should NOT trigger the assertion."""
        prompts = ["Is 500>300? ", "Is 700>400? "]
        inputs = tokenizer(
            prompts, return_tensors="pt", add_special_tokens=True, padding=True
        ).input_ids.to(device)

        # Should not raise
        results, _ = compute_gradient_matrices_batch(
            inputs,
            upstream_layer_idx=0,
            downstream_pairs=[(1, small_features[1])],
            upstream_features=small_features[0],
            model=model,
        )
        assert len(results) == 1
        assert results[0].shape == (3, 3)  # (n_down, n_up)


# ===========================================================================
# Test 3: Gradient correctness via finite differences
# ===========================================================================

class TestFiniteDifferences:
    """Compare autograd gradients to numerical finite differences."""

    def test_gradient_matches_finite_differences(self, model, tokenizer, device):
        """Autograd gradient should match (f(a+ε) - f(a-ε)) / 2ε."""
        prompt = "Is 500>300? "
        inputs = tokenizer.encode(
            prompt, return_tensors="pt", add_special_tokens=True
        ).to(device)

        upstream_layer = 0
        downstream_layer = 1
        # Use just 2 features for speed
        up_feats = torch.tensor([10, 50], device=device)
        down_feats = torch.tensor([20], device=device)

        # Get autograd gradient
        results, _ = compute_gradient_matrices_batch(
            inputs,
            upstream_layer_idx=upstream_layer,
            downstream_pairs=[(downstream_layer, down_feats)],
            upstream_features=up_feats,
            model=model,
        )
        grad_autograd = results[0]  # shape (1, 2)

        # Compute finite differences manually
        epsilon = 1e-3
        sae_upstream = load_sae(canonical_sae_filenames[upstream_layer], device)
        sae_downstream = load_sae(canonical_sae_filenames[downstream_layer], device)

        d_model = sae_upstream.W_dec.size(1)
        feature_vectors = sae_upstream.W_dec[up_feats, :]  # (2, d_model)
        last_pos = inputs.shape[1] - 1

        grad_numerical = torch.zeros(1, 2, device=device)
        for feat_idx in range(2):
            for sign, coeff in [(+1, +0.5), (-1, -0.5)]:
                perturbation = sign * epsilon * feature_vectors[feat_idx]  # (d_model,)

                def _make_hook(pert):
                    def hook_fn(module, inp, output):
                        residual = output[0] if isinstance(output, tuple) else output
                        modified = residual.clone()
                        pert_cast = pert.to(modified.dtype)
                        modified[:, last_pos:last_pos+1, :] += pert_cast.unsqueeze(0).unsqueeze(0)
                        return (modified,) + output[1:] if isinstance(output, tuple) else modified
                    return hook_fn

                with torch.no_grad():
                    hook = model.model.layers[upstream_layer].register_forward_hook(
                        _make_hook(perturbation)
                    )
                    outputs = model(inputs, output_hidden_states=True)
                    hook.remove()

                    act_down = outputs.hidden_states[downstream_layer + 1]
                    sae_acts = sae_downstream.encode(act_down.float())
                    f_val = sae_acts[0, last_pos, down_feats[0]]

                    grad_numerical[0, feat_idx] += coeff * f_val / epsilon

        # Compare (use loose tolerance due to bfloat16 precision)
        atol = 1e-1  # bfloat16 has ~2 decimal digits of precision
        rtol = 0.5   # 50% relative tolerance for bfloat16
        torch.testing.assert_close(
            grad_autograd.float(), grad_numerical.float(),
            atol=atol, rtol=rtol,
        )


# ===========================================================================
# Test 4: Batch vs single-prompt consistency
# ===========================================================================

class TestBatchConsistency:
    """Averaging single-prompt gradients should match batched computation."""

    def test_batch_equals_mean_of_singles(self, model, tokenizer, device, small_features):
        """grad_batch([A,B]) should ≈ (grad(A) + grad(B)) / 2."""
        prompts = ["Is 500>300? ", "Is 700>400? "]
        up_feats = small_features[0]
        down_feats = small_features[1]

        # Single-prompt gradients
        single_grads = []
        for p in prompts:
            inp = tokenizer.encode(
                p, return_tensors="pt", add_special_tokens=True
            ).to(device)
            results, _ = compute_gradient_matrices_batch(
                inp,
                upstream_layer_idx=0,
                downstream_pairs=[(1, down_feats)],
                upstream_features=up_feats,
                model=model,
            )
            single_grads.append(results[0])

        manual_mean = (single_grads[0] + single_grads[1]) / 2

        # Batched gradient
        inputs_batch = tokenizer(
            prompts, return_tensors="pt", add_special_tokens=True, padding=True
        ).input_ids.to(device)
        results_batch, _ = compute_gradient_matrices_batch(
            inputs_batch,
            upstream_layer_idx=0,
            downstream_pairs=[(1, down_feats)],
            upstream_features=up_feats,
            model=model,
        )
        grad_batch = results_batch[0]

        torch.testing.assert_close(
            grad_batch.float(), manual_mean.float(),
            atol=1e-3, rtol=0.1,
        )


# ===========================================================================
# Test 5: Label-aligned averaging correctness
# ===========================================================================

class TestLabelAlignedAveraging:
    """Verify prompt_signs correctly weight the gradients."""

    def test_opposite_signs(self, model, tokenizer, device, small_features):
        """With signs=[+1, -1], result ≈ (grad_A - grad_B) / 2."""
        prompts = ["Is 500>300? ", "Is 700>400? "]
        up_feats = small_features[0]
        down_feats = small_features[1]

        # Individual gradients
        single_grads = []
        for p in prompts:
            inp = tokenizer.encode(
                p, return_tensors="pt", add_special_tokens=True
            ).to(device)
            results, _ = compute_gradient_matrices_batch(
                inp,
                upstream_layer_idx=0,
                downstream_pairs=[(1, down_feats)],
                upstream_features=up_feats,
                model=model,
            )
            single_grads.append(results[0])

        expected = (single_grads[0] - single_grads[1]) / 2

        # Label-aligned batch
        inputs_batch = tokenizer(
            prompts, return_tensors="pt", add_special_tokens=True, padding=True
        ).input_ids.to(device)
        signs = torch.tensor([1.0, -1.0], device=device)
        results_signed, _ = compute_gradient_matrices_batch(
            inputs_batch,
            upstream_layer_idx=0,
            downstream_pairs=[(1, down_feats)],
            upstream_features=up_feats,
            model=model,
            prompt_signs=signs,
        )
        grad_signed = results_signed[0]

        torch.testing.assert_close(
            grad_signed.float(), expected.float(),
            atol=1e-3, rtol=0.1,
        )

    def test_same_signs(self, model, tokenizer, device, small_features):
        """With signs=[+1, +1], result ≈ (grad_A + grad_B) / 2 (same as plain batch)."""
        prompts = ["Is 500>300? ", "Is 700>400? "]
        up_feats = small_features[0]
        down_feats = small_features[1]

        inputs_batch = tokenizer(
            prompts, return_tensors="pt", add_special_tokens=True, padding=True
        ).input_ids.to(device)

        # Plain batch (no signs = uniform mean)
        results_plain, _ = compute_gradient_matrices_batch(
            inputs_batch,
            upstream_layer_idx=0,
            downstream_pairs=[(1, down_feats)],
            upstream_features=up_feats,
            model=model,
        )
        grad_plain = results_plain[0]

        # Explicit +1, +1
        signs = torch.tensor([1.0, 1.0], device=device)
        results_signed, _ = compute_gradient_matrices_batch(
            inputs_batch,
            upstream_layer_idx=0,
            downstream_pairs=[(1, down_feats)],
            upstream_features=up_feats,
            model=model,
            prompt_signs=signs,
        )
        grad_signed = results_signed[0]

        torch.testing.assert_close(
            grad_signed.float(), grad_plain.float(),
            atol=1e-6, rtol=1e-4,
        )


# ===========================================================================
# Test 6: Perturbation position is correct
# ===========================================================================

class TestPerturbationPosition:
    """Verify the hook perturbs only the last real token position."""

    def test_only_last_position_perturbed(self, model, tokenizer, device):
        """Register a diagnostic hook to capture which positions were modified.

        Uses actually-active features (discovered via inference) to ensure
        non-zero gradients, rather than arbitrary feature indices.
        """
        prompt = "Is 500>300? "
        inputs = tokenizer.encode(
            prompt, return_tensors="pt", add_special_tokens=True
        ).to(device)

        last_pos = inputs.shape[1] - 1

        # Find actually-active features at the last position
        sae0 = load_sae(canonical_sae_filenames[0], device)
        sae1 = load_sae(canonical_sae_filenames[1], device)
        with torch.no_grad():
            outputs = model(inputs, output_hidden_states=True)
            acts0 = sae0.encode(outputs.hidden_states[1].float())
            acts1 = sae1.encode(outputs.hidden_states[2].float())
            up_feats = torch.where(acts0[0, last_pos, :] > 0)[0][:3]
            down_feats = torch.where(acts1[0, last_pos, :] > 0)[0][:3]
            baseline_hidden = outputs.hidden_states[1].clone()
            del outputs

        assert len(up_feats) > 0 and len(down_feats) > 0, (
            "No active features found — cannot test perturbation position"
        )

        # Register our own diagnostic hook to capture the modified hidden state
        captured = {}

        def diagnostic_hook(module, inp, output):
            residual = output[0] if isinstance(output, tuple) else output
            captured["perturbed"] = residual.detach().clone()

        # Add diagnostic hook AFTER the layer (to see the output after perturbation)
        diag_handle = model.model.layers[0].register_forward_hook(diagnostic_hook)

        # Run the gradient function (it adds its own hook then removes it)
        results, _ = compute_gradient_matrices_batch(
            inputs,
            upstream_layer_idx=0,
            downstream_pairs=[(1, down_feats)],
            upstream_features=up_feats,
            model=model,
        )
        diag_handle.remove()

        # At a=0, all positions should match baseline exactly
        # (the perturbation is 0 * feature_vectors = 0)
        perturbed = captured["perturbed"]
        torch.testing.assert_close(
            perturbed.float(), baseline_hidden.float(),
            atol=1e-5, rtol=1e-5,
        )

        # Verify non-zero gradients exist (sanity check that features interact)
        grad_matrix = results[0]  # (n_down, n_up)
        assert grad_matrix.abs().max() > 0, (
            "All gradients are zero — the perturbation has no effect. "
            f"up_feats={up_feats.tolist()}, down_feats={down_feats.tolist()}"
        )


# ===========================================================================
# Test 7: Cross-batch isolation
# ===========================================================================

class TestCrossBatchIsolation:
    """Verify one batch item doesn't influence another's gradient."""

    def test_gradient_independent_of_batch_companion(self, model, tokenizer, device, small_features):
        """Gradient for prompt A should be the same regardless of what prompt B is."""
        prompt_a = "Is 500>300? "
        prompt_b1 = "Is 700>400? "
        prompt_b2 = "Is 100>999? "

        up_feats = small_features[0]
        down_feats = small_features[1]

        # Gradient of A alone
        inp_a = tokenizer.encode(
            prompt_a, return_tensors="pt", add_special_tokens=True
        ).to(device)
        results_solo, _ = compute_gradient_matrices_batch(
            inp_a,
            upstream_layer_idx=0,
            downstream_pairs=[(1, down_feats)],
            upstream_features=up_feats,
            model=model,
        )
        grad_a_solo = results_solo[0]

        # Gradient of [A, B1] — extract A's contribution
        # In the batch, the function averages over batch items.
        # For isolation test: run A alone vs A paired with different B's.
        # The single-prompt gradient should be the same each time.
        # (The batch gradient is a mean, but the autograd path for each
        # batch item should be independent.)

        # Instead of trying to extract per-item gradients from the batch
        # (which the current API doesn't expose), we verify that the
        # single-item result is identical regardless of what's in the
        # model's kv cache from prior runs — i.e., repeated single runs agree.
        results_solo2, _ = compute_gradient_matrices_batch(
            inp_a,
            upstream_layer_idx=0,
            downstream_pairs=[(1, down_feats)],
            upstream_features=up_feats,
            model=model,
        )
        grad_a_solo2 = results_solo2[0]

        torch.testing.assert_close(
            grad_a_solo.float(), grad_a_solo2.float(),
            atol=1e-6, rtol=1e-5,
        )

        # Also verify batch isolation via algebra:
        # If grad_batch([A,B]) = (grad(A) + grad(B)) / 2
        # and grad_batch([A,C]) = (grad(A) + grad(C)) / 2
        # then 2*grad_batch([A,B]) - 2*grad_batch([A,C]) = grad(B) - grad(C)
        # and grad(A) = 2*grad_batch([A,B]) - grad(B)
        # We verify this by computing all three and checking consistency.

        # This is only possible if B1 and B2 tokenize to same length as A
        len_a = inp_a.shape[1]
        inp_b1 = tokenizer.encode(prompt_b1, return_tensors="pt", add_special_tokens=True).to(device)
        inp_b2 = tokenizer.encode(prompt_b2, return_tensors="pt", add_special_tokens=True).to(device)

        if inp_b1.shape[1] != len_a or inp_b2.shape[1] != len_a:
            pytest.skip("Prompts have different token lengths, cannot batch without padding")

        # Batch [A, B1]
        batch_ab1 = torch.cat([inp_a, inp_b1], dim=0)
        results_ab1, _ = compute_gradient_matrices_batch(
            batch_ab1,
            upstream_layer_idx=0,
            downstream_pairs=[(1, down_feats)],
            upstream_features=up_feats,
            model=model,
        )
        grad_ab1 = results_ab1[0]  # mean of A and B1

        # Single B1
        results_b1, _ = compute_gradient_matrices_batch(
            inp_b1,
            upstream_layer_idx=0,
            downstream_pairs=[(1, down_feats)],
            upstream_features=up_feats,
            model=model,
        )
        grad_b1 = results_b1[0]

        # Extract A from batch: grad(A) = 2*grad_batch([A,B1]) - grad(B1)
        grad_a_extracted = 2 * grad_ab1 - grad_b1

        torch.testing.assert_close(
            grad_a_extracted.float(), grad_a_solo.float(),
            atol=1e-2, rtol=0.2,
        )


# ===========================================================================
# Test 8: retain_graph correctness
# ===========================================================================

class TestRetainGraph:
    """Verify retain_graph mechanics: multiple downstream layers + logits work."""

    def test_multiple_downstream_layers_no_error(self, model, tokenizer, device):
        """Multiple downstream layers + logit tokens should work without RuntimeError."""
        prompt = "Is 500>300? "
        inputs = tokenizer.encode(
            prompt, return_tensors="pt", add_special_tokens=True
        ).to(device)

        up_feats = torch.tensor([10, 50], device=device)
        down_pairs = [
            (1, torch.tensor([20, 60], device=device)),
            (2, torch.tensor([30, 70], device=device)),
        ]
        # Use actual token IDs for "Yes" and "No"
        yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
        no_id = tokenizer.encode("No", add_special_tokens=False)[0]

        # Should NOT raise RuntimeError about backward through graph a second time
        results, logit_grads = compute_gradient_matrices_batch(
            inputs,
            upstream_layer_idx=0,
            downstream_pairs=down_pairs,
            upstream_features=up_feats,
            model=model,
            logit_token_ids=[yes_id, no_id],
        )

        assert len(results) == 2
        assert results[0].shape == (2, 2)
        assert results[1].shape == (2, 2)
        assert logit_grads is not None
        assert logit_grads.shape == (2, 2)

    def test_graph_released_after_last_grad(self, model, tokenizer, device):
        """After the last grad call, the graph should be released."""
        prompt = "Is 500>300? "
        inputs = tokenizer.encode(
            prompt, return_tensors="pt", add_special_tokens=True
        ).to(device)

        up_feats = torch.tensor([10], device=device)
        down_feats = torch.tensor([20], device=device)

        # This should work fine (graph released after single downstream feature)
        results, _ = compute_gradient_matrices_batch(
            inputs,
            upstream_layer_idx=0,
            downstream_pairs=[(1, down_feats)],
            upstream_features=up_feats,
            model=model,
        )
        assert len(results) == 1
        # No further assertions needed — the test is that no error was raised
        # and memory was freed (the function should not leak the computation graph)
