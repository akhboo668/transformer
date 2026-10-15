# Step 10: Verification and Visualization

> **Status**: **Mostly complete** -- gradient checking and end-to-end
> accuracy are both done; attention visualization deferred.
>
> **Paper section**: Figures 3-5 (attention visualizations).
> **Depends on**: Step 9 (working inference).

[< Inference](09_inference.md) | [Overview](00_overview.md)

---

## What is in place

### Gradient checking (already done across Steps 1-8)

This was the headline item of Step 10's original plan. **It is already
covered in full** by the per-step pytest suite -- not as a separate
phase, but folded into each step's tests so regressions are caught at
the source. Specifically:

- Per-layer numerical-vs-analytical checks: `Linear`, `LayerNorm`,
  `Embedding`, `MultiHeadAttention`, `FeedForward`, etc.
- End-to-end through the `Encoder` (`test_encoder_full_backward_gradient_check`)
- End-to-end through the `Decoder` Post-LN and Pre-LN
  (`test_decoder_full_backward_gradient_check_post_ln`, `_pre_ln`)
- End-to-end through the `Transformer` with shared embeddings and
  three-way weight tying (`test_transformer_full_backward_gradient_check`)
- The cross-entropy + label-smoothing loss
  (`test_ce_gradient_finite_difference`)

Tolerance: `abs_err < 1e-7` or `rel_err < 1e-4` everywhere. If the code
is wrong, these tests catch it.

### End-to-end accuracy on the sort task (added in Step 9 / 10 code completion)

`evaluate_sort_accuracy` in `inference.py` plus the post-training
evaluation block in `scripts/train_sort.py` give the canonical
"did the model learn?" number. With default settings:

```
Exact-match accuracy on 1000 held-out examples: 100.0% (1000/1000)
```

All 1000 unseen sort examples are decoded correctly after 2000
training steps.

### Loss curve

`scripts/train_sort.py` writes `train_output/loss_curve.csv` with
`step, loss, lr` columns per training step, plotted easily in any
spreadsheet or notebook.

### Parameter-count assertions

Already covered by the test suite:
`test_encoder_full_param_count_matches_doc`,
`test_decoder_standalone_param_count_matches_doc`,
`test_transformer_full_param_count_matches_doc`.

---

## What is deferred

### Attention heatmap visualization (paper Figures 3-5)

Not implemented. Would require:

1. Modifying `MultiHeadAttention.forward` to optionally retain the
   attention weights (currently they are computed and discarded).
2. Adding `matplotlib` as a dependency (currently the project is
   pure-numpy and pure-stdlib).
3. A small script that decodes one input, captures the per-layer
   per-head per-sublayer attention matrices, and renders 6 heatmaps
   (encoder self-attention, decoder self-attention, decoder cross-
   attention; per layer).

Educational value is high -- watching cross-attention "look at" the
smallest unsorted digit when emitting the next token is satisfying --
but the lift is meaningful and was deliberately scoped out.

### A unified `verify` command

A single entry point that runs all gradient checks, accuracy
evaluation, and visualisations was sketched in the placeholder. It
would be largely redundant with the existing pytest suite plus the
training script's evaluation phase.

---

## How to verify the project right now

From the workspace root:

```
# 1. Run all 109 tests (~12 seconds; covers gradient checks, training,
#    inference, and accuracy thresholds on a small trained model):
cd nn
uv run --package numpy-transformer pytest numpy-transformer/tests/

# 2. Train a real model end-to-end and watch the accuracy report:
cd numpy-transformer
uv run --package numpy-transformer python scripts/train_sort.py
```

Expected outcomes:

- All tests pass.
- Training loss falls from ~2.56 (= log(13), the uniform-prior baseline)
  to ~0.58 (just above the label-smoothing floor of 0.5736).
- Exact-match accuracy on the 1000-example test split is 99-100%.

If those three things hold, every component built in Steps 1-9 is
working correctly.
