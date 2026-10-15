# Project Status Tracker

> Living progress log for the Transformer-from-scratch build.
> The conceptual map is [00_overview.md](00_overview.md); this file just tracks what
> is done. Update this file as each step lands so the overview stays stable.

---

## Legend

- **Complete** -- doc is written, code is implemented, and tests pass.
- **Code complete / Mostly complete** -- code and tests are fully in place; some
  conceptual deep-dive material was deliberately deferred. The doc itself
  explains exactly what is and isn't covered.
- **In progress** -- some artifacts exist but the step is not yet closed out.
- **Not started** -- placeholder only.

Code paths below are relative to the workspace root `nn/`.

---

## Per-step status

| # | Step | Doc | Code | Tests | Status |
|---|------|-----|------|-------|--------|
| 0 | Overview | [docs/00_overview.md](00_overview.md) | -- | -- | **Complete** |
| 1 | Math Primitives | [docs/01_math_primitives.md](01_math_primitives.md) | `numpy-transformer/src/numpy_transformer/layers.py` (`Linear`, `Softmax`, `ReLU`, `LayerNorm`, `Dropout`, `Embedding`) | `numpy-transformer/tests/test_layers.py` | **Complete** |
| 2 | Positional Encoding | [docs/02_positional_encoding.md](02_positional_encoding.md) + [walkthrough](../numpy-transformer/docs/embedding_and_positional_encoding_walkthrough.md) | `numpy-transformer/src/numpy_transformer/layers.py` (`SinusoidalPositionalEncoding`) | `numpy-transformer/tests/test_layers.py` (`test_sinusoidal_positional_encoding_doc_example`, `_backward_is_identity`, `_broadcast_batch`) | **Complete** |
| 3 | Scaled Dot-Product Attention | [docs/03_scaled_dot_product_attention.md](03_scaled_dot_product_attention.md) | `numpy-transformer/src/numpy_transformer/layers.py` (`ScaledDotProductAttention`) | `numpy-transformer/tests/test_layers.py` (`test_sdpa_*`) | **Complete** |
| 4 | Multi-Head Attention | [docs/04_multi_head_attention.md](04_multi_head_attention.md) | `numpy-transformer/src/numpy_transformer/layers.py` (`MultiHeadAttention` -- forward + backward) | `numpy-transformer/tests/test_layers.py` (`test_mha_section5_trace`, `_sort_task_shapes`, `_d_model_not_divisible_raises`, `_input_dim_mismatch_raises`, `_param_count_matches_doc`, `_cross_attention_rectangular`, `_backward_gradient_check`, `_W_k_bias_gradient_is_exactly_zero`, `_single_head_reduces_to_sdpa`, `_head_permutation_invariance`) | **Complete** |
| 5 | Feed-Forward Network | [docs/05_feedforward_network.md](05_feedforward_network.md) | `numpy-transformer/src/numpy_transformer/layers.py` (`FeedForward`) | `numpy-transformer/tests/test_layers.py` (`test_ffn_section7_trace`, `_position_wise_independence`, `_input_dim_mismatch_raises`, `_param_count_matches_doc`, `_backward_gradient_check`, `_relu_kills_all_hidden`) | **Complete** |
| 6 | Encoder | [docs/06_encoder.md](06_encoder.md) | `numpy-transformer/src/numpy_transformer/layers.py` (`EncoderLayer`, `Encoder`; both Post-LN default and Pre-LN flag) | `numpy-transformer/tests/test_layers.py` (`test_encoder_layer_shape_preservation`, `_section9_shape_trace`, `_post_ln_outputs_unit_normalized`, `_pre_ln_residual_unbounded`, `_residual_passes_zero_sublayer_pre_ln`, `_backward_gradient_check_post_ln`, `_backward_gradient_check_pre_ln`, `test_encoder_full_forward_shape`, `_full_param_count_matches_doc`, `_dropout_off_at_eval_is_deterministic`, `_full_backward_gradient_check`) | **Complete** |
| 7 | Decoder | [docs/07_decoder.md](07_decoder.md) | `numpy-transformer/src/numpy_transformer/layers.py` (`TiedLinear`, `DecoderLayer`, `Decoder`, `Transformer`; both Post-LN default and Pre-LN flag; `tie_embeddings=True` default; `Embedding.backward` switched to PyTorch-style accumulate; `Encoder` accepts `shared_embedding`) | `numpy-transformer/tests/test_layers.py` (`test_decoder_layer_shape_preservation`, `_causal_mask_blocks_future`, `_cross_attn_rectangular_unmasked`, `_residual_passes_zero_sublayer_pre_ln`, `_backward_gradient_check_post_ln`, `_backward_gradient_check_pre_ln`, `test_decoder_full_forward_shape`, `_section9_shape_trace`, `_standalone_param_count_matches_doc`, `_tied_vs_untied_param_count`, `_tied_output_uses_shared_embedding_weight`, `_tied_gradient_reaches_untouched_rows`, `_dropout_off_at_eval_is_deterministic`, `_full_backward_gradient_check`, `_full_backward_gradient_check_untied`, `test_transformer_forward_shape`, `_shared_embedding_three_way`, `_full_param_count_matches_doc`, `_full_backward_gradient_check`, `_eval_deterministic`; existing `test_embedding` updated for the new accumulate contract) | **Complete** |
| 8 | Training | [docs/08_training.md](08_training.md) | `numpy-transformer/src/numpy_transformer/training.py` (`log_softmax`, `cross_entropy_with_label_smoothing`, `noam_lr`, `Adam`, `zero_grads`, `train`, `label_smoothing_loss_floor`); `numpy-transformer/scripts/train_sort.py` (runnable CLI; converges to label-smoothing floor in ~9s on CPU) | `numpy-transformer/tests/test_training.py` (25 tests: log_softmax stability/shape, CE with/without smoothing, smoothing floor identity, gradient finite-difference check, PAD masking, Noam shape and continuity, Adam first-step / long-run / per-parameter-state / in-place identity / bias correction, zero_grads accumulation prevention, integration `test_train_loop_actually_learns`, step-1 uniform-prior baseline) | **Complete** |
| 9 | Inference | [docs/09_inference.md](09_inference.md) | `numpy-transformer/src/numpy_transformer/inference.py` (`greedy_decode`, `evaluate_sort_accuracy`); `numpy-transformer/scripts/train_sort.py` extended with post-training decode demo + accuracy report | `numpy-transformer/tests/test_inference.py` (9 tests: greedy_decode shape/EOS/max_len behavior, accuracy helper basic cases, headline `test_trained_model_actually_sorts` integration test asserting >70% exact-match on held-out 5-digit sort sequences, EOS emission for trained model) | **Code complete** -- conceptual deep-dive (autoregressive paradigm, exposure bias, decoding strategies, KV caching, batched inference) deferred for future study |
| 10 | Verification | [docs/10_verification.md](10_verification.md) | Gradient checks already distributed across `tests/test_layers.py` and `tests/test_training.py` (per-layer, encoder, decoder, transformer end-to-end, CE+LS); param-count assertions in same; accuracy via `evaluate_sort_accuracy` (Step 9); loss curve via `train_sort.py` CSV | Subsumed by Steps 1-9 test suites: 109 tests covering analytical-vs-numerical gradients (`abs_err<1e-7` or `rel_err<1e-4`), parameter counts, end-to-end training, end-to-end accuracy | **Mostly complete** -- attention heatmap visualization (paper Figures 3-5) and matplotlib-based loss curve plot deferred |
| 11 | Model Persistence | [docs/11_model_persistence.md](11_model_persistence.md) | `numpy-transformer/src/numpy_transformer/persistence.py` (`save_transformer`, `load_transformer`, `Storage` Protocol, `LocalStorage` with atomic tmp+rename, `FsspecStorage` for `s3://`/`gs://`/`azure://` via opt-in `[cloud]` extra, `FORMAT_VERSION = 1`, `meta.*`/`params.*` namespacing, in-place load preserving tied-embedding aliasing); `numpy-transformer/scripts/train_sort.py` extended with `--save-path` and `--no-save` flags; `numpy_transformer/__init__.py` re-exports the persistence API; `pyproject.toml` declares `[project.optional-dependencies] cloud = ["fsspec", "s3fs"]` | `numpy-transformer/tests/test_persistence.py` (17 tests: round-trip tied/untied, tied-aliasing preserved across encoder/decoder/transformer, metadata round-trip, `format_version` present + missing/future/`<1` rejected, missing-meta/missing-param/extra-param/shape/dtype mismatches each raise with offending name, unknown-namespace forward-compat, `LocalStorage` leaves no `.tmp` on success, in-memory `_DictStorage` round-trip, and end-to-end train-save-load-decode) | **Complete** |
| 12 | PyTorch Port | [docs/12_pytorch_port.md](12_pytorch_port.md) | TBD: `pytorch-transformer/src/pytorch_transformer/...` (educational port mirroring NumPy architecture; layer-by-layer with cross-implementation tests; `nn.Module`-based with autograd; same hyperparameters; persistence via `torch.save`+metadata-versioning) | TBD | **Design phase** -- doc shipped 2026-04-26 with full design rationale; build plan has 5 slices (scaffolding, layer port, training loop, validation on digit sort, persistence) |
| 13 | Lottery Floor Experiment | TBD: `docs/13_lottery_floor_experiment.md` | TBD: `pytorch-transformer/src/pytorch_transformer/lottery_data.py`; `pytorch-transformer/scripts/train_lottery.py` | TBD | **Not started** -- depends on Step 12. Will run two experiments empirically demonstrating the lottery-prediction-impossibility argument from `reference_capabilities_and_limits.md` Section 6: Experiment A (held-out prediction; predicted to plateau at `log(69)`/`log(26)`) and Experiment B (in-sample memorization; predicted to drop toward 0). Powerball format (Option B vocab: separate token spaces, `vocab_size=99`). |

---

## Side artifacts (not a numbered step)

These exist in the tree but belong to an upcoming step; listed here so they are
not lost.

| Artifact | Purpose | Belongs to |
|----------|---------|-----------|
| `numpy-transformer/src/numpy_transformer/sort_data.py` | Toy sort-task dataset generator | Step 8 (Training) |
| `numpy-transformer/train_input/sort_{train,test}.npz` + previews | Cached toy dataset (input fixtures) | Step 8 (Training) |
| `numpy-transformer/train_input/README.md` | Dataset description | Step 8 (Training) |
| `numpy-transformer/train_output/loss_curve.csv` | Runtime training log (one row per step) | Step 8 (Training) |
| `numpy-transformer/train_model/sort_model.npz` | Persisted trained Transformer | Step 11 (Persistence) |
| `numpy-transformer/tests/test_sort_data.py` | Tests for the dataset generator | Step 8 (Training) |
| [`docs/reference_tokenization.md`](reference_tokenization.md) | Reference: tokenization, BPE, vocabularies in real-world models | Reference (not a step) |
| [`docs/reference_capabilities_and_limits.md`](reference_capabilities_and_limits.md) | Reference: length generalization, vocabulary limits, why LLMs are bad at arithmetic, why no model can predict the lottery -- all unified by "training distribution = capability boundary" | Reference (post Step 11) |
| [`docs/reference_numpy.md`](reference_numpy.md) | Reference: NumPy fundamentals used in the numpy-transformer build (shapes, broadcasting, reshape/swapaxes, axis reductions, matmul, scatter, in-place ops, numerical-stability patterns) | Reference (not a step) |

---

## How to update this file

When a step lands:

1. Flip its row **Status** to `In progress` or `Complete`.
2. Fill in the **Code** column with the symbols added (e.g. class names).
3. Fill in the **Tests** column with the test file and, when useful, the test names.
4. If the step produced extra artifacts (walkthroughs, datasets, notebooks), add
   them to the row's Doc/Code cell or to the "Side artifacts" table above.
5. Do **not** edit [00_overview.md](00_overview.md) for status-only changes --
   the overview should stay conceptual.
