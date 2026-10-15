# Step 9: Inference -- Greedy Decoding

> **Status**: **Code complete** -- conceptual deep-dive deferred.
> Code, tests, and end-to-end demo all in place. The narrative
> sections (autoregressive paradigm, exposure bias, decoding
> strategies, KV caching) intentionally remain placeholders so the
> reader can return when ready for the deeper material.
>
> **Paper section**: 3 (auto-regressive), 6.1 (beam search mentioned).
> **Depends on**: Step 8 (a trained model).

[< Training](08_training.md) | [Overview](00_overview.md) | [Next: Verification >](10_verification.md)

---

## What is built

Two public functions in
`nn/numpy-transformer/src/numpy_transformer/inference.py`:

### `greedy_decode(transformer, source_ids, sos_id, eos_id, max_len)`

Single-sequence autoregressive decoding via argmax. The minimal loop:

```
1. transformer.eval()                     # dropout off, deterministic
2. enc_out = transformer.encoder.forward(source_ids)
3. target_ids = [[SOS]]
4. for _ in range(max_len):
5.     logits = transformer.decoder.forward(target_ids, enc_out)
6.     next_token = argmax(logits[0, -1, :])
7.     append next_token to target_ids
8.     if next_token == EOS: break
9. return target_ids[1:]                  # drop the seed SOS
```

The encoder runs once; the decoder runs once per emitted token. No KV
caching -- per-step work grows as the target length grows. For our
toy `n=6` task this is microseconds; for production use this would be
the first thing to optimise.

### `evaluate_sort_accuracy(transformer, encoder_inputs, decoder_targets, sos_id, eos_id, max_len, return_examples=0)`

Decode every example in a test split with `greedy_decode` and compare
element-wise against `decoder_targets`. Returns the exact-match rate
plus an optional list of `(source, predicted, expected)` triples.

## What "works" looks like

After a 2000-step training run (`scripts/train_sort.py` defaults), the
script's post-training evaluation prints something like:

```
Decoding the first 8 test examples (greedy):
  [OK ] input [4, 7, 8, 6, 1] -> predicted [1, 4, 6, 7, 8, EOS]  (expected [1, 4, 6, 7, 8, EOS])
  [OK ] input [2, 7, 1, 8, 3] -> predicted [1, 2, 3, 7, 8, EOS]  (expected [1, 2, 3, 7, 8, EOS])
  ...
Exact-match accuracy on 1000 held-out examples: 100.0% (1000/1000)
```

This is the canonical "Step 1-9 worked" artefact.

## Tests

`nn/numpy-transformer/tests/test_inference.py` (9 tests):

- `test_greedy_decode_returns_int_array_with_max_len_cap`
- `test_greedy_decode_stops_immediately_when_eos_dominates`
- `test_greedy_decode_respects_max_len_when_eos_never_emitted`
- `test_greedy_decode_accepts_2d_source_with_batch_one`
- `test_greedy_decode_rejects_batch_size_above_one`
- `test_evaluate_sort_accuracy_handles_zero_correct`
- `test_evaluate_sort_accuracy_returns_examples_when_requested`
- **`test_trained_model_actually_sorts`** -- headline integration test.
  Trains a 43k-parameter Transformer for 1500 steps and asserts that
  greedy decoding correctly sorts > 70% of held-out 5-digit sequences.
- `test_trained_model_outputs_eos_within_max_len`

---

## Topics deliberately deferred

The conceptual material listed below is **not** documented in this
file. They were the topics a full slice 1 deep-dive would have
covered; they are valuable but require their own focused study session.

- **Autoregressive paradigm and the train/inference asymmetry.**
  Training is parallel (teacher-forced through the causal mask);
  inference is sequential (one token at a time, feeding back the
  model's own predictions).
- **Exposure bias.** The train-test mismatch where training feeds the
  model ground truth at every step but inference feeds the model its
  own (possibly wrong) previous predictions. Errors compound during
  generation in a way they never do during training.
- **Decoding strategies.** Sampling (with temperature), top-k, top-p
  (nucleus), beam search with length penalty (paper section 6.1 uses
  `beam=4, alpha=0.6`). Greedy was sufficient for the deterministic
  sort task; for non-deterministic generation tasks the choice matters
  considerably.
- **KV caching.** Cache key/value projections in self-attention so the
  decoder does O(n) work per token instead of O(n²). The standard
  inference-time speedup; a meaningful re-architecture of the MHA
  forward path.
- **Batched inference.** Decoding multiple sequences in parallel with
  per-sequence EOS masking and dynamic stopping.

---

*Return here for the conceptual deep-dive whenever you want; the code
will still be in place.*

---

## Discussion Log

A chronological log of every conceptual exchange. Append, do not edit.

### 2026-04-28 -- The information barrier and asymmetric field usage (Q&A from chat)

The user, looking at `run_evaluation` in `scripts/train_sort.py` and
the `sort_test.npz` schema, asked two questions in sequence that each
strengthen the autoregressive paradigm intentionally deferred from the
main body of this doc:

1. "I thought test_data would not need a target."
2. "So `decoder_input` of test_data is never used, right?"

Both are exactly the right thing to be asking when first encountering
a held-out test set. Captured here near-verbatim so the reasoning is
preserved.

#### The information barrier between the model and the test labels

The intuition behind question 1 is correct: at test time, the model
must *not* read the answer. It must produce its prediction from the
encoder input alone (plus the seed `SOS`), and only after the
prediction is final do we look at the target -- to count whether the
prediction was right. This is sometimes called the **information
barrier** between the model and the test labels.

But this means the *target plays two completely different roles*
depending on which side of that barrier you are on:

| Role               | Who uses it           | When                                 |
|--------------------|-----------------------|--------------------------------------|
| **Input to model** | the model itself      | during the forward pass              |
| **Answer key**     | the evaluation script | comparing model's output to truth    |

The intuition "test_data shouldn't need a target" is exactly right for
**role 1**. But role 2 is unavoidable -- if you don't have ground truth
answers, you can't compute accuracy. So the test split *does* need to
ship the target, but the model never reads it. That contract is what
`run_evaluation` enforces:

- `greedy_decode(transformer, src, ...)` -- only `encoder_input` enters
  the forward pass. The function takes `sos_id` and `eos_id` as
  scalars, not the target sequence. There is no place to "leak"
  the answer in.
- `pred == expected` -- the target is touched only here, *after* the
  prediction is fixed. Used to count, not to compute.

A natural way to verify this in the F5 debug session: set a breakpoint
at the start of `greedy_decode` and watch the call arguments. You will
see `source_ids` enter, but `decoder_target` never appears anywhere in
the call stack until after the function returns and the comparison
runs.

#### Asymmetric field usage between train and test splits

The follow-up question ("is `decoder_input` of test_data ever used?")
revealed a real but subtle property of the codebase: train and test
consume the three `.npz` fields asymmetrically.

| Field          | Train (in `train()`)             | Test (in `run_evaluation`)            |
|----------------|----------------------------------|---------------------------------------|
| encoder_input  | fed to encoder                   | fed to encoder via `greedy_decode`    |
| decoder_input  | teacher-forced into decoder      | **not used** (only its shape is printed) |
| decoder_target | compared with logits for loss    | answer key for accuracy comparison    |

Confirmed by grep: `test_data['decoder_input']` is referenced in
*exactly one place* in the entire codebase -- a `print(...shape)` line
in the dataset-summary banner of `scripts/train_sort.py`. It is never
fed to any layer, never compared with anything, never read by
`greedy_decode`, never read by `evaluate_sort_accuracy`.

So why keep it on disk at all?

1. **Symmetry.** Both splits are produced by the same
   `generate_sort_split()` function in `sort_data.py`, so they share
   the same three-field schema. Special-casing the test split to omit
   `decoder_input` would add a code path for a few KB of disk savings
   -- not worth it.
2. **Optional teacher-forced test loss.** If a future iteration wanted
   to track the *generalization gap* (train CE vs test CE) alongside
   greedy-decode accuracy -- a useful overfitting diagnostic that
   accuracy alone can't surface -- it would need `decoder_input` on
   the test side to do a teacher-forced forward pass over the test
   set. The data is there, ready, for whoever wants it.

This entry pairs with the same-day entry in `08_training.md`'s
Discussion Log, which captures the train/test split design (same
generator, different seeds, different sizes) that triggered the
broader conversation. The two entries together give the full picture
of how the dataset is shaped and how each side consumes it.

#### Train-time vs test-time asymmetry, side by side

The visual model from chat that pulled the whole picture together:

```text
TRAINING (teacher forcing)                   TESTING (autoregressive)

encoder_input  -> encoder -> enc_out         encoder_input -> encoder -> enc_out
                                                                            |
decoder_input  -> decoder -> logits          [SOS]         -> decoder -> logits
   (= [SOS]+answer)             |                                          |
                                v                                       argmax
                  compare with                                             |
                  decoder_target -> loss     [SOS, t1]     -> decoder -> t2
                  (= answer+EOS)                                           |
                                             ... repeat until EOS ...
                                                                           |
                                                                           v
                                             compare predicted seq
                                             with decoder_target -> accuracy
```

Two different uses of "the answer":

- **Training.** Feed the answer in (`decoder_input`) so the decoder
  can learn the next-token distribution at every position in
  parallel. *Also* compare against `decoder_target` to compute loss.
  Both fields are used.
- **Testing.** Hide the answer from the model. Feed only
  `encoder_input`, then start from `[SOS]` and let the model generate
  one token at a time, feeding its *own predictions* back in. After
  it stops (or hits `max_len`), compare the generated sequence with
  `decoder_target` to score it.

Greedy decoding is the simplest realisation of this loop. Sampling,
top-k, top-p, and beam search (paper Section 6.1) are alternative
"how do you pick the next token from logits?" strategies that all
respect the same information barrier; they are deliberately deferred
to a future deep-dive (see "Topics deliberately deferred" above).

#### Cross-references

- Train/test split design and the dataset's three-field schema:
  `08_training.md` Discussion Log entry of the same date, plus
  `numpy-transformer/train_input/README.md`.
- Teacher-forcing mechanics on the train side: `08_training.md`
  Section 5 and `07_decoder.md` Section 5.
- The autoregressive loop in code: `greedy_decode` in
  `inference.py`.
- Verification that the model actually solves the task end-to-end:
  the `test_trained_model_actually_sorts` integration test in
  `tests/test_inference.py`.
