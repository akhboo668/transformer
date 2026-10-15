# Reference: Model Capabilities and Limits

> Supplementary reference. Not a numbered step. Records the conceptual
> exploration that happened on 2026-04-26 after Step 11 (Model Persistence)
> made it possible to load a trained model and probe its behavior on
> out-of-distribution inputs.
>
> **Three big questions, three big lessons:**
>
> 1. Can a model trained on length-5 sequences sort longer ones?
> 2. Can it sort multi-digit numbers (or numbers outside its vocabulary)?
> 3. Why is no model able to predict the next lottery draw?
>
> All three answers reduce to the same underlying principle, which is one
> of the most important things to internalize about supervised learning:
> **the training distribution is the capability boundary.**
>
> Bonus topic: why LLMs are notoriously bad at arithmetic and counting
> letters -- and why it is a tokenization problem, not a model-size problem.

[< Back to Overview](00_overview.md) | [Step 11 (Persistence)](11_model_persistence.md) | [Reference: Tokenization](reference_tokenization.md)

---

## 1. Why this doc exists

By the end of Step 11, we had a fully trained Transformer:

- 43,168 parameters, `d_model=32`, 2 encoder + 2 decoder layers
- Trained on 10,000 sequences of length 5 with digits 0-9
- 100% exact-match accuracy on 1000 held-out examples
- Persisted to `numpy-transformer/train_model/sort_model.npz`

Because Step 11 made the model loadable as a reusable artifact, we could
ask questions about its *behavior* without retraining. Three questions
came up naturally:

1. **What happens if we feed it a longer sequence?** ([Section 3](#3-length-generalization))
2. **What happens if we feed it numbers outside its 13-token vocabulary?** ([Section 4](#4-vocabulary-limits-and-multi-digit-numbers))
3. **Could it be used to predict lottery numbers?** ([Section 6](#6-why-lottery-prediction-is-impossible))

Probing the answers turned out to be surprisingly rich. This document
captures the experiments we ran, the results we observed, the principles
that explain them, and a fourth tangent (why LLMs are bad at arithmetic --
[Section 5](#5-why-llms-are-bad-at-arithmetic)) that fell out of the
vocabulary discussion.

---

## 2. The unifying principle

Every result in this document is an instance of one fact about supervised
learning:

> **A model can only learn patterns that exist in its training distribution.
> If a pattern is not present in the data, no amount of training, model
> capacity, or clever architecture can teach the model to use it.**

Stated more precisely:

> The training distribution defines the **capability boundary**. The model
> learns a function approximator for the conditional distribution
> `P(output | input)` *as observed in training*. Anything outside that
> distribution is undefined behavior. Anything that has zero
> input-to-output information in the training data is fundamentally
> unlearnable.

This principle has three corollaries that we will see in action:

| Corollary | Section |
|---|---|
| If the training data is all length-5, the model learns "always emit 5 tokens" -- not "match input length" | [3. Length generalization](#3-length-generalization) |
| If the training data uses vocabulary `{0..9}`, the model has no representation for `23` -- no amount of training adds it | [4. Vocabulary limits](#4-vocabulary-limits-and-multi-digit-numbers) |
| If the training data has zero input-output mutual information (i.i.d. random), the model can do no better than the marginal distribution | [6. Lottery prediction](#6-why-lottery-prediction-is-impossible) |

Keep this principle in mind throughout. Each section is essentially a
restatement of the same core idea applied to a different surface problem.

---

## 3. Length generalization

### 3.1 Setup

The model was trained on `seq_len=5` exclusively -- 10,000 examples, all
of length 5. We asked: *what happens if we feed it length-6, 7, 8, 9, or
10 inputs?*

The probe (paraphrased from `/tmp/length_generalization.py`):

```python
from numpy_transformer.persistence import load_transformer
from numpy_transformer.inference import greedy_decode

model = load_transformer("train_model/sort_model.npz")
src = np.array([6, 5, 4, 7, 8, 9, 3, 2, 5, 4], dtype=np.int64)  # length 10
pred = greedy_decode(model, src, sos_id=SOS_ID, eos_id=EOS_ID, max_len=15)
```

**Hypothesis before running:**

- *Mechanically* it should work. The architecture has no length-dependent
  shapes anywhere -- attention is `O(n^2)` but works for any `n`, and
  sinusoidal positional encoding is defined up to `max_len=5000`.
- *Behaviorally* it would probably fail in interesting ways. The model
  has only ever been rewarded for emitting exactly 5 tokens followed by
  EOS.

### 3.2 What actually happened

For input `[6, 5, 4, 7, 8, 9, 3, 2, 5, 4]` (expected sort: `[2, 3, 4, 4, 5, 5, 6, 7, 8, 9]`):

```
got = [2, 4, 5, 7, 9, EOS]
```

Five tokens, internally sorted, all from the input multiset, EOS emitted at
position 5. The model dropped half the elements (the duplicate `4`, the
duplicate `5`, plus `3`, `6`, `8`).

Across input lengths 5 through 10, the pattern is uncanny:

| Input len | Input | Output | Output len |
|---|---|---|---|
| 5 | `[8,1,0,6,3]` | `[0,1,3,6,8]` | 5 |
| 6 | `[4,0,3,6,3,8]` | `[0,3,4,6,8]` | 5 |
| 7 | `[7,7,9,7,1,8,6]` | `[1,6,7,7,9]` | 5 |
| 8 | `[0,2,1,9,7,9,2,6]` | `[0,2,6,7,9]` | 5 |
| 9 | `[6,7,1,5,6,8,6,4,4]` | `[1,4,5,6,8]` | 5 |
| 10 | `[3,1,2,1,2,8,5,0,4,2]` | `[0,1,2,4,8]` | 5 |

**Three things the model preserved across all out-of-distribution lengths:**

1. **Output length is hardcoded at 5.** EOS is always emitted after exactly 5 tokens.
2. **Output is internally sorted.** Even when wrong, the 5 emitted tokens are in increasing order.
3. **Output tokens come from the input multiset.** Never hallucinates a token that wasn't in the input.

So the model learned three rules: *increasing*, *subset-of-input*, and
*length-5*. It just didn't learn that the third one was *conditional* on
the input.

### 3.3 An emergent behavior: 5-quantile summary

A surprising side observation. Look at the length-10 example
`[3, 1, 2, 1, 2, 8, 5, 0, 4, 2]`:

- True sorted (10 elements): `[0, 1, 1, 2, 2, 2, 3, 4, 5, 8]` (positions 0-9)
- Model output (5 elements): `[0, 1, 2, 4, 8]`
- The model picked roughly sorted positions 0, 1-2, 3-5, 7, 9 -- **approximately the 5-quantile points** of the sorted input.

For `[6, 5, 4, 7, 8, 9, 3, 2, 5, 4]`:

- True sorted: `[2, 3, 4, 4, 5, 5, 6, 7, 8, 9]`
- Model output: `[2, 4, 5, 7, 9]`
- Picked sorted positions ~0, 2-3, 4-5, 7, 9 -- again roughly evenly-spaced quintiles.

The cross-attention mechanism, when faced with 10 source positions but
only 5 query slots in the decoder, is implicitly **averaging over chunks
of source positions**. Each output query attends to roughly two input
positions (since 10/5 = 2), and the resulting representation reads out a
value near the median of that chunk's sorted neighborhood. The model has
effectively reinvented a "5-point order-statistic summary" for OOD inputs --
not what we trained, but a coherent emergent generalization.

### 3.4 Why it happens: two failures, both about the training distribution

**Failure A: EOS is positional, not semantic.**

During training:

- Encoder always saw 5 input positions
- Decoder always saw 6 input positions: `[SOS, t_1, t_2, t_3, t_4, t_5]`
- The label at decoder position 5 was *always* EOS

The model has therefore learned, very confidently, "**at decoder position
5, emit EOS**." It was never given evidence that EOS should be conditional
on having emitted as many tokens as the encoder saw. Even if the encoder
positional encodings extrapolated perfectly to length 10, the decoder
would still emit EOS at position 5.

**Failure B: Encoder positions 5-9 are out-of-distribution.**

The sinusoidal positional encoding from the paper is *designed* to
extrapolate -- `PE(pos+k)` is a linear function of `PE(pos)`, in theory
giving the network a smooth way to handle unseen positions. In practice,
with a 32-d_model 2-layer model trained on 10K examples, the attention
heads' query/key projections were only ever optimized to discriminate
between 5 distinct positional encodings. They have no incentive to behave
sensibly on the unseen encodings for positions 5-9.

These compound: when given a length-10 input, the encoder produces 10
hidden states (5 in-distribution, 5 garbled), the decoder cross-attends
across all 10 but with queries only trained for 5 source positions, and
the decoder hits its hardcoded "emit EOS at position 5" behavior before
the input is fully consumed.

### 3.5 What would actually fix this

**Only one fix really works: retrain (or fine-tune) on variable-length data.**

| Strategy | Addresses Failure A? | Addresses Failure B? | Verdict |
|---|---|---|---|
| Train on variable lengths (e.g., `seq_len ~ Uniform(3, 12)`) | Yes -- model learns "output length = input length" | Yes -- model sees positions 0-11 during training | **The real fix** |
| RoPE / ALiBi position encoding | No (model still emits EOS at position 5) | Partial (relative positions extrapolate better) | Necessary but not sufficient |
| Inference-time decoding constraints (forbid EOS until len-N tokens) | Partial -- forces output length but the picks are still wrong | No | Hack that doesn't fix the root cause |
| Bigger model with same data | No -- still trained on length-5 only | No | Doesn't help; the data is the problem |

This is also why long-context LLMs (Claude, GPT-4 with 200K context) need
**training-time** exposure to long contexts, not just inference-time. The
same principle that limits our toy model limits production LLMs.

---

## 4. Vocabulary limits and multi-digit numbers

### 4.1 Setup

Our model has `vocab_size=13`:

| Token ID | Meaning |
|---|---|
| 0-9 | The digits 0 through 9 |
| 10 | `PAD_ID` |
| 11 | `SOS_ID` |
| 12 | `EOS_ID` |

The embedding matrix has shape `(13, 32)`. We asked: *what happens if we
pass numbers like 23, 56, or 156?* Two probes:

- **Probe A: Pass the number directly as a token id.** What does
  `greedy_decode(model, [23, 5, 14])` do?
- **Probe B: Tokenize multi-digit numbers digit-by-digit.** Encode
  `[23, 5, 14]` as `[2, 3, 5, 1, 4]` -- five digits, perfect for our
  length-5 model. What does the model produce?

### 4.2 Probe A: Direct out-of-vocabulary token

```python
src = np.array([23, 5, 14, 9, 33], dtype=np.int64)
greedy_decode(model, src, sos_id=SOS_ID, eos_id=EOS_ID, max_len=12)
# IndexError: index 23 is out of bounds for axis 0 with size 13
```

This is a **hard mechanical limit**, raised at the very first operation:
the embedding lookup. There is no token ID 23 in the embedding table.
`embedding.weight[23]` is undefined and NumPy refuses.

The vocabulary is the *substrate of perception* for the model. Anything
outside it doesn't exist as far as the model is concerned. No clever
prompting, no architectural tweak, no amount of additional training can
route around it.

### 4.3 Probe B: Digit-by-digit tokenization

```python
# Represent [23, 5, 14] as digit tokens
digits = np.array([2, 3, 5, 1, 4], dtype=np.int64)
pred = greedy_decode(model, digits, sos_id=SOS_ID, eos_id=EOS_ID, max_len=12)
# pred = [1, 2, 3, 4, 5, EOS]
```

No exception this time. The model produces `[1, 2, 3, 4, 5]`.

But this is **the wrong answer** for sorting numbers. We wanted the
sorted version of `[23, 5, 14]`, which is `[5, 14, 23]`. As digit tokens,
the correct answer would have been `[5, 1, 4, 2, 3]`. The model gave us
`[1, 2, 3, 4, 5]` -- a digit sort, not a number sort.

**Why?** Because once we tokenized `23` as `[2, 3]`, the grouping that
gave it meaning *as the number 23* was lost. The model sees five digits
and sorts them digit-by-digit. There is nothing in the input signaling
"these two adjacent digits belong together."

This is the deeper lesson of the experiment: **the tokenization scheme
defines the problem the model is solving.** Two different tokenizations
of the same numbers are two different problems, and the model has no way
to distinguish them.

### 4.4 Three options for handling multi-digit numbers

| Option | Idea | Cost |
|---|---|---|
| **1. Digit-by-digit + separator token** | Tokenize `[23, 5, 14]` as `[2, 3, COMMA, 5, COMMA, 1, 4]`. Model must learn that `1` followed by `4` between separators means "fourteen." | Conceptually small (one new token), but the model now needs to learn what is essentially **arithmetic**. Hard for small models. |
| **2. Extend the vocabulary** | Set `vocab_size=100` (or 50, or whatever). Each number is one token. | Embedding matrix grows. Requires retraining from scratch. Doesn't generalize beyond the chosen range. For lottery 1-49 this is the simplest answer. |
| **3. Subword tokenization** | What real LLMs do. Vocabulary of 32K-200K tokens, learned via BPE or similar. Common substrings become single tokens; rare ones split. | Significant infrastructure. Comes with the famous "LLMs are bad at arithmetic" downside (next section). |

Option 2 is what we'll use if we want a "lottery sorter" -- a fresh
vocabulary, fresh training run, same architecture.

---

## 5. Why LLMs are bad at arithmetic

This is a tangent that fell out of Section 4, and it is one of the most
important things to understand about real LLMs.

### 5.1 The setup

GPT-4, Claude, Llama, and friends use **subword tokenization** (typically
Byte-Pair Encoding). The vocabulary is learned from a training corpus
to be common subword units. This is brilliantly efficient for natural
language: common words like "the" are single tokens, rare words like
"defenestration" split into `["def", "en", "est", "ration"]`, and the
vocabulary stays manageable (~50K-200K tokens).

But numbers are **not natural language**, and BPE handles them in a
quirky way.

### 5.2 The quirk

How GPT-4 tokenizes some numbers (using its `cl100k_base` tokenizer):

```
"123"     -> ["123"]               1 token
"1234"    -> ["1234"]              1 token
"12345"   -> ["12", "345"]         2 tokens, split awkwardly
"123456"  -> ["123", "456"]        2 tokens, split differently
"1234567" -> ["123", "4567"]       2 tokens, split YET differently
"3.14159" -> ["3", ".", "141", "59"]    4 tokens
```

Notice the **same digits** appear in different groupings depending on
the surrounding context. The number 123 is a single token. The number
1234 is also a single token. But `12345` is suddenly **two tokens** --
and not in the obvious "12345 -> 1234 + 5" way. It splits as `12 + 345`.

### 5.3 Why this makes arithmetic catastrophically hard

Consider teaching the model that `12345 + 1 = 12346`.

What the model sees:

- Input: tokens `["12", "345", "+", "1", "="]`
- Expected output: tokens `["12", "346"]`

The model has to learn the rule:

> When you see the token pair `("12", "345")` followed by `+ 1`, output
> the token pair `("12", "346")`.

But that rule is **specific to that exact tokenization**. For
`12245 + 1 = 12246`, the input tokenizes as `["12", "245"]` and output
as `["12", "246"]` -- a *different* learned rule. For `99999 + 1 = 100000`,
the input is `["999", "99"]` and output is `["100", "000"]` -- a *very
different* learned rule that requires the model to "carry" across token
boundaries.

A model that genuinely understood addition would learn it once. An LLM
trained on tokens has to learn **a different version of addition for
every plausible token boundary configuration**. There are exponentially
many of these. Even with billions of parameters, the model never quite
nails it -- it gets simple cases right and messy cases wrong, with no
principled way to predict which is which.

### 5.4 Why this also explains "count the r's in strawberry"

A famous LLM failure mode:

```
User: How many r's are in "strawberry"?
LLM:  Two.
```

(There are three.)

Why? Because GPT-4 tokenizes `"strawberry"` as `["str", "aw", "berry"]`.
The model never sees the individual letters `s, t, r, a, w, b, e, r, r, y`.
It sees three tokens. To count `r`s, it would need to:

1. Decompose each token back into letters (knowledge it doesn't have a
   direct circuit for)
2. Count occurrences of `r` across the three letter-sets
3. Sum them

Step 1 is the killer. The model doesn't have access to its own tokens'
internal letter structure. The information is genuinely not present in
its input representation.

**Same principle as Section 4.3**: tokenization defines the problem the
model is solving. If the task requires sub-token information, the model
is operating with one hand tied behind its back -- not because of model
size, but because of the input representation.

### 5.5 The broader lesson

> Tokenization is the bedrock layer that everything else is built on.
> A transformer's reasoning is forever bounded by the symbols its
> vocabulary defines. Any pattern that requires sub-token information
> is invisible to the model, regardless of how big or well-trained it is.

This is one of the most important and underappreciated facts about LLMs.
The model isn't reading text the way you read text. It is processing a
sequence of opaque integer indices, each pointing to a learned embedding
vector. The internal structure of those tokens -- letters, digits,
syllables -- is fundamentally invisible to it.

This is also why character-level models (which avoid the tokenization
issue entirely) are an active research direction, why models like
**ByT5** exist, and why some specialized arithmetic models bypass BPE
entirely and use digit-level tokenization.

---

## 6. Why lottery prediction is impossible

### 6.1 The question

> Could this model (or any model) be used to predict next week's lottery numbers?

Two interpretations of the question hide here:

| Question | Example | Possible? |
|---|---|---|
| Sort a lottery draw | Given `[27, 5, 14, 9, 33]`, output `[5, 9, 14, 27, 33]` | Yes (with appropriate vocabulary -- see Section 4.4) |
| Predict the next draw | Given draws from the last 1000 weeks, predict next week's | **No, by information theory** |

Most people asking this are asking the second one. So that's what we
address.

### 6.2 Step 1: Lottery draws are designed to be i.i.d.

Modern lotteries use either:

- Physical mechanical ball machines, audited for mechanical bias, or
- Cryptographically-strong hardware random number generators

Both are explicitly engineered to produce sequences where each draw is
**independent and identically distributed (i.i.d.)** -- the past gives
you zero information about the future. Independence isn't an accident
or a happy property; it is the **engineered guarantee** that makes the
lottery legitimate. Operators get sued and shut down if they fail it.

### 6.3 Step 2: For i.i.d. data, the Bayes-optimal predictor is the marginal distribution

This is a theorem, not a heuristic. If draws are truly i.i.d., the best
possible predictor -- better than any neural network, better than any
algorithm of any complexity, period -- is to output the marginal
distribution of single draws.

For a 1-49 ball lottery, that is **1/49 probability for each number**.

The Bayes-optimal predictor is a computational ceiling. **No ML system
can do better than it. And the Bayes-optimal predictor for i.i.d.
lotteries is the same as a uniform random guess.**

### 6.4 Step 3: What our transformer would actually learn

If we trained this exact transformer architecture on a million historical
lotto draws, the loss function would converge to one specific outcome:

```
P(next number = k | past draws) = 1/49   for all k
```

The model would learn that **past draws contain no information** about
the next draw. Its output distribution would be uniform. The cross-entropy
loss would converge to `log(49) ~= 3.89` -- exactly what a fair die's loss
would be. The model's 43,168 parameters would settle into configurations
that ignore their inputs and emit a flat softmax. It would be an
extraordinarily expensive way to roll dice.

This is the fundamental difference from sorting:

- **Sorting**: `output = f(input)` -- input fully determines output
- **Lottery prediction**: `output ⊥ input` -- output is independent of input

A transformer is a function approximator. If the true function is
"output a uniform distribution regardless of input," the model will
faithfully learn that constant function. There is no useful pattern to
discover because no useful pattern exists.

### 6.5 Common misconceptions

People often *feel* there are patterns in lottery data because of
well-documented cognitive biases:

| Bias | What it makes you think | Why it's wrong |
|---|---|---|
| **Apophenia** | "Look, the number 7 has come up 3 of the last 5 weeks!" | Random sequences contain local clusters that look meaningful but are pure coincidence. "Random" does not mean "uniformly spread." |
| **Gambler's Fallacy** | "Number 23 hasn't come up in 6 months, it's overdue!" | The balls have no memory. Each draw is independent. Probability is 1/49 every week regardless of history. |
| **Survivorship bias** | "I read about a guy who won using a system!" | You don't read about the millions who used systems and lost. The base rate is overwhelmingly negative. |
| **Confirmation bias** | "I picked 17 once and won small -- something about that number!" | You forgot the 200 times you picked 17 and lost. Memory keeps the wins, discards the losses. |

### 6.6 The narrow exception: detectable physical bias

There is one historical loophole: if a lottery's physical machine has
mechanical bias (one ball slightly heavier, falling more often), that's
a tiny exploitable signal. Famous cases:

- **Joan Ginther** (Texas, 2000s): won $20M+ across multiple scratchers
  by analyzing scratch-card distributions and exploiting known printing
  biases.
- **The MIT Blackjack Team** (different game, same idea): found
  exploitable signal in nominally random casino games.

But:

- Modern lotteries are extensively audited specifically to detect such
  biases.
- Detected biases get fixed within weeks.
- Detecting a 1% bias requires millions of draws -- typically more than
  a single lottery's lifetime.
- This is **statistics**, not deep learning. You'd use a chi-squared
  test on draw frequencies. No transformer needed.

For a properly audited modern lottery, no exploitable bias exists, and
no ML model can extract a signal that is mathematically absent.

### 6.7 Where ML *does* work for prediction (and why)

For comparison, here is where ML genuinely predicts the future:

| Domain | Why ML works | Information present? |
|---|---|---|
| **Weather** | Atmosphere is deterministic but chaotic; current state strongly predicts near future | Yes -- physics determines tomorrow from today |
| **User behavior** | Humans have habits, preferences, daily/weekly patterns | Yes -- past behavior predicts future behavior |
| **Sports outcomes** | Player skill, team composition, conditions persist | Yes (with noise) |
| **Stock prices (intraday)** | Limited inefficiencies, order-flow patterns | Yes, but tiny edges; markets compete them away |
| **Protein folding** | Physical laws determine structure from sequence | Yes -- fully deterministic, just hard to compute |
| **Lottery** | Designed for independence | **No -- by engineering** |

The pattern: **ML works when the past contains information about the
future.** Lottery is the canonical case where it doesn't, by design.

---

## 7. The unifying principle, revisited

We can now see that all four investigations were the same principle in
different costumes:

| Investigation | Pattern that was missing from training | Result |
|---|---|---|
| Length generalization (Section 3) | Length-conditional output behavior | Model outputs constant length 5 |
| Vocabulary limits (Section 4) | Token IDs ≥ 13 | Model raises IndexError or misinterprets |
| LLM arithmetic (Section 5) | Sub-token information (digit-level structure of multi-digit numbers) | Model learns per-tokenization rules; never generalizes |
| Lottery prediction (Section 6) | Any input-to-output mutual information | Model converges to marginal distribution; no better than random |

In every case, the model faithfully learned the regularities present in
its training data. In every case, asking the model to do something
that required regularities **not** present in training led to failure.
The failures were not bugs -- they were the model behaving correctly
within the limits of what it could possibly know.

> **The training distribution is the capability boundary.** Build for
> what you actually need; you cannot retrieve what was never there.

This is the single most important fact to internalize when assessing
whether ML is the right tool for a problem. Before training a model,
ask:

1. **Is the function you're trying to learn even a function?** If the
   same input can produce wildly different outputs (e.g., next lottery
   draw given past draws), no model can do better than the marginal.
2. **Does your training data contain examples of the variation you need
   to handle?** If you need length-10 inputs at inference, you need
   length-10 (or near-it) examples at training time.
3. **Does your input representation contain the information the task
   requires?** If you're counting characters but tokenize to subwords,
   the relevant information has been thrown away before the model
   ever sees it.

If you can't answer "yes" to all three, the problem is not "we need a
bigger model." The problem is information-theoretic, distributional, or
representational -- and those problems are addressed *before* training,
not by training.

---

## 8. Discussion Log

### 2026-04-26: Initial exploration after Step 11

The conversation that produced this document went roughly:

1. Trained the sort model end-to-end (`--steps 2000`, all defaults). Got
   **100% (1000/1000)** accuracy. Loss settled at 0.6032 against floor
   0.5736 (the label-smoothing minimum). All 100 demonstration decodes
   were correct.
2. Verified save/load fidelity: loaded the persisted model in a fresh
   process, ran identical decoding, got bit-identical parameters across
   two independent loads (43,168 float64 elements verified), all
   tied-embedding aliasing preserved (encoder/decoder/output_projection
   all reference the same memory).
3. **Q: "What happens if I pass `[6, 5, 4, 7, 8, 9, 3, 2, 5, 4]` -- a
   length-10 sequence?"** -> Length generalization probe (Section 3).
   Surprised by the "5-quantile summary" emergent behavior.
4. **Q: "Does the model need to be retrained to handle different lengths?"**
   -> Yes, with discussion of why fine-tuning > from-scratch retraining
   (since persistence enables it). Connected to the principle "training
   distribution is the capability boundary."
5. **Q: "Can this model be used on lotto numbers?"** -> Distinguished
   sorting (yes, with vocab change) from prediction (no, by information
   theory). Section 6.
6. **Q: "Vocab size is 13 -- how does it handle numbers other than
   single digits?"** -> Section 4. Two probes: hard IndexError on
   out-of-vocab token IDs, and digit-tokenization producing
   semantically-wrong digit sorts.
7. The vocabulary discussion led naturally into "this is also why LLMs
   are bad at arithmetic" -- Section 5 was added because the same
   principle (tokenization-as-substrate) explains both behaviors.

### Key design decisions captured

- **The choice of file location**: `nn/docs/reference_*` (lowercase,
  no number) follows the precedent set by `reference_tokenization.md`.
  Reference docs are explicitly outside the numbered build steps but
  earn the same documentation rigor as steps.
- **The choice to combine four topics into one doc**: they share a
  unifying principle (Section 2 / Section 7). Splitting them into four
  separate docs would obscure the connection.
- **The choice not to include "what we'd build to fix it"**: this doc
  is conceptual reference, not implementation plan. Variable-length
  retraining and vocabulary-extension experiments belong in their own
  step docs if/when they are built.

### Open questions left for future exploration

These are good follow-ups that did not happen on 2026-04-26 but are
worth recording for later:

1. **Variable-length retraining experiment**: actually fine-tune the
   trained model on `seq_len ~ Uniform(3, 12)` and see how quickly the
   length-generalization gap closes. Predicted: ~500 fine-tune steps
   from the existing checkpoint should achieve >95% on lengths 3-12.
2. **Empirical demonstration of the lottery floor**: train this model
   on synthetic i.i.d. uniform "fake lottery" data and observe loss
   converging exactly to `log(vocab_size)`. Concretizes Section 6.4.
3. **Lottery sort task**: extend `vocab_size` to ~52 (for Lotto 6/49)
   or 73 (for Powerball), build appropriate dataset generator, retrain.
   Demonstrates Option 2 from Section 4.4 concretely.
4. **PyTorch port**: re-implement the same architecture in PyTorch using
   autograd, validate it produces matching results, then use it as the
   substrate for the lottery-sort task. Connects model persistence
   (Step 11) to cross-implementation comparison.

---

[< Back to Overview](00_overview.md) | [Step 11 (Persistence)](11_model_persistence.md) | [Reference: Tokenization](reference_tokenization.md)
