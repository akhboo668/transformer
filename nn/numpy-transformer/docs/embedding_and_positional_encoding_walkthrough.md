# Embedding + Positional Encoding -- Shape Walkthroughs

> **Scope**: Two small hand traces showing how **token embeddings** and **sinusoidal
> positional encoding** combine before attention; how **epoch** and **batch_size** relate in
> training. Same mechanics for any sequence task.
> **Implementation**: `SinusoidalPositionalEncoding` in `src/numpy_transformer/layers.py`.
> **Theory**: [02_positional_encoding.md](../../docs/02_positional_encoding.md) in `nn/docs`.

---

## Tensor axes (always the same layout)

For one stage of the encoder stack, token vectors are shaped like:

```text
(batch_size, seq_len, d_model)
```

| Axis | Meaning |
|------|---------|
| **batch_size** | How many independent sequences in this tensor (how many “rows” in the batch). |
| **seq_len** | How many token **positions** each sequence has along the sequence dimension (often padded to a common length). |
| **d_model** | Width of each token’s vector after lookup (and after adding PE). |

**Important:** `seq_len` is **not** the batch size. It is the length of each sequence (number of
tokens), not the number of samples.

---

## Example A -- Language-style tokens (“the cat sat”)

### Setup

- One sequence in the batch: the words **“the”**, **“cat”**, **“sat”**.
- Stand-in token ids (arbitrary): **`[2, 9, 4]`**.
- **`batch_size = 1`**, **`seq_len = 3`**, **`d_model = 4`** (tiny, for tracing only).

After embedding lookup, activations have shape **`(1, 3, 4)`**.

### Step 1 -- Token ids and positions

| Position `pos` | 0 | 1 | 2 |
|----------------|---|---|---|
| Token | the | cat | sat |
| Token id | 2 | 9 | 4 |

`pos` is the **index along this sentence**. It is **not** the vocabulary id.

### Step 2 -- Embedding lookup

The embedding matrix has one row per vocabulary id. Each id selects a vector of length
`d_model`.

Illustrative values (placeholders for learned weights):

| `pos` | Token | Id | Example embedding vector |
|------:|-------|---:|---------------------------|
| 0 | the | 2 | `[0.10, -0.20, 0.30, 0.40]` |
| 1 | cat | 9 | `[0.50, 0.10, -0.10, 0.20]` |
| 2 | sat | 4 | `[-0.20, 0.60, 0.05, -0.30]` |

Stacked as **`(1, 3, 4)`** for batch 0:

```text
row 0 (the): [0.10, -0.20, 0.30, 0.40]
row 1 (cat): [0.50, 0.10, -0.10, 0.20]
row 2 (sat): [-0.20, 0.60, 0.05, -0.30]
```

### Step 3 -- Positional encoding

Sinusoidal PE depends **only** on `pos` and `d_model`, not on which word it is. For
`d_model = 4`, the first three rows of the PE table look like:

| `pos` | `PE(pos)` (see math doc for formulas) |
|------:|----------------------------------------|
| 0 | `[0, 1, 0, 1]` |
| 1 | `[sin(1), cos(1), sin(0.01), cos(0.01)]` (angles from Section 3.5) |
| 2 | `[sin(2), cos(2), sin(0.02), cos(0.02)]` |

The stored table has shape **`(max_len, d_model)`**; for a sequence of length 3 you use the
slice **`PE[:3, :]`**, shape **`(3, 4)`**, which broadcasts when added to **`(1, 3, 4)`**.

### Step 4 -- Input to the first attention layer

Per position:

```text
input[pos] = embedding[token_id at pos] + PE(pos)
```

So the model sees **“which token”** (embedding) and **“where in the sentence”** (PE) together.
If the same vocabulary id appeared at two positions, the embeddings would match but **`PE(pos)`**
would not, so the two positions would still differ after the sum.

---

## Example B -- Sorting task input `[3, 1, 4, 1, 5]`

### Setup

- Input sequence (token ids): **`[3, 1, 4, 1, 5]`** (each id is a digit symbol in the vocab).
- **`batch_size = 1`**, **`seq_len = 5`**, **`d_model = 4`** (still tiny, for tracing).

After embedding lookup: shape **`(1, 5, 4)`**.

### Step 1 -- Token ids and positions

| `pos` | 0 | 1 | 2 | 3 | 4 |
|-------|---|---|---|---|---|
| Token id | 3 | 1 | 4 | 1 | 5 |

The id **`1` appears at `pos = 1` and `pos = 3`**. The embedding table returns the **same**
vector for id 1 both times. **Positional encoding** is what distinguishes the first `1` from the
second `1`.

### Step 2 -- Embedding lookup (same id → same row)

Illustrative vectors:

| `pos` | Id | Example embedding |
|------:|---:|-------------------|
| 0 | 3 | `[0.2, 0.0, -0.1, 0.4]` |
| 1 | 1 | `[0.5, -0.3, 0.2, 0.1]` |
| 2 | 4 | `[-0.1, 0.4, 0.0, 0.3]` |
| 3 | 1 | `[0.5, -0.3, 0.2, 0.1]` ← same row as `pos = 1` |
| 4 | 5 | `[0.0, 0.2, -0.2, 0.5]` |

### Step 3 -- Positional encoding

Use **`PE(0)` … `PE(4)`**, each a length-4 vector, stacked as **`(5, 4)`**.

### Step 4 -- Input to attention

| `pos` | Id | Vector entering the stack |
|------:|---:|----------------------------|
| 0 | 3 | `E[3] + PE(0)` |
| 1 | 1 | `E[1] + PE(1)` |
| 2 | 4 | `E[4] + PE(2)` |
| 3 | 1 | `E[1] + PE(3)` — same **`E[1]`** as row 1, **different** `PE` |
| 4 | 5 | `E[5] + PE(4)` |

Because **`PE(1) ≠ PE(3)`**, the two occurrences of digit **`1`** become **different** inputs
to attention. That separation of **value** (id / embedding) and **place in the list** (`pos` /
PE) is what makes permutation-sensitive tasks like sorting learnable.

### Shapes recap

| Tensor | Shape |
|--------|-------|
| Token ids (one batch row) | `(1, 5)` |
| After `Embedding.forward` | `(1, 5, 4)` |
| PE slice used | `(5, 4)` (broadcasts over batch dim 0) |

---

## Batching note (optional mental picture)

If you had **two** sort examples in one batch, both padded to length 5, you would have
**`batch_size = 2`**, **`seq_len = 5`**, same **`d_model`**, shape **`(2, 5, d_model)`**. Each
batch index `b` has its own token ids; **PE** is the same **`PE[:5]`** added to **every** row,
because position `k` always means “`k`th slot in the sequence,” not “which problem in the
batch.”

---

## Epoch and `batch_size` (how they relate)

These terms describe **training data throughput**, not the inside of a single sequence. They are
orthogonal to **`seq_len`**: a batch tensor can be **`(batch_size, seq_len, d_model)`**; changing
`batch_size` does not change what `seq_len` means.

### `batch_size`

**`batch_size`** is how many **training examples** you put in **one** optimizer step (one
forward, one backward, one parameter update), **for that step**.

In this project, one “example” is usually **one input sequence** (e.g. one list to sort). So
**`batch_size = 32`** means: each step processes 32 different sequences in parallel, each with
its own row in the batch dimension.

### Epoch

An **epoch** is **one full pass** over the **training set**: every training example is used once
(in expectation, if you shuffle; exactly once in the naive ordered case).

If the training set has **`N`** examples and you use **`batch_size = B`**, then one epoch
contains about

```text
steps_per_epoch = ceil(N / B)
```

**steps** (minibatches). Each step consumes **`B`** examples (except possibly the last step,
which can be smaller if **`N`** is not divisible by **`B`**).

### Relationship (numeric)

Suppose **`N = 1000`** training sequences and **`B = 50`**.

- **`steps_per_epoch = ceil(1000 / 50) = 20`**
- **One epoch** = **20** optimizer updates, and those 20 batches cover all 1000 examples.
- **Ten epochs** = **200** steps total, and each example has been seen **10** times (again modulo
  exact counting if you drop/duplicate the last partial batch).

Larger **`batch_size`** → **fewer** steps per epoch for the same **`N`**, but more memory and
often different gradient noise. Smaller **`batch_size`** → **more** steps per epoch.

### Summary table

| Term | Counts what |
|------|----------------|
| **`batch_size`** | Examples **per step** (width of batch dim 0). |
| **`steps_per_epoch`** | About **`ceil(N / batch_size)`** for **`N`** training examples. |
| **Epoch** | **`N`** examples seen once through the training set → **`steps_per_epoch`** updates. |
| **`seq_len`** | Tokens **per sequence**; unrelated to how many sequences are in a batch. |
