# Sort task input data

This folder (`numpy-transformer/train_input/`) holds the dataset
fixtures that `scripts/train_sort.py` reads. It is the **input** half of
the project's runtime layout, alongside its two output siblings:

- `train_input/`  -- this folder; dataset fixtures (committed to git).
- `train_output/` -- runtime logs (`loss_curve.csv`).
- `train_model/`  -- persisted Transformer `.npz` files.

## Why the `.npz` files look unreadable

`sort_train.npz` and `sort_test.npz` are **NumPy compressed archives** (binary). Your editor
will not show them as a table. Use them in Python for training:

```python
from pathlib import Path
from numpy_transformer.sort_data import default_input_dir, load_sort_npz

d = load_sort_npz(default_input_dir() / "sort_train.npz")
x = d["encoder_input"]      # (N, 5) int64
di = d["decoder_input"]     # (N, 6) int64 — SOS + sorted digits
dt = d["decoder_target"]    # (N, 6) int64 — sorted digits + EOS
```

## Human-readable previews

Open these in the IDE or Excel:

- **`sort_train_preview.csv`** — first 30 training rows
- **`sort_test_preview.csv`** — first 30 test rows

Column names: `encoder_0..4` (unsorted digit ids 0–9), `decoder_in_0..5` (11 = SOS, then sorted
digits), `decoder_tgt_0..5` (sorted digits then 12 = EOS).

---

## Vocabulary: example, batch, epoch, training vs optimization

These definitions use **this** repo’s default files: **`sort_train.npz`** has **N = 10 000**
rows (10 000 independent sort problems). Each row is **one example** (one `(encoder_input,
decoder_input, decoder_target)` triple).

### One **example**

One **training example** = **one row** of the dataset: one unsorted length-5 input and its
teacher-forcing decoder pair. It is **only data**. It is **not** automatically one full
forward+backward+update by itself (unless your batch size is 1).

### **`batch_size`**

**`batch_size`** = how many examples you **stack into one tensor** for **one** forward pass and
**one** backward pass, before applying **one** optimizer update.

Example (numbers are illustrative): if **`batch_size = 32`**, then **one training step** uses
examples #0–31, then the next step uses #32–63, and so on. The tensor might look like
**`(32, 5)`** for encoder inputs (32 sequences, length 5).

Default **10 000** examples and **`batch_size = 32`** ⇒ one **epoch** needs
**`ceil(10 000 / 32) = 313`** steps (312 full batches of 32 + one batch of 16). Each step does
forward + backward + **one** weight update.

The **preview CSV** shows **30 examples** for humans to read; it is **not** “30 batches” or
“30 epochs.”

### One **epoch**

One **epoch** = one full sweep through the **entire training set** (all **10 000** examples),
each used **once** in that epoch (order may be shuffled). In terms of steps:

```text
steps_per_epoch ≈ ceil(N / batch_size)   with N = 10_000 for sort_train.npz
```

So **epoch** counts **how many times you have seen the whole training file**, not how many rows
are in the CSV preview.

### What **“training”** means vs **optimizing parameters**

| Term | Meaning |
|------|---------|
| **Training** | The **whole process**: repeatedly draw batches from the training data, run the model (**forward**), compare predictions to targets (**loss**), propagate gradients (**backward**), and **update weights**. Usually many **epochs**. May also include validation, logging, checkpoints. |
| **Optimizing / updating parameters** | The **narrow step**: change weights using gradients (SGD, Adam, …). This happens **once per batch** (per training step). It is **part of** training, not a separate unrelated activity. |

So: **training** = the full loop over data + loss + gradients + updates (and extras).
**Parameter optimization** = the **update** slice inside each training step. You do not
“optimize instead of train”; you **train by** repeatedly **optimizing** on batches.

### Tiny checklist

- **Example** → one sort problem in `sort_train.npz` (one of 10 000).
- **`batch_size`** → how many of those problems per forward/backward/**single** update.
- **Epoch** → all **10 000** examples consumed once (split across many batches).
- **Training** → the overall procedure; **optimizer step** → one weight update after one batch.

---

## Strategy: splitting and sizing training data

### Three roles: train / validation / test

| Split | Typical use |
|-------|----------------|
| **Train** | Update weights (forward, loss, backward, optimizer). |
| **Validation** (hold-out or dev) | Tune hyperparameters, early stopping, model selection — **run forward + loss**, **no** gradient updates on this split. |
| **Test** | Final honest estimate of generalization — touch **only after** you are done iterating on architecture and hyperparameters. |

For a small toy project, **train + test** alone is acceptable; add **validation** when you start
comparing runs or doing early stopping.

**Ratios** (rule of thumb when data is cheap to generate, as here): **80 / 10 / 10** or
**70 / 15 / 15** train / val / test. What matters more than the exact ratio is that **test**
examples are **never** used for any training decision, including picking the “best” epoch by
eye on test loss.

### For this sort task specifically

- **IID random sequences** (independent draws of five digits) match the **training
  distribution** you care about. Separate files with **different RNG seeds** (as
  `write_default_datasets` does for train vs test) give **independent** samples — fine for a
  toy benchmark.
- You do **not** need stratified splits unless you explicitly care about rare multiset shapes;
  for learning “sort five digits,” volume of random examples usually suffices.
- If you add **validation**, generate a third `.npz` (e.g. `sort_val.npz`, **N ≈ 1–2k**) with
  another seed and **never** mix it into training batches.

### How big should **N** be?

- **Larger `N`** → smoother loss curves, better coverage of the input space, slower disk/epoch.
- **Smaller `N`** → faster iteration while debugging code paths.
- For **digit sort, length 5**, **10k train** is already a comfortable toy scale; scale up if
  loss plateaus early or you see overfit to a small pool.

### During training (not stored in the files)

- **Shuffle** example indices (or shuffle batches) **each epoch** so batch composition varies
  and order bias does not accumulate.
- Keep **`batch_size`** fixed for comparability between runs; change it deliberately when
  studying throughput vs gradient noise.

### Reproducibility

- Fix **seeds** for `generate_sort_split` / `write_default_datasets` so splits and rows are
  reproducible across machines and reruns.
- Record in notes or config: **`N_train`, `N_val`, `N_test`, seeds, `batch_size`, epochs.**

---

## Regenerate everything

From the `numpy-transformer` directory:

```text
uv run python -c "from numpy_transformer.sort_data import write_default_datasets; write_default_datasets()"
```

This rewrites the `.npz` files and the two preview CSVs (default 10k train / 1k test).
