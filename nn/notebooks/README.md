# Transformer basics — standalone notebooks

Each notebook is self-contained (only `numpy` + `matplotlib`) and meant to be run cell by cell, top to bottom.
Suggested order (00 is a reference you can return to any time):

| # | Notebook | Concepts |
|---|---|---|
| 00 | `00_numpy_guide` | every NumPy operation used in `numpy-transformer`: axes, `keepdims`, broadcasting, `reshape`/`swapaxes`, batched `@`, masks, `np.add.at` (start here if NumPy is new) |
| 01 | `01_vocabulary_and_tokens` | token, vocabulary, ids, special tokens, subwords/BPE |
| 02 | `02_batch_and_sequence` | batch size `B`, sequence length `T`, padding & masks, steps/epochs, teacher-forcing shift |
| 03 | `03_embeddings` | embedding table, one-hot equivalence, meaning from context, `√d_model` scaling |
| 04 | `04_positional_encoding` | why order is lost, sinusoidal encoding, relative offsets |
| 05 | `05_attention` | Q/K/V, softmax, `√d_k`, causal & padding masks, self vs cross attention |
| 06 | `06_multihead_and_encoder_block` | heads, FFN, LayerNorm, residuals, full encoder layer, parameter count |
| 07 | `07_decoder_loss_and_generation` | decoder, logits, cross-entropy, greedy decoding, temperature |

Run (from `nn/`): `uv run --with jupyter --with numpy --with matplotlib jupyter lab notebooks/`
Or in VS Code just open a `.ipynb` and pick a Python kernel.
