"""
Step 8 runnable: train the toy Transformer on the sort task.

Usage (from nn/numpy-transformer/):
    uv run --package numpy-transformer python scripts/train_sort.py
    uv run --package numpy-transformer python scripts/train_sort.py --steps 1000

Default project layout (all siblings of `src/` under
`nn/numpy-transformer/`):
    train_input/sort_{train,test}.npz   dataset fixtures (input)
    train_output/loss_curve.csv         one row per step: step,loss,lr
    train_model/sort_model.npz          persisted trained Transformer
    stdout                              periodic progress + final eval

`train_output/` and `train_model/` are auto-created on first run. The
input fixtures live in `train_input/`; runtime artifacts live in the
two output folders so each can be wiped (or git-tracked) independently.
Override with `--input-dir`, `--out-csv`, and `--save-path` if you want
the artifacts somewhere else.

The architecture matches the toy build documented in nn/docs/00_overview.md
and the parameter-count tables in 06/07. Defaults are tuned to converge on
the sort task within a few minutes on CPU.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from numpy_transformer.inference import evaluate_sort_accuracy, greedy_decode
from numpy_transformer.layers import Transformer
from numpy_transformer.persistence import save_transformer
from numpy_transformer.sort_data import (
    EOS_ID,
    PAD_ID,
    SOS_ID,
    VOCAB_SIZE,
    default_input_dir,
    load_sort_npz,
    write_default_datasets,
)
from numpy_transformer.training import (
    label_smoothing_loss_floor,
    train,
)


def default_output_dir() -> Path:
    """Project `train_output/` folder, sibling of `src/` and `train_input/`.

    Lives at `nn/numpy-transformer/train_output/`. Holds runtime
    training artifacts (loss curves, future plots) -- distinct from the
    `train_input/` fixtures folder so artifacts can be wiped
    independently.
    """
    return Path(__file__).resolve().parent.parent / "train_output"


def default_model_dir() -> Path:
    """Project `train_model/` folder, sibling of `src/` and `train_input/`.

    Lives at `nn/numpy-transformer/train_model/`. Holds persisted
    Transformer `.npz` files written by `save_transformer`. Separate
    from `train_output/` so model artifacts (which one might want to
    keep across runs) and ephemeral logs can have different
    git-tracking policies.
    """
    return Path(__file__).resolve().parent.parent / "train_model"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the toy Transformer on the sort task.")
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--d-model', type=int, default=32)
    p.add_argument('--num-heads', type=int, default=4)
    p.add_argument('--d-ff', type=int, default=64)
    p.add_argument('--num-layers', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--label-smoothing', type=float, default=0.1)
    p.add_argument('--warmup-steps', type=int, default=400)
    p.add_argument('--log-every', type=int, default=50)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument(
        '--no-tied-embeddings', action='store_true',
        help="Disable input/output weight tying."
    )
    p.add_argument(
        '--input-dir', type=Path, default=None,
        help="Directory containing sort_{train,test}.npz. "
             "Default: <project>/train_input/."
    )
    p.add_argument(
        '--out-csv', type=Path, default=None,
        help="Where to write loss_curve.csv. "
             "Default: <project>/train_output/loss_curve.csv."
    )
    p.add_argument(
        '--save-path', type=Path, default=None,
        help="Where to save the trained model (.npz). "
             "Default: <project>/train_model/sort_model.npz."
    )
    p.add_argument(
        '--no-save', action='store_true',
        help="Skip persisting the trained model after training."
    )
    p.add_argument(
        '--no-evaluate', action='store_true',
        help="Skip post-training accuracy / decode demo on sort_test.npz."
    )
    p.add_argument(
        '--demo-examples', type=int, default=5,
        help="Number of test examples to print with predicted vs expected."
    )
    return p.parse_args()


def load_or_generate_data(input_dir: Path) -> tuple[dict, dict]:
    train_path = input_dir / 'sort_train.npz'
    test_path = input_dir / 'sort_test.npz'
    if not train_path.exists() or not test_path.exists():
        print(f"sort_train.npz / sort_test.npz not found in {input_dir}; generating defaults...")
        write_default_datasets(input_dir=input_dir)
    return load_sort_npz(train_path), load_sort_npz(test_path)


def _format_seq(ids) -> str:
    """Render a sequence with friendly aliases for SOS/EOS/PAD."""
    parts = []
    for i in ids:
        i = int(i)
        if i == SOS_ID:
            parts.append("SOS")
        elif i == EOS_ID:
            parts.append("EOS")
        elif i == PAD_ID:
            parts.append("PAD")
        else:
            parts.append(str(i))
    return "[" + ", ".join(parts) + "]"


def run_evaluation(transformer, test_data: dict, demo_examples: int) -> None:
    """Decode `demo_examples` test sequences, then report exact-match
    accuracy on the full test set."""
    print()
    print("Evaluating on sort_test.npz ...")

    enc_in = test_data['encoder_input']
    dec_tgt = test_data['decoder_target']
    n_test = enc_in.shape[0]

    if demo_examples > 0:
        n_demo = min(demo_examples, n_test)
        print(f"  Decoding the first {n_demo} test examples (greedy):")
        for i in range(n_demo):
            src = enc_in[i]
            expected = dec_tgt[i]
            pred = greedy_decode(
                transformer, src,
                sos_id=SOS_ID, eos_id=EOS_ID, max_len=src.shape[0] + 3,
            )
            ok = pred.shape == expected.shape and (pred == expected).all()
            mark = "OK " if ok else "BAD"
            print(f"    [{mark}] input {_format_seq(src):<24} "
                  f"-> predicted {_format_seq(pred):<32} "
                  f"(expected {_format_seq(expected)})")

    accuracy, _ = evaluate_sort_accuracy(
        transformer,
        enc_in,
        dec_tgt,
        sos_id=SOS_ID, eos_id=EOS_ID,
        max_len=enc_in.shape[1] + 3,
    )
    print()
    print(f"  Exact-match accuracy on {n_test} held-out examples: "
          f"{accuracy:.1%} ({int(round(accuracy * n_test))}/{n_test})")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir or default_input_dir()
    out_csv = args.out_csv or (default_output_dir() / 'loss_curve.csv')
    save_path = args.save_path or (default_model_dir() / 'sort_model.npz')

    print("=" * 60)
    print(" Step 8: Training the toy Transformer on the sort task")
    print("=" * 60)
    print(f"  vocab_size       = {VOCAB_SIZE}")
    print(f"  d_model          = {args.d_model}")
    print(f"  num_heads        = {args.num_heads}")
    print(f"  d_ff             = {args.d_ff}")
    print(f"  num_layers       = {args.num_layers} (encoder == decoder)")
    print(f"  dropout          = {args.dropout}")
    print(f"  tie_embeddings   = {not args.no_tied_embeddings}")
    print(f"  label smoothing  = {args.label_smoothing}")
    print(f"  warmup_steps     = {args.warmup_steps}")
    print(f"  batch_size       = {args.batch_size}")
    print(f"  steps            = {args.steps}")
    print(f"  seed             = {args.seed}")
    print()

    floor = label_smoothing_loss_floor(args.label_smoothing, VOCAB_SIZE)
    print(f"  label-smoothing loss floor (per token): {floor:.4f}")
    print(f"  initial-uniform-prior loss (-log(V)):    {np.log(VOCAB_SIZE):.4f}")
    print()

    train_data, test_data = load_or_generate_data(input_dir)
    print(f"  loaded data from {input_dir}/")
    print(f"    encoder_input  : {train_data['encoder_input'].shape} (train), "
          f"{test_data['encoder_input'].shape} (test)")
    print(f"    decoder_input  : {train_data['decoder_input'].shape} (train), "
          f"{test_data['decoder_input'].shape} (test)")
    print(f"    decoder_target : {train_data['decoder_target'].shape} (train), "
          f"{test_data['decoder_target'].shape} (test)")
    print()

    np.random.seed(args.seed)
    transformer = Transformer(
        vocab_size=VOCAB_SIZE,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=args.num_layers,
        p_drop=args.dropout,
        tie_embeddings=not args.no_tied_embeddings,
    )

    n_params = sum(p.size for (p, _, _) in transformer.params_and_grads())
    print(f"  learnable parameters: {n_params:,}")
    print()
    print("Training...")

    t0 = time.time()
    losses, lrs = train(
        transformer,
        train_data,
        n_steps=args.steps,
        batch_size=args.batch_size,
        eps=args.label_smoothing,
        vocab_size=VOCAB_SIZE,
        pad_id=PAD_ID,
        d_model=args.d_model,
        warmup_steps=args.warmup_steps,
        log_every=args.log_every,
        seed=args.seed,
    )
    elapsed = time.time() - t0
    print(f"\nTraining finished in {elapsed:.1f}s "
          f"({args.steps / max(elapsed, 1e-6):.1f} steps/s)")

    initial_avg = np.mean(losses[: max(20, args.steps // 50)])
    final_avg = np.mean(losses[-max(20, args.steps // 50):])
    print(f"  initial loss (first ~20 steps avg): {initial_avg:.4f}")
    print(f"  final   loss (last ~20 steps avg):  {final_avg:.4f}")
    print(f"  smoothing floor:                    {floor:.4f}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', encoding='utf-8') as fh:
        fh.write("step,loss,lr\n")
        for i, (loss, lr) in enumerate(zip(losses, lrs), start=1):
            fh.write(f"{i},{loss:.6f},{lr:.8f}\n")
    print(f"\nWrote {len(losses)} rows to {out_csv}")

    if not args.no_save:
        save_transformer(transformer, save_path)
        size = save_path.stat().st_size
        print(f"Saved trained model to {save_path} ({size:,} bytes)")
    else:
        print("Skipping model save (--no-save).")

    if not args.no_evaluate:
        run_evaluation(transformer, test_data, demo_examples=args.demo_examples)


if __name__ == '__main__':
    main()
