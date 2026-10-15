"""
Step 13: Lottery number prediction experiment.

Two experiments that together demonstrate memorisation ≠ prediction:

  Experiment A (prediction):
    Train on draws 0..N-n_eval-1, evaluate on held-out draws N-n_eval..N-1.
    Expected: exact-match 0/n_eval regardless of training loss.

  Experiment B (memorisation):
    Train on ALL draws including the eval window.
    Expected: loss → 0; exact-match → n_eval/n_eval.

Usage
-----
  python scripts/train_lottery.py --data train_input/powerball_draws.csv
  python scripts/train_lottery.py --data train_input/powerball_draws.csv --exp a
  python scripts/train_lottery.py --data train_input/powerball_draws.csv --steps 5000
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pytorch_transformer.inference import greedy_decode
from pytorch_transformer.layers import Transformer
from pytorch_transformer.lottery_data import (
    CONTEXT_DRAWS,
    EOS_ID,
    PAD_ID,
    SOS_ID,
    VOCAB_SIZE,
    load_draws_from_csv,
    make_dataloader,
    make_experiment_a,
    make_experiment_b,
)
from pytorch_transformer.persistence import save_transformer
from pytorch_transformer.training import train

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
MODEL_KWARGS = dict(
    vocab_size=VOCAB_SIZE,
    d_model=64,
    num_heads=4,
    d_ff=128,
    num_encoder_layers=2,
    num_decoder_layers=2,
    p_drop=0.1,
    tie_embeddings=True,
)

LOG69 = math.log(69)
LOG26 = math.log(26)
UNIFORM_RANDOM_LOSS = (5 * LOG69 + LOG26) / 6   # ≈ 4.071

TRAIN_MODEL_DIR  = Path(__file__).resolve().parents[1] / "train_model"
TRAIN_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "train_output"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _eval_exact_match(model: Transformer, eval_ds, label: str) -> tuple[int, int]:
    enc_tensors, _, dec_tgt_tensors = eval_ds.tensors
    n = enc_tensors.size(0)
    correct = 0
    for i in range(n):
        pred = greedy_decode(model, enc_tensors[i], SOS_ID, EOS_ID, max_len=8)
        tgt  = dec_tgt_tensors[i]
        if pred.shape == tgt.shape and (pred == tgt).all():
            correct += 1
    pct = 100.0 * correct / n
    print(f"  {label}: exact-match {correct}/{n}  ({pct:.1f}%)")
    return correct, n


def _run_experiment(
    name: str,
    train_ds,
    eval_ds,
    n_steps: int,
    warmup_steps: int,
    save_path: Path,
) -> None:
    print(f"\n{'='*60}")
    print(f"  Experiment {name}")
    print(f"  train examples : {len(train_ds)}")
    print(f"  eval  examples : {len(eval_ds)}")
    print(f"  steps          : {n_steps}")
    print(f"  uniform-random loss : {UNIFORM_RANDOM_LOSS:.3f}  "
          f"(= [5·log69 + log26] / 6)")
    print(f"{'='*60}")

    model = Transformer(**MODEL_KWARGS)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {total_params:,}")

    dl = make_dataloader(train_ds, batch_size=64, shuffle=True)
    losses, lrs = train(
        model, dl,
        n_steps=n_steps,
        ignore_index=PAD_ID,
        label_smoothing=0.1,
        warmup_steps=warmup_steps,
        log_every=max(1, n_steps // 10),
    )

    final_loss = losses[-1]
    print(f"\n  final loss          : {final_loss:.4f}")
    print(f"  uniform-random loss : {UNIFORM_RANDOM_LOSS:.4f}")

    _eval_exact_match(model, eval_ds, name)

    TRAIN_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_transformer(model, save_path)
    rel = save_path.relative_to(Path(__file__).resolve().parents[1])
    print(f"  model saved → {rel}")

    TRAIN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TRAIN_OUTPUT_DIR / f"lottery_{name.lower()}_losses.csv"
    with csv_path.open("w") as f:
        f.write("step,loss,lr\n")
        for step, (l, lr) in enumerate(zip(losses, lrs), start=1):
            f.write(f"{step},{l:.6f},{lr:.8f}\n")
    rel_csv = csv_path.relative_to(Path(__file__).resolve().parents[1])
    print(f"  losses saved → {rel_csv}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Lottery prediction experiments")
    parser.add_argument(
        "--data", required=True,
        help="Path to CSV file with columns: date,w1,w2,w3,w4,w5,pb",
    )
    parser.add_argument("--exp",    choices=["a", "b", "both"], default="both")
    parser.add_argument("--steps",  type=int, default=3000,
                        help="Training steps per experiment (default: 3000)")
    parser.add_argument("--warmup", type=int, default=400,
                        help="Warmup steps for Noam schedule (default: 400)")
    parser.add_argument("--n-eval", type=int, default=10,
                        help="Held-out evaluation draws (default: 10)")
    parser.add_argument("--context", type=int, default=CONTEXT_DRAWS,
                        help=f"Context draws for encoder input (default: {CONTEXT_DRAWS})")
    args = parser.parse_args()

    draws = load_draws_from_csv(args.data)
    print(f"Loaded {len(draws)} draws from {args.data}")
    print(f"Theoretical uniform-random loss: "
          f"log(69)={LOG69:.3f} (white), log(26)={LOG26:.3f} (PB)")

    run_a = args.exp in ("a", "both")
    run_b = args.exp in ("b", "both")

    if run_a:
        train_a, eval_a = make_experiment_a(draws, args.n_eval, args.context)
        _run_experiment(
            "A", train_a, eval_a,
            n_steps=args.steps,
            warmup_steps=args.warmup,
            save_path=TRAIN_MODEL_DIR / "lottery_a.pt",
        )

    if run_b:
        train_b, eval_b = make_experiment_b(draws, args.n_eval, args.context)
        _run_experiment(
            "B", train_b, eval_b,
            n_steps=args.steps,
            warmup_steps=args.warmup,
            save_path=TRAIN_MODEL_DIR / "lottery_b.pt",
        )

    if run_a and run_b:
        print("\n" + "="*60)
        print("  LESSON: Experiment A loss ≈ random floor → model cannot")
        print("  predict lottery draws.  Experiment B loss → 0 → model")
        print("  memorised the training set.  Memorisation ≠ Prediction.")
        print("="*60)


if __name__ == "__main__":
    main()
