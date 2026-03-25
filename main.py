import torch
import gc
import argparse
import sys

from model_setup import load_model_and_tokenizer
from dataset_prep import prepare_dataset
from reward_func import format_reward_func, correctness_reward_func
from grpo_config import setup_grpo_config
from trainer import run_training
from evaluation import rigorous_evaluation, evaluate_model


def get_dataset_mode():
    """Get dataset mode from CLI args, or prompt interactively if possible."""
    parser = argparse.ArgumentParser(description="RLVR Training Pipeline")
    parser.add_argument(
        "--mode", type=int, choices=[1, 2, 3], default=None,
        help="Dataset mode: 1=Real Only, 2=Synthetic Only, 3=Both (default)"
    )
    args = parser.parse_args()

    if args.mode is not None:
        choice = str(args.mode)
    elif sys.stdin.isatty():
        # Terminal is interactive — safe to prompt
        print("\nWhich dataset mode would you like to use for training?")
        print("1: Real Data Only (HuggingFace tau2-bench trace)")
        print("2: Synthetic Data Only (Generated via rules)")
        print("3: Both (50/50 Mix - Recommended)")
        choice = input("Enter 1, 2, or 3: ").strip()
    else:
        print("No --mode flag provided and stdin is not interactive. Defaulting to mode 3 (Both).")
        choice = "3"

    if choice == "1":
        return "real"
    elif choice == "2":
        return "synthetic"
    else:
        return "both"


if __name__ == "__main__":
    print("🚀 Starting RLVR Pipeline...")

    dataset_mode = get_dataset_mode()
    print(f"Selected dataset mode: {dataset_mode}")

    # 1. Load model & tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # 2. Prepare dataset
    train_dataset = prepare_dataset(tokenizer, num_samples=50, mode=dataset_mode)

    # 3. Define reward functions
    reward_funcs = [format_reward_func, correctness_reward_func]

    # 4. Setup GRPO config
    training_args = setup_grpo_config()

    # 5. Train
    trainer = run_training(model, tokenizer, train_dataset, reward_funcs, training_args)

    # 6. Rigorous A/B evaluation (runs before clearing training model)
    rigorous_evaluation(adapter_path=training_args.output_dir)

    # 7. Clear training memory
    del model
    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    # 8. Final single-prompt evaluation
    evaluate_model(adapter_path=training_args.output_dir)

    print("✅ Pipeline Complete!")