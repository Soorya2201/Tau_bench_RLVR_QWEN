from trl import GRPOConfig

def setup_grpo_config():
    print("Initializing GRPO Configuration...")
    return GRPOConfig(
        output_dir="./qwen-taubench-rlvr",
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        logging_steps=2,
        max_steps=75,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_generations=4,
        max_completion_length=128,
        beta=0.04,
        bf16=True,
        optim="adamw_torch",
        report_to="none",
        # ── Evaluation & early stopping ──
        eval_strategy="steps",
        eval_steps=5,
        save_strategy="steps",
        save_steps=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
    )