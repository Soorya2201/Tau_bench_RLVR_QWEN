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
        beta=0.01,
        bf16=True,
        optim="adamw_torch",
        report_to="none"
    )