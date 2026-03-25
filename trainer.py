from trl import GRPOTrainer

def run_training(model, tokenizer, train_dataset, reward_funcs, training_args):
    print("Initializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
    )
    print("Starting Training Loop...")
    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    print(f"Model saved to {training_args.output_dir}")
    return trainer