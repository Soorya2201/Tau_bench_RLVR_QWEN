from trl import GRPOTrainer
from transformers import EarlyStoppingCallback


def run_training(model, tokenizer, train_dataset, reward_funcs, training_args,
                 eval_dataset=None):
    print("Initializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    print("Starting Training Loop...")
    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    print(f"Model saved to {training_args.output_dir}")
    return trainer