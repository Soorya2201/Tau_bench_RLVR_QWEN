# RLVR Pipeline for AfterQuery

Fine-tunes **Qwen2.5-7B-Instruct** on the tau2-bench customer support benchmark using **GRPO** (Group Relative Policy Optimization) via HuggingFace TRL. 

The model is trained to output structured JSON tool-call sequences instead of conversational responses.

---



### Install dependencies

```bash
pip install -r requirements.txt
```
## Run my Work 

```bash
python main.py
```

---


## Project Structure

```
.
├── main.py            # Entry point — runs the full pipeline
├── model_setup.py     # Loads Qwen2.5-7B with 4-bit quantization + LoRA
├── dataset_prep.py    # tau2-bench dataset loading + synthetic task generation
├── reward_func.py     # GRPO reward functions (format + correctness)
├── grpo_config.py     # All training hyperparameters
├── trainer.py         # GRPOTrainer training loop
├── evaluation.py      # A/B evaluation: base model vs fine-tuned model
├── requirements.txt   # Python dependencies
└── README.md
```

---

## Requirements

- Python 3.10+
- CUDA-capable GPU with **at least 16GB VRAM** (24GB recommended)
- CUDA 11.8 or higher



> **Note:** `bitsandbytes` requires a CUDA-enabled GPU. CPU-only machines are not supported.

---


On launch, you will be prompted to choose a dataset mode:

```
Which dataset mode would you like to use for training?
1: Real Data Only (HuggingFace tau2-bench trace)
2: Synthetic Data Only (Generated via rules)
3: Both (50/50 Mix - Recommended)

Enter 1, 2, or 3:
```

The pipeline will then run end-to-end:

1. Load Qwen2.5-7B-Instruct (4-bit NF4 quantization)
2. Apply LoRA adapters
3. Prepare the training dataset
4. Run GRPO training
5. Save the fine-tuned adapter to `./qwen-taubench-rlvr`
6. Run A/B evaluation comparing base vs fine-tuned model
7. Run a final single-prompt sanity check

---

## Reward Functions

Two reward functions are composed additively during training:

| Function | What it measures | Reward Range |
|---|---|---|
| `format_reward_func` | Valid JSON structure with a `name` key | -1.0 to +0.5 |
| `correctness_reward_func` | Correct tool selected for the user intent | -2.0 to +1.0 |

---

## Key Hyperparameters

| Parameter | Value |
|---|---|
| Base Model | Qwen/Qwen2.5-7B-Instruct |
| Quantization | 4-bit NF4 (BitsAndBytes) |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| Learning Rate | 1e-5 |
| LR Scheduler | Cosine |
| Training Steps | 75 |
| Batch Size | 1 + 8 gradient accumulation steps |
| GRPO Generations | 4 per prompt |
| Max Completion Tokens | 128 |
| KL Beta | 0.01 |
| Optimizer | AdamW (torch) |

---

## Output

After training, the fine-tuned LoRA adapter is saved to:

```
./qwen-taubench-rlvr/
```

Evaluation results are printed to stdout:

```
=== Evaluation Results ===
Base Model JSON Formatting Accuracy: 40.0%
Finetuned Model JSON Formatting Accuracy: 80.0%
```

---

## Scaling

**Larger models (13B–70B):** Lower learning rate to `5e-6`, increase LoRA rank to `32–64`, keep GRPO generations at 4.

**Smaller models (1B–3B):** Raise learning rate to `3e-5`, increase GRPO generations to `8–16`, increase batch size to `4–8`.

---

## Author

Soorya Narayanan V — AfterQuery ML/AI Research Internship Take-Home