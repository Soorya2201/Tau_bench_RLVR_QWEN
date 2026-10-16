# Tau-bench RLVR: GRPO fine-tuning Qwen2.5-7B for structured tool calls

**Reinforcement learning from verifiable rewards (RLVR) that teaches Qwen2.5-7B-Instruct to answer customer-support requests with the correct sequence of JSON tool calls instead of conversational prose. It runs on a single 16 GB GPU with 4-bit QLoRA and GRPO.**

![License](https://img.shields.io/badge/license-MIT-yellow.svg)

## Highlights

- **Four rule-based, verifiable reward functions** instead of a learned reward model. Each scores a different failure mode of tool-calling agents: malformed JSON, wrong or hallucinated tools, drifted arguments, and wrong call order.
- **GRPO via Hugging Face TRL.** There is no critic network: four sampled completions per prompt are scored against each other, which is cheaper and more stable than PPO for short, sparse-reward outputs.
- **Fits on one GPU.** Qwen2.5-7B is loaded in 4-bit NF4 with LoRA (r = 16, α = 32) on all attention and MLP projections.
- **Real and synthetic data.** It mixes tau2-bench traces (`Jarrodbarnes/tau2-sft-v4-dataset`) with procedurally generated retail-support tasks that have known ground-truth tool sequences.
- **Built-in A/B check.** The base model and the fine-tuned adapter are compared on held-out prompts after training.

Originally built as a take-home for an ML research internship (AfterQuery).

## Why RLVR for tool calling

[tau2-bench](https://github.com/sierra-research/tau2-bench) measures whether an LLM can act as a support agent: read a request and emit the right tool calls (`get_order_status`, `refund_order`, `exchange_item`). Instruction-tuned models chat well but are unreliable at strict, structured output. They wrap JSON in prose, invent fields, or pick a plausible tool instead of the correct one. All of those failures can be **checked by a program**, so the reward can be computed exactly, with no human preference labels.

## Reward design (`reward_func.py`)

| Reward | Checks | Range |
|---|---|---|
| `format_reward_func` | Output parses as JSON tool calls; every call has `name` (+0.5) and `arguments` (+1.0) | -1.0 to 1.0 |
| `correctness_reward_func` | Recall of ground-truth tool names, minus a penalty for hallucinated tools | -0.5 to 1.0 |
| `argument_fidelity_reward_func` | Fuzzy match (`SequenceMatcher`) of each call's arguments against ground truth | -0.5 to 1.0 |
| `trajectory_order_reward_func` | Longest-common-subsequence ratio between predicted and expected call order | 0.0 to 1.0 |

The four rewards are passed to `GRPOTrainer` together. `compute_total_reward` also provides a weighted composite (1.0 / 1.0 / 0.75 / 0.5, clipped to ±2) for analysis.

## Pipeline

```
python main.py --mode 3
  1. Load Qwen2.5-7B-Instruct in 4-bit NF4, attach LoRA adapters        model_setup.py
  2. Build 50 tasks (real / synthetic / 50-50), 80/20 train/val split    dataset_prep.py
  3. GRPO: 4 generations per prompt, scored by the four rewards           trainer.py, grpo_config.py
     eval and checkpoint every 5 steps, early stopping (patience 3)
  4. A/B evaluation: base vs. fine-tuned on held-out prompts              evaluation.py
  5. Save the adapter to ./qwen-taubench-rlvr
```

## Hyperparameters (`grpo_config.py`)

| Parameter | Value |
|---|---|
| Base model | Qwen/Qwen2.5-7B-Instruct |
| Quantization | 4-bit NF4 (bitsandbytes), bf16 compute |
| LoRA | r = 16, α = 32, q/k/v/o + gate/up/down projections |
| Optimizer / LR | AdamW, 1e-5, cosine schedule |
| Steps | 75 max, batch 1 × 8 gradient accumulation |
| GRPO generations | 4 per prompt, max 128 completion tokens |
| KL β | 0.04 |

## Results

```
=== Evaluation Results ===
Base Model JSON Formatting Accuracy: 40.0%
Finetuned Model JSON Formatting Accuracy: 80.0%
```

**Scope of this result:** the A/B check runs on **5 held-out prompts**, so 40% → 80% is 2/5 → 4/5 valid JSON tool calls after a 75-step run on 40 training tasks. It measures *format* compliance, not tool-choice correctness. It shows that the reward signal moves the model in the right direction; it is not a benchmark score.

**Next steps:** evaluate on the full tau2-bench test split; report tool-choice, argument and ordering accuracy alongside format; run multiple seeds; and scale the training set beyond 50 tasks.

## Run it

Requires Python 3.10+ and a CUDA GPU with 16 GB+ VRAM (24 GB recommended). `bitsandbytes` does not run on CPU.

```bash
pip install -r requirements.txt
python main.py --mode 3      # 1 = real tau2 traces, 2 = synthetic, 3 = 50/50 mix
```

## Repository layout

```
main.py            entry point: data → GRPO training → A/B evaluation
model_setup.py     4-bit NF4 Qwen2.5-7B + LoRA
dataset_prep.py    tau2 trace formatting + synthetic task generator
reward_func.py     four verifiable reward functions + weighted composite
grpo_config.py     GRPO hyperparameters
trainer.py         GRPOTrainer with early stopping
evaluation.py      base vs. fine-tuned A/B evaluation
```

## License

MIT — see [LICENSE](LICENSE).

Soorya Narayanan · [GitHub](https://github.com/Soorya2201) · [LinkedIn](https://www.linkedin.com/in/soorya-venkat)
