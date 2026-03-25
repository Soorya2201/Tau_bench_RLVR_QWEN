import torch
import numpy as np
import re
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def evaluate_model(adapter_path="./qwen-taubench-rlvr"):
    print("\n--- Starting Evaluation ---")
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=bnb_config
    )

    print("Merging RLVR LoRA adapters...")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    test_messages = [
        {"role": "system", "content": "You are a helpful agent. You have access to the tools: {'name': 'get_order_status'}, {'name': 'refund_order'}. Output in valid JSON."},
        {"role": "user", "content": "Can you check the status of order #X999 and refund it?"}
    ]
    prompt = tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print("Generating response...")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.1, do_sample=True)

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print("\n--- Final Model Output ---")
    print(response)
    print("--------------------------")


def rigorous_evaluation(adapter_path="./qwen-taubench-rlvr", num_test_samples=5):
    print("\n--- Starting Rigorous A/B Evaluation ---")
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", quantization_config=bnb_config
    )

    test_prompts = [
        "Can you check the status of order #123 and then refund it?",
        "I need to exchange order #456 for a blue one.",
        "Where is my package for order #789?",
        "My item arrived broken, order #321, please refund.",
        "Can I get the status of order #555?"
    ]

    system_msg = "You are a helpful agent. You have access to the tools: {'name': 'get_order_status'}, {'name': 'refund_order'}, {'name': 'exchange_item'}. Output in valid JSON."

    base_model_outputs = []
    finetuned_outputs = []

    print("\n[Running Inference: Base Model]")
    for user_msg in test_prompts:
        messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.1)
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        base_model_outputs.append(response)

    print("\n[Running Inference: Finetuned Model]")
    model = PeftModel.from_pretrained(model, adapter_path)

    for user_msg in test_prompts:
        messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.1)
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        finetuned_outputs.append(response)

    def calculate_formatting_score(completions):
        scores = []
        for completion in completions:
            match = re.search(r'\{.*\}', completion, re.DOTALL)
            if match:
                try:
                    json.loads(match.group(0))
                    scores.append(1.0)
                except:
                    scores.append(0.0)
            else:
                scores.append(0.0)
        return np.mean(scores)

    base_score = calculate_formatting_score(base_model_outputs)
    finetuned_score = calculate_formatting_score(finetuned_outputs)

    print("\n=== Evaluation Results ===")
    print(f"Base Model JSON Formatting Accuracy: {base_score * 100}%")
    print(f"Finetuned Model JSON Formatting Accuracy: {finetuned_score * 100}%")
    print("\nSample Base Output:\n", base_model_outputs[0])
    print("\nSample Finetuned Output:\n", finetuned_outputs[0])