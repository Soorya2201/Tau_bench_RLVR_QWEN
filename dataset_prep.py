import random
import datasets
from datasets import load_dataset

def generate_synthetic_tau2_task(tokenizer):
    first_names = ["Alice", "Soorya", "Charlie", "Diana", "Sabesh", "Fiona"]
    items = ["wireless headphones", "mechanical keyboard", "running shoes", "coffee maker", "smartwatch"]
    issues = [
        ("arrived damaged, I need a refund", ["get_order_status", "refund_order"]),
        ("hasn't arrived yet, can you check the status?", ["get_order_status"]),
        ("is the wrong color, I wanted black instead", ["get_order_status", "exchange_item"]),
        ("was charged twice to my account, please refund the duplicate", ["get_order_status", "refund_order"])
    ]

    name = random.choice(first_names)
    item = random.choice(items)
    order_id = f"ORD-{random.randint(100000, 999999)}"
    issue_text, expected_tools = random.choice(issues)

    user_content = f"Hi, my name is {name}. I recently ordered a {item} (Order #{order_id}), but it {issue_text}."
    system_content = (
        "You are an AI customer support assistant for a retail company. "
        "You have access to the following tools to assist the user:\n"
        "- get_order_status(order_id: str): Returns the current status and details of an order.\n"
        "- refund_order(order_id: str): Initiates a refund for the specified order.\n"
        "- exchange_item(order_id: str, new_item_details: str): Initiates an exchange for an item.\n"
        "Analyze the user's request and output the correct sequence of tool calls to assist them."
    )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return {"prompt": prompt_text}


def prepare_dataset(tokenizer, num_samples=200, mode="both"):
    print(f"Preparing {num_samples} tasks in '{mode}' mode...")

    if mode in ["real", "both"]:
        real_ds = load_dataset("Jarrodbarnes/tau2-sft-v4-dataset", split="train")

        def format_for_grpo(example):
            system_msg = next(msg["content"] for msg in example["prompt"] if msg["role"] == "system")
            user_msg = next(msg["content"] for msg in example["prompt"] if msg["role"] == "user")
            messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}]
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return {"prompt": prompt_text}

        real_ds = real_ds.map(format_for_grpo).select_columns(["prompt"])
        real_list = list(real_ds)
        random.shuffle(real_list)

    final_data = []

    if mode == "real":
        final_data = real_list[:num_samples]
    elif mode == "synthetic":
        final_data = [generate_synthetic_tau2_task(tokenizer) for _ in range(num_samples)]
    elif mode == "both":
        half = num_samples // 2
        real_part = real_list[:half]
        synth_part = [generate_synthetic_tau2_task(tokenizer) for _ in range(num_samples - half)]
        final_data = real_part + synth_part
        random.shuffle(final_data)
    else:
        raise ValueError("Invalid mode selected.")

    return datasets.Dataset.from_list(final_data)