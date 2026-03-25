import re
import json

def format_reward_func(prompts, completions, **kwargs):
    rewards = []
    for completion in completions:
        match = re.search(r'\{.*\}', completion, re.DOTALL)
        if match:
            try:
                parsed_json = json.loads(match.group(0))
                rewards.append(0.5 if 'name' in parsed_json else 0.1)
            except json.JSONDecodeError:
                rewards.append(-0.5)
        else:
            rewards.append(-1.0)
    return rewards


def correctness_reward_func(prompts, completions, **kwargs):
    rewards = []
    for prompt, completion in zip(prompts, completions):
        if "check its status" in prompt or "hasn't arrived" in prompt:
            if "get_order_status" in completion:
                rewards.append(1.0)
            elif "refund_order" in completion:
                rewards.append(-2.0)
            else:
                rewards.append(0.0)
        else:
            rewards.append(0.0)
    return rewards