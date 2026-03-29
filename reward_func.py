import re
import json
from typing import Optional, List
from difflib import SequenceMatcher


def _parse_tool_calls(completion: str) -> Optional[List[dict]]:
    """Try to parse a completion string into a list of tool-call dicts."""
    text = completion.strip()
    # Try the raw string first, then fall back to extracting a JSON array
    for candidate in [text, re.search(r'\[.*\]', text, re.DOTALL)]:
        if candidate is None:
            continue
        try:
            parsed = json.loads(candidate if isinstance(candidate, str) else candidate.group(0))
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return [parsed]
        except (json.JSONDecodeError, TypeError):
            continue
    return None


# Reward Function 1

def format_reward_func(completions, **kwargs):
    """Penalises structurally invalid or schema-violating tool-call JSON."""
    rewards = []
    for completion in completions:
        parsed = _parse_tool_calls(completion)
        if parsed is None:
            rewards.append(-1.0)
            continue
        if not isinstance(parsed, list) or len(parsed) == 0:
            rewards.append(-0.5)
            continue
        all_have_name_and_args = all(
            isinstance(item, dict) and "name" in item and "arguments" in item
            for item in parsed
        )
        all_have_name = all(
            isinstance(item, dict) and "name" in item
            for item in parsed
        )
        if all_have_name_and_args:
            rewards.append(1.0)
        elif all_have_name:
            rewards.append(0.5)
        else:
            rewards.append(-0.5)
    return rewards



# ─────────────────────────────────────────────────────────────────────────────
def correctness_reward_func(completions, **kwargs):
    """Penalises hallucinated tool names and rewards F1-style recall."""
    ground_truth_tools = kwargs.get("ground_truth_tools", None)
    if ground_truth_tools is None:
        return [0.0] * len(completions)
    rewards = []
    for completion, gt_tools in zip(completions, ground_truth_tools):
        parsed = _parse_tool_calls(completion)
        if parsed is None:
            rewards.append(-0.5)
            continue
        gt_names = [t["name"] for t in gt_tools if isinstance(t, dict) and "name" in t]
        pred_names = [t["name"] for t in parsed if isinstance(t, dict) and "name" in t]
        if not gt_names or not pred_names:
            rewards.append(-0.5)
            continue
        hits = sum(1 for name in pred_names if name in gt_names)
        hallucinations = len(pred_names) - hits
        reward = (hits / len(gt_names)) - 0.5 * (hallucinations / len(pred_names))
        rewards.append(max(-0.5, min(1.0, reward)))
    return rewards


# Reward Function 3

def argument_fidelity_reward_func(completions, **kwargs):
    """Penalises argument drift via fuzzy string matching on serialised args."""
    ground_truth_tools = kwargs.get("ground_truth_tools", None)
    if ground_truth_tools is None:
        return [0.0] * len(completions)
    rewards = []
    for completion, gt_tools in zip(completions, ground_truth_tools):
        parsed = _parse_tool_calls(completion)
        if parsed is None:
            rewards.append(-0.5)
            continue
        gt_by_name = {}
        for t in gt_tools:
            if isinstance(t, dict) and "name" in t:
                gt_by_name[t["name"]] = t.get("arguments", {})
        scores = []
        for pred in parsed:
            if not isinstance(pred, dict) or "name" not in pred:
                scores.append(0.0)
                continue
            if pred["name"] in gt_by_name:
                pred_args = json.dumps(pred.get("arguments", {}), sort_keys=True)
                gt_args = json.dumps(gt_by_name[pred["name"]], sort_keys=True)
                scores.append(SequenceMatcher(None, pred_args, gt_args).ratio())
            else:
                scores.append(0.0)
        if not scores:
            rewards.append(-0.5)
            continue
        reward = (sum(scores) / len(scores)) - 0.1
        rewards.append(max(-0.5, min(1.0, reward)))
    return rewards


# Reward Function 4

def trajectory_order_reward_func(completions, **kwargs):
    """Penalises incorrect tool invocation ordering via LCS ratio."""
    ground_truth_tools = kwargs.get("ground_truth_tools", None)
    if ground_truth_tools is None:
        return [0.0] * len(completions)

    def _lcs_length(seq_a, seq_b):
        m, n = len(seq_a), len(seq_b)
        dp: List[List[int]] = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq_a[i - 1] == seq_b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    rewards = []
    for completion, gt_tools in zip(completions, ground_truth_tools):
        parsed = _parse_tool_calls(completion)
        if parsed is None:
            rewards.append(0.0)
            continue
        pred_seq = [t["name"] for t in parsed if isinstance(t, dict) and "name" in t]
        gt_seq = [t["name"] for t in gt_tools if isinstance(t, dict) and "name" in t]
        if not pred_seq or not gt_seq:
            rewards.append(0.0)
            continue
        lcs_len = _lcs_length(pred_seq, gt_seq)
        reward = lcs_len / max(len(pred_seq), len(gt_seq))
        rewards.append(max(0.0, min(1.0, reward)))
    return rewards


# REWARD WEIGHTS & COMPOSITE SCORE

REWARD_WEIGHTS = {
    "format":      1.0,
    "correctness": 1.0,
    "arguments":   0.75,
    "ordering":    0.5,
}


def compute_total_reward(format_r, correctness_r, argument_r, ordering_r):
    """Combines per-axis rewards into a single clipped scalar for GRPO."""
    raw = (REWARD_WEIGHTS["format"] * format_r +
           REWARD_WEIGHTS["correctness"] * correctness_r +
           REWARD_WEIGHTS["arguments"] * argument_r +
           REWARD_WEIGHTS["ordering"] * ordering_r)
    return round(max(-2.0, min(2.0, raw)), 4)