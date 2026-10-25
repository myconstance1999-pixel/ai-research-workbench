"""数据、mask 与可独立核验的损失约定；CPU 自检不依赖 torch。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

IGNORE_INDEX = -100
PROMPT_TEMPLATE = "Question: {prompt}\nAnswer:"


def canonical_hash(value) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def file_hash(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_records(rows, name="data"):
    if not rows:
        raise ValueError(f"{name}: empty dataset")
    ids, prompts = set(), set()
    clean = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{name}[{i}]: expected object")
        for key in ("id", "prompt", "completion"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"{name}[{i}]: {key} must be a nonempty string")
        key = " ".join(row["prompt"].split()).casefold()
        if row["id"] in ids or key in prompts:
            raise ValueError(f"{name}[{i}]: duplicate id or normalized prompt")
        ids.add(row["id"])
        prompts.add(key)
        clean.append({k: row[k] for k in ("id", "prompt", "completion")})
    return clean


def load_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{Path(path).name}:{number}: invalid JSON") from exc
    return validate_records(rows, Path(path).name)


def ensure_disjoint(train, validation):
    train_ids = {r["id"] for r in train}
    val_ids = {r["id"] for r in validation}
    train_prompts = {" ".join(r["prompt"].split()).casefold() for r in train}
    val_prompts = {" ".join(r["prompt"].split()).casefold() for r in validation}
    if train_ids & val_ids or train_prompts & val_prompts:
        raise ValueError("Train/validation share an id or normalized prompt")


def encode_record(row, tokenizer, max_length=512):
    """分段编码并拼接：训练和生成共用完全相同的 prompt token 序列。

    不用 chat template（示例是 Base 模型）。不拼接整串后猜测 BPE 边界。
    response 带一个前导空格，末尾显式 EOS；prompt 的 label 为 -100。
    超长样本整条剔除，避免悄悄把最终答案截掉；不做 packing 或 padding。
    """
    if max_length < 2:
        raise ValueError("max_length must be >= 2")
    prompt = PROMPT_TEMPLATE.format(prompt=row["prompt"])
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(" " + row["completion"], add_special_tokens=False)
    if not prompt_ids or not response_ids:
        raise ValueError(f"{row['id']}: tokenizer returned an empty segment")
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id")
    # 一个末尾 EOS。若输入中有显式特殊 token，拒绝而非重复插入或隐式结束。
    if tokenizer.eos_token_id in prompt_ids + response_ids:
        raise ValueError(f"{row['id']}: literal EOS in input is unsupported")
    response_ids = response_ids + [tokenizer.eos_token_id]
    input_ids = prompt_ids + response_ids
    if len(input_ids) > max_length:
        return None, {"id": row["id"], "reason": "overlength_dropped", "length": len(input_ids)}
    labels = [IGNORE_INDEX] * len(prompt_ids) + response_ids
    result = {
        "id": row["id"], "input_ids": input_ids, "labels": labels,
        "prompt_ids": prompt_ids, "target_tokens": len(response_ids),
    }
    if sum(v != IGNORE_INDEX for v in labels[1:]) != result["target_tokens"]:
        raise ValueError("Shift/mask contract violated")
    return result, None


def encode_dataset(rows, tokenizer, max_length=512):
    kept, dropped = [], []
    for row in rows:
        item, issue = encode_record(row, tokenizer, max_length)
        if item is not None:
            kept.append(item)
        else:
            dropped.append(issue)
    if not kept:
        raise ValueError("No usable examples remain after length filtering")
    return kept, dropped


def make_order(n_rows, total_microsteps, seed):
    """固定一份抽样顺序供两个分支使用；每轮洗牌，少量数据可能重复。"""
    if n_rows < 1 or total_microsteps < 1:
        raise ValueError("Nonempty rows and positive microsteps required")
    rng, result = random.Random(seed), []
    while len(result) < total_microsteps:
        indices = list(range(n_rows))
        rng.shuffle(indices)
        result.extend(indices)
    return result[:total_microsteps]


def numpy_causal_loss(logits, labels, objective="sft"):
    """NumPy 参考值与解析梯度；对齐 torch 实现，但不是训练实现。

    logits: [batch, sequence, vocabulary]。logits[:, :-1] 对 labels[:, 1:]。
    DFT 梯度是 p(target).detach() * grad(CE)，不是对 -p log p 求全导。
    """
    import numpy as np
    if objective not in ("sft", "dft"):
        raise ValueError("objective must be sft or dft")
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    if logits.ndim != 3 or labels.shape != logits.shape[:2] or logits.shape[1] < 2:
        raise ValueError("Expected [B,T,V] logits, matching [B,T] labels, T>=2")
    if not np.issubdtype(labels.dtype, np.integer) or not np.isfinite(logits).all():
        raise ValueError("Expected integer labels and finite logits")
    shifted_labels = labels[:, 1:]
    valid = shifted_labels != IGNORE_INDEX
    count = int(valid.sum())
    if count == 0:
        raise ValueError("No supervised tokens after shifting")
    targets = shifted_labels[valid]
    if (targets < 0).any() or (targets >= logits.shape[-1]).any():
        raise ValueError("Target token outside vocabulary")
    selected = logits[:, :-1, :][valid]
    shifted = selected - selected.max(axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    probs = np.exp(log_probs)
    target_logp = log_probs[np.arange(count), targets]
    target_p = np.exp(target_logp)
    nll = -target_logp
    weights = target_p if objective == "dft" else np.ones_like(target_p)
    gradient_selected = probs.copy()
    gradient_selected[np.arange(count), targets] -= 1
    gradient_selected *= weights[:, None] / count
    gradient = np.zeros_like(logits)
    gradient[:, :-1, :][valid] = gradient_selected
    return {
        "objective": objective,
        "loss": float((weights * nll).mean()),
        "unweighted_nll": float(nll.mean()),
        "mean_target_probability": float(target_p.mean()),
        "target_tokens": count,
        "gradient": gradient,
    }
