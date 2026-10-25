"""真实训练调用的 PyTorch loss；import 时不加载模型。"""


def causal_loss_sums(logits, labels, objective="sft"):
    import torch
    import torch.nn.functional as F

    if objective not in ("sft", "dft"):
        raise ValueError("objective must be sft or dft")
    if logits.ndim != 3 or labels.shape != logits.shape[:2] or logits.shape[1] < 2:
        raise ValueError("Expected logits [B,T,V], labels [B,T], T>=2")
    targets = labels[:, 1:]
    valid = targets != -100
    count = int(valid.sum().item())
    if count == 0:
        raise ValueError("No supervised tokens after shifting")
    selected_logits = logits[:, :-1, :][valid].float()
    selected_targets = targets[valid]
    # 所有概率与CE在float32计算；仅有效位置参与，不用pad/eos相等来推断mask。
    nll = F.cross_entropy(selected_logits, selected_targets, reduction="none")
    target_probability = torch.exp(-nll.detach())
    weights = target_probability if objective == "dft" else torch.ones_like(nll)
    return {
        "loss_sum": (weights * nll).sum(),
        "nll_sum": nll.detach().sum(),
        "probability_sum": target_probability.sum(),
        "target_tokens": count,
    }
