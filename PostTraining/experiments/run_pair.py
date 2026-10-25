"""显式、单GPU的 SFT/DFT LoRA smoke 对照；默认仅预检，不训练。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import time

from .contracts import (PROMPT_TEMPLATE, canonical_hash, encode_dataset, ensure_disjoint,
                        file_hash, load_jsonl, make_order)
from .preflight import environment_report
from .recovery import archive_run, snapshot_sources, unique_run_dir


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--model-revision", default="main", help="运行前解析为commit，两个分支共享该commit")
    parser.add_argument("--train-jsonl", type=Path)
    parser.add_argument("--val-jsonl", type=Path)
    parser.add_argument("--output", type=Path, default=None, help="默认生成唯一run目录；已有目录不覆盖")
    parser.add_argument("--recovery-dir", type=Path, help="可选：已存在、已挂载的恢复包目录；不自动登录")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=20, help="每个分支optimizer步数，硬上限20")
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--confirm-smoke-training", action="store_true", help="启用模型下载与CUDA训练")
    return parser


def checked_inputs(args):
    if not 1 <= args.max_steps <= 20:
        raise ValueError("Smoke run requires 1 <= max_steps <= 20")
    if not 2 <= args.max_length <= 512:
        raise ValueError("Smoke run requires 2 <= max_length <= 512")
    if not 1 <= args.gradient_accumulation <= 8:
        raise ValueError("Smoke run requires 1 <= gradient_accumulation <= 8")
    if not 1 <= args.max_new_tokens <= 128:
        raise ValueError("Smoke run requires 1 <= max_new_tokens <= 128")
    if not math.isfinite(args.learning_rate) or not 0 < args.learning_rate <= 1e-3:
        raise ValueError("Smoke learning rate must be finite and in (0, 0.001]")
    if bool(args.train_jsonl) != bool(args.val_jsonl):
        raise ValueError("Provide both --train-jsonl and --val-jsonl")
    bundled = Path(__file__).parent / "data"
    train_path = args.train_jsonl or bundled / "smoke_train.jsonl"
    val_path = args.val_jsonl or bundled / "smoke_val.jsonl"
    train, validation = load_jsonl(train_path), load_jsonl(val_path)
    if len(train) > 256 or len(validation) > 32:
        raise ValueError("Smoke cap: <=256 train examples, <=32 validation examples")
    ensure_disjoint(train, validation)
    return train_path, val_path, train, validation


def seeded(seed, torch, np):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    # 固定随机数与顺序，但不声称跨GPU/驱动逐位一致。


def adapter_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.named_parameters()):
        if value.requires_grad:
            digest.update(name.encode("utf-8"))
            digest.update(value.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


def tensors(item, torch):
    return {key: torch.tensor([item[key]], device="cuda", dtype=torch.long)
            for key in ("input_ids", "labels")}


def evaluate(model, encoded, torch):
    """两个分支统一报告 unweighted NLL；不比较两种不同目标的数值高低。"""
    from .torch_losses import causal_loss_sums
    model.eval()
    nll, count = 0.0, 0
    with torch.no_grad():
        for item in encoded:
            batch = tensors(item, torch)
            logits = model(input_ids=batch["input_ids"], use_cache=False).logits
            stats = causal_loss_sums(logits, batch["labels"], "sft")
            nll += float(stats["nll_sum"].item())
            count += stats["target_tokens"]
    return {"unweighted_nll": nll / count, "target_tokens": count, "examples": len(encoded)}


def generate_examples(model, tokenizer, rows, encoded, torch, max_new_tokens):
    """相同保留样本、相同prompt token、greedy解码；不提供误导性的任务正确率。"""
    model.eval()
    row_lookup = {row["id"]: row for row in rows}
    results = []
    with torch.no_grad():
        for item in encoded[:4]:
            prompt = torch.tensor([item["prompt_ids"]], device="cuda", dtype=torch.long)
            generated = model.generate(
                input_ids=prompt, attention_mask=torch.ones_like(prompt),
                do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            new_ids = generated[0, prompt.shape[1]:].tolist()
            results.append({"id": item["id"], "prompt": row_lookup[item["id"]]["prompt"],
                            "reference": row_lookup[item["id"]]["completion"],
                            "generated": tokenizer.decode(new_ids, skip_special_tokens=True),
                            "generated_token_ids": new_ids})
    return results


def run_branch(objective, args, model_sha, tokenizer, train, validation, val_rows, order,
               dtype, initial_hash, torch, np):
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    from .torch_losses import causal_loss_sums

    branch = args.output / objective
    branch.mkdir()
    started = now()
    branch_start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    seeded(args.seed, torch, np)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=model_sha, trust_remote_code=False,
        torch_dtype=dtype, attn_implementation="sdpa",
    ).to("cuda")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"], bias="none", init_lora_weights=True,
    ))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    current_hash = adapter_hash(model)
    if initial_hash is not None and current_hash != initial_hash:
        raise RuntimeError("Initial adapter mismatch: abort comparison")
    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in name for name in trainable_names):
        raise RuntimeError("Unexpected trainable parameters; expected LoRA only")
    write_json(branch / "initialization.json", {
        "started_utc": started, "initial_adapter_sha256": current_hash,
        "trainable_parameters": sum(p.numel() for p in trainable),
        "trainable_names": trainable_names,
    })
    before = evaluate(model, validation, torch)
    before_generation = generate_examples(model, tokenizer, val_rows, validation, torch, args.max_new_tokens)
    write_json(branch / "before.json", {"validation": before, "generation": before_generation})
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)
    model.train()
    # 基线评估可能消耗内部状态；每个分支训练前再次统一随机数状态。
    seeded(args.seed, torch, np)
    torch.cuda.synchronize()
    start_training = time.perf_counter()
    with (branch / "train_log.jsonl").open("w", encoding="utf-8") as log:
        for step in range(args.max_steps):
            indices = order[step * args.gradient_accumulation:(step + 1) * args.gradient_accumulation]
            denominator = sum(train[i]["target_tokens"] for i in indices)
            optimizer.zero_grad(set_to_none=True)
            objective_sum = nll_sum = probability_sum = 0.0
            for index in indices:
                batch = tensors(train[index], torch)
                output = model(input_ids=batch["input_ids"], use_cache=False)
                stats = causal_loss_sums(output.logits, batch["labels"], objective)
                if not torch.isfinite(stats["loss_sum"]):
                    raise FloatingPointError("Nonfinite objective; abort rather than saving invalid result")
                # 按整个累计组的有效token总数归一化。长短样本不会获得无意的等样本权重。
                (stats["loss_sum"] / denominator).backward()
                objective_sum += stats["loss_sum"].detach().item()
                nll_sum += stats["nll_sum"].item()
                probability_sum += stats["probability_sum"].item()
                del output, stats, batch
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            optimizer.step()
            row = {"step": step + 1, "objective": objective,
                   "optimization_loss": objective_sum / denominator,
                   "unweighted_nll": nll_sum / denominator,
                   "mean_target_probability": probability_sum / denominator,
                   "supervised_tokens": denominator, "microbatches": len(indices),
                   "grad_norm_before_clip": float(grad_norm.item()),
                   "learning_rate": args.learning_rate}
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
            print(json.dumps(row, allow_nan=False), flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_training
    after = evaluate(model, validation, torch)
    after_generation = generate_examples(model, tokenizer, val_rows, validation, torch, args.max_new_tokens)
    model.save_pretrained(branch / "adapter", safe_serialization=True)
    tokenizer.save_pretrained(branch / "tokenizer")
    write_json(branch / "after.json", {"validation": after, "generation": after_generation})
    result = {"objective": objective, "status": "smoke_complete_not_research_evidence",
              "initial_adapter_sha256": current_hash,
              "before_validation": before, "after_validation": after,
              "training_seconds": elapsed,
              "branch_wall_seconds": time.perf_counter() - branch_start,
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
              "finished_utc": now()}
    write_json(branch / "summary.json", result)
    del optimizer, trainable, model
    gc.collect()
    torch.cuda.empty_cache()
    return result, current_hash, before_generation


def main():
    total_start = time.perf_counter()
    args = argument_parser().parse_args()
    if args.output is None:
        args.output = unique_run_dir()
    if args.recovery_dir is not None and not args.recovery_dir.is_dir():
        raise FileNotFoundError("--recovery-dir必须是已存在、已挂载的目录")
    train_path, val_path, rows_train, rows_val = checked_inputs(args)
    env = environment_report()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    if not args.confirm_smoke_training:
        print(json.dumps({"status": "preflight_only_no_download_no_training", "config": config,
                          "train_examples": len(rows_train), "validation_examples": len(rows_val),
                          "environment": env}, ensure_ascii=False, indent=2))
        return
    if not env["cuda_available"]:
        raise RuntimeError("CUDA GPU required for this smoke script. CPU notebook remains usable.")
    if args.output.exists():
        raise FileExistsError("Output already exists; choose a new path to preserve previous records")
    for package in ("numpy", "torch", "transformers", "peft", "accelerate", "huggingface-hub"):
        if not env["packages"].get(package):
            raise RuntimeError(f"Missing dependency: {package}; see experiments/requirements-colab.txt")
    args.output.mkdir(parents=True)
    manifest = {"created_utc": now(), "status": "preparing", "config": config, "environment": env,
                "purpose": "SFT/DFT implementation smoke; not paper reproduction or novelty evidence",
                "dataset": {"kind": "user_local_jsonl" if args.train_jsonl else "synthetic_smoke_only",
                            "train_file_sha256": file_hash(train_path), "val_file_sha256": file_hash(val_path),
                            "train_records_sha256": canonical_hash(rows_train), "val_records_sha256": canonical_hash(rows_val),
                            "train_rows": len(rows_train), "validation_rows": len(rows_val),
                            "external_dataset_revision": None, "test_split_accessed": False},
                "training_contract": {"batch_size": 1, "packing": False, "padding": False,
                    "quantization": False, "prompt_template": PROMPT_TEMPLATE,
                    "tokenization": "separate prompt/space-prefixed completion + one EOS",
                    "masking": "prompt labels=-100; completion and EOS supervised; next-token shift once",
                    "overlength": "drop whole example and record; never truncate completion",
                    "normalization": "sum loss / supervised tokens per full accumulation group",
                    "dft_weight": "exp(-token_nll.detach()); no weight-sum normalization",
                    "optimizer": "AdamW; constant lr; weight_decay=0; clip_grad_norm=1",
                    "lora": {"r": 8, "alpha": 16, "dropout": 0.0, "targets": ["q_proj", "v_proj"]}},
                "generation": {"do_sample": False, "num_beams": 1, "max_new_tokens": args.max_new_tokens,
                               "validation_examples_max": 4},
                "source_hashes": {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}}
    write_json(args.output / "manifest.json", manifest)
    snapshot_sources(Path(__file__).resolve().parents[1], args.output)
    snapshot = args.output / "data_snapshot"
    snapshot.mkdir()
    shutil.copyfile(train_path, snapshot / "train.jsonl")
    shutil.copyfile(val_path, snapshot / "validation.jsonl")
    try:
        import numpy as np
        import torch
        from huggingface_hub import HfApi
        from transformers import AutoTokenizer
        # 只读取公开模型元数据。不登录，不写入Hub。model+tokenizer共享同一完整SHA。
        model_sha = HfApi(token=False).model_info(args.model, revision=args.model_revision).sha
        if not model_sha or not re.fullmatch(r"[0-9a-f]{40}", model_sha):
            raise ValueError("Unable to resolve immutable model commit")
        manifest["model_commit"] = model_sha
        manifest["tokenizer_commit"] = model_sha
        write_json(args.output / "manifest.json", manifest)
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=model_sha, trust_remote_code=False)
        train, dropped_train = encode_dataset(rows_train, tokenizer, args.max_length)
        validation, dropped_val = encode_dataset(rows_val, tokenizer, args.max_length)
        order = make_order(len(train), args.max_steps * args.gradient_accumulation, args.seed)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        manifest["dtype"] = str(dtype)
        manifest["tokenized_data"] = {"train_kept": len(train), "validation_kept": len(validation),
            "train_dropped": dropped_train, "validation_dropped": dropped_val,
            "train_sha256": canonical_hash(train), "validation_sha256": canonical_hash(validation),
            "order_sha256": canonical_hash(order), "order_example_ids": [train[i]["id"] for i in order]}
        write_json(args.output / "tokenized_snapshot.json", {"train": train, "validation": validation})
        write_json(args.output / "manifest.json", manifest)
        frozen = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
        # freeze通常只包含包名版本；去掉可能包含本机路径、私有URL或凭据的direct URL行。
        safe_freeze = "\n".join(line for line in frozen.stdout.splitlines() if re.fullmatch(r"[A-Za-z0-9_.-]+==[^\s/@]+", line))
        (args.output / "installed-packages.txt").write_text(safe_freeze + "\n", encoding="utf-8")
        results, first_hash, first_generation = [], None, None
        for objective in ("sft", "dft"):
            manifest["status"] = f"running_{objective}"
            write_json(args.output / "manifest.json", manifest)
            result, initial_hash, before_generation = run_branch(
                objective, args, model_sha, tokenizer, train, validation, rows_val,
                order, dtype, first_hash, torch, np)
            if first_hash is None:
                first_hash, first_generation = initial_hash, before_generation
            elif before_generation != first_generation:
                raise RuntimeError("Baseline greedy generation differs; comparison invalid")
            results.append(result)
            manifest["completed_branches"] = [r["objective"] for r in results]
            manifest["total_wall_seconds"] = time.perf_counter() - total_start
            write_json(args.output / "manifest.json", manifest)
            archive_run(args.output, args.recovery_dir)
        manifest["status"] = "smoke_complete_not_research_evidence"
        manifest["finished_utc"] = now()
        write_json(args.output / "comparison.json", {"status": manifest["status"], "results": results,
                   "interpretation": "No claim about research quality. Different optimization losses are not comparable."
                   })
        write_json(args.output / "manifest.json", manifest)
        print(f"Saved smoke records to {args.output}")
    except BaseException as exc:
        manifest["status"] = ("interrupted_partial_outputs_not_results" if isinstance(exc, KeyboardInterrupt)
                              else "failed_partial_outputs_not_results")
        manifest["failure_type"] = type(exc).__name__
        manifest["finished_utc"] = now()
        write_json(args.output / "manifest.json", manifest)
        raise
    finally:
        manifest["total_wall_seconds"] = time.perf_counter() - total_start
        write_json(args.output / "manifest.json", manifest)
        try:
            archive_run(args.output, args.recovery_dir)
        except Exception as exc:
            print(f"恢复包归档/复制失败：{type(exc).__name__}；原始记录保留在 {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
