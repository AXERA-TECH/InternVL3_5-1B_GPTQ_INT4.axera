#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_to_gptq.py

说明:
  - 使用 GPTQModel 将 HuggingFace 的 InternVL3_5-1B 量化为 GPTQ Int4
  - 支持验证量化前后 embedding 相似度
  - 单卡跑就不会报错, 多卡会在中途报错
"""

import os
import sys
import time
import json
import glob
import shutil
import argparse
import inspect
import hashlib
from typing import List, Tuple

os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["DISABLE_TF32"] = "1"

# -------------------------
# imports
# -------------------------
try:
    import torch
    import torch.nn.functional as F
    import numpy as np
    from transformers import AutoTokenizer, AutoModel, AutoConfig
    from datasets import load_dataset
    from gptqmodel import GPTQModel, QuantizeConfig, get_best_device
    from gptqmodel.utils.model import get_module_by_name_prefix


    def ensure_internvl_chat_supported():
        """Register InternVLChat as a Qwen3-compatible QModel for gptqmodel."""
        try:
            from gptqmodel.models import auto as gptq_auto
            from gptqmodel.models.definitions.qwen3 import Qwen3QModel
        except Exception:
            return

        if "internvl_chat" in getattr(gptq_auto, "MODEL_MAP", {}):
            return

        class InternVLChatQModel(Qwen3QModel):
            lm_head = "language_model.lm_head"
            pre_lm_head_norm_module = "language_model.model.norm"
            module_tree = [
                "language_model",
                "model",
                "layers",
                "#",
                {
                    "input_layernorm": ("input_layernorm:!",),
                    "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
                    "post_attention_layernorm": ("post_attention_layernorm:!",),
                    "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
                },
            ]

        gptq_auto.MODEL_MAP["internvl_chat"] = InternVLChatQModel
        supported = getattr(gptq_auto, "SUPPORTED_MODELS", None)
        if isinstance(supported, list) and "internvl_chat" not in supported:
            supported.append("internvl_chat")


    TOKENIZER_FILES = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "tokenizer.model",
        "spiece.model",
    ]


    def _copy_if_exists(src: str, dst: str):
        if os.path.exists(src):
            shutil.copy2(src, dst)


    def _copy_tokenizer_artifacts(model_dir: str, target_dir: str):
        for name in TOKENIZER_FILES:
            src = os.path.join(model_dir, name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(target_dir, name))


    def _collect_language_only_state(source_dir: str, target_dir: str) -> Tuple[str, str]:
        safetensor_files = sorted(glob.glob(os.path.join(source_dir, "*.safetensors")))
        use_safetensors = len(safetensor_files) > 0
        weight_files = safetensor_files if use_safetensors else sorted(glob.glob(os.path.join(source_dir, "*.bin")))
        if not weight_files:
            raise FileNotFoundError(f"未在 {source_dir} 中找到权重文件")

        prefix = "language_model."
        state = {}

        if use_safetensors:
            try:
                from safetensors import safe_open
                from safetensors.torch import save_file as safe_save
            except Exception as exc:
                raise RuntimeError("缺少 safetensors 依赖, 无法提取权重") from exc

            for wf in weight_files:
                with safe_open(wf, framework="pt") as f:
                    for key in f.keys():
                        if key.startswith(prefix):
                            trimmed = key[len(prefix):]
                            state[trimmed] = f.get_tensor(key)
            target_file = "model.safetensors"
            safe_save(state, os.path.join(target_dir, target_file))
        else:
            import torch

            for wf in weight_files:
                shard = torch.load(wf, map_location="cpu")
                for key, value in shard.items():
                    if key.startswith(prefix):
                        trimmed = key[len(prefix):]
                        state[trimmed] = value
                del shard
            target_file = "pytorch_model.bin"
            torch.save(state, os.path.join(target_dir, target_file))

        del state
        return target_file, weight_files[0]


    def ensure_internvl_chat_llm_submodel(model_id: str) -> str:
        try:
            cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        except Exception:
            return model_id

        config_path = os.path.join(os.path.abspath(cfg._name_or_path), "config.json")
        if not os.path.exists(config_path):
            return model_id

        with open(config_path, "r", encoding="utf-8") as f:
            raw_cfg = json.load(f)

        if raw_cfg.get("model_type") != "internvl_chat":
            return model_id

        llm_cfg = raw_cfg.get("llm_config")
        if llm_cfg is None:
            raise ValueError("InternVL 配置中缺少 llm_config 字段, 无法提取 Qwen3 子模型")

        source_dir = os.path.abspath(cfg._name_or_path)
        target_dir = os.path.join(source_dir, "_qwen3_llm")
        os.makedirs(target_dir, exist_ok=True)

        cfg_sha = hashlib.sha256(json.dumps(llm_cfg, sort_keys=True).encode("utf-8")).hexdigest()
        flag_path = os.path.join(target_dir, "build_meta.json")

        if os.path.exists(flag_path):
            try:
                with open(flag_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("config_sha") == cfg_sha:
                    weight_candidate = os.path.join(target_dir, meta.get("weight_file", "model.safetensors"))
                    if os.path.exists(weight_candidate):
                        return target_dir
            except Exception:
                pass

        print("[STEP] 检测到 InternVL 多模态模型, 正在提取 Qwen3 语言子模型用于 GPTQ ...")
        shutil.rmtree(target_dir, ignore_errors=True)
        os.makedirs(target_dir, exist_ok=True)

        llm_cfg = llm_cfg.copy()
        llm_cfg.setdefault("model_type", llm_cfg.get("model_type", "qwen3"))
        llm_cfg.setdefault("architectures", ["Qwen3ForCausalLM"])

        with open(os.path.join(target_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(llm_cfg, f, indent=2, ensure_ascii=False)

        _copy_tokenizer_artifacts(source_dir, target_dir)
        _copy_if_exists(os.path.join(source_dir, "generation_config.json"), os.path.join(target_dir, "generation_config.json"))

        weight_file, _ = _collect_language_only_state(source_dir, target_dir)

        with open(flag_path, "w", encoding="utf-8") as f:
            json.dump({"config_sha": cfg_sha, "source": source_dir, "weight_file": weight_file}, f, indent=2, ensure_ascii=False)

        return target_dir


    def materialize_meta_parameters(q_model, context=""):
        if not hasattr(q_model, "model"):
            return
        meta_params = []
        for name, param in q_model.model.named_parameters():
            if getattr(param, "is_meta", False):
                meta_params.append(name)
        if not meta_params:
            return
        if context:
            print(f"[WARN] {context}: 检测到 {len(meta_params)} 个 meta 参数, 正在 materialize...")
        else:
            print(f"[WARN] 检测到 {len(meta_params)} 个 meta 参数, 正在 materialize...")
        seen = set()
        for param_name in meta_params:
            module_name = param_name.rsplit('.', 1)[0] if '.' in param_name else param_name
            if module_name in seen:
                continue
            seen.add(module_name)
            module, _ = get_module_by_name_prefix(q_model.model, module_name)
            if module is None:
                continue
            try:
                q_model.shell_module_materialize(module, q_model.quantize_config.device)
            except Exception as exc:
                print(f"[WARN] materialize 模块 {module_name} 失败: {exc}")


    ensure_internvl_chat_supported()
except Exception as e:
    print("缺少依赖或导入失败, 请先安装必要包 (gptqmodel, transformers, datasets, torch, numpy). ")
    print("示例安装命令:")
    print("  pip install gptqmodel transformers datasets torch numpy")
    raise

if hasattr(torch, "compile"):
    torch.compile = lambda *args, **kwargs: args[0]

if hasattr(torch, "_inductor") and hasattr(torch._inductor, "config"):
    try:
        torch._inductor.config.triton = False
    except Exception:
        pass

def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    if (attention_mask[:, -1].sum().item() == attention_mask.shape[0]):
        return last_hidden_states[:, -1]
    else:
        seq_lens = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device)
        return last_hidden_states[batch_idx, seq_lens]

def compute_embeddings_with_model(model, tokenizer, texts: List[str], device, max_length=1024, batch_size=8):
    if hasattr(model, "encode") and callable(getattr(model, "encode")):
        embs = model.encode(texts)
        return np.array(embs)

    all_embs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i: i + batch_size]
            batch = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            if hasattr(out, "last_hidden_state"):
                last_hidden = out.last_hidden_state
            elif isinstance(out, (tuple, list)) and len(out) > 0:
                last_hidden = out[0]
            else:
                raise RuntimeError("无法从模型输出中获取 last_hidden_state")
            pooled = last_token_pool(last_hidden, batch["attention_mask"])
            pooled = F.normalize(pooled, p=2, dim=1).cpu().numpy()
            all_embs.append(pooled)
    return np.concatenate(all_embs, axis=0)

def cosine_similarities(a: np.ndarray, b: np.ndarray):
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    sims = (a_n * b_n).sum(axis=1)
    return sims

def safe_cuda_empty_cache():
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="InternVL3_5-1B")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--bits", type=int, default=8, choices=[4, 8])
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--calib_size", type=int, default=512)
    parser.add_argument("--eval_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=1024)
    args = parser.parse_args()

    model_id = args.model_id
    prepared_model_id = ensure_internvl_chat_llm_submodel(model_id)
    if prepared_model_id != model_id:
        print(f"[INFO] 已生成 Qwen3 语言子模型目录: {prepared_model_id}")
        model_id = prepared_model_id
    out_dir = args.out_dir or f"{model_id.replace('/', '_')}-gptq-int{args.bits}"
    os.makedirs(out_dir, exist_ok=True)
    done_flag = os.path.join(out_dir, "quantize_done.txt")

    # select device
    if args.device:
        device = torch.device(args.device)
    else:
        try:
            device = torch.device(get_best_device())
        except Exception:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] target device = {device}")

    # load tokenizer
    print("[STEP] 加载 tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left", use_fast=True, trust_remote_code=True)

    # calibration data
    print(f"[STEP] 准备 calibration 数据 (size={args.calib_size}) ...")
    try:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = [t for t in ds["text"] if isinstance(t, str) and len(t.strip()) > 10]
        calib_texts = texts[:args.calib_size]
    except Exception:
        calib_texts = ["Deep learning is transforming AI.", "Qwen is a family of embedding models."] * args.calib_size
        calib_texts = calib_texts[:args.calib_size]
    calib_tokenized = [tokenizer(t, truncation=True, max_length=args.max_length) for t in calib_texts]

    # 量化（若未完成）
    if not os.path.exists(done_flag):
        print("[STEP] 构建 QuantizeConfig ...")
        qc_sig = inspect.signature(QuantizeConfig)
        qc_kwargs = {}
        if 'bits' in qc_sig.parameters:
            qc_kwargs['bits'] = args.bits
        if 'group_size' in qc_sig.parameters:
            qc_kwargs['group_size'] = args.group_size
        if 'calibration_enable_gpu_cache' in qc_sig.parameters:
            qc_kwargs['calibration_enable_gpu_cache'] = False
        # if 'device' in qc_sig.parameters: # 配置下面这两组参数都会导致程序出错
        #     qc_kwargs['device'] = str(device)
        # if 'offload_to_disk' in qc_sig.parameters:
        #     qc_kwargs['offload_to_disk'] = False
        if 'desc_act' in qc_sig.parameters:
            qc_kwargs['desc_act'] = False
        if 'use_accelerate' in qc_sig.parameters:
            qc_kwargs['use_accelerate'] = False
        if 'static_groups' in qc_sig.parameters:
            qc_kwargs['static_groups'] = False
        if 'sym' in qc_sig.parameters: # 启用对称量化
            qc_kwargs['sym'] = True
        if 'true_sequential' in qc_sig.parameters: # "true_sequential" 选项在量化配置中表示是否采用真正的顺序量化方法. 顺序量化是一种逐层量化的方法, 其中每一层的量化都依赖于前一层的量化结果. 启用这个选项可以确保量化过程更加严格地遵循模型的层次结构, 从而可能提高量化后模型的性能和准确性.  如果禁用这个选项, 量化过程可能会更加松散, 可能会导致量化后模型的性能下降. 默认情况下, 这个选项通常是启用的, 以确保量化过程的严谨性. 
            qc_kwargs['true_sequential'] = True
        if 'damp_percent' in qc_sig.parameters: # 启用对称量化
            qc_kwargs['damp_percent'] = 0.01 # 0.01 的阻尼系数有助于稳定量化过程,  防止过度拟合量化参数, 从而提升量化后模型的泛化能力.

        quant_config = QuantizeConfig(**qc_kwargs)

        # quant_config = QuantizeConfig( # Resolved: 前面的 config 存在问题
        #     bits=8,                  # 量化为 8-bit
        #     group_size=128,          # group size 128 依据模型的 config.json "head_dim": 128
        #     damp_percent=0.01,       # Dampening
        #     desc_act=False,          # 设为 False 可提升速度和兼容性
        #     static_groups=False,     # 不设置静态组
        #     sym=True,                # 对称量化
        #     true_sequential=True,    # 真正的顺序量化
        #     # 根据 gptqmodel 文档可能还有其他参数
        # )

        print("[STEP] 调用 GPTQModel.load ...")
        # model = GPTQModel.load(model_id, quant_config, trust_remote_code=True)
        model = GPTQModel.from_pretrained(model_id, quant_config, trust_remote_code=True, device_map="auto")

        safe_cuda_empty_cache()
        print("[STEP] 开始量化 ...")
        try:
            model.quantize(calib_tokenized, batch_size=args.batch_size)
            materialize_meta_parameters(model, "GPU save")
            model.save(out_dir)
            with open(done_flag, "w") as f:
                f.write("done\n")
            print("[INFO] GPU 量化完成")
        except Exception as e_gpu:
            print("[WARN] GPU 量化失败, 尝试 CPU:", repr(e_gpu))
            safe_cuda_empty_cache()
            if 'device' in qc_sig.parameters:
                qc_kwargs['device'] = 'cpu'
            quant_config_cpu = QuantizeConfig(**qc_kwargs)
            model_cpu = GPTQModel.load(model_id, quant_config_cpu, trust_remote_code=True)
            model_cpu.quantize(calib_tokenized, batch_size=1)
            materialize_meta_parameters(model_cpu, "CPU save")
            model_cpu.save(out_dir)
            with open(done_flag, "w") as f:
                f.write("done(cpu)\n")
            print("[INFO] CPU 量化完成")

    # 验证阶段
    print("[STEP] 加载参考模型 ...")
    dtype = torch.float16 if "cuda" in str(device) else None
    ref_model = AutoModel.from_pretrained(model_id, torch_dtype=dtype, trust_remote_code=True)
    ref_model.to(device)

    print("[STEP] 加载量化模型 ...")
    try:
        quant_model = AutoModel.from_pretrained(out_dir, trust_remote_code=True)
        quant_model.to(device)
    except Exception as e_auto:
        print("[WARN] AutoModel 无法加载量化模型, 尝试 GPTQModel.load:", repr(e_auto))
        quant_model = GPTQModel.load(out_dir)
        raise

    eval_texts = calib_texts[:args.eval_size]

    print("[STEP] 计算参考模型 embeddings ...")
    ref_embs = compute_embeddings_with_model(ref_model, tokenizer, eval_texts, device, args.max_length, batch_size=8)

    print("[STEP] 计算量化模型 embeddings ...")
    q_embs = compute_embeddings_with_model(quant_model, tokenizer, eval_texts, device, args.max_length, batch_size=8)

    sims = cosine_similarities(ref_embs, q_embs)
    summary = {
        "cosine_mean": float(np.mean(sims)),
        "cosine_std": float(np.std(sims)),
        "cosine_min": float(np.min(sims)),
        "cosine_max": float(np.max(sims)),
    }
    print("=== 验证结果 ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    results_path = os.path.join(out_dir, "quant_eval_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("[INFO] 结果保存到:", results_path)

if __name__ == "__main__":
    """
    CUDA_VISIBLE_DEVICES=0 python convert_to_gptq.py \
        --model_id InternVL3_5-1B \
        --out_dir ./InternVL3_5-1B_GPTQ_INT4 \
        --bits 4
    """
    main()
