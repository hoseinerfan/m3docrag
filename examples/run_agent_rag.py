"""Minimal agent wrapper over M3DocRAG retrieval + VQA.

Usage (example):
    conda run -p ./.conda/m3docrag python examples/run_agent_rag.py \
        --question "What is the main contribution?" \
        --embeddings /path/to/docid2embs.pt \
        --max-turns 3 --pages-per-turn 3

Note: this is a CPU-friendly scaffold; wire in your own llm_call that points to
an LLM endpoint (OpenAI, vLLM, or local) returning a short text reply.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Optional
import warnings

import torch

from m3docrag.agent import run_agent_session
from m3docrag.rag import MultimodalRAGModel
from m3docrag.retrieval import ColPaliRetrievalModel
from m3docrag.utils.paths import LOCAL_MODEL_DIR


def load_doc_embs(path: Path):
    obj = torch.load(path, map_location="cpu")
    return obj


def _candidate_key_variants(doc_id: str, page_idx: int) -> list[str]:
    return [
        f"{doc_id}_page{page_idx}",
        f"{doc_id}#p{page_idx}",
        f"{doc_id}:{page_idx}",
        f"{doc_id}/{page_idx}",
    ]


def _extract_context_text(obj) -> Optional[str]:
    if obj is None:
        return None
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        for key in ("summary", "text", "snippet", "page_summary", "content"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def load_context_map(path: Path) -> dict[str, str]:
    """Load page context snippets from JSON or JSONL.

    Accepted formats:
    - JSON object: {"<doc_id>_page<idx>": "...", ...}
    - JSON list / JSONL rows with keys: doc_id, page_idx/page/page_id and summary/text/snippet
    """

    if not path.exists():
        raise FileNotFoundError(f"Context file not found: {path}")

    suffixes = {s.lower() for s in path.suffixes}
    entries = []
    if ".jsonl" in suffixes:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entries.append(json.loads(line))
    else:
        with path.open() as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            # Already keyed mapping, but values may be nested dicts.
            out = {}
            for key, value in loaded.items():
                text = _extract_context_text(value)
                if text:
                    out[str(key)] = text
            return out
        if isinstance(loaded, list):
            entries = loaded
        else:
            raise ValueError(f"Unsupported context JSON payload type: {type(loaded)}")

    out: dict[str, str] = {}
    for row in entries:
        if not isinstance(row, dict):
            continue
        doc_id = row.get("doc_id") or row.get("document_id")
        page_idx = row.get("page_idx", row.get("page", row.get("page_id")))
        text = _extract_context_text(row)
        if doc_id is None or page_idx is None or not text:
            continue
        try:
            page_idx = int(page_idx)
        except Exception:
            continue
        out[f"{doc_id}_page{page_idx}"] = text
    return out


def move_doc_embs(docid2embs, device: str):
    if device == "cpu":
        return docid2embs
    # Smoke tests use a small embedding bundle, so moving it to the GPU is acceptable.
    return {doc_id: embs.to(device) for doc_id, embs in docid2embs.items()}


def configure_warning_filters():
    # Suppress known noisy warnings from transformers/model wrappers during smoke runs.
    patterns = [
        r"`config\.hidden_act` is ignored",
        r"Gemma's activation function will be set to `gelu_pytorch_tanh`",
        r"You are passing both `text` and `images` to `PaliGemmaProcessor`",
        r"`Qwen2VLRotaryEmbedding` can now be fully parameterized",
    ]
    for pattern in patterns:
        warnings.filterwarnings("ignore", message=pattern)


def _sanitize_generation_config(model):
    cfg = getattr(model, "generation_config", None)
    if cfg is None:
        return
    # Qwen checkpoints may ship sampling params that trigger warnings in greedy mode.
    for attr in ("temperature", "top_p", "top_k"):
        if hasattr(cfg, attr):
            setattr(cfg, attr, None)


def make_llm_call_stub():
    def _call(prompt: str) -> str:
        # TODO: replace with real LLM call; for now echo a continue.
        # This keeps the loop running and showcases the plumbing.
        return "CONTINUE QUERY: " + prompt.split("QUESTION:")[-1].strip().split("\n")[0]

    return _call


def _normalize_policy_output(raw: str, prompt: str) -> str:
    if raw is None:
        raw = ""
    line = raw.strip().splitlines()[0].strip() if raw.strip() else ""
    lower = line.lower()
    if (
        lower.startswith("answer:")
        or lower.startswith("continue query:")
        or lower.startswith("unanswerable")
    ):
        return line
    question = prompt.split("QUESTION:")[-1].strip().split("\n")[0]
    return f"CONTINUE QUERY: {question}"


def _resolve_policy_model_path(model_name_or_path: str) -> str:
    p = Path(model_name_or_path)
    if p.exists():
        resolved = p
    else:
        resolved = Path(LOCAL_MODEL_DIR) / model_name_or_path
    if not resolved.exists():
        raise FileNotFoundError(
            f"Policy model path does not exist: {resolved}. "
            "Pass a full local path or a folder name under LOCAL_MODEL_DIR."
        )
    return str(resolved)


def make_llm_call_local_hf(model_name_or_path: str, device: str = "cuda"):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    resolved = _resolve_policy_model_path(model_name_or_path)
    config = AutoConfig.from_pretrained(resolved, trust_remote_code=True)

    if device.startswith("cuda"):
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    system_prompt = (
        "Return exactly one line in one of these formats only: "
        "ANSWER: <text> OR CONTINUE QUERY: <text> OR UNANSWERABLE: <reason>."
    )

    if getattr(config, "model_type", "") == "qwen2_vl":
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        processor = AutoProcessor.from_pretrained(resolved, trust_remote_code=True)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            resolved,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).eval()
        _sanitize_generation_config(model)
        if device != "cpu":
            model = model.to(device)

        def _call(prompt: str) -> str:
            if hasattr(processor, "apply_chat_template"):
                text = processor.apply_chat_template(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                text = f"{system_prompt}\n\n{prompt}"

            inputs = processor(
                text=[text],
                padding=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=False,
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], outputs)
            ]
            generated = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            return _normalize_policy_output(generated, prompt)

        return _call

    tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        resolved,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).eval()
    _sanitize_generation_config(model)
    if device != "cpu":
        model = model.to(device)

    def _call(prompt: str) -> str:
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = f"{system_prompt}\n\n{prompt}"

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
            )

        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        return _normalize_policy_output(generated, prompt)

    return _call


def build_rag_model(device: str = "cpu"):
    retrieval_model = ColPaliRetrievalModel(
        backbone_name_or_path=f"{LOCAL_MODEL_DIR}/colpaligemma-3b-pt-448-base",
        adapter_name_or_path=f"{LOCAL_MODEL_DIR}/colpali-v1.2",
    )
    retrieval_model.model = retrieval_model.model.to(device)
    rag_model = MultimodalRAGModel(retrieval_model=retrieval_model, vqa_model=None)
    return rag_model


def to_jsonable(obj):
    if dataclasses.is_dataclass(obj):
        return {k: to_jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def make_candidate_context_fn(
    context_map: dict[str, str],
    max_chars: int = 400,
):
    def _lookup(doc_id: str, page_idx: int) -> Optional[str]:
        for key in _candidate_key_variants(doc_id, page_idx):
            value = context_map.get(key)
            if value:
                value = " ".join(value.split())
                if len(value) > max_chars:
                    value = value[: max_chars - 3].rstrip() + "..."
                return value
        return None

    return _lookup


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--question", required=True)
    p.add_argument("--embeddings", type=Path, required=True, help="Path to docid2embs.pt")
    p.add_argument("--max-turns", type=int, default=4)
    p.add_argument("--pages-per-turn", type=int, default=3)
    p.add_argument("--n-return-pages", type=int, default=6)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--context-file",
        type=Path,
        default=None,
        help="Optional JSON/JSONL file containing page summaries/snippets keyed by doc_id+page.",
    )
    p.add_argument(
        "--context-max-chars",
        type=int,
        default=400,
        help="Max characters per candidate summary/snippet inserted into the policy prompt.",
    )
    p.add_argument(
        "--policy-backend",
        default="stub",
        choices=["stub", "local-hf"],
        help="Policy backend for agent decisions.",
    )
    p.add_argument(
        "--policy-model",
        default=None,
        help="Local HF model path or model folder name under LOCAL_MODEL_DIR (used when --policy-backend local-hf).",
    )
    p.add_argument(
        "--policy-device",
        default=None,
        help="Device for policy model (defaults to --device). Use cpu if GPU memory is tight.",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the agent result JSON (also prints to stdout).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    configure_warning_filters()

    docid2embs = move_doc_embs(load_doc_embs(args.embeddings), args.device)

    rag_model = build_rag_model(device=args.device)
    candidate_context_fn = None
    if args.context_file is not None:
        context_map = load_context_map(args.context_file)
        candidate_context_fn = make_candidate_context_fn(
            context_map=context_map,
            max_chars=args.context_max_chars,
        )

    policy_device = args.policy_device or args.device
    if args.policy_backend == "stub":
        llm_call = make_llm_call_stub()
    elif args.policy_backend == "local-hf":
        if not args.policy_model:
            raise ValueError("--policy-model is required when --policy-backend local-hf")
        llm_call = make_llm_call_local_hf(args.policy_model, device=policy_device)
    else:
        raise ValueError(f"Unknown policy backend: {args.policy_backend}")

    result = run_agent_session(
        query=args.question,
        rag_model=rag_model,
        docid2embs=docid2embs,
        token2pageuid=None,
        all_token_embeddings=None,
        max_turns=args.max_turns,
        pages_per_turn=args.pages_per_turn,
        n_return_pages=args.n_return_pages,
        llm_call=llm_call,
        candidate_context_fn=candidate_context_fn,
    )

    payload = to_jsonable(result)
    rendered = json.dumps(payload, indent=2)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
