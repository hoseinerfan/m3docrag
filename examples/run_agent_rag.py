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

import torch

from m3docrag.agent import run_agent_session
from m3docrag.rag import MultimodalRAGModel
from m3docrag.retrieval import ColPaliRetrievalModel
from m3docrag.utils.paths import LOCAL_MODEL_DIR


def load_doc_embs(path: Path):
    obj = torch.load(path, map_location="cpu")
    return obj


def move_doc_embs(docid2embs, device: str):
    if device == "cpu":
        return docid2embs
    # Smoke tests use a small embedding bundle, so moving it to the GPU is acceptable.
    return {doc_id: embs.to(device) for doc_id, embs in docid2embs.items()}


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
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).eval()
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--question", required=True)
    p.add_argument("--embeddings", type=Path, required=True, help="Path to docid2embs.pt")
    p.add_argument("--max-turns", type=int, default=4)
    p.add_argument("--pages-per-turn", type=int, default=3)
    p.add_argument("--n-return-pages", type=int, default=6)
    p.add_argument("--device", default="cpu")
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
    return p.parse_args()


def main():
    args = parse_args()

    docid2embs = move_doc_embs(load_doc_embs(args.embeddings), args.device)

    rag_model = build_rag_model(device=args.device)

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
    )

    print(json.dumps(to_jsonable(result), indent=2))


if __name__ == "__main__":
    main()
