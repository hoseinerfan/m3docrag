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
    return p.parse_args()


def main():
    args = parse_args()

    docid2embs = move_doc_embs(load_doc_embs(args.embeddings), args.device)

    rag_model = build_rag_model(device=args.device)

    result = run_agent_session(
        query=args.question,
        rag_model=rag_model,
        docid2embs=docid2embs,
        token2pageuid=None,
        all_token_embeddings=None,
        max_turns=args.max_turns,
        pages_per_turn=args.pages_per_turn,
        n_return_pages=args.n_return_pages,
        llm_call=make_llm_call_stub(),
    )

    print(json.dumps(to_jsonable(result), indent=2))


if __name__ == "__main__":
    main()
