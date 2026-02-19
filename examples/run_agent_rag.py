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


def make_llm_call_stub():
    def _call(prompt: str) -> str:
        # TODO: replace with real LLM call; for now echo a continue.
        # This keeps the loop running and showcases the plumbing.
        return "CONTINUE QUERY: " + prompt.split("QUESTION:")[-1].strip().split("\n")[0]

    return _call


def build_rag_model(device: str = "cpu"):
    retrieval_model = ColPaliRetrievalModel.from_pretrained(
        model_name_or_path=f"{LOCAL_MODEL_DIR}/colpaligemma-3b-pt-448-base",
        adapter_name_or_path=f"{LOCAL_MODEL_DIR}/colpali-v1.2",
        device=device,
    )
    rag_model = MultimodalRAGModel(retrieval_model=retrieval_model, vqa_model=None)
    return rag_model


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

    docid2embs = load_doc_embs(args.embeddings)

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

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

