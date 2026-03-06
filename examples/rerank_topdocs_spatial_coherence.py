#!/usr/bin/env python3
# Copyright 2024 Bloomberg Finance L.P.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

try:
    import jsonlines
except Exception:  # pragma: no cover
    jsonlines = None


_DEFAULT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
    "which",
    "who",
    "whom",
    "whose",
    "what",
    "where",
    "when",
    "why",
    "how",
    "among",
    "into",
    "about",
    "than",
}


@dataclass
class SpatialLayout:
    offset: int
    side: int
    spatial_token_count: int


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Rerank existing topdocs/top-pages results using MaxSim spatial coherence "
            "(no retrieval rerun)."
        )
    )
    p.add_argument("--topdocs-json", type=Path, required=False, help="Input *_topdocs.json file.")
    p.add_argument(
        "--retrieval-parquet",
        type=Path,
        required=False,
        help="Input retrieval parquet directory/file with page-level candidates.",
    )
    p.add_argument("--output-json", type=Path, required=False, help="Output reranked json.")
    p.add_argument("--retrieval-qid-col", type=str, default=None, help="Optional parquet qid column override.")
    p.add_argument("--retrieval-doc-col", type=str, default=None, help="Optional parquet doc_id column override.")
    p.add_argument("--retrieval-page-col", type=str, default=None, help="Optional parquet page_idx column override.")
    p.add_argument(
        "--retrieval-score-col",
        type=str,
        default=None,
        help="Optional parquet score column override.",
    )
    p.add_argument(
        "--retrieval-rank-col",
        type=str,
        default=None,
        help="Optional parquet rank column override (used if score is absent).",
    )
    p.add_argument(
        "--mmqa-jsonl",
        type=Path,
        default=None,
        help="MMQA split jsonl used to map qid -> question.",
    )
    p.add_argument(
        "--qid",
        type=str,
        default=None,
        help="Optional: rerank only a single qid.",
    )
    p.add_argument(
        "--query",
        type=str,
        default=None,
        help="Optional query text for --qid; overrides MMQA lookup.",
    )
    p.add_argument(
        "--embedding-dir",
        type=Path,
        default=None,
        help="Directory with per-doc safetensors embeddings (<doc_id>.safetensors).",
    )
    p.add_argument(
        "--retrieval-model-name-or-path",
        type=str,
        default="colpali-v1.2-backbone",
        help="Retriever backbone path or model name under LOCAL_MODEL_DIR.",
    )
    p.add_argument(
        "--retrieval-adapter-model-name-or-path",
        type=str,
        default="colpali-v1.2",
        help="Retriever adapter path or model name under LOCAL_MODEL_DIR.",
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Retriever model dtype.",
    )
    p.add_argument(
        "--topk-candidates",
        type=int,
        default=1000,
        help="How many existing candidates to rerank per qid from top_pages.",
    )
    p.add_argument(
        "--save-top-k",
        type=int,
        default=1000,
        help="How many reranked pages to save per qid.",
    )
    p.add_argument(
        "--coherence-lambda",
        type=float,
        default=0.85,
        help="Final score = z(base) + lambda * z(coherence).",
    )
    p.add_argument(
        "--cluster-radius",
        type=float,
        default=2.5,
        help="Neighborhood radius (grid cells) for largest-cluster mass.",
    )
    p.add_argument(
        "--spatial-token-offset",
        type=int,
        default=None,
        help="Manual prefix-token offset before square patch grid. If omitted, inferred.",
    )
    p.add_argument(
        "--max-offset-search",
        type=int,
        default=16,
        help="Max offset searched when inferring square patch grid.",
    )
    p.add_argument(
        "--idf-json",
        type=Path,
        default=None,
        help="Optional token->idf json for informative-token weighting.",
    )
    p.add_argument(
        "--gold-doc-id",
        type=str,
        default=None,
        help="Optional gold doc id for qid-level before/after rank report.",
    )
    p.add_argument(
        "--gold-page-idx",
        type=int,
        default=None,
        help="Optional gold page idx for qid-level before/after rank report.",
    )
    p.add_argument(
        "--debug-qid-topn",
        type=int,
        default=10,
        help="Number of top pages to keep per-token debug for the selected --qid.",
    )
    p.add_argument(
        "--debug-qid-json",
        type=Path,
        default=None,
        help="Optional path to save per-token alignment debug for selected --qid.",
    )
    p.add_argument(
        "--self-test",
        action="store_true",
        help="Run synthetic smoke test (no model/data required).",
    )
    return p.parse_args()


def _load_topdocs(path: Path) -> dict[str, list[dict[str, Any]]]:
    with path.open() as f:
        obj = json.load(f)
    top_pages = obj.get("top_pages", {})
    if not isinstance(top_pages, dict):
        raise ValueError(f"Invalid top_pages in {path}")
    out: dict[str, list[dict[str, Any]]] = {}
    for qid, rows in top_pages.items():
        normalized = []
        for r in rows or []:
            d = str(r.get("doc_id"))
            p = int(r.get("page_idx"))
            s = float(r.get("score", 0.0))
            normalized.append({"doc_id": d, "page_idx": p, "score": s})
        out[str(qid)] = normalized
    return out


def _pick_column(cols: list[str], override: str | None, candidates: list[str], name: str) -> str:
    if override:
        if override in cols:
            return override
        raise ValueError(f"{name} override '{override}' not found. Available cols: {cols}")
    for c in candidates:
        if c in cols:
            return c
    raise ValueError(f"Could not infer {name}. Available cols: {cols}")


def _load_retrieval_parquet(
    parquet_path: Path,
    qid_filter: set[str] | None,
    qid_col_override: str | None,
    doc_col_override: str | None,
    page_col_override: str | None,
    score_col_override: str | None,
    rank_col_override: str | None,
) -> dict[str, list[dict[str, Any]]]:
    try:
        import pyarrow.dataset as ds
    except Exception as e:
        raise ImportError("pyarrow is required to load --retrieval-parquet") from e

    dataset = ds.dataset(str(parquet_path), format="parquet")
    cols = list(dataset.schema.names)

    qid_col = _pick_column(cols, qid_col_override, ["qid", "query_id", "question_id"], "qid column")
    doc_col = _pick_column(cols, doc_col_override, ["doc_id", "document_id"], "doc_id column")
    page_col = _pick_column(cols, page_col_override, ["page_idx", "page_id", "page"], "page_idx column")

    score_col = None
    if score_col_override:
        score_col = _pick_column(cols, score_col_override, [score_col_override], "score column")
    else:
        for c in ["score", "sim", "similarity", "maxsim_score"]:
            if c in cols:
                score_col = c
                break

    rank_col = None
    if rank_col_override:
        rank_col = _pick_column(cols, rank_col_override, [rank_col_override], "rank column")
    else:
        for c in ["rank", "retrieval_rank", "source_rank"]:
            if c in cols:
                rank_col = c
                break

    read_cols = [qid_col, doc_col, page_col]
    if score_col:
        read_cols.append(score_col)
    if rank_col:
        read_cols.append(rank_col)

    table = None
    if qid_filter:
        try:
            table = dataset.to_table(columns=read_cols, filter=ds.field(qid_col).isin(list(qid_filter)))
        except Exception:
            table = dataset.to_table(columns=read_cols)
    else:
        table = dataset.to_table(columns=read_cols)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in table.to_pylist():
        qid = row.get(qid_col)
        doc_id = row.get(doc_col)
        page_idx = row.get(page_col)
        if qid is None or doc_id is None or page_idx is None:
            continue
        qid = str(qid)
        if qid_filter and qid not in qid_filter:
            continue
        score = row.get(score_col) if score_col else None
        rank = row.get(rank_col) if rank_col else None
        if score is None:
            if rank is None:
                score = 0.0
            else:
                try:
                    score = -float(rank)
                except Exception:
                    score = 0.0
        grouped[qid].append(
            {
                "doc_id": str(doc_id),
                "page_idx": int(page_idx),
                "score": float(score),
                "_rank": None if rank is None else rank,
            }
        )

    out: dict[str, list[dict[str, Any]]] = {}
    for qid, rows in grouped.items():
        has_rank = any(r.get("_rank") is not None for r in rows)
        if has_rank:
            def rank_key(r: dict[str, Any]) -> tuple[float, float]:
                rv = r.get("_rank")
                try:
                    rr = float(rv)
                except Exception:
                    rr = float("inf")
                return (rr, -float(r["score"]))
            rows = sorted(rows, key=rank_key)
        else:
            rows = sorted(rows, key=lambda x: float(x["score"]), reverse=True)
        for r in rows:
            r.pop("_rank", None)
        out[qid] = rows
    return out


def _load_qid2query(mmqa_jsonl: Path | None) -> dict[str, str]:
    if mmqa_jsonl is None:
        return {}
    if jsonlines is None:
        raise ImportError("jsonlines is required to read MMQA jsonl. Please install jsonlines.")
    qmap: dict[str, str] = {}
    with jsonlines.open(mmqa_jsonl) as reader:
        for obj in reader:
            qid = obj.get("qid") or obj.get("query_id") or obj.get("id")
            q = obj.get("question")
            if qid is None or q is None:
                continue
            qmap[str(qid)] = str(q)
    return qmap


def _resolve_model_path(name_or_path: str) -> str:
    p = Path(name_or_path)
    if p.exists():
        return str(p)
    try:
        from m3docrag.utils.paths import LOCAL_MODEL_DIR

        alt = Path(LOCAL_MODEL_DIR) / name_or_path
        if alt.exists():
            return str(alt)
    except Exception:
        pass
    return name_or_path


def _resolve_dtype(dtype: str) -> torch.dtype:
    if dtype == "float16":
        return torch.float16
    if dtype == "float32":
        return torch.float32
    return torch.bfloat16


def _get_tokenizer(processor: Any) -> Any:
    for attr in ("tokenizer", "text_tokenizer"):
        if hasattr(processor, attr):
            return getattr(processor, attr)
    if hasattr(processor, "processor"):
        inner = getattr(processor, "processor")
        for attr in ("tokenizer", "text_tokenizer"):
            if hasattr(inner, attr):
                return getattr(inner, attr)
    raise RuntimeError("Tokenizer not found on retrieval processor.")


def _query_pieces(processor: Any, query: str) -> list[str]:
    tok = _get_tokenizer(processor)
    ids = None
    try:
        batch = processor.process_queries([query])
        ids = batch.get("input_ids")
    except Exception:
        ids = None
    if ids is None:
        enc = tok(query, add_special_tokens=True, return_tensors=None)
        ids = enc["input_ids"] if isinstance(enc, dict) else enc
    if hasattr(ids, "cpu"):
        ids = ids.cpu()
    if hasattr(ids, "numpy"):
        ids = ids.numpy()
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, (list, tuple)) and ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return tok.convert_ids_to_tokens(ids)


def _clean_piece(tok: str) -> str:
    return str(tok).lstrip("▁Ġ_").strip()


def _is_informative_piece(tok: str, stopwords: set[str]) -> bool:
    t = _clean_piece(tok)
    if not t:
        return False
    if t.startswith("<") and t.endswith(">"):
        return False
    if re.fullmatch(r"[\W_]+", t):
        return False
    low = t.lower()
    if low in stopwords:
        return False
    if len(low) <= 1 and not low.isdigit():
        return False
    return True


def _load_idf_map(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    with path.open() as f:
        obj = json.load(f)
    out: dict[str, float] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k is None:
                continue
            try:
                out[str(k).lower()] = float(v)
            except Exception:
                continue
    return out


def _idf_weight(tok: str, idf_map: dict[str, float], idf_max: float) -> float:
    if not idf_map:
        return 1.0
    low = _clean_piece(tok).lower()
    raw = float(idf_map.get(low, 0.0))
    if idf_max <= 0.0:
        return 1.0
    return 1.0 + raw / idf_max


def _infer_spatial_layout(num_tokens: int, explicit_offset: int | None, max_search: int) -> SpatialLayout:
    if explicit_offset is not None:
        n = max(0, num_tokens - explicit_offset)
        side = int(math.sqrt(n))
        if side * side != n:
            raise ValueError(
                f"spatial-token-offset={explicit_offset} leaves {n} tokens, not a perfect square."
            )
        return SpatialLayout(offset=explicit_offset, side=side, spatial_token_count=n)

    for off in range(max(0, max_search) + 1):
        n = num_tokens - off
        if n <= 0:
            continue
        side = int(math.sqrt(n))
        if side * side == n and side >= 8:
            return SpatialLayout(offset=off, side=side, spatial_token_count=n)

    side = int(math.sqrt(num_tokens))
    n = side * side
    return SpatialLayout(offset=0, side=side, spatial_token_count=n)


def _idx_to_xy(idx: int, layout: SpatialLayout) -> tuple[float, float] | None:
    local = idx - layout.offset
    if local < 0 or local >= layout.spatial_token_count:
        return None
    row = local // layout.side
    col = local % layout.side
    return float(col), float(row)


def _largest_cluster_mass(coords: list[tuple[float, float]], weights: list[float], radius: float) -> float:
    if not coords:
        return 0.0
    n = len(coords)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    r2 = radius * radius
    for i in range(n):
        xi, yi = coords[i]
        for j in range(i + 1, n):
            xj, yj = coords[j]
            d2 = (xi - xj) * (xi - xj) + (yi - yj) * (yi - yj)
            if d2 <= r2:
                union(i, j)

    mass = defaultdict(float)
    total = 0.0
    for i, w in enumerate(weights):
        root = find(i)
        wi = float(max(w, 0.0))
        mass[root] += wi
        total += wi
    if total <= 0.0:
        return 0.0
    return max(mass.values()) / total


def _weighted_entropy(ids: list[int], weights: list[float]) -> float:
    if not ids:
        return 1.0
    mass = defaultdict(float)
    total = 0.0
    for i, w in zip(ids, weights):
        wi = float(max(w, 0.0))
        mass[int(i)] += wi
        total += wi
    if total <= 0.0:
        return 1.0
    probs = [v / total for v in mass.values() if v > 0.0]
    if len(probs) <= 1:
        return 0.0
    ent = -sum(p * math.log(p + 1e-12) for p in probs)
    return float(ent / math.log(len(probs)))


def _phrase_groups(informative_positions: list[int]) -> list[list[int]]:
    if not informative_positions:
        return []
    groups: list[list[int]] = []
    cur = [informative_positions[0]]
    for p in informative_positions[1:]:
        if p - cur[-1] <= 1:
            cur.append(p)
        else:
            if len(cur) >= 2:
                groups.append(cur)
            cur = [p]
    if len(cur) >= 2:
        groups.append(cur)
    return groups


def _phrase_cohesion_score(
    pos2coord: dict[int, tuple[float, float]],
    pos2w: dict[int, float],
    phrase_groups: list[list[int]],
    side: int,
) -> float:
    if not phrase_groups:
        return 0.0
    diag = math.sqrt(2.0) * max(side, 1)
    total_group_weight = 0.0
    score_sum = 0.0
    for grp in phrase_groups:
        coords = []
        ws = []
        for p in grp:
            c = pos2coord.get(p)
            if c is None:
                continue
            w = float(max(pos2w.get(p, 0.0), 0.0))
            coords.append(c)
            ws.append(w)
        if len(coords) < 2:
            continue
        wsum = sum(ws)
        if wsum <= 0.0:
            continue
        cx = sum(c[0] * w for c, w in zip(coords, ws)) / wsum
        cy = sum(c[1] * w for c, w in zip(coords, ws)) / wsum
        d = sum(math.sqrt((c[0] - cx) ** 2 + (c[1] - cy) ** 2) * w for c, w in zip(coords, ws)) / wsum
        cohesion = max(0.0, 1.0 - d / max(diag, 1e-6))
        score_sum += cohesion * wsum
        total_group_weight += wsum
    if total_group_weight <= 0.0:
        return 0.0
    return score_sum / total_group_weight


def _zscore(xs: list[float]) -> list[float]:
    if not xs:
        return []
    mean = sum(xs) / len(xs)
    var = sum((x - mean) * (x - mean) for x in xs) / len(xs)
    std = math.sqrt(var)
    if std < 1e-12:
        return [0.0 for _ in xs]
    return [(x - mean) / std for x in xs]


class EmbeddingStore:
    def __init__(self, emb_dir: Path):
        self.emb_dir = emb_dir
        self._cache: dict[str, torch.Tensor] = {}

    def _load_doc(self, doc_id: str) -> torch.Tensor:
        import safetensors

        if doc_id in self._cache:
            return self._cache[doc_id]
        fp = self.emb_dir / f"{doc_id}.safetensors"
        if not fp.exists():
            raise FileNotFoundError(f"Missing embedding file: {fp}")
        with safetensors.safe_open(fp, framework="pt", device="cpu") as f:
            doc_embs = f.get_tensor("embeddings")
        self._cache[doc_id] = doc_embs
        return doc_embs

    def page_tokens(self, doc_id: str, page_idx: int) -> torch.Tensor:
        d = self._load_doc(doc_id)
        if page_idx < 0 or page_idx >= d.shape[0]:
            raise IndexError(f"Invalid page_idx={page_idx} for doc={doc_id} with n_pages={d.shape[0]}")
        return d[page_idx].float()


def _encode_query_tokens(retrieval: Any, query: str) -> torch.Tensor:
    q = retrieval.encode_queries([query], batch_size=1, use_tqdm=False, to_cpu=False)[0]
    if q.dim() == 3:
        q = q.squeeze(0)
    return q.float()


def _page_features(
    q_tokens: torch.Tensor,
    q_pieces: list[str],
    p_tokens: torch.Tensor,
    idf_map: dict[str, float],
    layout: SpatialLayout,
    cluster_radius: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    q = torch.nn.functional.normalize(q_tokens, p=2, dim=-1)
    p = torch.nn.functional.normalize(p_tokens, p=2, dim=-1)
    sim = q @ p.T  # [Lq, Lp]
    if sim.shape[1] >= 2:
        top2_vals, top2_idx = torch.topk(sim, k=2, dim=1)
        top1 = top2_vals[:, 0]
        top1_idx = top2_idx[:, 0]
        top2 = top2_vals[:, 1]
    else:
        top1 = sim[:, 0]
        top1_idx = torch.zeros_like(top1, dtype=torch.long)
        top2 = torch.zeros_like(top1)

    informative_positions = []
    idf_max = max([1.0] + [float(v) for v in idf_map.values()]) if idf_map else 1.0
    weights = []
    coords = []
    picked_ids = []
    margins = []
    pos2coord: dict[int, tuple[float, float]] = {}
    pos2w: dict[int, float] = {}
    alignments = []

    for i, tok in enumerate(q_pieces):
        if i >= top1.shape[0]:
            break
        if not _is_informative_piece(tok, _DEFAULT_STOPWORDS):
            continue
        informative_positions.append(i)
        s1 = float(top1[i].item())
        s2 = float(top2[i].item())
        idx = int(top1_idx[i].item())
        margin = max(0.0, s1 - s2)
        idfw = _idf_weight(tok, idf_map, idf_max)
        w = idfw * (max(s1, 0.0) + 1e-6)
        xy = _idx_to_xy(idx, layout)
        if xy is not None:
            coords.append(xy)
            weights.append(w)
            picked_ids.append(idx)
            margins.append(margin)
            pos2coord[i] = xy
            pos2w[i] = w
        alignments.append(
            {
                "query_pos": i,
                "query_token": _clean_piece(tok),
                "page_token_idx": idx,
                "sim": s1,
                "margin": margin,
                "xy": None if xy is None else [xy[0], xy[1]],
            }
        )

    if not informative_positions:
        return (
            {
                "cluster_mass": 0.0,
                "token_entropy": 1.0,
                "phrase_cohesion": 0.0,
                "margin_conf": 0.0,
                "spatial_coverage": 0.0,
                "coherence": 0.0,
            },
            {"alignments": alignments},
        )

    n_inf = len(informative_positions)
    coverage = len(coords) / max(n_inf, 1)

    if not coords:
        return (
            {
                "cluster_mass": 0.0,
                "token_entropy": 1.0,
                "phrase_cohesion": 0.0,
                "margin_conf": 0.0,
                "spatial_coverage": coverage,
                "coherence": 0.05 * coverage,
            },
            {"alignments": alignments},
        )

    if sum(weights) <= 0.0:
        weights = [1.0 for _ in weights]

    cluster_mass = _largest_cluster_mass(coords=coords, weights=weights, radius=cluster_radius)
    entropy = _weighted_entropy(picked_ids, weights)
    focus = 1.0 - entropy
    phrase_groups = _phrase_groups(informative_positions)
    phrase_cohesion = _phrase_cohesion_score(
        pos2coord=pos2coord,
        pos2w=pos2w,
        phrase_groups=phrase_groups,
        side=layout.side,
    )
    margin_conf = (sum(margins) / len(margins)) if margins else 0.0
    margin_conf = float(max(0.0, min(1.0, margin_conf)))

    coherence = (
        0.45 * cluster_mass
        + 0.25 * phrase_cohesion
        + 0.20 * focus
        + 0.10 * margin_conf
        + 0.05 * coverage
    )
    coherence = float(max(0.0, min(1.0, coherence)))

    feats = {
        "cluster_mass": float(cluster_mass),
        "token_entropy": float(entropy),
        "phrase_cohesion": float(phrase_cohesion),
        "margin_conf": float(margin_conf),
        "spatial_coverage": float(coverage),
        "coherence": float(coherence),
    }
    debug = {
        "alignments": alignments,
        "layout": {
            "offset": layout.offset,
            "side": layout.side,
            "spatial_token_count": layout.spatial_token_count,
        },
    }
    return feats, debug


def _gold_rank(rows: list[dict[str, Any]], gold_doc_id: str | None, gold_page_idx: int | None) -> int | None:
    if not gold_doc_id:
        return None
    for i, r in enumerate(rows, start=1):
        if str(r["doc_id"]) != str(gold_doc_id):
            continue
        if gold_page_idx is None or int(r["page_idx"]) == int(gold_page_idx):
            return i
    return None


def _run_self_test() -> int:
    torch.manual_seed(7)
    q = torch.randn(12, 16)
    q = torch.nn.functional.normalize(q, p=2, dim=-1)
    pieces = [f"tok{i}" for i in range(12)]
    idf = {}

    layout = SpatialLayout(offset=6, side=32, spatial_token_count=1024)
    total_tokens = layout.offset + layout.spatial_token_count

    def mk_page(kind: str) -> torch.Tensor:
        p = torch.randn(total_tokens, 16)
        p = torch.nn.functional.normalize(p, p=2, dim=-1)
        if kind == "coherent":
            hot = [6 + 32 * 6 + 6, 6 + 32 * 6 + 7, 6 + 32 * 22 + 20, 6 + 32 * 22 + 21]
            src = [0, 1, 2, 3]
            for t, qi in zip(hot, src):
                p[t] = q[qi]
        else:
            hot = [6 + 32 * 2 + 2, 6 + 32 * 11 + 25, 6 + 32 * 24 + 3, 6 + 32 * 29 + 28]
            src = [0, 1, 2, 3]
            for t, qi in zip(hot, src):
                p[t] = q[qi]
        return p

    cands = [
        {"doc_id": "wrong_doc", "page_idx": 0, "score": 1.00, "tokens": mk_page("scattered")},
        {"doc_id": "gold_doc", "page_idx": 0, "score": 0.95, "tokens": mk_page("coherent")},
        {"doc_id": "wrong_doc2", "page_idx": 0, "score": 0.90, "tokens": mk_page("scattered")},
    ]

    rows = []
    for c in cands:
        feats, _ = _page_features(
            q_tokens=q,
            q_pieces=pieces,
            p_tokens=c["tokens"],
            idf_map=idf,
            layout=layout,
            cluster_radius=2.5,
        )
        rows.append(
            {
                "doc_id": c["doc_id"],
                "page_idx": c["page_idx"],
                "base_score": c["score"],
                "coherence": feats["coherence"],
            }
        )

    bz = _zscore([r["base_score"] for r in rows])
    cz = _zscore([r["coherence"] for r in rows])
    for i, r in enumerate(rows):
        r["final_score"] = bz[i] + 0.85 * cz[i]
    reranked = sorted(rows, key=lambda x: x["final_score"], reverse=True)

    before = _gold_rank(rows, "gold_doc", 0)
    after = _gold_rank(reranked, "gold_doc", 0)

    print("[self-test] before rank:", before)
    print("[self-test] after rank :", after)
    print("[self-test] top-3 after:")
    for i, r in enumerate(reranked, start=1):
        print(
            f"  {i}. {r['doc_id']} page={r['page_idx']} "
            f"base={r['base_score']:.4f} coh={r['coherence']:.4f} final={r['final_score']:.4f}"
        )
    if after is None or after > before:
        print("[self-test] FAILED: coherence did not improve gold rank.")
        return 1
    print("[self-test] OK")
    return 0


def main() -> int:
    args = _parse_args()
    if args.self_test:
        return _run_self_test()

    if args.output_json is None or args.embedding_dir is None:
        raise ValueError(
            "--output-json and --embedding-dir are required (unless --self-test)."
        )
    if args.topdocs_json is None and args.retrieval_parquet is None:
        raise ValueError("Provide either --topdocs-json or --retrieval-parquet.")
    if args.topdocs_json is not None and args.retrieval_parquet is not None:
        raise ValueError("Use only one input source: --topdocs-json OR --retrieval-parquet.")

    qid_filter = {args.qid} if args.qid else None
    if args.topdocs_json is not None:
        qid2pages = _load_topdocs(args.topdocs_json)
    else:
        qid2pages = _load_retrieval_parquet(
            parquet_path=args.retrieval_parquet,
            qid_filter=qid_filter,
            qid_col_override=args.retrieval_qid_col,
            doc_col_override=args.retrieval_doc_col,
            page_col_override=args.retrieval_page_col,
            score_col_override=args.retrieval_score_col,
            rank_col_override=args.retrieval_rank_col,
        )
    qid2query = _load_qid2query(args.mmqa_jsonl)
    idf_map = _load_idf_map(args.idf_json)

    if args.qid is not None:
        if args.qid not in qid2pages:
            raise KeyError(f"qid={args.qid} not found in topdocs file.")
        qids = [args.qid]
    else:
        qids = sorted(qid2pages.keys())

    if not qids:
        raise ValueError("No qids available to rerank.")

    # Lazy import so --self-test remains lightweight.
    from m3docrag.retrieval.colpali import ColPaliRetrievalModel

    retrieval_model = ColPaliRetrievalModel(
        backbone_name_or_path=_resolve_model_path(args.retrieval_model_name_or_path),
        adapter_name_or_path=_resolve_model_path(args.retrieval_adapter_model_name_or_path),
        dtype=_resolve_dtype(args.dtype),
    )
    emb_store = EmbeddingStore(args.embedding_dir)

    reranked_top_pages: dict[str, list[dict[str, Any]]] = {}
    reranked_top_docs: dict[str, list[str]] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    debug_payload: dict[str, Any] = {}

    for qid in qids:
        query = args.query if (args.qid == qid and args.query is not None) else qid2query.get(qid)
        if query is None:
            raise ValueError(
                f"No query text for qid={qid}. Provide --query with --qid or --mmqa-jsonl."
            )

        cands = qid2pages[qid][: args.topk_candidates]
        if not cands:
            reranked_top_pages[qid] = []
            reranked_top_docs[qid] = []
            diagnostics[qid] = {"n_candidates": 0}
            continue

        q_tokens = _encode_query_tokens(retrieval_model, query)
        q_pieces = _query_pieces(retrieval_model.processor, query)
        if len(q_pieces) != q_tokens.shape[0]:
            if len(q_pieces) < q_tokens.shape[0]:
                q_pieces = list(q_pieces) + ["<pad>"] * (q_tokens.shape[0] - len(q_pieces))
            else:
                q_pieces = q_pieces[: q_tokens.shape[0]]

        rows: list[dict[str, Any]] = []
        qid_debug_pages = []

        for i, row in enumerate(cands):
            d = row["doc_id"]
            pidx = int(row["page_idx"])
            base_s = float(row["score"])
            page_tokens = emb_store.page_tokens(d, pidx)
            layout = _infer_spatial_layout(
                num_tokens=page_tokens.shape[0],
                explicit_offset=args.spatial_token_offset,
                max_search=args.max_offset_search,
            )
            feats, dbg = _page_features(
                q_tokens=q_tokens,
                q_pieces=q_pieces,
                p_tokens=page_tokens,
                idf_map=idf_map,
                layout=layout,
                cluster_radius=args.cluster_radius,
            )
            rec = {
                "doc_id": d,
                "page_idx": pidx,
                "base_score": base_s,
                "coherence": feats["coherence"],
                "features": feats,
            }
            rows.append(rec)

            if args.qid == qid and args.debug_qid_json is not None and i < args.debug_qid_topn:
                qid_debug_pages.append(
                    {
                        "doc_id": d,
                        "page_idx": pidx,
                        "base_score": base_s,
                        "coherence": feats["coherence"],
                        "features": feats,
                        "debug": dbg,
                    }
                )

        base_z = _zscore([r["base_score"] for r in rows])
        coh_z = _zscore([r["coherence"] for r in rows])
        for i, r in enumerate(rows):
            r["final_score"] = float(base_z[i] + args.coherence_lambda * coh_z[i])

        reranked = sorted(rows, key=lambda x: x["final_score"], reverse=True)
        save_rows = reranked[: args.save_top_k]
        reranked_top_pages[qid] = [
            {
                "doc_id": r["doc_id"],
                "page_idx": int(r["page_idx"]),
                "score": float(r["final_score"]),
                "base_score": float(r["base_score"]),
                "coherence": float(r["coherence"]),
                "features": r["features"],
            }
            for r in save_rows
        ]

        seen = set()
        docs = []
        for r in save_rows:
            d = str(r["doc_id"])
            if d not in seen:
                seen.add(d)
                docs.append(d)
        reranked_top_docs[qid] = docs

        before_rank = _gold_rank(cands, args.gold_doc_id if args.qid == qid else None, args.gold_page_idx)
        after_rank = _gold_rank(reranked, args.gold_doc_id if args.qid == qid else None, args.gold_page_idx)
        diagnostics[qid] = {
            "n_candidates": len(cands),
            "gold_before_rank": before_rank,
            "gold_after_rank": after_rank,
            "coherence_lambda": args.coherence_lambda,
        }

        if qid_debug_pages:
            debug_payload[qid] = {
                "qid": qid,
                "query": query,
                "top_debug_pages": qid_debug_pages,
                "gold_doc_id": args.gold_doc_id if args.qid == qid else None,
                "gold_page_idx": args.gold_page_idx if args.qid == qid else None,
                "gold_before_rank": before_rank,
                "gold_after_rank": after_rank,
            }

    out = {
        "meta": {
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "method": "maxsim_spatial_coherence_rerank",
            "source_topdocs_json": None if args.topdocs_json is None else str(args.topdocs_json),
            "source_retrieval_parquet": None if args.retrieval_parquet is None else str(args.retrieval_parquet),
            "topk_candidates": args.topk_candidates,
            "save_top_k": args.save_top_k,
            "coherence_lambda": args.coherence_lambda,
            "cluster_radius": args.cluster_radius,
            "spatial_token_offset": args.spatial_token_offset,
            "max_offset_search": args.max_offset_search,
            "idf_json": None if args.idf_json is None else str(args.idf_json),
            "retrieval_model_name_or_path": args.retrieval_model_name_or_path,
            "retrieval_adapter_model_name_or_path": args.retrieval_adapter_model_name_or_path,
            "dtype": args.dtype,
        },
        "top_docs": reranked_top_docs,
        "top_pages": reranked_top_pages,
        "diagnostics": diagnostics,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w") as f:
        json.dump(out, f, indent=2)

    if args.debug_qid_json is not None and debug_payload:
        args.debug_qid_json.parent.mkdir(parents=True, exist_ok=True)
        with args.debug_qid_json.open("w") as f:
            json.dump(debug_payload, f, indent=2)

    if args.qid is not None:
        d = diagnostics.get(args.qid, {})
        print(
            f"[qid={args.qid}] candidates={d.get('n_candidates')} "
            f"gold_before={d.get('gold_before_rank')} gold_after={d.get('gold_after_rank')}"
        )
    print(f"Saved reranked topdocs: {args.output_json}")
    if args.debug_qid_json is not None and debug_payload:
        print(f"Saved qid debug: {args.debug_qid_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
