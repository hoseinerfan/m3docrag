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

import datetime
import hashlib
import json
import time
from pathlib import Path

import accelerate
import numpy as np
import pytz
import torch
import transformers
from accelerate import Accelerator
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from m3docrag.datasets.m3_docvqa import M3DocVQADataset, evaluate_prediction_file
from m3docrag.rag import MultimodalRAGModel
from m3docrag.retrieval import ColPaliRetrievalModel
from m3docrag.utils.args import parse_args
from m3docrag.utils.distributed import (
    barrier,
    global_rank,
    is_distributed,
    local_rank,
    log_runtime_info,
    print_gpu_stats,
    supports_flash_attention,
)
from m3docrag.utils.paths import (
    LOCAL_DATA_DIR,
    LOCAL_EMBEDDINGS_DIR,
    LOCAL_MODEL_DIR,
)
from m3docrag.utils.prompts import short_answer_template
from m3docrag.utils.tar import extract_tarfile
from m3docrag.vqa import VQAModel


def _get_doc_ids_signature(doc_ids: list[str]) -> str:
    hasher = hashlib.sha1()
    for doc_id in doc_ids:
        hasher.update(str(doc_id).encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def _get_retrieval_cache_paths(args):
    if args.retrieval_cache_dir:
        cache_dir = Path(args.retrieval_cache_dir)
    else:
        cache_dir = Path(LOCAL_EMBEDDINGS_DIR) / args.embedding_name / "_runtime_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_tag = f"{args.retrieval_model_type}_{args.split}"
    return {
        "dir": cache_dir,
        "meta": cache_dir / f"{cache_tag}_meta.json",
        "all_token_embeddings": cache_dir / f"{cache_tag}_all_token_embeddings.float32.npy",
        "token2pageidx": cache_dir / f"{cache_tag}_token2pageidx.int32.npy",
        "page_doc_ids": cache_dir / f"{cache_tag}_page_doc_ids.json",
        "page_indices": cache_dir / f"{cache_tag}_page_indices.int32.npy",
    }


def _load_retrieval_cache(cache_paths, expected_meta: dict):
    required_paths = [
        cache_paths["meta"],
        cache_paths["all_token_embeddings"],
        cache_paths["token2pageidx"],
        cache_paths["page_doc_ids"],
        cache_paths["page_indices"],
    ]
    if not all(path.exists() for path in required_paths):
        return None

    try:
        with open(cache_paths["meta"], "r") as f:
            cache_meta = json.load(f)
    except Exception:
        return None

    for key, value in expected_meta.items():
        if cache_meta.get(key) != value:
            return None

    try:
        all_token_embeddings = np.load(cache_paths["all_token_embeddings"], mmap_mode="r")
        token2pageidx = np.load(cache_paths["token2pageidx"], mmap_mode="r")
        with open(cache_paths["page_doc_ids"], "r") as f:
            page_doc_ids = json.load(f)
        page_indices = np.load(cache_paths["page_indices"], mmap_mode="r")
    except Exception:
        return None

    n_pages = len(page_doc_ids)
    if n_pages == 0:
        return None
    if page_indices.shape[0] != n_pages:
        return None
    if token2pageidx.shape[0] != all_token_embeddings.shape[0]:
        return None

    token2pageuid = (token2pageidx, page_doc_ids, page_indices)
    return all_token_embeddings, token2pageuid


def _save_retrieval_cache(
    cache_paths,
    cache_meta: dict,
    all_token_embeddings,
    token2pageidx,
    page_doc_ids,
    page_indices,
):
    np.save(cache_paths["all_token_embeddings"], all_token_embeddings)
    np.save(cache_paths["token2pageidx"], token2pageidx)
    with open(cache_paths["page_doc_ids"], "w") as f:
        json.dump(page_doc_ids, f)
    np.save(cache_paths["page_indices"], page_indices)
    with open(cache_paths["meta"], "w") as f:
        json.dump(cache_meta, f, indent=2)


def run_model(
    rag_model: MultimodalRAGModel,
    datum,
    dataset: M3DocVQADataset,
    docid2embs: dict[str, torch.Tensor],
    docid2lens=None,
    index=None,
    token2pageuid=None,
    all_token_embeddings=None,
    n_return_pages=1,
    args=None,
):
    # if type(datum['num_pages']) == list:
    batch = datum
    datum = {}
    for k, v in batch.items():
        datum[k] = v[0]

    query = datum["question"]

    out_dict = {}

    start = time.perf_counter()

    # Stage 1: Page retrieval
    # [(doc_id, page_idx, scores)...]
    top_n_page_retrieval_results = rag_model.retrieve_pages_from_docs(
        query=query,
        docid2embs=docid2embs,
        docid2lens=docid2lens,
        index=index,
        token2pageuid=token2pageuid,
        all_token_embeddings=all_token_embeddings,
        n_return_pages=n_return_pages,
        show_progress=True,
    )
    logger.info(top_n_page_retrieval_results)
    out_dict["page_retrieval_results"] = top_n_page_retrieval_results

    end = time.perf_counter()

    time_retrieval = end - start
    logger.info(f"time_retrieval: {time_retrieval}")

    start = time.perf_counter()

    if args.retrieval_only:
        pred_answer = ""
        out_dict["pred_answer"] = pred_answer
    else:
        # Stage 2: QA on the retrived page
        # Obtain images from the page retrieval results
        images = []
        for doc_id, page_idx, scores in top_n_page_retrieval_results:
            page_images = dataset.get_images_from_doc_id(doc_id)
            page_image = page_images[page_idx]
            images += [page_image]

        logger.info(len(images))

        # Run VQA
        if "florence" in args.model_name_or_path.lower():
            text_input = query
        else:
            text_input = short_answer_template.substitute({"question": query})
        pred_answer = rag_model.run_vqa(images=images, question=text_input)

        assert isinstance(pred_answer, str)
        out_dict["pred_answer"] = pred_answer

    end = time.perf_counter()

    time_qa = end - start
    logger.info(f"time_qa: {time_qa}")

    out_dict["time_retrieval"] = time_retrieval
    out_dict["time_qa"] = time_qa

    logger.info(query)
    logger.info(pred_answer)
    logger.info(datum["answers"])

    return out_dict


def evaluate(data_loader, rag_model, index=None, data_len=None, args=None, **kwargs):
    if data_len is not None:
        logger.info(f"eval on the first {data_len} items")

    # docid2embs = data_loader.dataset.load_all_embeddings()

    logger.info("Preparing doc indices")

    docid2embs = {}
    all_token_embeddings = None
    token2pageuid = None
    docid2lens = None

    if args.retrieval_model_type == "colpali":
        can_use_cache = bool(index is not None and args.retrieval_use_cache)
        cache_paths = None
        expected_cache_meta = None
        if can_use_cache:
            cache_paths = _get_retrieval_cache_paths(args)
            expected_cache_meta = {
                "embedding_name": args.embedding_name,
                "split": args.split,
                "retrieval_model_type": args.retrieval_model_type,
                "doc_ids_signature": _get_doc_ids_signature(data_loader.dataset.all_supporting_doc_ids),
                "num_docs": len(data_loader.dataset.all_supporting_doc_ids),
            }
            logger.info(f"Retrieval cache dir: {cache_paths['dir']}")

        if can_use_cache and not args.retrieval_rebuild_cache:
            cached_artifacts = _load_retrieval_cache(cache_paths, expected_cache_meta)
            if cached_artifacts is not None:
                all_token_embeddings, token2pageuid = cached_artifacts
                logger.info(
                    f"Loaded retrieval cache with {all_token_embeddings.shape[0]} token embeddings"
                )

        if all_token_embeddings is None or token2pageuid is None:
            docid2embs = data_loader.dataset.load_all_embeddings()

            #  reduce_embeddings(docid2embs=docid2embs)
            # docid2embs_page_reudced = reduce_embeddings(docid2embs, dim='page')
            # docid2embs_token_reudced = reduce_embeddings(docid2embs, dim='token')
            # docid2embs_page_token_reudced = reduce_embeddings(docid2embs, dim='page_token')

            if index is None:
                logger.info("Index not loaded; skipping flattened token artifact prep")
                all_token_embeddings = None
                token2pageuid = None
            else:
                all_token_embeddings_chunks = []
                token2pageidx_chunks = []
                page_doc_ids = []
                page_indices = []
                page_global_idx = 0

                for doc_id, doc_emb in tqdm(docid2embs.items(), total=len(docid2embs)):
                    # e.g., doc_emb - torch.Size([9, 1030, 128])
                    for page_id in range(len(doc_emb)):
                        page_emb = doc_emb[page_id].view(-1, 128)
                        all_token_embeddings_chunks.append(page_emb)
                        token2pageidx_chunks.append(
                            np.full(page_emb.shape[0], page_global_idx, dtype=np.int32)
                        )
                        page_doc_ids.append(doc_id)
                        page_indices.append(page_id)
                        page_global_idx += 1

                logger.info(len(all_token_embeddings_chunks))

                all_token_embeddings = torch.cat(all_token_embeddings_chunks, dim=0)
                all_token_embeddings = all_token_embeddings.float().numpy()
                token2pageidx = np.concatenate(token2pageidx_chunks, axis=0)
                page_indices = np.asarray(page_indices, dtype=np.int32)
                token2pageuid = (token2pageidx, page_doc_ids, page_indices)

                logger.info("Created flattened token embeddings / token2page mapping")

                if can_use_cache:
                    cache_meta = dict(expected_cache_meta)
                    cache_meta.update(
                        {
                            "num_pages": len(page_doc_ids),
                            "num_tokens": int(all_token_embeddings.shape[0]),
                            "emb_dim": int(all_token_embeddings.shape[1]),
                        }
                    )
                    _save_retrieval_cache(
                        cache_paths=cache_paths,
                        cache_meta=cache_meta,
                        all_token_embeddings=all_token_embeddings,
                        token2pageidx=token2pageidx,
                        page_doc_ids=page_doc_ids,
                        page_indices=page_indices,
                    )
                    logger.info("Saved retrieval cache artifacts")

                # Indexed retrieval path does not need full docid2embs after flattening.
                docid2embs = {}
    else:
        docid2embs = data_loader.dataset.load_all_embeddings()

    qid2result = {}

    total_time_retrieval = 0
    total_time_qa = 0

    for batch_idx, batch in enumerate(tqdm(data_loader)):
        bs = len(batch["question"])

        # Single batch
        assert bs == 1
        qid = batch["qid"][0]

        with torch.no_grad():
            outputs = run_model(
                rag_model,
                batch,
                dataset=data_loader.dataset,
                docid2embs=docid2embs,
                # docid2embs=docid2embs_page_reudced,
                # docid2embs=docid2embs_token_reudced,
                # docid2embs=docid2embs_page_token_reudced,
                index=index,
                token2pageuid=token2pageuid,
                all_token_embeddings=all_token_embeddings,
                n_return_pages=args.n_retrieval_pages,
                args=args,
            )

            pred_answer = outputs["pred_answer"]
            assert isinstance(pred_answer, str), type(pred_answer)

            total_time_qa += outputs["time_qa"]
            total_time_retrieval += outputs["time_retrieval"]

        qid2result[qid] = outputs

    logger.info(total_time_qa)
    logger.info(total_time_retrieval)

    avg_time_qa = total_time_qa / len(data_loader)
    avg_time_retrieval = total_time_retrieval / len(data_loader)
    logger.info(avg_time_qa)
    logger.info(avg_time_retrieval)

    return qid2result


def main():
    args = parse_args()

    logger.info(torch.__version__)
    logger.info(transformers.__version__)
    logger.info(accelerate.__version__)

    log_runtime_info()
    print_gpu_stats()

    accelerator = Accelerator()

    if not is_distributed() or global_rank() == 0:
        logger.info(f"Process {global_rank()}:{local_rank()} - args {args}")

    local_data_dir = Path(LOCAL_DATA_DIR) / args.data_name
    local_embedding_dir = Path(LOCAL_EMBEDDINGS_DIR) / args.embedding_name
    local_model_dir = Path(LOCAL_MODEL_DIR) / args.model_name_or_path
    local_retrieval_model_dir = (
        Path(LOCAL_MODEL_DIR) / args.retrieval_model_name_or_path
    )
    local_retrieval_adapter_model_dir = (
        Path(LOCAL_MODEL_DIR) / args.retrieval_adapter_model_name_or_path
    )
    # local_ft_model_dir = Path(LOCAL_MODEL_DIR) / args.ft_model_name_or_path

    if args.faiss_index_dir:
        local_index_dir = Path(args.faiss_index_dir)
    else:
        local_index_dir = (
            Path(LOCAL_EMBEDDINGS_DIR)
            / f"{args.embedding_name}_pageindex_{args.faiss_index_type}"
        )
    logger.info(f"Using FAISS index dir: {local_index_dir}")
    # local_answer_extraction_model_dir =  Path(LOCAL_MODEL_DIR) / args.answer_extraction_model_name_or_path

    if is_distributed():
        barrier()

    if not is_distributed() or global_rank() == 0:
        if not local_data_dir.exists():
            raise ValueError(f"Data directory {local_data_dir} does not exist")

        if not local_embedding_dir.exists():
            raise ValueError(
                f"Embedding directory {local_embedding_dir} does not exist"
            )

        if local_model_dir.exists() or args.retrieval_only:
            logger.info("Model exists - pass")
        else:
            raise ValueError(
                f"Model directory {local_model_dir} does not exist"
            )

        if args.use_retrieval:
            if not local_retrieval_model_dir.exists():
                raise ValueError(
                    f"Retrieval model directory {local_retrieval_model_dir} does not exist"
                )

            if not local_retrieval_adapter_model_dir.exists():
                raise ValueError(
                    f"Retrieval adapter model directory {local_retrieval_adapter_model_dir} does not exist"
                )

        if not local_index_dir.exists():
            raise ValueError(
                f"Index directory {local_index_dir} does not exist"
            )

    if is_distributed():
        barrier()

    # Create Retrieval Model (Step 1)
    assert args.use_retrieval
    if args.retrieval_model_type == "colpali":
        colpali_model = ColPaliRetrievalModel(
            backbone_name_or_path=local_retrieval_model_dir,
            adapter_name_or_path=local_retrieval_adapter_model_dir,
        )
        retrieval_model = colpali_model

    logger.info(f"loaded Retrieval model -: {local_retrieval_model_dir}")

    # Create QA / VQA Model (Step 2)

    if args.retrieval_only:
        rag_model = MultimodalRAGModel(retrieval_model=retrieval_model, vqa_model=None)
        logger.info("skipping QA model")

    else:
        if "florence" in args.model_name_or_path.lower():
            model_type = "florence2"
        elif "idefics2" in args.model_name_or_path.lower():
            model_type = "idefics2"
        elif "idefics3" in args.model_name_or_path.lower():
            model_type = "idefics3"
        elif "internvl2" in args.model_name_or_path.lower():
            model_type = "internvl2"
        elif "qwen2" in args.model_name_or_path.lower():
            model_type = "qwen2"
        else:
            raise KeyError(f"model type unknown for: {args.model_name_or_path}")

        use_flash_attn = True
        attn_implementation = "flash_attention_2"
        if not supports_flash_attention():
            use_flash_attn = False
            attn_implementation = "eager"

        try:
            vqa_model = VQAModel(
                model_name_or_path=local_model_dir,
                model_type=model_type,
                bits=args.bits,
                use_flash_attn=use_flash_attn,
                attn_implementation=attn_implementation,
            )
        except ImportError as e:
            # Some environments have compatible GPUs but no flash_attn package.
            # Fall back to eager attention automatically for usability.
            if use_flash_attn and "flash_attn" in str(e).lower():
                logger.warning(
                    "FlashAttention requested but unavailable; retrying with eager attention."
                )
                use_flash_attn = False
                attn_implementation = "eager"
                vqa_model = VQAModel(
                    model_name_or_path=local_model_dir,
                    model_type=model_type,
                    bits=args.bits,
                    use_flash_attn=use_flash_attn,
                    attn_implementation=attn_implementation,
                )
            else:
                raise

        logger.info(f"loaded VQA model - {model_type}: {local_model_dir}")

        rag_model = MultimodalRAGModel(
            retrieval_model=retrieval_model, vqa_model=vqa_model
        )

    logger.info("Created RAG model")

    dataset = M3DocVQADataset(args=args)
    logger.info("loaded dataset")

    index = None
    if local_index_dir.exists():
        logger.info("Loading faiss index")
        import faiss

        index = faiss.read_index(str(local_index_dir / "index.bin"))
        try:
            ivf_index = faiss.extract_index_ivf(index)
            ivf_index.nprobe = int(args.faiss_nprobe)
            logger.info(f"Configured FAISS IVF nprobe={ivf_index.nprobe}")
        except Exception:
            logger.info("FAISS index is non-IVF; faiss_nprobe ignored")
        logger.info("Loading faiss index -- done")

    def list_collate_fn(batch):
        batch = {
            k: [dic[k] for dic in batch] for k in batch[0]
        }  # List of dictionaries to dict of lists.
        return batch

    data_loader = DataLoader(
        dataset, batch_size=1, shuffle=False, collate_fn=list_collate_fn
    )

    if args.retrieval_only:
        retrieval_model.model, data_loader = accelerator.prepare(
            retrieval_model.model, data_loader
        )
    else:
        retrieval_model.model, data_loader, vqa_model.model = accelerator.prepare(
            retrieval_model.model, data_loader, vqa_model.model
        )

    eval_out = evaluate(
        data_loader=data_loader,
        rag_model=rag_model,
        index=index,
        data_len=args.data_len,
        args=args,
    )

    samples = eval_out

    EST = pytz.timezone("US/Eastern")
    experiment_date = (
        datetime.datetime.now().astimezone(EST).strftime("%Y-%m-%d_%H-%M-%S")
    )

    save_dir = Path(args.output_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    logger.info("Results will be saved at:", save_dir)

    if args.retrieval_model_type == "colpali":
        ret_name = args.retrieval_adapter_model_name_or_path
    else:
        ret_name = args.retrieval_model_name_or_path

    index_tag = Path(args.faiss_index_dir).name if args.faiss_index_dir else args.faiss_index_type
    if args.retrieval_only:
        pred_save_fname = f"{ret_name}_{index_tag}_ret{args.n_retrieval_pages}_{experiment_date}.json"
    else:
        pred_save_fname = f"{ret_name}_{index_tag}_ret{args.n_retrieval_pages}_{args.model_name_or_path}_{experiment_date}.json"
    results_file = save_dir / pred_save_fname
    with open(results_file, "w") as f:
        json.dump(samples, f, indent=4)

    logger.info(f"Prediction results saved at: {results_file}")

    if args.skip_eval:
        logger.info("Skipping evaluation because --skip_eval=True")
    else:
        # Evaluation
        all_eval_scores = evaluate_prediction_file(
            samples,
            dataset.mmqa_data_path,  # '/job/datasets/m3-docvqa/MMQA_dev.jsonl'
        )
        if args.retrieval_only:
            eval_save_fname = f"{ret_name}_{index_tag}_ret{args.n_retrieval_pages}_{experiment_date}_eval_results.json"
        else:
            eval_save_fname = f"{ret_name}_{index_tag}_ret{args.n_retrieval_pages}_{args.model_name_or_path}_{experiment_date}_eval_results.json"
        results_file = save_dir / eval_save_fname
        with open(results_file, "w") as f:
            json.dump(all_eval_scores, f, indent=4)

        logger.info(f"Evaluation results saved at: {results_file}")

    if is_distributed():
        barrier()


if __name__ == "__main__":
    main()
