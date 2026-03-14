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

from PIL import Image
import torch
from typing import List
from transformers import AutoConfig, AutoProcessor, BitsAndBytesConfig


def init(
    model_name_or_path,
    model_type="qwen2",
    dtype=torch.bfloat16,
    bits=16,
    attn_implementation="flash_attention_2",
    use_fast_processor=None,
    **kwargs,
):
    if bits == 4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype
        )
    else:
        bnb_config = None
    model_type_l = str(model_type).lower()
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    config_model_type = str(getattr(config, "model_type", "")).lower()
    effective_model_type = model_type_l
    if config_model_type in {"qwen2_5_vl", "qwen2_vl"}:
        effective_model_type = config_model_type

    # Qwen2.5-VL has been unstable with flash-attn on some cluster stacks.
    # Prefer eager attention for reliability in offline summary generation.
    if effective_model_type == "qwen2_5_vl" and attn_implementation == "flash_attention_2":
        attn_implementation = "eager"

    if effective_model_type == "qwen2_5_vl":
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as QwenVLForConditionalGeneration
        except Exception as exc:
            raise RuntimeError(
                "Transformers build does not expose Qwen2_5_VLForConditionalGeneration. "
                "Upgrade transformers to a version that supports qwen2_5_vl."
            ) from exc
    elif effective_model_type in {"qwen2_vl", "qwen2"}:
        from transformers import Qwen2VLForConditionalGeneration as QwenVLForConditionalGeneration
    else:
        raise ValueError(
            f"Unsupported Qwen VL model type: requested={model_type_l} config={config_model_type}"
        )

    # Load the model in half-precision on the available device(s)
    model = QwenVLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation=attn_implementation,
        quantization_config=bnb_config,
    )
    model.eval()
    processor_kwargs = {}
    # Keep Qwen2.5-VL aligned with the known-good path used in prior successful runs.
    if effective_model_type == "qwen2_5_vl":
        if use_fast_processor is None:
            processor_kwargs["use_fast"] = True
        else:
            processor_kwargs["use_fast"] = bool(use_fast_processor)
    processor = AutoProcessor.from_pretrained(model_name_or_path, **processor_kwargs)

    return {
        'model': model,
        'processor': processor
    }

def generate(
    model,
    processor,
    question,
    images
) -> List[str]:
    if not images:
        return [""]

    def _resize_images(max_side: int | None):
        resized = []
        for image in images:
            if not isinstance(image, Image.Image):
                resized.append(image)
                continue
            img = image.convert("RGB")
            if max_side is not None:
                w, h = img.size
                if max(w, h) > max_side:
                    scale = max_side / float(max(w, h))
                    img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)
            resized.append(img)
        return resized

    def _is_retryable_cuda_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        retry_signals = (
            "cuda driver error: invalid argument",
            "device-side assert",
            "cublas",
            "cuda error",
        )
        return any(token in msg for token in retry_signals)

    last_exc = None
    # Retry with progressively smaller images when CUDA kernels are unstable.
    # Try original page resolution first (matches prior successful behavior),
    # then progressively downscale on retry.
    for max_side in (None, 1536, 1344, 1120, 960, 768):
        resized_images = _resize_images(max_side=max_side)
        image_content = [{"type": "image", "image": "dummy_content"}] * len(resized_images)
        messages = [
            {
                "role": "user",
                "content": image_content + [{"type": "text", "text": question}]
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text],
            images=resized_images,
            padding=True,
            return_tensors="pt",
        )

        p = next(iter(model.parameters()))
        inputs = inputs.to(p.device)

        try:
            generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            assert isinstance(output_text, list), output_text
            return output_text
        except RuntimeError as exc:
            last_exc = exc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if _is_retryable_cuda_error(exc) and max_side != 768:
                continue
            raise

    if last_exc is not None:
        raise last_exc
    return [""]
