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
    if effective_model_type == "qwen2_5_vl":
        # Slow processor path is more stable for Qwen2.5-VL in this environment.
        processor_kwargs["use_fast"] = False
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

    # Downscale very large pages to reduce visual-kernel instability.
    resized_images = []
    for image in images:
        if not isinstance(image, Image.Image):
            resized_images.append(image)
            continue
        img = image.convert("RGB")
        w, h = img.size
        max_side = 1344
        if max(w, h) > max_side:
            scale = max_side / float(max(w, h))
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)
        resized_images.append(img)

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
    # image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=resized_images,
        # videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    p = next(iter(model.parameters()))

    inputs = inputs.to(p.device)

    # Inference
    generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    assert isinstance(output_text, list), output_text

    return output_text
