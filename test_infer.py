# -*- coding: utf-8 -*-
import os
import torch
from PIL import Image
import copy
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX
from transformers import CLIPImageProcessor  # 🔹加這行

# === 1. 指定你本地的模型資料夾 ===
model_path = "/home/itris3/SD-VLM/SD-VLM-7B"

# === 2. 指定本地 vision tower 路徑 ===
local_clip_path = "/home/itris3/SD-VLM/llava/model/clip-vit-large-patch14-336"

# === 3. 載入模型 ===
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)

# === 4. 驗證 Vision Tower ===
vision_tower = model.get_model().get_vision_tower()
vision_tower.vision_tower_name = "/home/itris3/SD-VLM/llava/model/clip-vit-large-patch14-336"
print("✅ Vision Tower loaded:", type(vision_tower))
print("✅ Has .vision_tower:", hasattr(vision_tower, "vision_tower"))
print("✅ Has .image_processor:", hasattr(vision_tower, "image_processor"))

# === 5. 若未載入 → 手動載入你的本地 clip-vit 模型 ===
if hasattr(vision_tower, "load_model") and not getattr(vision_tower, "is_loaded", False):
    print("⚙️ Loading local vision tower from:", local_clip_path)
    vision_tower.load_model(device_map=None)
    # 👇手動指定 image_processor 來源為你的本地 clip-vit
    vision_tower.image_processor = CLIPImageProcessor.from_pretrained(local_clip_path)
    print("✅ Vision tower fully initialized.")
    print("✅ Has .vision_tower:", hasattr(vision_tower, "vision_tower"))
    print("✅ Has .image_processor:", hasattr(vision_tower, "image_processor"))

# === 6. 使用 vision_tower 的 image_processor ===
image_processor = vision_tower.image_processor

# === 7. 準備 prompt 與圖片 ===
prompt = "how far is the chair in this image?"
image_folder = "/home/itris3/SD-VLM/images"
image_file = "test.jpg"

input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
ori_img = copy.deepcopy(image)
image_tensor = process_images([image], image_processor, model.config)[0]


model = model.to("cuda")

# === 8. 推論 ===
with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        images=image_tensor.unsqueeze(0).half().to(input_ids.device),
        image_sizes=[image.size],
        do_sample=False,
        temperature=0.2,
        num_beams=1,
        ori_imgs=[ori_img],
        max_new_tokens=512,
        use_cache=True,
    )

response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
print("🟢 Model response:", response)
