# -*- coding: utf-8 -*-
import os
import torch
from PIL import Image
import copy
import numpy as np
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX

print("🚀 Running SD-VLM pipeline with built-in Depth-Anything v2 ...")

# ========== 1️⃣ 裝置設定 ==========
device = torch.device("cpu")   # 改成 "cuda" 若可用 GPU
torch.set_default_device(device)
torch.set_num_threads(8)
print(f"✅ Using device: {device}")

# ========== 2️⃣ 載入 SD-VLM 主模型 ==========
model_path = "/home/itris3/SD-VLM/SD-VLM-7B"
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)
model.to(device)
print("✅ SD-VLM model loaded")
# ✅ 徹底轉成 float32，避免 Float / Half 混用錯誤
for name, param in model.named_parameters():
    if param.dtype == torch.float16 or param.dtype == torch.bfloat16:
        param.data = param.data.float()
        if param.grad is not None:
            param.grad = param.grad.float()
print("✅ 全模型權重已轉為 float32 (CPU safe)")


# ========== 3️⃣ 載入內建 Depth Anything ==========
# ========== 3️⃣ 載入內建 Depth Anything ==========
print("⚙️ Loading built-in Depth Anything v2 (from local checkpoint)...")
# ✅ 初始化 DepthAnything 模型（手動載入權重）
from llava.model.depth.depth_anything_v2.dpt import DepthAnythingV2

# 🔧 指向你本地的模型權重
depth_ckpt_path = "/home/itris3/SD-VLM/llava/model/depth/depth_anything_v2/depth_anything_v2_vitl.pth"

# ✅ 建立 DepthAnythingV2 模型（不傳入 model_type）
vit_depth = DepthAnythingV2()  # 官方 dpt.py 不需 model_type

# ✅ 載入本地權重
checkpoint = torch.load(depth_ckpt_path, map_location=device)
# 有些版本的 checkpoint 是 {'model': state_dict} 結構，所以做個防呆：
if "model" in checkpoint:
    vit_depth.load_state_dict(checkpoint["model"])
else:
    vit_depth.load_state_dict(checkpoint)

vit_depth.to(device)
vit_depth.eval()

# ✅ 附加到 SD-VLM 模型中
model.vit_depth = vit_depth
print("✅ Attached built-in depth module to model.vit_depth (from local .pth)")


# ========== 4️⃣ 確保 vision_tower 正常 ==========
vision_tower = model.get_model().get_vision_tower()
if hasattr(vision_tower, "load_model") and not getattr(vision_tower, "is_loaded", False):
    print("⚙️ Loading CLIP vision tower ...")
    vision_tower.load_model(device_map=None)
    print("✅ Vision tower ready.")
image_processor = vision_tower.image_processor

# ========== 5️⃣ 準備輸入 ==========
prompt = "How far is the chair in this image?"
image_folder = "/home/itris3/SD-VLM/images"
image_file = "test.jpg"
image_path = os.path.join(image_folder, image_file)
assert os.path.exists(image_path), f"❌ 找不到圖片: {image_path}"

image = Image.open(image_path).convert("RGB")


ori_img = copy.deepcopy(image)
input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

# ========== 6️⃣ 圖像處理 ==========
image_tensor = process_images([image], image_processor, model.config)[0].unsqueeze(0).to(device)
if image_tensor.dtype != torch.float32:
    image_tensor = image_tensor.float()

# ========== 7️⃣ 推論 ==========
print("🧠 Running SD-VLM + Depth-Anything inference ... (CPU may take 3–8 mins)")
# ✅ 將 PIL Image 轉成 Tensor，符合 resize_depth() 預期輸入
from torchvision import transforms

to_tensor = transforms.ToTensor()
ori_img_tensor = to_tensor(ori_img).to(device).float()

with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=[image.size],
        do_sample=False,
        temperature=0.2,
        num_beams=1,
        ori_imgs=[ori_img_tensor],  # ✅ 改成 tensor
        max_new_tokens=64,
        use_cache=True,
    )
# ========== 8️⃣ 輸出 ==========
response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
print("🟢 Model response:", response)
