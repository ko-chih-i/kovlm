# -*- coding: utf-8 -*-
import os
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import sys
sys.path.append("./")
import torch
import json
import pandas as pd
import argparse
import types
from tqdm import tqdm

# ===== SD-VLM / LLaVA 模組 =====
from llava.constants import IMAGE_TOKEN_INDEX
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.model.multimodal_projector.builder import build_vision_projector
from llava.model.depth.depth_anything_v2.dpt import DepthAnythingV2
import copy
from transformers import __version__ as hf_version
print("Transformers version:", hf_version)
def base64_to_pil(image_base64):
    try:
        return Image.open(BytesIO(base64.b64decode(image_base64)))
    except Exception as e:
        print(f"❌ Base64 轉換失敗: {e}")
        return None


def eval_sdvlm(args):
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # === 基本設定 ===
    model_path = "/home/itris3/SD-VLM/SD-VLM-7B"
    data_path = "/home/itris3/SD-VLM/test.parquet"
    vision_tower_path = "/home/itris3/SD-VLM/llava/model/clip-vit-large-patch14-336"
    depth_ckpt = "/home/itris3/SD-VLM/llava/model/depth/depth_anything_v2/depth_anything_v2_vitl.pth"
    gt_depth, use_depth = False, True
    answers_file = "./preds.json"
    temperature, num_beams, max_new_tokens = 0.2, 1, 1024

    # === 模型載入 ===
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name, mm_vision_tower=vision_tower_path
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device, dtype=torch.float16)

    # === 修復 Vision Tower 與 image_processor ===
    try:
        vision_tower_obj = model.get_model().get_vision_tower()
    except Exception:
        vision_tower_obj = None

    if vision_tower_obj is None or not getattr(vision_tower_obj, "is_loaded", False):
        print("⚠️ Vision tower 尚未掛入，嘗試手動載入...")
        args_stub = types.SimpleNamespace()
        args_stub.mm_vision_select_layer = -2
        args_stub.mm_vision_select_feature = "patch"
        args_stub.unfreeze_mm_vision_tower = False
        from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
        vision_tower_obj = CLIPVisionTower(vision_tower_path, args_stub, delay_load=False)
        vision_tower_obj.load_model(device_map=None)
        model.get_model().vision_tower = vision_tower_obj
        print("✅ 手動載入 Vision tower 完成。")

    if image_processor is None:
        print("⚠️ image_processor is None，嘗試從 Vision Tower 取回...")
        if hasattr(vision_tower_obj, "image_processor"):
            image_processor = vision_tower_obj.image_processor
            print("✅ image_processor 已成功從 vision tower 還原")
        else:
            raise RuntimeError("❌ 無法從 vision tower 取得 image_processor")

    # === 掛上 DepthAnything ===
    model.vit_depth = DepthAnythingV2().to(device, dtype=torch.float16)
    state = torch.load(depth_ckpt, map_location=device)
    missing, unexpected = model.vit_depth.load_state_dict(state, strict=False)
    print(f"✅ DepthAnything checkpoint loaded ({depth_ckpt})")
    print(f"   Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")

    # === 模型設定 ===
    model.use_depth, model.gt_depth = use_depth, gt_depth
    model.config.mm_use_images = True
    print(f"✅ Model loaded: {model_name}")

    # === 投影層修復 ===
    hidden_size = getattr(model.get_model().config, "hidden_size", 4096)
    projector = getattr(model.get_model(), "mm_projector", None)
    if projector is None:
        projector = build_vision_projector(model.get_model().config)
        model.get_model().mm_projector = projector
    model.get_model().mm_projector = model.get_model().mm_projector.to(device, dtype=torch.float16)
    print(f"🎯 Detected hidden_size = {hidden_size}")

    # === Dataset ===
    print(f"📂 Loading dataset from: {data_path}")
    data = pd.read_parquet(data_path, columns=["image_base64", "image_size", "image_mode", "conversations", "type"])
    data = data.iloc[:min(args.max_samples, len(data))]
    ans_file = open(answers_file, "w", encoding="utf-8")
    ans_file.write(json.dumps({"model": model_path}) + "\n")

    # === 推論迴圈 ===
    for i in tqdm(range(len(data))):
    
        sample = data.iloc[i]
        prompt = sample["conversations"][0]["value"]
        gt = sample["conversations"][1]["value"]
        image = base64_to_pil(sample["image_base64"])
        if image is None:
            continue

        print("\n" + "=" * 80)
        print(f"🧠 DEBUG | 第 {i+1} 筆樣本")
        print(f"🗨️ Prompt: {repr(prompt)}")

        # --- Tokenize ---
        if "<image>" not in prompt:
            prompt = "<image>\n" + prompt.strip()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        print(input_ids)
        # --- 圖像處理 ---
        if image.mode != "RGB":
            image = image.convert("RGB")
        processed_images = process_images([image], image_processor, model.config)
        image_tensor = processed_images[0].unsqueeze(0).to(device, dtype=torch.float16)
        ori_imgs = [copy.deepcopy(image)]
        
        print(f"✅ 圖像載入成功, image_tensor shape={tuple(image_tensor.shape)}")

        # --- 深度 debug ---
        try:
            if hasattr(model, "encode_depth"):
                with torch.no_grad():
                    depths = model.encode_depth(ori_imgs, target_size=(24, 24))
                print(f"🔍 encode_depth() 產生 {len(depths)} 張深度圖")
                print(f"🧩  depths type={type(depths)}")
                d = depths[0]
                print(f"   ➤ depth shape={tuple(d.shape)}, min={d.min():.4f}, max={d.max():.4f}")
            else:
                print("⚠️ model 沒有 encode_depth()")
        except Exception as e:
            print(f"❌ encode_depth() 發生錯誤: {e}")

        # 1️⃣ 確保 Vision Tower 已載入
        try:
            vision_tower_obj = model.get_model().get_vision_tower()
        except Exception:
            vision_tower_obj = None
        if vision_tower_obj is None or not getattr(vision_tower_obj, "is_loaded", False):
            from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
            args_stub = types.SimpleNamespace()
            args_stub.mm_vision_select_layer = -2
            args_stub.mm_vision_select_feature = "patch"
            vision_tower_obj = CLIPVisionTower("/home/itris3/SD-VLM/llava/model/clip-vit-large-patch14-336", args_stub, delay_load=False)
            vision_tower_obj.load_model()
            model.get_model().vision_tower = vision_tower_obj
            print("✅ Vision tower reloaded.")
        
                # 2️⃣ 確保 Projector 有掛且在正確 device
        from llava.model.multimodal_projector.builder import build_vision_projector
        
        if not hasattr(model.get_model(), "mm_projector") or model.get_model().mm_projector is None:
            model.get_model().mm_projector = build_vision_projector(model.get_model().config)
            print("✅ Projector rebuilt.")
        
        # 🔧 強制移動到 GPU
        model.get_model().mm_projector = model.get_model().mm_projector.to(device, dtype=torch.float16)
        
        # 🔧 同時確保 vision tower 在 GPU
        vt = model.get_model().get_vision_tower()
        if hasattr(vt, "to"):
            vt.to(device)
        print("✅ VisionTower & Projector 都在", device)

        
        # 3️⃣ 確保 image_tensor 不是 None
        assert image_tensor is not None and hasattr(image_tensor, "shape"), "❌ image_tensor is None!"



                # ---- Vision tower forward test ----
        with torch.no_grad():
            vt = model.get_model().get_vision_tower()
            feat = vt(image_tensor)
            print("✅ vision_tower output:", getattr(feat, "shape", None))
        
        # ---- Projector test ----
        if hasattr(model.get_model(), "mm_projector") and model.get_model().mm_projector is not None:
            try:
                projected = model.get_model().mm_projector(feat)
                print("✅ mm_projector output:", getattr(projected, "shape", None))
            except Exception as e:
                print("❌ mm_projector forward failed:", e)
        else:
            print("⚠️ mm_projector is None")
        

        # --- Generate ---
        try:
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids, 
                    images=image_tensor,
                    ori_imgs=ori_imgs,
                    depth_features=depths,  
                    image_sizes=[image.size],
                    do_sample=temperature > 0,
                    temperature=temperature,
                    num_beams=num_beams,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                )
       
            
            
            response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            print(f"🟢 pred: {response}")
            ans_file.write(json.dumps({
                "question": prompt, "pred": response, "gt": gt, "type": sample["type"]
            }) + "\n")
        except Exception as e:
            print(f"❌ model.generate 發生錯誤: {e}")
            continue

    ans_file.close()
    print("✅ All predictions saved to preds.json")


# === Main ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=2)
    args = parser.parse_args()
    eval_sdvlm(args)
