 #Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape
import torch.nn.functional as F
import numpy as np

from torchvision.transforms import ToPILImage
from PIL import Image
import cv2
class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_depth(self):
        pass

    def rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


    def depth_sincos_encoding(self,img_features,depth_features):

        #if not isinstance(image_features,list):
         #   img_features = [img_features]
        depth_features = torch.cat(depth_features,dim=0)
        depth_features =depth_features.reshape(depth_features.shape[0],-1)
        B, L, dim = img_features.shape
        assert dim % 2 == 0, "wrong dim"
        position_embedding = torch.zeros(B,L, dim, dtype=img_features.dtype).to(depth_features.device)

        omega = torch.arange(dim//2, dtype=img_features.dtype)
        omega /= dim/2.
        omega = 1./(10000**omega)

        sita = depth_features[:,:,None] @ omega[None,:].to(depth_features.device).to(depth_features.dtype)
        emb_sin = torch.sin(sita)
        emb_cos = torch.cos(sita)

        position_embedding[:,:,0::2] = emb_sin
        position_embedding[:,:,1::2] = emb_cos

        #if True:
          #  return torch.cat([img_features,position_embedding.to(img_features.device)],dim=1)
        return position_embedding.to(img_features.device) + img_features
    
    
    def depth_sincos_encoding_fixed(self,depth_features,shape):

        #if not isinstance(image_features,list):
         #   img_features = [img_features]
        depth_features = torch.cat(depth_features,dim=0)
        depth_features =depth_features.reshape(depth_features.shape[0],-1)
        B, L, dim = shape
        assert dim % 2 == 0, "wrong dim"
        position_embedding = torch.zeros(B,L, dim).to(depth_features.device)

        omega = torch.arange(dim//2, dtype=torch.float)
        omega /= dim/2.
        omega = 1./(10000**omega)

        sita = depth_features[:,:,None] @ omega[None,:].to(depth_features.device).to(depth_features.dtype)
        emb_sin = torch.sin(sita)
        emb_cos = torch.cos(sita)

        position_embedding[:,:,0::2] = emb_sin
        position_embedding[:,:,1::2] = emb_cos

        return position_embedding

    def depth_projection(self,img_features,depth_features):

        depth_features = torch.cat(depth_features,dim=0)
        if len(depth_features.shape)==3:
            depth_features = depth_features.unsqueeze(1)

        depth_embeddings = self.get_model().get_vision_tower()(depth_features.repeat(1,3,1,1).to(img_features.device))
        depth_embeddings = self.get_model().mm_projector(depth_embeddings.half())
        return torch.cat([img_features, depth_embeddings],dim=1)

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images,depths=None):

        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features
   
    def encode_depth(self, images, target_size, alpha=100):

        with torch.no_grad():
            depths =[]
            for image in images:

                if image==None:
                    depth = torch.zeros(1,target_size[0],target_size[1]).cuda()
                else:
                    tmp = np.asarray(image)[:,:,[2,1,0]]

                    tmp, (h, w) = self.depth.image2tensor(tmp)


                    depth = self.depth(tmp.to(torch.float16).cuda())


                depth = torch.nn.AdaptiveAvgPool2d(target_size)(depth)  
                

                #normalize to 0 to 1
                data_min = depth.min()
                data_max = depth.max()
                
                depth = (depth-data_min)/(data_max-data_min+1e-9)

                depths.append(depth*alpha)

        return depths
    


    def custom_adaptive_avg_pool2d(self,input_tensor, output_size):
        # 获取输入张量的尺寸
        batch_size, channels, height, width = input_tensor.size()
        
        # 计算每个维度的缩放因子
        scale_h = height / output_size[0]
        scale_w = width / output_size[1]
        
        # 初始化输出张量
        output = torch.zeros(batch_size, channels, output_size[0], output_size[1])
        
        # 遍历每个输出位置
        for b in range(batch_size):
            for c in range(channels):
                for i in range(output_size[0]):
                    for j in range(output_size[1]):
                        # 计算输入张量中对应的区域
                        h_start = int(i * scale_h)
                        h_end = int((i + 1) * scale_h)
                        w_start = int(j * scale_w)
                        w_end = int((j + 1) * scale_w)
                        
                        # 获取当前区域的数据
                        region = input_tensor[b, c, h_start:h_end, w_start:w_end]
                        
                        # 计算非零元素的平均值
                        non_zero_elements = region[region != 0]
                        if non_zero_elements.numel() > 0:
                            avg = non_zero_elements.mean()
                        else:
                            avg = 0
                        
                        output[b, c, i, j] = avg
        
        return output
    
    
    def resize_depth(self,images,target_size,alpha=100):
        depths =[]
        for image in images:
            if self.vit_depth:
                tmp = image.clone().detach()
                depth = self.custom_adaptive_avg_pool2d(tmp.unsqueeze(0),target_size)
            else:
                tmp2 = torch.from_numpy(np.asarray(image).copy())
                tmp2 = tmp2.cuda().half().unsqueeze(0)


                depth = self.custom_adaptive_avg_pool2d(tmp2.unsqueeze(0),target_size)

            #normalize to 0 to 1
            data_min = depth.min()
            data_max = depth.max()
            
            depth = (depth-data_min)/(data_max-data_min+1e-9)

            depths.append(depth*alpha)


        return depths

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, ori_imgs, image_sizes=None,depth_features=None
    ):
        print("\n================= [DEBUG: multimodal flow check] =================")
    
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            
            
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
    
        # ======================================================
        # Stage 1️⃣ Encode image + depth features
        # ======================================================
        print("\n[Stage 1] 🔍 Extracting image & depth features...")
    
        if isinstance(images, list) or images.ndim == 5:
            if isinstance(images, list):
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = vision_tower(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features = self.encode_images(images)
    
        print(f"✅ image_features[0] shape = {list(image_features[0].shape)}")
        print(f"🧩 image_features type before return = {type(image_features)}")
        if isinstance(image_features, list):
            image_features = torch.stack(image_features, dim=0)
            print(f"🧩 強制轉成 tensor, shape = {list(image_features.shape)}")
        

        print(f"🧩 self.gt_depth={self.gt_depth}, self.use_depth={self.use_depth}")
        print(f"🔍 images type={type(images)}, "
          f"islist={isinstance(images, list)}, "
          f"elem0_type={type(images[0]) if isinstance(images, list) else 'N/A'}")
        print(f"🔍 ori_imgs type={type(ori_imgs)}, "
          f"islist={isinstance(ori_imgs, list)}, "
          f"elem0_type={type(ori_imgs[0]) if isinstance(ori_imgs, list) else 'N/A'}")

        
        print(f"🧩 ori_imgs type={type(ori_imgs)}")
        # ======================================================
# Stage 1️⃣ Encode image + depth features
# ======================================================
        if depth_features is None:
            print("🧩 沒有外部 depth_features，執行 encode_depth() ...")
            target_size = (24, 24)
            if self.gt_depth:
                depth_features = self.resize_depth(ori_imgs, target_size)
            else:
                depth_features = self.encode_depth(ori_imgs, target_size)
        else:
            print(f"🧩 使用外部傳入的 depth_features，len={len(depth_features)} shape={list(depth_features[0].shape)}")

    
        # === depth fusion ===
        if self.use_depth:
            print("⚙️ 開始執行 depth_sincos_encoding() ...")
            image_features = self.depth_sincos_encoding(image_features, depth_features)
            # ⬇️ 這裡加入統計檢查
            if isinstance(image_features, list):
                sample_feat = image_features[0]
            else:
                sample_feat = image_features
            print(f"✅ depth_sincos_encoding() 融合成功，特徵 shape={list(sample_feat[0].shape) if sample_feat.ndim==3 else list(sample_feat.shape)}")
            print(f"   mean={sample_feat.cpu().mean().item():.4f}, std={sample_feat.cpu().std().item():.4f}")

        else:
            print("⚠️ self.use_depth=False，跳過深度融合")
    
        # ✅ Safe convert: 確保 image_features 不會是 list，否則 model.generate 會報錯
        if isinstance(image_features, list):
            if len(image_features) == 1:
                image_features = image_features[0].unsqueeze(0)  # [1, patch_num, hidden_dim]
                print(f"🩵 image_features list -> tensor (single), shape={list(image_features.shape)}")
            else:
                try:
                    image_features = torch.stack(image_features, dim=0)
                    print(f"🩵 image_features list -> tensor (stacked), shape={list(image_features.shape)}")
                except Exception as e:
                    print(f"⚠️ image_features stack failed: {e}")
    
        print("----------------------------------------------------------")

        # ======================================================
        # Stage 4️⃣ Embedding + Replacement flow
        # ======================================================
       
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)
         
        _input_ids = input_ids
        print(_input_ids)
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
      
    
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
    
        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            print(f"🧩 Batch {batch_idx} - num_images={num_images}")
            if num_images == 0:
                print("⚠️ No -200 token in this batch, skip image feature insertion.")
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
    
            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1:image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1:image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
    
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],),
                                                     IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
    
           # cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
    
            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
         
        print("✅ 替換階段完成，成功生成 new_input_embeds 與 new_labels")
        print(new_input_embeds)
        print( new_labels)
        print("================================================================\n")
      
        # ======================================================
        # Stage 5️⃣ Padding & Return
        # ======================================================
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None


        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels


    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False