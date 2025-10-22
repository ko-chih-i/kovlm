#    Copyright 2023 Haotian Liu
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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.models.clip.modeling_clip import CLIPVisionEmbeddings

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from ..depth.depth_anything_v2.dpt import DepthAnythingV2
import numpy as np


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout=0.1, activation="gelu"):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.linear3 = nn.Linear(d_model,100)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)        # 可选：防止过拟合
        x = self.linear2(x)
        x = self.linear3(x)
        return x

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

def load_depth():
    # instantiate the model



    from llava.model.depth.depth_anything_v2.dpt import DepthAnythingV2

    model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

    encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

    model = DepthAnythingV2(**model_configs[encoder])

    model.to_empty(device=('cuda' if torch.cuda.is_available() else 'cpu'))

    model.eval()

    for para in model.parameters():
        para.requires_grad = False
    return model

class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"
    use_depth = False
    gt_depth = False



class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)

        self.model = LlavaLlamaModel(config)
        self.depth = load_depth()
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.use_depth = config.use_depth
        self.gt_depth = config.gt_depth

        # Initialize weights and apply final processing
        self.post_init()
        
        # Initialize depth projector
        self.get_depth_module(config)

    def get_depth_module(self,config):
        '''
        modules = []
       
        modules.append(nn.Conv2d(1, 1024, kernel_size=3, stride=1,padding=1))
        modules.append(LayerNorm2d(1024))
        modules.append(nn.GELU())
        modules.append(nn.Conv2d(1024, 4096, kernel_size=14, stride=14))
        modules.append(LayerNorm2d(4096))
        modules.append(nn.GELU())
        '''

        class TMP:
            def __init__(self):
                self.patch_size = 14
                self.image_size = 336
                self.hidden_size = 4096
                self.num_channels = 1
        tmp = TMP()

        self.depth_projector =  CLIPVisionEmbeddings(tmp)


    def get_model(self):
        return self.model

    def get_depth(self):
        return self.depth



    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        ori_imgs: Optional[np.ndarray] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                ori_imgs,
                
                image_sizes
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        ori_imgs: Optional[torch.Tensor] = None,depth_features=None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                input_ids,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                ori_imgs,
                depth_features=depth_features ,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        print("\n====================== [DEBUG] prepare_inputs_for_generation ======================")
    
        # 🧩 1️⃣ 基本資訊
        print(f"🔹 input_ids shape = {tuple(input_ids.shape) if input_ids is not None else 'None'}")
        print(f"🔹 inputs_embeds shape = {tuple(inputs_embeds.shape) if inputs_embeds is not None else 'None'}")
    
        # 🧠 2️⃣ cache 狀態
        if past_key_values is None:
            print("⚪ past_key_values = None → 第一次 forward")
        else:
            print(f"🟢 past_key_values 已建立，層數 = {len(past_key_values)}")
            try:
                k_shape = past_key_values[0][0].shape
                v_shape = past_key_values[0][1].shape
                print(f"   Layer[0] K shape: {k_shape}")
                print(f"   Layer[0] V shape: {v_shape}")
                print(f"   🔸 cache_seq_len = {k_shape[2]}")
            except Exception as e:
                print(f"   ⚠️ 無法讀取 cache shape: {e}")
    
        # 🪄 3️⃣ 呼叫父類別 (HuggingFace)
        inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
    
        # 🧮 4️⃣ attention mask 檢查
        attn_mask = inputs.get("attention_mask", None)
        if attn_mask is not None:
            print(f"🧮 attention_mask shape after super(): {tuple(attn_mask.shape)}")
        else:
            print("⚠️ attention_mask = None after super()")
    
        # 🚦 5️⃣ 控制行為
        if past_key_values is None:
            print("✅ 第一次 forward → 保留 multimodal 輸入")
            for key in ["images", "image_sizes", "ori_imgs", "depth_features"]:
                if key in kwargs:
                    inputs[key] = kwargs[key]
        else:
            print("🚫 使用 cache → 僅處理新 token")
    
            # ⚙️ 關鍵：清除殘留的舊 embedding
            inputs["inputs_embeds"] = None
            if "inputs_embeds" in kwargs:
                kwargs.pop("inputs_embeds")
    
            # ⚠️ 確認 Hugging Face 已自動裁切 input_ids[:, -1:]
            if input_ids.shape[1] > 1:
                print(f"⚠️ 注意：HF 傳入時 input_ids={input_ids.shape}，理論上 forward 只應取最後一個 token")
    
            # 🧩 Debug 檢查 cache / mask 同步
            cache_len = past_key_values[0][0].shape[2]
            mask_len = attn_mask.shape[1] if attn_mask is not None else -1
            print(f"[CHECK] cache={cache_len}, mask={mask_len}")
    
            # 🧩 額外安全檢查每層 K/V 長度
            for i, (k, v) in enumerate(past_key_values):
                if k.shape[2] != v.shape[2]:
                    print(f"🚨 層 {i} K/V 不符: K={k.shape}, V={v.shape}")
                    raise RuntimeError(f"❌ 第 {i} 層 cache K/V 長度不符")
    
        print("===================================================================================\n")
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)