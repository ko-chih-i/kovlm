# 👷SD-VLM: Spatial Measuring and Understanding with Depth-Encoded Vision-Language Models


[📢 [[Project Page](https://cpystan.github.io/SD_VLM_pages/)] [[Arxiv](https://arxiv.org/abs/2509.17664)]  [[Data](https://huggingface.co/datasets/cpystan/MSMU)] 



> **We are excited to announce that our paper is accepted by NeurIPS 2025!**

---

## Install

1. Clone this repository
```bash
git clone https://github.com/cpystan/SD-VLM.git
cd SD-VLM
```

2. Install Package
```Shell
conda create -n sdvlm python=3.10 -y
conda activate sdvlm
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for training cases
```
pip install -e ".[train]"
```
## MSMU (Massive Spatial Measuring and Understanding) Dataset
For instruction tuning, please download the train.parquet of MSMU Dataset from [[HuggingFace](https://huggingface.co/datasets/cpystan/MSMU)]. 
For evaluation on MSMU-Bench, please download the test.parquet of MSMU Dataset from [[HuggingFace](https://huggingface.co/datasets/cpystan/MSMU)]. 



## 🚧 Status: Coming Soon
More details are coming soon.

