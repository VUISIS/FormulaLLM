### Finetuning and Inference
To be able to run and finetune the models you need a >6GB VRAM Nvidia GPU with CUDA. 

```bash
Xturing, vLLM, and QLora are used in finetuning and running inference. 

Install into separate virtual environments as they each use different library versions.

conda create -n xt python=3.10
conda activate xt
pip install xturing==0.1.4

conda create -n vl python=3.10
conda activate vl
pip install vllm==0.1.2

conda create -n ql python=3.10
conda activate ql
git clone https://github.com/artidoro/qlora
pip install -U -r requirements.txt
```
