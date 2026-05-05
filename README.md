# KidVis Evaluation

This repository provides evaluation code for **KidVis**, a benchmark for assessing foundational visual primitives in multimodal large language models (MLLMs).

## 1. Download the KidVis Dataset

First, install the Hugging Face CLI:

```bash
pip install -U "huggingface_hub[cli]"
```

If you are in mainland China, you can use the Hugging Face mirror:
```bash
export HF_ENDPOINT=https://hf-mirror.com
```
Download the KidVis dataset:
```bash
hf download Jack-2026/KidVis --repo-type dataset --local-dir ./KidVis
```
After downloading, the dataset directory should look like this:
```bash
KidVis/  
├── metadata.csv  
└── images/  
├── Question_1/  
├── Question_2/  
└── ...
```

## 2. Download the Model
Taking Qwen3-VL-4B-Instruct as an example, download the model from Hugging Face:
```bash
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct --local-dir ./Qwen3-VL-4B-Instruct

```
## 3. Run Evaluation
3.1 For Qwen3-VL-4B-Instruct
```bash
pip install "transformers>=4.57.0" torch datasets accelerate tqdm pillow torchvision "qwen_vl_utils>=0.0.14"
```
3.2 Evaluate with the Hugging Face model name
For Chinese prompts
```bash
python run_eval_qwen3vl.py \  
--model_name Qwen/Qwen3-VL-4B-Instruct \  
--data_dir /your/data/path/KidVis \  
--lang zh
```
3.3 Evaluate with a local model path
For Chinese prompts:
```bash
python run_eval_qwen3vl.py \  
--model_name /your/model/path/Qwen3-VL-4B-Instruct \  
--data_dir /your/data/path/KidVis \  
--lang zh
```
3.4 Evaluate with English prompts
If you want to run evaluation with English prompts, simply replace:
```bash
--lang zh
```
with:
```bash
--lang en
```
## 4. API Evaluation
Evaluate with API Models
```bash
python run_eval_API.py \
  --base_url --your base_url \
  --api_key --your api key \
  --model_name --your model name \
  --data_dir ./KidVis \
  --lang zh \
  --output_dir results
  ```
## 5. Output

The evaluation script will save:

* per-sample predictions
* summary results
* overall accuracy
* subset-level results
* capability-level results

The outputs are typically written to the `results/` directory.
