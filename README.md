
# **GraphEdit: Large Language Models for Graph Structure Learning**
<a href='https://github.com/HKUDS/GraphEdit'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='https://arxiv.org/abs/2402.15183'><img src='https://img.shields.io/badge/arXiv-2402.15183-b31b1b'></a>
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/b9f51e64959e4999ad469c2ca437373a.png#pic_center)
## Code Structure
```
.
├── README.md
├── GNN
│   ├── GNNs
│   │   ├── GCN
│   │   │   └── model.py
│   │   ├── MLP
│   │   │   └── model.py
│   │   ├── RevGAT
│   │   │   ├── eff_gcn_modules/rev
│   │   │   │   ├── __init__.py
│   │   │   │   ├── gcn_revop.py
│   │   │   │   ├── memgcn.py
│   │   │   │   └── rev_layer.py
│   │   │   ├── __init__.py
│   │   │   └── model.py
│   │   ├── SAGE
│   │   │   └── model.py
│   │   ├── gnn_trainer.py
│   │   └── gnn_utils.py
│   ├── datasets
│   │   ├── dataset.py
│   │   ├── load.py
│   │   ├── load_citeseer.py
│   │   ├── load_cora.py
│   │   ├── load_pubmed.py
│   │   └── utils.py
│   ├── main.py
│   ├── predict_edge.py
│   ├── train_edge_predictor.py
│   └── utils.py
└── LLM
    ├── graphedit
    │   ├── data
    │   │   ├──__init__.py
    │   │   ├──clean_sharegpt.py
    │   │   ├──convert_alpaca.py
    │   │   ├──extract_gpt4_only.py
    │   │   ├──extract_single_round.py
    │   │   ├──filter_wrong_format.py
    │   │   ├──get_stats.py
    │   │   ├──hardcoded_questions.py
    │   │   ├──inspect_data.py
    │   │   ├──merge.py
    │   │   ├──optional_clean.py
    │   │   ├──optional_replace.py
    │   │   ├──prepare_all.py
    │   │   ├──pretty_json.py
    │   │   ├──sample.py
    │   │   ├──split_long_conversation.py
    │   │   └── split_train_test.py
    │   ├── eval   
    │   │   └── eval_model.py
    │   ├── model
    │   │   ├── GraphEdit.py
    │   │   ├── __init__.py
    │   │   ├── apply_delta.py
    │   │   ├── apply_lora.py
    │   │   ├── compression.py
    │   │   ├── convert_fp16.py
    │   │   ├── llama_condense_monkey_patch.py
    │   │   ├── make_delta.py
    │   │   ├── model_adapter.py
    │   │   ├── model_chatglm.py
    │   │   ├── model_codet5p.py
    │   │   ├── model_exllama.py
    │   │   ├── model_falcon.py
    │   │   ├── model_registry.py
    │   │   ├── monkey_patch_non_inplace.py
    │   │   ├── rwkv_model.py
    │   │   └── upload_hub.py
    │   ├── modules
    │   │   ├── __init__.py
    │   │   ├── awq.py
    │   │   ├── exllama.py
    │   │   └── gptq.py
    │   ├── protocol
    │   │   ├── api_protocol.py
    │   │   └── openai_api_protocol.py
    │   ├── serve
    │   │   ├── gateway
    │   │   │   ├── README.md
    │   │   │   └── nginx.conf
    │   │   ├── monitor
    │   │   │   ├── dataset_release_scripts
    │   │   │   │   ├── arena_33k
    │   │   │   │   │   ├── count_unique_users.py
    │   │   │   │   │   ├── filter_bad_conv.py
    │   │   │   │   │   ├── merge_field.py
    │   │   │   │   │   ├── sample.py
    │   │   │   │   │   └── upload_hf_dataset.py
    │   │   │   │   └── lmsys_chat_1m
    │   │   │   │       ├── approve_all.py
    │   │   │   │       ├── compute_stats.py
    │   │   │   │       ├── filter_bad_conv.py
    │   │   │   │       ├── final_post_processing.py
    │   │   │   │       ├── instructions.md
    │   │   │   │       ├── merge_oai_tag.py
    │   │   │   │       ├── process_all.sh
    │   │   │   │       ├── sample.py
    │   │   │   │       └── upload_hf_dataset.py
    │   │   │   ├── basic_stats.py
    │   │   │   ├── clean_battle_data.py
    │   │   │   ├── clean_chat_data.py
    │   │   │   ├── elo_analysis.py
    │   │   │   ├── inspect_conv.py
    │   │   │   ├── intersect_conv_file.py
    │   │   │   ├── leaderboard_csv_to_html.py
    │   │   │   ├── monitor.py
    │   │   │   ├── summarize_cluster.py
    │   │   │   ├── tag_openai_moderation.py
    │   │   │   └── topic_clustering.py
    │   │   ├── __init__.py
    │   │   ├── api_provider.py
    │   │   ├── base_model_worker.py
    │   │   ├── cli.py
    │   │   ├── controller.py
    │   │   ├── gradio_block_arena_anony.py
    │   │   ├── gradio_block_arena_named.py
    │   │   ├── gradio_web_server.py
    │   │   ├── gradio_web_server_multi.py
    │   │   ├── huggingface_api.py
    │   │   ├── huggingface_api_worker.py
    │   │   ├── inference.py
    │   │   ├── launch_all_serve.py
    │   │   ├── model_worker.py
    │   │   ├── multi_model_worker.py
    │   │   ├── openai_api_server.py
    │   │   ├── register_worker.py
    │   │   ├── shutdown_serve.py
    │   │   ├── test_message.py
    │   │   ├── test_throughput.py
    │   │   └── vllm_worker.py
    │   ├── train
    │   │   ├── GraphEdit_trainer.py
    │   │   ├── llama2_flash_attn_monkey_patch.py
    │   │   ├── llama_flash_attn_monkey_patch.py
    │   │   ├── llama_xformers_attn_monkey_patch.py
    │   │   ├── train.py
    │   │   ├── train_baichuan.py
    │   │   ├── train_flant5.py
    │   │   ├── train_lora.py
    │   │   ├── train_lora_t5.py
    │   │   ├── train_mem.py
    │   │   └── train_xformers.py
    │   ├── __init__.py
    │   ├── constants.py
    │   ├── conversation.py
    │   └── utils.py
    ├── playground
    │   ├── test_embedding
    │   │   ├── README.md
    │   │   ├── test_classification.py
    │   │   ├── test_semantic_search.py
    │   │   └── test_sentence_similarity.py
    │   ├── deepspeed_config_s2.json
    │   └── deepspeed_config_s3.json
    ├── scripts
    │   ├── apply_lora.py
    │   ├── create_ins.py
    │   ├── eval.sh
    │   ├── get_embs.py
    │   ├── result2np.py
    │   └── train_lora.sh
    ├── tests
    │   ├── killall_python.sh    
    │   ├── launch_openai_api_test_server.py
    │   ├── test_cli.py
    │   ├── test_cli_inputs.txt
    │   ├── test_openai_api.py
    │   └── test_openai_langchain.py
    ├── .pylintrc
    ├── LICENSE
    ├── format.sh
    └── pyproject.toml
```
## 0. Python Environment Setup
* Packed conda environment is provided [here](https://drive.google.com/file/d/1eeLKFiDU4CbOjb3uzl1Ur0jHXAEUyh5j/view?usp=drive_link) (NVIDIA GeForce RTX 3090)
```bash
conda create --name GraphEdit python=3.8
conda activate GraphEdit

pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0
pip install torch_geometric
pip install dgl
pip install transformers==4.31.0
pip install flash_attn==1.0.4
```

## 1. Download TAG datasets
| Dataset | Description |
|--|--|
| Pubmed | Download the dataset [here](https://drive.google.com/file/d/11OVDmP_DaM3urAswIlMLjiby28X8-8_Z/view?usp=drive_link), unzip and move it to `GNN/datasets/pubmed` |
| Citeseer | Download the dataset [here](https://drive.google.com/file/d/1KtFjg95p3tPRWQ5XCqTtjJh9nXLylcbQ/view?usp=drive_link), unzip and move it to `GNN/datasets/citeseer` |
| Cora | Download the dataset [here](https://drive.google.com/file/d/1fO9tAX2yUoQ74WBE25bAw943nRCKaBqj/view?usp=drive_link), unzip and move it to `GNN/datasets/cora` |

## 2. Getting Started

* Replace the system path in `eval_model.py`, `train_lora.py` and `get_embs.py`  with your path.
### Stage-1: Instruction tuning the LLM
* Vicuna-7b can get from the [huggingface](https://huggingface.co/lmsys/vicuna-7b-v1.5-16k).
* Trained Lora models are provided [here](https://drive.google.com/drive/folders/15MO09sVetHaEPBAYM2M2kZ4eyuPdL-Ng?usp=drive_link).
```bash
cd GraphEdit/LLM/
sh scripts/train_lora.sh

python scripts/apply_lora.py
```
### Stage-2: Get the candidate structure
* Trained edge predictors are provided [here](https://drive.google.com/drive/folders/1bJ5rArLRa-MMbqytioZFQt2HRWMquIHl?usp=drive_link)
```bash
python scripts/get_embs.py

cd ../GNN/
python train_edge_predictor.py
python predict_edge.py --combine True
```
### Stage-3: Refine the candidate structure
```bash
cd ../LLM/
python scripts/create_ins.py
sh scripts/eval.sh

python scripts/result2np.py
```

### Stage-4: Eval the refined structure
* Refined structrues are provided [here](https://drive.google.com/drive/folders/1EeggwedsQraVVIqxkqQDOGBOH4qwVvLU?usp=drive_link)
```bash
cd ../GNN/
python main.py
```

## 3. Instruction Template
> Pubmed

```
Based on the title and abstract of the two papers. Do they belong to the same category among Diabetes Mellitus Type 1, Diabetes Mellitus Type 2, or Diabetes Mellitus, Experimental? If the answer is \"True\", answer \"True\" and the category, otherwise answer \"False\". The first paper: {pubmed.raw_texts[paperID_0]} The second paper: {pubmed.raw_texts[paperID_1]}.
```

> Citeseer

```
Based on the title and abstract of the two papers. Do they belong to the same category among Agent, ML, IR, DB, HCI and AI? If the answer is \"True\", answer \"True\" and the category, otherwise answer \"False\". The first paper: {citeseer.raw_texts[paperID_0]} The second paper: {citeseer.raw_texts[paperID_1]}.
```
> Cora

```
Based on the title and abstract of the two papers. Do they belong to the same category among Rule_Learning, Neural_Networks, Case_Based, Genetic_Algorithms, Theory, Reinforcement_Learning or Probabilistic_Methods? If the answer is \"True\", answer \"True\" and the category, otherwise answer \"False\". If there is insufficient text information, answer \"True\". The first paper: Title: {cora.raw_text[paperID_0].split(':')[0]}  Abstract: {cora.raw_text[paperID_0].split(':')[1]}  The second paper: Title: {cora.raw_text[paperID_1].split(':')[0]}  Abstract: {cora.raw_text[paperID_1].split(':')[1]}.
```
## Citation

```
@article{guo2024graphedit,
title={GraphEdit: Large Language Models for Graph Structure Learning}, 
author={Zirui Guo and Lianghao Xia and Yanhua Yu and Yuling Wang and Zixuan Yang and Wei Wei and Liang Pang and Tat-Seng Chua and Chao Huang},
year={2024},
eprint={2402.15183},
archivePrefix={arXiv},
primaryClass={cs.CL}
}
```

## Acknowledgement
The structure of the LLM in this code is largely based on [FastChat](https://github.com/lm-sys/FastChat). And the original TAG datasets are provided by [Graph-LLM](https://github.com/CurryTang/Graph-LLM). Thanks for their work.
