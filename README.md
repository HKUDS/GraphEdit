# **GraphEdit: Large Language Models for Graph Structure Learning**
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/b9f51e64959e4999ad469c2ca437373a.png#pic_center)

## 0. Python Environment Setup
```bash
conda create --name GraphEdit python=3.8
conda activate GraphEdit

pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0

pip install torch_geometric
pip install dgl
```

## 1. Download TAG datasets
| Dataset | Description |
|--|--|
| Pubmed | Download the dataset here, unzip and move it to `GNN/datasets/pubmed` |
| Citeseer | Download the dataset here, unzip and move it to `GNN/datasets/citeseer` |
| Cora | Download the dataset here, unzip and move it to `GNN/datasets/cora` |

## 2. Getting Started
### Stage-1: Instruction tuning the LLM
* Vicuna-7b can get from the [huggingface](https://huggingface.co/lmsys/vicuna-7b-v1.5-16k).
```bash
cd GraphEdit/LLM/
sh script/train_lora.sh

python script/apply_lora.py
```
### Stage-2: Get the candidate structure
```bash
python script/get_embs.py

cd ../GNN/
python train_edge_predictor.py
python predict_edge.py --combine
```
### Stage-3: Refine the candidate structure
```bash
cd ../LLM/
python script/create_ins.py
sh script/eval.sh

python script/result2np.py
```

### Stage-4: Eval the refined structure
```bash
cd ../GNN/
python main.py
```
