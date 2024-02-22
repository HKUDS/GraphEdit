import torch
import numpy as np
from tqdm import tqdm

import sys
sys.path.append("/xxx/xxx/LLM")

from transformers import AutoTokenizer
from graphedit.model.GraphEdit import GraphEditForCausalLM

path = "./vicuna_7b_pubmed"

tokenizer = AutoTokenizer.from_pretrained(path)
model = GraphEditForCausalLM.from_pretrained(path, torch_dtype=torch.float16, use_cache=True, low_cpu_mem_usage=True).cuda()

texts = np.load('../GNN/datasets/pubmed/pubmed_text.npy')

embs = []

for text in tqdm(texts):
    inputs = tokenizer(text)
    input_ids = torch.as_tensor(inputs.input_ids).to('cuda')
    emb = model.model.embed_tokens(input_ids)
    emb = torch.mean(emb, axis=0)
    embs.append(emb.cpu().detach().numpy())

np.save('../GNN/datasets/pubmed/pubmed_embs', np.array(embs))