import torch
import numpy as np

import json

from tqdm import tqdm, trange

data = torch.load("../GNN/datasets/pubmed/pubmed_fixed_tfidf.pt")
data.edge_index = np.load('../GNN/datasets/pubmed/pubmed_edges_add_3.npy')
transformed_data = []


for index in trange(len(data.edge_index[0])):
    if data.edge_index[0][index] < data.edge_index[1][index]: 
        paperID_0 = data.edge_index[0][index]
        paperID_1 = data.edge_index[1][index]

        # template = f"Based on the title and abstract of the two papers. Do they belong to the same category among Diabetes Mellitus Type 1, Diabetes Mellitus Type 2, or Diabetes Mellitus, Experimental? Only give me the True or False. The first paper: {data.raw_texts[paperID_0]} The second paper: {data.raw_texts[paperID_1]}"
        template = f"Based on the title and abstract of the two papers. Do they belong to the same category among Diabetes Mellitus Type 1, Diabetes Mellitus Type 2, or Diabetes Mellitus, Experimental? If the answer is \"True\", answer \"True\" and the category, otherwise answer \"False\". The first paper: {data.raw_texts[paperID_0]} The second paper: {data.raw_texts[paperID_1]}"
        
        if data.y[paperID_0] == data.y[paperID_1]:
            # result = True
            result = f"True. {data.category_names[paperID_0]}."
        else:
            result = False


        temp =  {
                    "id": str(index),
                    "conversations": [
                        {
                            "from": "human",
                            "value": template
                        },
                        {
                            "from": "gpt",
                            "value": str(result)
                        }
                    ]
                }

        transformed_data.append(temp)

file_path = f"../GNN/datasets/pubmed/pubmed_template_add_3.json"
with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(transformed_data, f, ensure_ascii=False, indent=4)