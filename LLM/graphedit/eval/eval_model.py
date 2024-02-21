import argparse
from transformers import AutoTokenizer
import torch
import os
import json

import ray
import sys
import os.path as osp
sys.path.append("/xxx/xxx/LLM")

from graphedit.model.GraphEdit import GraphEditForCausalLM
from graphedit.conversation import SeparatorStyle, get_conv_template
from graphedit.utils import disable_torch_init

from tqdm import tqdm

def load_eval_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def run_eval(args, num_gpus):
    # split question file into num_gpus files
    prompt_file = load_eval_file(args.eval_file)

    if args.end_id == -1:
        args.end_id = len(prompt_file)
    else:
        args.end_id += 1

    prompt_file = prompt_file[args.start_id:args.end_id]
    chunk_size = len(prompt_file) // num_gpus
    ans_handles = []
    split_list = list(range(args.start_id, args.end_id, chunk_size))
    idx_list = list(range(0, len(prompt_file), chunk_size))

    if len(split_list) == num_gpus: 
        split_list.append(args.end_id)
        idx_list.append(len(prompt_file))
    elif len(split_list) == num_gpus + 1: 
        split_list[-1] = args.end_id
        idx_list[-1] = len(prompt_file)
    else: 
        raise ValueError('error in the number of list')
    
    if osp.exists(args.output_res_path) is False: 
        os.mkdir(args.output_res_path)
    
    for idx in range(len(idx_list) - 1):
        start_idx = idx_list[idx]
        end_idx = idx_list[idx + 1]
        
        start_split = split_list[idx]
        end_split = split_list[idx + 1]
        # eval_model(
        #         args, prompt_file[start_idx:end_idx], start_split, end_split
        #     )
        ans_handles.append(
            eval_model.remote(
                args, prompt_file[start_idx:end_idx], start_split, end_split
            )
        )

    ans_jsons = []
    for ans_handle in ans_handles:
        ans_jsons.extend(ray.get(ans_handle))

@ray.remote(num_gpus=1)
@torch.inference_mode()
def eval_model(args, prompt_file, start_idx, end_idx):
    disable_torch_init()
    print('start loading')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    print('finish loading')

    print('start loading')
    model = GraphEditForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16, use_cache=True, low_cpu_mem_usage=True).cuda()
    print('finish loading')

    res_data = []
    print(f'total: {len(prompt_file)}')
    for instruct_item in tqdm(prompt_file):
        conv = get_conv_template("graphedit")
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

        for sentence in instruct_item["conversations"][:-1]:
            role = roles[sentence["from"]]
            conv.append_message(role, sentence["value"])

        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        inputs = tokenizer([prompt])

        input_ids = torch.as_tensor(inputs.input_ids).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.ADD_COLON_TWO else conv.sep2

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        res_data.append({"id": instruct_item["id"], "res": outputs}.copy())
        with open(osp.join(args.output_res_path, '{}.json'.format(args.eval_file.split('.')[1].split('/')[-1])), "w") as fout:
            json.dump(res_data, fout, indent=4)

    return res_data
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="facebook/opt-350m")

    parser.add_argument("--eval_file", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)

    parser.add_argument("--output_res_path", type=str, default=None)
    parser.add_argument("--num_gpus", type=int, default=4)

    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--end_id", type=int, default=0)

    args = parser.parse_args()
    ray.init()
    run_eval(args, args.num_gpus)