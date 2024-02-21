export CUDA_VISIBLE_DEVICES=0

python ./graphedit/eval/eval_model.py \
    --model_name_or_path ./vicuna_7b_pubmed \
    --eval_file ../GNN/datasets/pubmed/pubmed_template_add_3.json \
    --output_res_path result \
    --start_id 0 \
    --end_id -1 \
    --num_gpus 1