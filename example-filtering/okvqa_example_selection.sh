
# filtering okvqa training examples
python gpt3_example_selection.py \
--apikey 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 \
--dataset_type okvqa \
--partition train \
--train_caption_file ../data/raw_captions/okvqa_w_gpt3_caption_code002.json \
--train_dataset ../data/task_data/okvqa_w_coco_caption.json \
--val_caption_file ../data/raw_captions/okvqa_w_gpt3_caption_code002.json \
--val_dataset ../data/task_data/okvqa_w_coco_caption.json \
--output_fn okvqa_w_gpt3_caption_code002_filtered.json


# filtering okvqa val examples
python gpt3_example_selection.py \
--apikey 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 \
--dataset_type okvqa \
--partition val \
--train_caption_file ../data/raw_captions/okvqa_w_gpt3_caption_code002.json \
--train_dataset ../data/task_data/okvqa_w_coco_caption.json \
--val_caption_file ../data/raw_captions/okvqa_w_gpt3_caption_val_code002.json \
--val_dataset ../data/task_data/okvqa_w_coco_caption_val.json \
--output_fn okvqa_w_gpt3_caption_val_code002_filtered.json