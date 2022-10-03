python gpt3_direct_answer.py \
--apikey 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 \
--dataset_type okvqa \
--train_caption_file ../data/predicted_captions/okvqa_train_1003.json \
--train_dataset ../data/task_data/okvqa_w_coco_caption.json \
--val_caption_file ../data/predicted_captions/okvqa_val_1003.json \
--val_dataset ../data/task_data/okvqa_w_coco_caption_val.json \
--output_fn okvqa_1003_log.json


# python gpt3_direct_answer.py \
# --apikey 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 \
# --dataset_type aokvqa \
# --train_caption_file ../data/predicted_captions/aokvqa_train_1003.json \
# --train_dataset ../data/task_data/aokvqa_w_coco_caption.json \
# --val_caption_file ../data/predicted_captions/aokvqa_val_1003.json \
# --val_dataset ../data/task_data/aokvqa_w_coco_caption_val.json \
# --output_fn aokvqa_1003_log.json
