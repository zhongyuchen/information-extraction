export CUDA_VISIBLE_DEVICES=0,1,2

python bin/subject_object_labeling/sequence_labeling_data_manager.py

python run_sequnce_labeling.py \
--task_name=SKE_2019 \
--do_train=true \
--do_eval=false \
--data_dir=bin/subject_object_labeling/sequence_labeling_data \
--vocab_file=pretrained_model/chinese_L-12_H-768_A-12/vocab.txt \
--bert_config_file=pretrained_model/chinese_L-12_H-768_A-12/bert_config.json \
--init_checkpoint=pretrained_model/chinese_L-12_H-768_A-12/bert_model.ckpt \
--max_seq_length=320 \
--train_batch_size=8 \
--learning_rate=2e-5 \
--num_train_epochs=3.0 \
--output_dir=./output/sequnce_labeling_model/
