

python bin/predicate_classifiction/prepare_data_for_labeling_infer.py


export CUDA_VISIBLE_DEVICES=1,2
python run_sequnce_labeling.py \
  --task_name=SKE_2019 \
  --do_predict=true \
  --data_dir=bin/subject_object_labeling/sequence_labeling_data \
  --vocab_file=pretrained_model/chinese_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=pretrained_model/chinese_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=output/sequnce_labeling_model/model.ckpt-79000 \
  --max_seq_length=320 \
  --output_dir=./output/sequnce_infer_out/ckpt79000
