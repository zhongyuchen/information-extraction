export CUDA_VISIBLE_DEVICES=2

export BASE_DIR=./
export BERT_BASE_DIR=${BASE_DIR}/chinese_L-12_H-768_A-12 # or multilingual_L-12_H-768_A-12
export DATA_DIR=${BASE_DIR}/data
export MODEL_DIR=${BASE_DIR}/model
export INIT_DIR=${BERT_BASE_DIR}/bert_model.ckpt # ${MODEL_DIR}/model.ckpt-134405

python run_classifier.py \
  --task_name=p_classification \
  --do_train=true \
  --do_eval=false \
  --data_dir=$DATA_DIR \
  --train_file=train_data.json \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$INIT_DIR \
  --max_seq_length=320 \
  --train_batch_size=3 \
  --learning_rate=5e-5 \
  --num_train_epochs=2.0 \
  --output_dir=$MODEL_DIR

# num_train_epochs=0.1 for debugging, 2.0 for fine-tuning

