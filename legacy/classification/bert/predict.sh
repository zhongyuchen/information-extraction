export BASE_DIR=./
export BERT_BASE_DIR=${BASE_DIR}/chinese_L-12_H-768_A-12 # or multilingual_L-12_H-768_A-12
export DATA_DIR=${BASE_DIR}/data
export TEST_FILE=test1_data_postag.json
export INIT_DIR=${BASE_DIR}/model/model.ckpt-230810
export RESULT_DIR=${BASE_DIR}/result
export RESULT_FILE=dev_data.prob
export OUTPUT_FILE=dev_data.p

python run_classifier.py \
 --task_name=p_classification \
 --do_train=false \
 --do_eval=false \
 --do_predict=true \
 --data_dir=$DATA_DIR \
 --test_file=$TEST_FILE \
 --vocab_file=$BERT_BASE_DIR/vocab.txt \
 --bert_config_file=$BERT_BASE_DIR/bert_config.json \
 --init_checkpoint=$INIT_DIR \
 --max_seq_length=320 \
 --output_dir=$RESULT_DIR \
 --output_file=$RESULT_FILE

python prob2res.py \
  --output_dir=$RESULT_DIR \
  --prob_file=$RESULT_FILE \
  --output_file=$OUTPUT_FILE \
  --threshold=0.5

