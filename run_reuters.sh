########################################
# R8
########################################

# Full data
#python main.py -c experiment_params/gcn_full.ini --experiment_name r8_full_gcn --path_to_train_set data/r8-train-all-terms.txt --path_to_test_set data/r8-test-all-terms.txt --tokenizer_type whitespace --language english --percentage_dev 0.1
python main.py -c experiment_params/self_training.ini  --experiment_name r8_full_self_training --path_to_train_set data/r8-train-all-terms.txt --path_to_test_set data/r8-train-all-terms.txt --tokenizer_type whitespace --language english --percentage_dev 0.1
#python main.py -c experiment_params/unsup.ini  --experiment_name r8_full_unsup --path_to_train_set data/r8-train-all-terms.txt --path_to_test_set data/r8-train-all-terms.txt --tokenizer_type whitespace --language english --percentage_dev 0.1
#python main.py -c experiment_params/bbpe.ini  --experiment_name r8_full_bbpe --path_to_train_set data/r8-train-all-terms.txt --path_to_test_set data/r8-train-all-terms.txt --tokenizer_type whitespace --language english --percentage_dev 0.1
python main.py -c experiment_params/textgcnII.ini --experiment_name r8_90_textgcnII --path_to_train_set data/r8-train-all-terms.txt --path_to_test_set data/r8-test-all-terms.txt --tokenizer_type whitespace --language english --percentage_dev 0.1
#
## 0.99% dev
#python main.py -c experiment_params/gcn_full.ini --experiment_name r8_01_gcn --path_to_train_set data/r8-train-all-terms.txt --path_to_test_set data/r8-test-all-terms.txt --tokenizer_type whitespace --language english --percentage_dev 0.99
python main.py -c experiment_params/self_training.ini  --experiment_name r8_01_self_training --path_to_train_set data/r8-train-all-terms.txt --path_to_test_set data/r8-test-all-terms.txt --tokenizer_type whitespace --language english --percentage_dev 0.99
#python main.py -c experiment_params/unsup.ini  --experiment_name r8_01_unsup --path_to_train_set data/r8-train-all-terms.txt --path_to_test_set data/r8-test-all-terms.txt --tokenizer_type whitespace --language english --percentage_dev 0.99
#python main.py -c experiment_params/bbpe.ini  --experiment_name r8_01_bbpe --path_to_train_set data/r8-train-all-terms.txt --path_to_test_set data/r8-test-all-terms.txt --tokenizer_type whitespace --language english --percentage_dev 0.99
python main.py -c experiment_params/textgcnII.ini --experiment_name r8_01_textgcnII --path_to_train_set data/r8-train-all-terms.txt --path_to_test_set data/r8-test-all-terms.txt --tokenizer_type whitespace --language english --percentage_dev 0.99
##
## 98% dev
#python main.py -c experiment_params/gcn_full.ini --experiment_name r8_02_gcn --path_to_train_set data/r8-train-all-terms.txt --path_to_test_set data/r8-test-all-terms.txt --tokenizer_type whitespace --language english --percentage_dev 0.98
python main.py -c experiment_params/self_training.ini  --experiment_name r8_02_self_training --path_to_train_set data/r8-train-all-terms.txt --path_to_test_set data/r8-test-all-terms.txt --tokenizer_type whitespace --language english --percentage_dev 0.98
#python main.py -c experiment_params/unsup.ini  --experiment_name r8_02_unsup --path_to_train_set data/r8-train-all-terms.txt --path_to_test_set data/r8-test-all-terms.txt --tokenizer_type whitespace --language english --percentage_dev 0.98
#python main.py -c experiment_params/bbpe.ini  --experiment_name r8_02_bbpe --path_to_train_set data/r8-train-all-terms.txt --path_to_test_set data/r8-test-all-terms.txt --tokenizer_type whitespace --language english --percentage_dev 0.98
python main.py -c experiment_params/textgcnII.ini --experiment_name r8_02_textgcnII --path_to_train_set data/r8-train-all-terms.txt --path_to_test_set data/r8-test-all-terms.txt --tokenizer_type whitespace --language english --percentage_dev 0.98

## 95% dev
#python main.py -c experiment_params/gcn_full.ini --experiment_name r8_05_gcn --path_to_train_set data/r8-train-all-terms.txt --path_to_test_set data/r8-test-all-terms.txt --tokenizer_type whitespace --language english --percentage_dev 0.95
python main.py -c experiment_params/self_training.ini  --experiment_name r8_05_self_training --path_to_train_set data/r8-train-all-terms.txt --path_to_test_set data/r8-test-all-terms.txt --tokenizer_type whitespace --language english --percentage_dev 0.95
#python main.py -c experiment_params/unsup.ini  --experiment_name r8_05_unsup --path_to_train_set data/r8-train-all-terms.txt --path_to_test_set data/r8-test-all-terms.txt --tokenizer_type whitespace --language english --percentage_dev 0.95
#python main.py -c experiment_params/bbpe.ini  --experiment_name r8_05_bbpe --path_to_train_set data/r8-train-all-terms.txt --path_to_test_set data/r8-train-all-terms.txt --tokenizer_type whitespace --language english --percentage_dev 0.95
python main.py -c experiment_params/textgcnII.ini --experiment_name r8_05_textgcnII --path_to_train_set data/r8-train-all-terms.txt --path_to_test_set data/r8-test-all-terms.txt --tokenizer_type whitespace --language english --percentage_dev 0.95

## 90% dev
#python main.py -c experiment_params/gcn_full.ini --experiment_name r8_10_gcn --path_to_train_set data/r8-train-all-terms.txt --path_to_test_set data/r8-test-all-terms.txt --tokenizer_type whitespace --language english --percentage_dev 0.9
python main.py -c experiment_params/self_training.ini  --experiment_name r8_10_self_training --path_to_train_set data/r8-train-all-terms.txt --path_to_test_set data/r8-test-all-terms.txt --tokenizer_type whitespace --language english --percentage_dev 0.9
#python main.py -c experiment_params/unsup.ini  --experiment_name r8_10_unsup --path_to_train_set data/r8-train-all-terms.txt --path_to_test_set data/r8-test-all-terms.txt --tokenizer_type whitespace --language english --percentage_dev 0.9
#python main.py -c experiment_params/bbpe.ini  --experiment_name r8_10_bbpe --path_to_train_set data/r8-train-all-terms.txt --path_to_test_set data/r8-test-all-terms.txt --tokenizer_type whitespace --language english --percentage_dev 0.9
python main.py -c experiment_params/textgcnII.ini --experiment_name r8_10_textgcnII --path_to_train_set data/r8-train-all-terms.txt --path_to_test_set data/r8-test-all-terms.txt --tokenizer_type whitespace --language english --percentage_dev 0.90

##########################################
# R52
##########################################
# Full data
python main.py -c experiment_params/gcn_full.ini --experiment_name r52_full_gcn --path_to_train_set data/r52-train-all-terms.txt --path_to_test_set data/r52-test-all-terms.txt --tokenizer_type whitespace --language english --percentage_dev 0.1
python main.py -c experiment_params/self_training.ini  --experiment_name r52_full_self_training --path_to_train_set data/r52-train-all-terms.txt --path_to_test_set data/r52-train-all-terms.txt --tokenizer_type whitespace --language english --percentage_dev 0.1
python main.py -c experiment_params/unsup.ini  --experiment_name r52_full_unsup --path_to_train_set data/r52-train-all-terms.txt --path_to_test_set data/r52-train-all-terms.txt --tokenizer_type whitespace --language english --percentage_dev 0.1
python main.py -c experiment_params/bbpe.ini  --experiment_name r52_full_bbpe --path_to_train_set data/r52-train-all-terms.txt --path_to_test_set data/r52-train-all-terms.txt --tokenizer_type whitespace --language english --percentage_dev 0.1
python main.py -c experiment_params/textgcnII.ini --experiment_name r52_90_textgcnII --path_to_train_set data/r52-train-all-terms.txt --path_to_test_set data/r52-test-all-terms.txt --tokenizer_type whitespace --language english --percentage_dev 0.1
#
## 0.99% dev
python main.py -c experiment_params/gcn_full.ini --experiment_name r52_01_gcn --path_to_train_set data/r52-train-all-terms.txt --path_to_test_set data/r52-test-all-terms.txt --tokenizer_type whitespace --language english --percentage_dev 0.99
python main.py -c experiment_params/self_training.ini  --experiment_name r52_01_self_training --path_to_train_set data/r52-train-all-terms.txt --path_to_test_set data/r52-test-all-terms.txt --tokenizer_type whitespace --language english --percentage_dev 0.99
python main.py -c experiment_params/unsup.ini  --experiment_name r52_01_unsup --path_to_train_set data/r52-train-all-terms.txt --path_to_test_set data/r52-test-all-terms.txt --tokenizer_type whitespace --language english --percentage_dev 0.99
python main.py -c experiment_params/bbpe.ini  --experiment_name r52_01_bbpe --path_to_train_set data/r52-train-all-terms.txt --path_to_test_set data/r52-test-all-terms.txt --tokenizer_type whitespace --language english --percentage_dev 0.99
python main.py -c experiment_params/textgcnII.ini --experiment_name r52_01_textgcnII --path_to_train_set data/r52-train-all-terms.txt --path_to_test_set data/r52-test-all-terms.txt --tokenizer_type whitespace --language english --percentage_dev 0.99
#
## 98% dev
python main.py -c experiment_params/gcn_full.ini --experiment_name r52_02_gcn --path_to_train_set data/r52-train-all-terms.txt --path_to_test_set data/r52-test-all-terms.txt --tokenizer_type whitespace --language english --percentage_dev 0.98
python main.py -c experiment_params/self_training.ini  --experiment_name r52_02_self_training --path_to_train_set data/r52-train-all-terms.txt --path_to_test_set data/r52-test-all-terms.txt --tokenizer_type whitespace --language english --percentage_dev 0.98
python main.py -c experiment_params/unsup.ini  --experiment_name r52_02_unsup --path_to_train_set data/r52-train-all-terms.txt --path_to_test_set data/r52-test-all-terms.txt --tokenizer_type whitespace --language english --percentage_dev 0.98
python main.py -c experiment_params/bbpe.ini  --experiment_name r52_02_bbpe --path_to_train_set data/r52-train-all-terms.txt --path_to_test_set data/r52-test-all-terms.txt --tokenizer_type whitespace --language english --percentage_dev 0.98
python main.py -c experiment_params/textgcnII.ini --experiment_name r52_02_textgcnII --path_to_train_set data/r52-train-all-terms.txt --path_to_test_set data/r52-test-all-terms.txt --tokenizer_type whitespace --language english --percentage_dev 0.98
#
## 95% dev
python main.py -c experiment_params/gcn_full.ini --experiment_name r52_05_gcn --path_to_train_set data/r52-train-all-terms.txt --path_to_test_set data/r52-test-all-terms.txt --tokenizer_type whitespace --language english --percentage_dev 0.95
python main.py -c experiment_params/self_training.ini  --experiment_name r52_05_self_training --path_to_train_set data/r52-train-all-terms.txt --path_to_test_set data/r52-test-all-terms.txt --tokenizer_type whitespace --language english --percentage_dev 0.95
python main.py -c experiment_params/unsup.ini  --experiment_name r52_05_unsup --path_to_train_set data/r52-train-all-terms.txt --path_to_test_set data/r52-test-all-terms.txt --tokenizer_type whitespace --language english --percentage_dev 0.95
python main.py -c experiment_params/bbpe.ini  --experiment_name r52_05_bbpe --path_to_train_set data/r52-train-all-terms.txt --path_to_test_set data/r52-train-all-terms.txt --tokenizer_type whitespace --language english --percentage_dev 0.95
python main.py -c experiment_params/textgcnII.ini --experiment_name r52_05_textgcnII --path_to_train_set data/r52-train-all-terms.txt --path_to_test_set data/r52-test-all-terms.txt --tokenizer_type whitespace --language english --percentage_dev 0.95
# 90% dev
python main.py -c experiment_params/gcn_full.ini  --experiment_name r52_10_gcn --path_to_train_set data/r52-train-all-terms.txt --path_to_test_set data/r52-test-all-terms.txt --tokenizer_type whitespace --language english --percentage_dev 0.9
python main.py -c experiment_params/self_training.ini  --experiment_name r52_10_self_training --path_to_train_set data/r52-train-all-terms.txt --path_to_test_set data/r52-test-all-terms.txt --tokenizer_type whitespace --language english --percentage_dev 0.9
python main.py -c experiment_params/unsup.ini  --experiment_name r52_10_unsup --path_to_train_set data/r52-train-all-terms.txt --path_to_test_set data/r52-test-all-terms.txt --tokenizer_type whitespace --language english --percentage_dev 0.9
python main.py -c experiment_params/bbpe.ini  --experiment_name r52_10_bbpe --path_to_train_set data/r52-train-all-terms.txt --path_to_test_set data/r52-test-all-terms.txt --tokenizer_type whitespace --language english --percentage_dev 0.9
python main.py -c experiment_params/textgcnII.ini --experiment_name r52_10_textgcnII --path_to_train_set data/r52-train-all-terms.txt --path_to_test_set data/r52-test-all-terms.txt --tokenizer_type whitespace --language english --percentage_dev 0.90


