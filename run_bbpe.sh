# Full data experiments
python main.py -c experiment_params/bbpe.ini --experiment_name zn_bbpe --path_to_train_set data/MLDoc/mldoc_chinese_train.tsv --path_to_test_set data/MLDoc/mldoc_chinese_test.tsv --tokenizer_type chinese --language chinese --percentage_dev 0.1
python main.py -c experiment_params/bbpe.ini --experiment_name ja_bbpe --path_to_train_set data/MLDoc/mldoc_japanese_train.tsv --path_to_test_set  data/MLDoc/mldoc_japanese_test.tsv --tokenizer_type japanese --language japanese --percentage_dev 0.1
python main.py -c experiment_params/bbpe.ini --experiment_name de_bbpe --path_to_train_set data/MLDoc/mldoc_german_train.tsv --path_to_test_set  data/MLDoc/mldoc_german_test.tsv --tokenizer_type whitespace --language german --percentage_dev 0.1
python main.py -c experiment_params/bbpe.ini --experiment_name es_bbpe --path_to_train_set data/MLDoc/mldoc_spanish_train.tsv --path_to_test_set  data/MLDoc/mldoc_spanish_test.tsv --tokenizer_type whitespace --language german --percentage_dev 0.1
python main.py -c experiment_params/bbpe.ini --experiment_name fr_bbpe --path_to_train_set data/MLDoc/mldoc_french_train.tsv --path_to_test_set  data/MLDoc/mldoc_french_test.tsv --tokenizer_type whitespace --language french --percentage_dev 0.1
python main.py -c experiment_params/bbpe.ini --experiment_name it_bbpe --path_to_train_set data/MLDoc/mldoc_italian_train.tsv --path_to_test_set  data/MLDoc/mldoc_italian_test.tsv --tokenizer_type whitespace --language italian --percentage_dev 0.1
python main.py -c experiment_params/bbpe.ini --experiment_name ru_bbpe --path_to_train_set data/MLDoc/mldoc_russian_train.tsv --path_to_test_set  data/MLDoc/mldoc_russian_test.tsv --tokenizer_type whitespace --language russian --percentage_dev 0.1
#
## 100 samples
python main.py -c experiment_params/bbpe.ini --experiment_name zn_bbpe_100 --path_to_train_set data/MLDoc/mldoc_chinese_train.tsv --path_to_test_set data/MLDoc/mldoc_chinese_test.tsv --tokenizer_type chinese --language chinese --percentage_dev 0.9
python main.py -c experiment_params/bbpe.ini --experiment_name ja_bbpe_100 --path_to_train_set data/MLDoc/mldoc_japanese_train.tsv --path_to_test_set  data/MLDoc/mldoc_japanese_test.tsv --tokenizer_type japanese --language japanese --percentage_dev 0.9
python main.py -c experiment_params/bbpe.ini --experiment_name de_bbpe_100 --path_to_train_set data/MLDoc/mldoc_german_train.tsv --path_to_test_set  data/MLDoc/mldoc_german_test.tsv --tokenizer_type whitespace --language german --percentage_dev 0.9
python main.py -c experiment_params/bbpe.ini --experiment_name es_bbpe_100 --path_to_train_set data/MLDoc/mldoc_spanish_train.tsv --path_to_test_set  data/MLDoc/mldoc_spanish_test.tsv --tokenizer_type whitespace --language german --percentage_dev 0.9
python main.py -c experiment_params/bbpe.ini --experiment_name fr_bbpe_100 --path_to_train_set data/MLDoc/mldoc_french_train.tsv --path_to_test_set  data/MLDoc/mldoc_french_test.tsv --tokenizer_type whitespace --language french --percentage_dev 0.9
python main.py -c experiment_params/bbpe.ini --experiment_name it_bbpe_100 --path_to_train_set data/MLDoc/mldoc_italian_train.tsv --path_to_test_set  data/MLDoc/mldoc_italian_test.tsv --tokenizer_type whitespace --language italian --percentage_dev 0.9
python main.py -c experiment_params/bbpe.ini --experiment_name ru_bbpe_100 --path_to_train_set data/MLDoc/mldoc_russian_train.tsv --path_to_test_set  data/MLDoc/mldoc_russian_test.tsv --tokenizer_type whitespace --language russian --percentage_dev 0.9
#
#
## 50 samples
python main.py -c experiment_params/bbpe.ini --experiment_name zn_bbpe_50 --path_to_train_set data/MLDoc/mldoc_chinese_train.tsv --path_to_test_set data/MLDoc/mldoc_chinese_test.tsv --tokenizer_type chinese --language chinese --percentage_dev 0.95
python main.py -c experiment_params/bbpe.ini --experiment_name ja_bbpe_50 --path_to_train_set data/MLDoc/mldoc_japanese_train.tsv --path_to_test_set  data/MLDoc/mldoc_japanese_test.tsv --tokenizer_type japanese --language japanese --percentage_dev 0.95
python main.py -c experiment_params/bbpe.ini --experiment_name de_bbpe_50 --path_to_train_set data/MLDoc/mldoc_german_train.tsv --path_to_test_set  data/MLDoc/mldoc_german_test.tsv --tokenizer_type whitespace --language german --percentage_dev 0.95
python main.py -c experiment_params/bbpe.ini --experiment_name es_bbpe_50 --path_to_train_set data/MLDoc/mldoc_spanish_train.tsv --path_to_test_set  data/MLDoc/mldoc_spanish_test.tsv --tokenizer_type whitespace --language german --percentage_dev 0.95
python main.py -c experiment_params/bbpe.ini --experiment_name fr_bbpe_50 --path_to_train_set data/MLDoc/mldoc_french_train.tsv --path_to_test_set  data/MLDoc/mldoc_french_test.tsv --tokenizer_type whitespace --language french --percentage_dev 0.95
python main.py -c experiment_params/bbpe.ini --experiment_name it_bbpe_50 --path_to_train_set data/MLDoc/mldoc_italian_train.tsv --path_to_test_set  data/MLDoc/mldoc_italian_test.tsv --tokenizer_type whitespace --language italian --percentage_dev 0.95
python main.py -c experiment_params/bbpe.ini --experiment_name ru_bbpe_50 --path_to_train_set data/MLDoc/mldoc_russian_train.tsv --path_to_test_set  data/MLDoc/mldoc_russian_test.tsv --tokenizer_type whitespace --language russian --percentag0e_dev 0.95

# 20 samples
python main.py -c experiment_params/bbpe.ini --experiment_name zn_bbpe_unsup_20 --path_to_train_set data/MLDoc/mldoc_chinese_train.tsv --path_to_test_set data/MLDoc/mldoc_chinese_test.tsv --tokenizer_type chinese --language chinese --percentage_dev 0.98
python main.py -c experiment_params/bbpe.ini --experiment_name ja_bbpe_unsup_20 --path_to_train_set data/MLDoc/mldoc_japanese_train.tsv --path_to_test_set  data/MLDoc/mldoc_japanese_test.tsv --tokenizer_type japanese --language japanese --percentage_dev 0.98
python main.py -c experiment_params/bbpe.ini --experiment_name de_bbpe_unsup_20 --path_to_train_set data/MLDoc/mldoc_german_train.tsv --path_to_test_set  data/MLDoc/mldoc_german_test.tsv --tokenizer_type whitespace --language german --percentage_dev 0.98
python main.py -c experiment_params/bbpe.ini --experiment_name es_bbpe_unsup_20 --path_to_train_set data/MLDoc/mldoc_spanish_train.tsv --path_to_test_set  data/MLDoc/mldoc_spanish_test.tsv --tokenizer_type whitespace --language german --percentage_dev 0.98
python main.py -c experiment_params/bbpe.ini --experiment_name fr_bbpe_unsup_20 --path_to_train_set data/MLDoc/mldoc_french_train.tsv --path_to_test_set  data/MLDoc/mldoc_french_test.tsv --tokenizer_type whitespace --language french --percentage_dev 0.98
python main.py -c experiment_params/bbpe.ini --experiment_name it_bbpe_unsup_20 --path_to_train_set data/MLDoc/mldoc_italian_train.tsv --path_to_test_set  data/MLDoc/mldoc_italian_test.tsv --tokenizer_type whitespace --language italian --percentage_dev 0.98
python main.py -c experiment_params/bbpe.ini --experiment_name ru_bbpe_unsup_20 --path_to_train_set data/MLDoc/mldoc_russian_train.tsv --path_to_test_set  data/MLDoc/mldoc_russian_test.tsv --tokenizer_type whitespace --language russian --percentage_dev 0.98
