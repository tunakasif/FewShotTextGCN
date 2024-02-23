#!/usr/bin/env bash
# Check if exactly one argument is provided
if [ "$#" -ne 1 ]; then
	echo "Usage: $0 {r8|r52|mr|ohsumed|20ng|cola|sst2}"
	exit 1
fi

# Check if the provided argument is one of the pre-defined values
case "$1" in
r8 | r52 | mr | ohsumed | 20ng | cola | sst2)
	echo "Valid argument: $1"
	dataset=$1
	dev_splits=(0.01 0.80 0.90 0.95 0.99)
	for item in "${dev_splits[@]}"; do
		python main.py -c experiment_params/self_training.ini --experiment_name "$dataset" --path_to_train_set data/"$dataset"-train.tsv --path_to_test_set data/"$dataset"-test.tsv --tokenizer_type whitespace --language english --percentage_dev "$item" --num_seeds 1 --max_epochs 1000
	done
	;;
*)
	echo "Invalid argument: $1. Please provide one of the following values: r8, r52, mr, ohsumed, 20ng, cola, sst2."
	exit 1
	;;
esac
