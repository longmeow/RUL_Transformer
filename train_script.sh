#!/bin/bash

CONFIG_DIR="configs"

for STR in $CONFIG_DIR/*.yml; do
	substring=true;
	for SUB in "$@"; do
		if [[ "$STR" != *"$SUB"* ]]; then
			substring=false;
		fi;
	done;
	if [ "$substring" == true ]; then
		echo "Training with config $STR";
		python train.py --config $STR;
		if [ $? -eq 0 ]; then
			echo "Testing with config $STR";
			python inference.py --config $STR;
			if [ $? -eq 0 ]; then
				mv -v $STR $CONFIG_DIR/done;
			fi;
            printf "\n";
			printf '=%.0s' {1..100};
            printf "\n";
		fi;
	fi;
done
