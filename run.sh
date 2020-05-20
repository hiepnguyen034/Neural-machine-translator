#!/bin/bash

if [ "$1" = "train" ]; then
	CUDA_VISIBLE_DEVICES=0 python run.py train --train-src=./en_es_data/train.es --train-tgt=./en_es_data/train.en --dev-src=./en_es_data/dev.es --dev-tgt=./en_es_data/dev.en --vocab=vocab.json --cuda
elif [ "$1" = "test" ]; then
        CUDA_VISIBLE_DEVICES=0 python run.py decode model.bin ./en_es_data/test.es ./en_es_data/test.en outputs/test_outputs.txt --cuda
elif [ "$1" = "train_local_cuda" ]; then
	python run.py train --train-src=./eng_vie_data/vie.txt --train-tgt=./eng_vie_data/eng.txt --dev-src=./eng_vie_data/vie_test.txt --dev-tgt=./eng_vie_data/eng_test.txt --vocab=eng_vn_vocab.json --cuda --batch-size=16
    python run.py decode model.bin ./eng_vie_data/vie_test.txt  ./eng_vie_data/eng_test.txt test_outputs.txt
elif [ "$1" = "vocab" ]; then	
	python vocab.py --train-src=./eng_vie_data/vie.txt --train-t	gt=./eng_vie_data/eng.txt eng_vn_vocab.json
else
	echo "Invalid Option Selected"
fi
