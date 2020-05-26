#!/bin/bash

elif [ "$1" = "train_local_cuda" ]; then
	python run.py train --train-src=./eng_vie_data/vie.txt --train-tgt=./eng_vie_data/eng.txt --dev-src=./eng_vie_data/vie.txt --dev-tgt=./eng_vie_data/eng.txt --vocab=eng_vn_vocab.json --cuda --batch-size=16
    python run.py decode model.bin ./eng_vie_data/vie_test.txt  ./eng_vie_data/eng_test.txt test_outputs.txt

elif [ "$1" = "vocab" ]; then	
	python vocab.py --train-src=./eng_vie_data/vie.txt --train-tgt=./eng_vie_data/eng.txt eng_vn_vocab.json

else
	echo "Invalid Option Selected"
fi
