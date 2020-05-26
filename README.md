# Vietnamese - English NMT
You can use your own pre-trained word-embeddings or use

(i) GloVe embeddings (download [here](https://nlp.stanford.edu/projects/glove/))

(ii) Vietnamese wiki embeddings (download [here](https://drive.google.com/file/d/0B1GKSX6YCHXlakkzQ2plZVdUUE0/view))

use `python vocab.py --train-src=[vietnamese texts] --train-tgt =[english texts] [output_file]` to generate vocab dictionaries as json

use `sh run.sh train_local_cuda` to train using pre-configured settings on the given toy dataset, or

use `python run.py train --train-src =[vietnamese training data] --train-tgt =[english training data] --dev-src =[vietnamese dev data] --dev-tgt =[english dev data] --vocab = [json vocab file] ` (Optional: include `--cuda` to train on GPU)

use `python run.py decode [model_path] [Vietnamese text file] [English text file] [output_file]` to perform prediction and validation

use `python run.py translate [model_path] [ input vietnamese text file]` to translate 

This implementation is based on the starter code given by Stanford cs224n's assignment 4

