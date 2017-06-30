## Word2Vec on CLEVR questions

Trains word2vec on CLEVR questions and answers

### To run:

```
usage: CLEVR_word2vec.py [-h] --data_file DATA_FILE [--log_dir LOG_DIR]

required arguments:
  --data_file DATA_FILE     The file with question data

optional arguments:
  -h, --help                show this help message and exit
  --log_dir LOG_DIR         Summaries log directory
```


### To view embeddings:
In another terminal, run `tensorboard --logdir LOG_DIR` and go to localhost:6006 in your browser. To show the labels, click on the 'Load data' and upload metadata.tsv, which is in LOG_DIR