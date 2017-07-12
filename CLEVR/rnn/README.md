## LSTM Model on CLEVR Dataset

A tensorflow implementation of the LSTM model described [here](https://arxiv.org/pdf/1612.06890.pdf).

### To run:

The first time it runs, the CLEVR question files must be passed to the program and a tfrecords file will be generated from it. To run, type `python CLEVR_lstm.py --question_file [path to question_file]`. On subsequent runs, the `--question_file` is not needed. Run `python CLEVR_lstm.py -h` to view additional options.

To view the tensorboard output, run `tensorboard --logdir [path to log directory]`
