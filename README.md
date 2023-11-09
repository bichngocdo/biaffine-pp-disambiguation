# PP Attachment Disambiguation with Biaffine Attention and BERT Embeddings

This repository contains code for the neural PP Attachment disambiguation system (`PP-BIAFFINE`) from the paper
[Parsers Know Best: German PP Attachment Revisited](https://aclanthology.org/2020.coling-main.185/)
published at COLING 2020.

## Usage

### Requirements
* Python 3.6
* Clone the `bert` repository:
  ```shell
  git clone https://github.com/google-research/bert.git
  ```
* Install dependencies in [requirements.txt](requirements.txt)

### Training
Run:
```shell
PYTHONPATH=`pwd` python pp/model/experiment.py train --help
```
to see possible arguments.

For example, to train a model on the sample dataset, run:

```shell
PYTHONPATH=`pwd` python pp/model/experiment.py train \
  --train_file data/sample/train.jsonl \
  --dev_file data/sample/dev.jsonl \
  --test_file data/sample/test.jsonl \
  --model_dir runs/sample \
  --word_dim 5 \
  --tag_dim 4 \
  --topological_field_dim 3 \
  --hidden_dim 7 \
  --num_lstms 2 \
  --num_mlps 1 \
  --mlp_dim 8 \
  --clf bilinear \
  --use_scores True \
  --interval 1 \
  --num_train_buckets 2 \
  --num_dev_buckets 2 \
  --max_epoch 1
```

### Evaluation
Run:
```shell
PYTHONPATH=`pwd` python pp/model/experiment.py eval --help
```
to see possible arguments.

For example, to evaluate a trained model on the sample dataset, run:
```shell
PYTHONPATH=`pwd` python pp/model/experiment.py eval \
  --test_file data/sample/test.jsonl \
  --output_file runs/sample/result.jsonl \
  --model_dir runs/sample
```

### Tensorboard

```shell
tensorboard --logdir runs/sample/log
```

### Scripts

* [reattach.py](scripts%2Freattach.py): reattach PP attachment disambiguation result
  of the system to the parser's output
* [pp_eval_conll09.py](scripts%2Fpp_eval_conll09.py): evaluate PP attachment disambiguation results
  with CoNLL format


## Reproduction

All trained models contain:
* File `config.cfg` that records all parameters used to produce the model.
* Folder `log` records training and evaluation metrics, which can be viewed by `tensorboard`.
* First, reattach the disambiguation result to the predicted parse trees.
  Then evaluate the disambiguation/reattachment result using the script
  [pp_eval_conll09.py](scripts%2Fpp_eval_conll09.py).
* See more information at [data](data) and [models](models).


## Citation

```bib
@inproceedings{do-rehbein-2020-parsers,
    title = "Parsers Know Best: {G}erman {PP} Attachment Revisited",
    author = "Do, Bich-Ngoc and Rehbein, Ines",
    editor = "Scott, Donia and Bel, Nuria and Zong, Chengqing",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2020.coling-main.185",
    doi = "10.18653/v1/2020.coling-main.185",
    pages = "2049--2061",
}
```