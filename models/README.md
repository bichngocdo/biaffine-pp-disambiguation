# Models

All models are trained on the train and dev sets from the dataset `spmrl-pp-gold-obj-jsonl`
and tested on the test set of the dataset `spmrl-pp-pred-obj-jsonl`.

| Model                     | BERT | Topological Field | Auxiliary Distributions |
|---------------------------|:----:|:-----------------:|:-----------------------:|
| pp-biaffine               |      |                   |                         |
| pp-biaffine-topo          |      |         X         |                         |
| pp-biaffine-topo-aux      |      |         X         |            X            |
| pp-biaffine-bert          |  X   |                   |                         |
| pp-biaffine-bert-topo     |  X   |         X         |                         |
| pp-biaffine-bert-topo-aux |  X   |         X         |            X            |
