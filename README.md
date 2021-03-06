# NRMS-BERT

This repository serves the PyTorch implementation of non-official [NRMS](https://www.aclweb.org/anthology/D19-1671/) (Neural News Recommendation with Multi-Head Self-Attention) model using BERT. More precisely, it uses DistilBERT to save training time.

NRMS has shown dominant performance in [MIND dataset competition](https://msnews.github.io/index.html). This repository is a simple baseline without any handcrafted features.

| | MIND Large Val | Test |
| ---  | ---  | --- |
| AUC | 0.71146 | 0.7103 |

## Getting Started

Download MIND dataset. The directory structure will be like below.

```
# Only tsv files will be used in this repository.
$ find data/ -name "*.tsv"
data/mind-demo/train/behaviors.tsv
data/mind-demo/train/news.tsv
data/mind-demo/valid/behaviors.tsv
data/mind-demo/valid/news.tsv
data/mind-large/test/behaviors.tsv
data/mind-large/test/news.tsv
data/mind-large/train/behaviors.tsv
data/mind-large/train/news.tsv
data/mind-large/valid/behaviors.tsv
data/mind-large/valid/news.tsv
```

### Training

In test set, some behaviours do not have history. So, their candidates must be predicted from popularity etc.

```
$ cd src
$ python 001.train-nrms.py params/main/002.yaml
$ python 002.train-popularity.py params/popularity/002.yaml
```

## Credits

* [aqweteddy/NRMS-Pytorch](https://github.com/aqweteddy/NRMS-Pytorch)
