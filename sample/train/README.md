# Training a new model

## IWSLT'14 German to English (Transformer)

The following instructions can be used to train a Transformer model on the [IWSLT'14 German to English dataset](http://workshop2014.iwslt.org/downloads/proceeding.pdf).

Step 1: Download and extract the data:

*We provide the bpe code for better reproducibility, the source and target vocabulary are shared with 10,000 merges.*

```bash
# Download and prepare the data
cd sample/train/
IWSLT_PATH=iwslt14.tokenized.de-en
tar -zxvf $IWSLT_PATH.tar.gz
IWSLT_PATH=sample/train/$IWSLT_PATH

# Binarize the data
cd ../..
python3 tools/PrepareParallelData.py \
  -src $IWSLT_PATH/train.de -tgt $IWSLT_PATH/train.en \
  -src_vocab $IWSLT_PATH/vocab.de -tgt_vocab $IWSLT_PATH/vocab.en \
  -output $IWSLT_PATH/train.data
python3 tools/PrepareParallelData.py \
  -src $IWSLT_PATH/valid.de -tgt $IWSLT_PATH/valid.en \
  -src_vocab $IWSLT_PATH/vocab.de -tgt_vocab $IWSLT_PATH/vocab.en \
  -output $IWSLT_PATH/valid.data
```
*You may extract the data manually on Windows.*


Step 2: Train the model with default configurations 
(6 encoder/decoder layer, 512 model size, 50 epoches):

```bash
bin/NiuTrans.NMT \
  -dev 0 \
  -nepoch 50 \
  -model model.bin \
  -maxcheckpoint 10 \
  -train $IWSLT_PATH/train.data \
  -valid $IWSLT_PATH/valid.data
```

Step 3: Average the last 10 checkpoints:

```bash
python tools/Ensemble.py -src 'model.bin*' -tgt model.ensemble
```

It costs about 310s per epoch on a GTX 1080 Ti.

Expected BLEU score (lenalpha=0.6, maxlenalpha=1.2):

| Model type      | Beam Search     | Greedy Search   |
| --------------- | --------------- | --------------- |
| Single model    | 33.94 (beam=4)  | 33.33    |
| Ensemble model  | 34.69 (beam=4)  | 34.07    |

We provide pretrained models using the default configurations:

[Google Drive](https://drive.google.com/file/d/1A5Z40lAWCxO54zJ2VFHQLT_AaI00J6bj) (Single model)

[Google Drive](https://drive.google.com/file/d/1NOafDYtlnYJFMop5PEhO6gICSUXN9hy9) (Ensemble model)