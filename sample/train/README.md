# Training a new model

## IWSLT'14 German to English (Transformer)

The following instructions can train a Transformer model on the [IWSLT'14 German to English dataset](http://workshop2014.iwslt.org/downloads/proceeding.pdf).

Step 1: Prepare the training data:

*We provide the BPE code for better reproducibility. The source and target vocabulary are shared with 10,000 merges.*

```bash
# Extract the data
cd sample/train/
IWSLT_PATH=iwslt14.tokenized.de-en
tar -zxvf $IWSLT_PATH.tar.gz
IWSLT_PATH=sample/train/$IWSLT_PATH

# Binarize the data
cd ../..
python3 tools/GetVocab.py \
  -raw $IWSLT_PATH/bpevocab \
  -new $IWSLT_PATH/vocab.de
python3 tools/GetVocab.py \
  -raw $IWSLT_PATH/bpevocab \
  -new $IWSLT_PATH/vocab.en
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

Step 3: Average the last ten checkpoints:

```bash
python tools/Ensemble.py -input 'model.bin.*' -output model.ensemble
```

It costs about 280 second per epoch on a GTX 1080 Ti.

Expected BLEU score (lenalpha=0.6, maxlenalpha=1.2):

| Model type      | Beam Search     | Greedy Search   |
| --------------- | --------------- | --------------- |
| Single model    | 34.05 (beam=4)  | 33.35    |
| Ensemble model  | 34.48 (beam=4)  | 34.01    |

We provide models trained with the default configurations:

[Google Drive](https://drive.google.com/drive/folders/10W89cx60Q7A9nGyg5fwLP21Sg53n6NXV?usp=sharing)

[Baidu Cloud](https://pan.baidu.com/s/1LbkV8kuaDWNunVR2jwOhRg) (password: bdwp)