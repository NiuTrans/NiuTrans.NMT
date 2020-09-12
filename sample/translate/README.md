# Translating with pre-trained models

## IWSLT'14 En-De Models

The following instructions can be used to translate with a pre-trained Transformer model.

You can evaluate models trained in the [training example](../sample/train) by two steps.

Step 1: Translate the IWSLT14 De-En test set (tokenized) on the GPU:
```
IWSLT_PATH=sample/train/iwslt14.tokenized.de-en
bin/NiuTrans.NMT \
-dev 0 \
 -test $IWSLT_PATH/test.de \
 -model model.bin \
 -sbatch 64 \
 -beamsize 1  \
 -srcvocab $IWSLT_PATH/vocab.de \
 -tgtvocab $IWSLT_PATH/vocab.en \
 -output output.atat
sed -r 's/(@@ )|(@@ ?$)//g' < output.atat > output
```

You can also set `-dev -1` to use the CPU.

Step 2: Check the translation with [multi-bleu](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl):
```
perl multi-bleu.perl $IWSLT_PATH/test.en < output
```

It takes about 15s for translating test.de (6,750 sentences) on a GTX 1080 Ti with a greedy search.

## WNGT 2020 Models

The models here are the submissions to the [WNGT 2020 efficiency task](https://sites.google.com/view/wngt20/efficiency-task), which focuses on developing efficient MT systems.

The WNGT 2020 efficiency task constrains systems to translate 1 million sentences on CPUs and GPUs under the condition of the [WMT 2019 English-German news](http://statmt.org/wmt19/translation-task.html) translation task.

- For CPUs, the performance was measured on an [AWS c5.metal instance](https://aws.amazon.com/cn/blogs/aws/now-available-new-c5-instance-sizes-and-bare-metal-instances/) with 96 logical Cascade Lake processors and 192 GB memory. We submitted one system (9-1-tiny) running with all CPU cores.

- For GPUs, the performance was measured on an [AWS g4dn.xlarge instance](https://aws.amazon.com/cn/ec2/instance-types/g4/) with an NVIDIA T4 GPU and 16 GB memory. We submitted four systems (9-1, 18-1, 35-1, 35-6) running with FP16.

We list the results of all submissions. See [the official results](https://docs.google.com/spreadsheets/d/1M82S5wPSIM543Gh20d71Zs0FNHJQ3JdiJzDECiYJNlE/edit#gid=0) for more details.

| Model type | Time (s) | File size (MiB) | BLEU | Word per second |
| ---------- | -------- | --------------- | ---- | --------------- |
| 9-1-tiny*  | 810      | 66.8            | 27.0 | 18518 |           
| 9-1        | 977      | 99.3            | 31.1 | 15353 |
| 18-1       | 1355     | 156.1           | 31.4 | 11070 |
| 35-1       | 2023     | 263.3           | 32.0 | 7418  |
| 35-6       | 3166     | 305.4           | 32.2 | 4738  |


<em>* means run on CPUs. </em>

Description:

* `Model type` - Number of encoder and decoder layers, e.g., 9-1 means that the model consists of 9 encoder layers and 1 decoder layer. The model size is 512 except for the *tiny* model, whose size is 256.
* `Time` - Real time took for translating the whole test set, which contains about 1 million sentences with ~15 million tokens. The time of the `tiny` model was measured on CPUs, while other models were measured on GPUs.
* `File size` - All models are stored in FP16 except for the `tiny` model stored in FP32.
* `BLEU` - We report the averaged sacre BLEU score across wmt10 to wmt19, wmt12 is excluded. BLEU+case.mixed+lang.en-de+numrefs.1+smooth.exp+test.wmt10+tok.13a+version.1.4.9 (for wmt10, similar for others).


All these models and docker images are available at:

[Baidu Cloud](https://pan.baidu.com/s/1J8kRoF3d5P-XA4Qd3YT4ZQ) password: bdwp

[Google Drive](https://drive.google.com/file/d/1tgCUN8TnUsbcI7BCYFQkj30rCvk68YRb) (docker images only)