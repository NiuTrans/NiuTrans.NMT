# NiuTrans.NMT

  - [Features](#features)
  - [Installation](#installation)
    - [Requirements](#requirements)
    - [Build from Source](#build-from-source)
      - [Compile on Linux](#compile-on-linux)
      - [Compile on Windows](#compile-on-windows)
      - [Compilation Example](#compilation-example)
  - [Usage](#usage)
    - [Training](#training)
      - [Commands](#commands)
      - [An Example](#an-example)
    - [Translating](#translating)
      - [Commands](#commands-1)
      - [An Example](#an-example-1)
  - [Converting Models from Fairseq](#converting-models-from-fairseq)
  - [A Model Zoo](#a-model-zoo)
  - [Papers](#papers)
  - [Team Members](#team-members)

## Features
NiuTrans.NMT is a lightweight and efficient Transformer-based neural machine translation system. Its main features are:
* Few dependencies. It is implemented with pure C++, and all dependencies are optional.
* Fast decoding. It supports various decoding acceleration strategies, such as batch pruning and dynamic batch size.
* Advanced NMT models, such as [Deep Transformer](https://www.aclweb.org/anthology/P19-1176) and [RPR attention](https://arxiv.org/abs/1803.02155).
* Flexible running modes. The system can be run on various systems and devices (Linux vs. Windows, CPUs vs. GPUs, and FP32 vs. FP16, etc.).
* Framework agnostic. It supports various models trained with other tools, e.g., fairseq models.
* The code is simple and friendly to beginners.

## Installation

### Requirements
OS: Linux or Windows

[Cmake](https://cmake.org/download/) >= 2.8

[CUDA](https://developer.nvidia.com/cuda-92-download-archive) >= 9.2 (optional)

[MKL](https://software.intel.com/content/www/us/en/develop/tools/math-kernel-library.html) latest version (optional)

[OpenBLAS](https://github.com/xianyi/OpenBLAS) latest version (optional)

<!-- [Visual Studio](https://visualstudio.microsoft.com/zh-hans/) >= vs 2015 (optional) -->

### Build from Source

Clone a copy from the web:

```
git clone https://github.com/NiuTrans/NiuTrans.NMT.git
mkdir nmt/build && cd nmt/build
cmake ..
```

NiuTrans.NMT supports acceleration with MKL, OpenBLAS or CUDA.
Please note that you can only select at most **one** of MKL or OpenBLAS.

*Use MKL (optional):*

Add ``-DUSE_MKL=ON`` and ``-DINTEL_ROOT=$MKL_PATH`` to the CMake command, where ``$MKL_PATH`` is the path of MKL.

Example: 
```
cmake -DUSE_MKL=ON -DINTEL_ROOT='/home/nlplab/intelcompilers_and_libraries_2020.2.254/linux' ..
```

*Use OpenBLAS (optional):*

Add ``-DUSE_OPENBLAS=ON`` and ``-DOPENBLAS_ROOT=$OPENBLAS_PATH`` to the CMake command, where ``$OPENBLAS_PATH`` is the path of OpenBLAS.

Example: 
```
cmake -DUSE_OPENBLAS=ON -DOPENBLAS_ROOT='/home/nlplab/openblas/' ..
```

*Use CUDA (required for training):*

Add ``-DUSE_CUDA=ON`` and ``-DCUDA_TOOLKIT_ROOT_DIR=$CUDA_PATH`` to the CMake command, where ``$CUDA_PATH`` is the path of CUDA toolkit.

You can also add ``-DUSE_FP16=ON`` to the CMake command to get half precision supported.

Example: 
```
cmake -DUSE_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR='/home/nlplab/cuda9.2/' -DUSE_FP16=ON ..
```

*Note that half precision requires Pascal or newer architectures on GPUs.*

#### Compile on Linux
```
make -j
```

#### Compile on Windows

Add ``-A 64`` to the CMake command.

It will generate a visual studio project on windows, i.e., ``NiuTrans.NMT.sln`` and you can open & build it with Visual Studio (>= Visual Studio 2015).

If the compilation succeeds, you will get an executable file **`NiuTrans.NMT`** in the 'bin' directory.

#### Compilation Example

Here is some compilation example on Linux with MKL, OpenBLAS, or CUDA supported. Replace the path in your environment.

*Compile with MKL supported:*
```
git clone https://github.com/NiuTrans/NiuTrans.NMT.git
mkdir nmt/build && cd nmt/build
cmake -DUSE_MKL=ON -DINTEL_ROOT='/home/nlplab/intel/compilers_and_libraries_2020.2.254/linux' ..
make -j
```

*Compile with OpenBLAS supported:*
```
git clone https://github.com/NiuTrans/NiuTrans.NMT.git
mkdir nmt/build && cd nmt/build
cmake -DUSE_OPENBLAS=ON -DOPENBLAS_ROOT='/home/nlplab/openblas/' ..
make -j
```

*Compile with CUDA supported:*
```
git clone https://github.com/NiuTrans/NiuTrans.NMT.git
mkdir nmt/build && cd nmt/build
cmake -DUSE_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR='/home/nlplab/cuda9.2/' ..
make -j
```

*Compile with CUDA and FP16 supported:*
```
git clone https://github.com/NiuTrans/NiuTrans.NMT.git
mkdir nmt/build && cd nmt/build
cmake -DUSE_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR='/home/nlplab/cuda9.2/' -DUSE_FP16=ON ..
make -j
```

## Usage

### Training

#### Commands

*Make sure compiling the program with CUDA because training on CPUs is not supported now.*

Step 1: Prepare the training data.

```command
python3 tools/PrepareParallelData.py \ 
-src $srcFile \
-tgt $tgtFile \
-src_vocab $srcVocab \
-tgt_vocab $tgtVocab \
-output $trainingFile \
```

Description:
* `src` - Path of the source language data. One sentence per line with tokens separated by spaces or tabs.
* `tgt` - Path of the target language data. The same format as the source language data.
* `sv` - Path of the source language vocabulary. Its first line is the vocabulary size, followed by a word and its index in each following line.
* `tv` - Path of the target language vocabulary. The same format as the source language vocabulary.
* `output` - Path of the training data to be saved. Each line consists of a source sentence and a target sentence, separated by `|||`.

You can load a [BPE](https://github.com/rsennrich/subword-nmt) vocabulary by running:
```
python3 tools/GetVocab.py -src $bpeVocab -tgt $niutransVocab
```

Description:
* `src` - Path of the BPE vocabulary.
* `tgt` - Path of the NiuTrans.NMT vocabulary to be saved.

Step 2: Train the model

```command
bin/NiuTrans.NMT \
-dev $deviceID \
-model $modelPath \
-train $trainingData \
-valid $validData \
```

Option:

* `dev` - Device id (>= 0 for GPUs). Default: 0.
* `model` - The model file required for training.
* `train` - Path to the training file. The same format as the output file in step 1.
* `valid` - Path to the validation file. The same format as the output file in step 1.
* `wbatch` - Word batch size. Default: 2048.
* `sbatch` - Sentence batch size. Default: 1.
* `mt` - Indicates whether the model runs for machine translation. Default: true.
* `dropout` - Dropout rate for the model. Default: 0.3.
* `fnndrop` - Dropout rate for fnn layers. Default: 0.1.
* `attdrop` - Dropout rate for attention layers. Default: 0.1.
* `lrate`- Learning rate. Default: 1.0F.
* `lrbias` - The parameter that controls the maximum learning rate in training. Default: 0.
* `nepoch` - Training epoch number. Default: 50.
* `nstep` - Traing step number. Default: 100000.
* `nwarmup` - Step number of warm-up for training. Default: 8000.
* `adam` - Indicates whether Adam is used. Default: true.
* `adambeta1` - Hyper parameters of Adam. Default: 0.9F.
* `adambeta2` - Hyper parameters of Adam. Default: 0.98F.
* `adambeta` - Hyper parameters of Adam. Default: 1e-8F.
* `shuffled` - Indicates whether the data file is shuffled for training. Default: true.
* `labelsmoothing` - Label smoothing factor. Default: 0.1.
* `nstepcheckpoint` - Number of steps after which we make a checkpoint. Default: -1.
* `epochcheckpoint` - Indicates whether we make a checkpoint after each training epoch. Default: false.
* `updatestep` - Number of batches that we collect for model update. Default: 1 (one can set > 1 for gradient accumulation).
* `debug` - Indicates whether we intend to debug the net. Default: false.
* `sorted` - Indicates whether the sequence is sorted by length. Default: false.
* `bufsize` - Buffer size for the batch loader. Default: 50000.
* `doubledend` - Indicates whether we double the </s> symbol for the output of LM. Default: false.
* `smallbatch` - Indicates whether we use batchsize = max *  sc
       rather rather than batchsize = word-number, where max is the maximum
       length and sc is the sentence number. Default: true.
* `bigbatch` - Counterpart of "isSmallBatch". Default: false.
* `randbatch` - Randomize batches. Default: false.
* `bucketsize` - Bucket size for the batch loader. Default: 50000.

#### An Example

Step 1: Download and extract the IWLST14 De-En data (tokenized, bpe vocabulary shared):

*We provide the bpe code for better reproducibility, the source and target vocabulary are shared and the size is about 10,000.*

```
IWSLT_PATH=sample/train/iwslt14.tokenized.de-en
tar -zxvf $IWSLT_PATH.tar.gz
python3 tools/PrepareParallelData.py \
  -src $IWSLT_PATH/train.de \
  -tgt $IWSLT_PATH/train.en \
  -src_vocab $IWSLT_PATH/vocab.de \
  -tgt_vocab $IWSLT_PATH/vocab.en \
  -output $IWSLT_PATH/train.data
python3 tools/PrepareParallelData.py \
  -src $IWSLT_PATH/valid.de \
  -tgt $IWSLT_PATH/valid.en \
  -src_vocab $IWSLT_PATH/vocab.de \
  -tgt_vocab $IWSLT_PATH/vocab.en \
  -output $IWSLT_PATH/valid.data
```
*You may extract the data manually on Windows.*


Step 2: Train the model with default configurations 
(6 encoder/decoder layer, 512 model size, 50 epoches):
```
bin/NiuTrans.NMT \
  -dev 0 \
  -model model.bin \
  -train $IWSLT_PATH/train.data \
  -valid $IWSLT_PATH/valid.data
```

It costs about 330s per epoch on a GTX 1080 Ti.

### Translating

*Make sure compiling the program with CUDA and FP16 if you want to translate with FP16 on GPUs.*

#### Commands

```command
bin/NiuTrans.NMT \
 -dev $deviceID \
 -test $inputFile \
 -model $modelPath \
 -sbatch $batchSize \
 -beamsize $beamSize \
 -srcvocab $srcVocab \
 -tgtvocab $tgtVocab \
 -output $outputFile
```
Description:

* `model` - Path of the model.
* `sbatch` - Sentence batch size. Default: 8.
* `dev` - Device id (-1 for CPUs, and >= 0 for GPUs). Default: 0.
* `beamsize` - Size of the beam. 1 for the greedy search.
* `test` - Path of the input file. One sentence per line with tokens separated by spaces.
* `output` - Path of the output file to be saved. The same format as the input file.
* `srcvocab` - Path of the source language vocabulary. Its first line is the vocabulary size, followed by a word and its index in each following line.
* `tgtvocab` - Path of the target language vocabulary. The same format as the source language vocabulary.
* `fp16 (optional)` - Inference with FP16. This will not work if the model is stored in FP32. Default: false.
* `lenalpha` - The alpha parameter controls the length preference. Default: 0.6.
* `maxlenalpha` - Scalar of the input sequence (for the max number of search steps). Default: 2.0.

#### An Example

You can evaluate models trained in the [training example](#an-example) by two steps.

Step 1: Translate the IWSLT14 De-En test set (tokenized) on the GPU:
```
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

You can also remove the `-dev 0` to use the CPU.

Step 2: Check the translation with [SacreBLEU](https://github.com/mjpost/sacrebleu):
```
cat output | sacrebleu $IWSLT_PATH/test.en
```

It takes about 15s for translating test.de (6,750 sentences) on a GTX 1080 Ti. The expected sacreBLEU score is about 31.4 (without detokenizing).


## Converting Models from Fairseq

The core implementation is framework agnostic, so we can easily convert models trained with other frameworks to a binary format for efficient inference. 

The following frameworks and models are currently supported:

|     | [fairseq (0.6.2)](https://github.com/pytorch/fairseq/tree/v0.6.2) |
| --- | :---: |
| Transformer ([Vaswani et al. 2017](https://arxiv.org/abs/1706.03762)) | ✓ |
| RPR attention ([Shaw et al. 2018](https://arxiv.org/abs/1803.02155)) | ✓ |
| Deep Transformer ([Wang et al. 2019](https://www.aclweb.org/anthology/P19-1176/)) | ✓ |

*Refer to [this page](https://fairseq.readthedocs.io/en/latest/getting_started.html#training-a-new-model) for the details about training models with fairseq.*

After training, you can convert the fairseq models and vocabulary with the following steps.

Step 1: Converting parameters of a single fairseq model
```command
python3 tools/ModelConverter.py -src $src -tgt $tgt
```
Description:

* `src` - Path of the fairseq checkpoint, [refer to this for more details](https://fairseq.readthedocs.io/en/latest/).
* `tgt` - Path to save the converted model parameters. All parameters are stored in a binary format.
* `fp16 (optional)` - Save the parameters with 16-bit data type. Default: disabled.

Step 2: Converting the vocabulary:
```command
python3 tools/VocabConverter.py -src $fairseqVocabPath -tgt $newVocabPath
```
Description:

* `src` - Path of the fairseq vocabulary, [refer to this for more details](https://fairseq.readthedocs.io/en/latest/).
* `tgt` - Path to save the converted vocabulary. Its first line is the vocabulary size, followed by a word and its index in each following line.

*You may need to convert both the source language vocabulary and the target language vocabulary if they are not shared.*

## A Model Zoo

Several models (for WNGT 2020) are provided to test the system. All models and runnable systems are packaged into docker files so that one can easily reproduce our result. The models here are the submissions to the [WNGT 2020 efficiency task](https://sites.google.com/view/wngt20/efficiency-task), which focuses on developing efficient MT systems.

The WNGT 2020 efficiency task constrains systems to translate 1 million sentences on CPUs and GPUs under the condition of the [WMT 2019 English-German news](http://statmt.org/wmt19/translation-task.html) translation task.

- For CPUs, the performance was measured on an [AWS c5.metal instance](https://aws.amazon.com/cn/blogs/aws/now-available-new-c5-instance-sizes-and-bare-metal-instances/) with 96 logical Cascade Lake processors and 192 GB memory. We submitted one system (9-1-tiny) running with all CPU cores.

- For GPUs, the performance was measured on an [AWS g4dn.xlarge instance](https://aws.amazon.com/cn/ec2/instance-types/g4/) with an NVIDIA T4 GPU and 16 GB memory. We submitted four systems (9-1, 18-1, 35-1, 35-6) running with FP16.

We list the results of all submissions, see [the official results](https://docs.google.com/spreadsheets/d/1M82S5wPSIM543Gh20d71Zs0FNHJQ3JdiJzDECiYJNlE/edit#gid=0) for more details.

| Model type | Time (s) | File size (MiB) | BLEU | Word per second |
| ---------- | -------- | --------------- | ---- | --------------- |
| 9-1-tiny*  | 810      | 66.8            | 27.0 | 18518
| 9-1        | 977      | 99.3            | 31.1 | 15353
| 18-1       | 1355     | 156.1           | 31.4 | 11070
| 35-1       | 2023     | 263.3           | 32.0 | 7418
| 35-6       | 3166     | 305.4           | 32.2 | 4738


<em>* means run on CPUs. </em>

Description:

* `Model type` - Number of encoder and decoder layers, e.g., 9-1 means that the model consists of 9 encoder layers and 1 decoder layer. The model size is 512 except for the *tiny* model, whose size is 256.
* `Time` - Real time took for translating the whole test set, which contains about 1 million sentences with ~15 million tokens. The time of the `tiny` model was measured on CPUs, while other models were measured on GPUs.
* `File size` - All models are stored in FP16 except for the `tiny` model stored in FP32.
* `BLEU` - We report the averaged sacre BLEU score across wmt10 to wmt19, wmt12 is excluded. BLEU+case.mixed+lang.en-de+numrefs.1+smooth.exp+test.wmt10+tok.13a+version.1.4.9 (for wmt10, similar for others).


All these models and docker images are available at:

[Baidu Cloud](https://pan.baidu.com/s/1J8kRoF3d5P-XA4Qd3YT4ZQ) password: bdwp

[Google Drive](https://drive.google.com/file/d/1tgCUN8TnUsbcI7BCYFQkj30rCvk68YRb) (docker images only)

## Papers

Here are the papers related to this project

[Deep Transformer Models for Machine Translation.](https://www.aclweb.org/anthology/P19-1176) Qiang Wang, Bei Li, Tong Xiao, Jingbo Zhu, Changliang Li, Derek F. Wong, Lidia S. Chao. 2019. Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics.

[The NiuTrans System for WNGT 2020 Efficiency Task.](https://www.aclweb.org/anthology/2020.ngt-1.24)  Chi Hu, Bei Li, Yinqiao Li, Ye Lin, Yanyang Li, Chenglong Wang, Tong Xiao, Jingbo Zhu. 2020. Proceedings of the Fourth Workshop on Neural Generation and Translation.

## Team Members

This project is maintained by a joint team from NiuTrans Research and NEU NLP Lab. Current team members are

*Chi Hu, Bei Li, Yinqiao Li, Ye Lin, Yanyang Li, Quan Du, Tong Xiao and Jingbo Zhu*

Please contact niutrans@mail.neu.edu.cn if you have any questions.

