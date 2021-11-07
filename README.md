# NiuTrans.NMT

- [NiuTrans.NMT](#niutransnmt)
  - [Features](#features)
  - [Recent Updates](#recent-updates)
  - [Installation](#installation)
    - [Requirements](#requirements)
    - [Build from Source](#build-from-source)
      - [Configure with cmake](#configure-with-cmake)
      - [Configuration Example](#configuration-example)
      - [Compile on Linux](#compile-on-linux)
      - [Compile on Windows](#compile-on-windows)
  - [Usage](#usage)
    - [Training](#training)
      - [Commands](#commands)
      - [An Example](#an-example)
    - [Translating](#translating)
      - [Commands](#commands-1)
      - [An Example](#an-example-1)
  - [Low Precision Inference](#low-precision-inference)
  - [Converting Models from Fairseq](#converting-models-from-fairseq)
  - [A Model Zoo](#a-model-zoo)
  - [Papers](#papers)
  - [Team Members](#team-members)

## Features
NiuTrans.NMT is a lightweight and efficient Transformer-based neural machine translation system. [中文介绍](./README_zh.md)


Its main features are:
* Few dependencies. It is implemented with pure C++, and all dependencies are optional.
* High efficiency. It is heavily optimized for fast decoding, see [our WMT paper](https://arxiv.org/pdf/2109.08003.pdf) for more details.
* Flexible running modes. The system can run with various systems and devices (Linux vs. Windows, CPUs vs. GPUs, and FP32 vs. FP16, etc.).
* Framework agnostic. It supports various models trained with other tools, e.g., fairseq models.

## Recent Updates
November 2021: Released the code of our submissions to the [WMT21 efficiency task](http://statmt.org/wmt21/efficiency-task.html). We speed up the inference by 3 times on the GPU (up to 250k words/s with an NVIDIA A100)!

December 2020: Added support for the training of [DLCL](https://arxiv.org/abs/1906.01787) and [RPR Attention](https://arxiv.org/abs/1803.02155)

December 2020: Heavily reduced the memory footprint of training by optimizing the backward functions

## Installation

### Requirements
* OS: Linux or Windows

* [GCC/G++](https://gcc.gnu.org/) >=4.8.5 (on Linux)

* [VC++](https://www.microsoft.com/en-us/download/details.aspx?id=48145) >=2015 (on Windows)

* [cmake](https://cmake.org/download/) >= 3.5

* [CUDA](https://developer.nvidia.com/cuda-92-download-archive) >= 10.2 (optional)

* [MKL](https://software.intel.com/content/www/us/en/develop/tools/math-kernel-library.html) latest version (optional)

* [OpenBLAS](https://github.com/xianyi/OpenBLAS) latest version (optional)


### Build from Source

#### Configure with cmake

The default configuration enables compiling for the **pure CPU** version.

```bash
# Download the code
git clone https://github.com/NiuTrans/NiuTrans.NMT.git
git clone https://github.com/NiuTrans/NiuTensor.git
# Merge with NiuTrans.Tensor
mv NiuTensor/source NiuTrans.NMT/source/niutensor
rm NiuTrans.NMT/source/niutensor/Main.cpp
rm -rf NiuTrans.NMT/source/niutensor/sample NiuTrans.NMT/source/niutensor/tensor/test
mkdir NiuTrans.NMT/build && cd NiuTrans.NMT/build
# Run cmake
cmake ..
```

You can add compilation options to the cmake command to support accelerations with MKL, OpenBLAS, or CUDA.

*Please note that you can only select at most one of MKL or OpenBLAS.*

* Use CUDA (required for training)

  Add ``-DUSE_CUDA=ON``, ``-DCUDA_TOOLKIT_ROOT=$CUDA_PATH`` and ``DGPU_ARCH=$GPU_ARCH`` to the cmake command, where ``$CUDA_PATH`` is the path of the CUDA toolkit and ``$GPU_ARCH`` is the GPU architecture.

  Supported GPU architectures are listed as below:
  K：Kepler
  M：Maxwell
  P：Pascal
  V：Volta
  T：Turing
  A：Ampere

  See the [NVIDIA's official page](https://developer.nvidia.com/cuda-gpus#compute) for more details.

  You can also add ``-DUSE_HALF_PRECISION=ON`` to the cmake command to get half-precision supported.

* Use MKL (optional)

  Add ``-DUSE_MKL=ON`` and ``-DINTEL_ROOT=$MKL_PATH`` to the cmake command, where ``$MKL_PATH`` is the path of MKL.

* Use OpenBLAS (optional)

  Add ``-DUSE_OPENBLAS=ON`` and ``-DOPENBLAS_ROOT=$OPENBLAS_PATH`` to the cmake command, where ``$OPENBLAS_PATH`` is the path of OpenBLAS.


*Note that half-precision requires Pascal or newer GPU architectures.*

#### Configuration Example

We provide [several examples](./sample/compile/README.md) to build the project with different options. 

#### Compile on Linux

```bash
make -j && cd ..
```

#### Compile on Windows

Add ``-A 64`` to the cmake command and it will generate a visual studio project on windows, i.e., ``NiuTrans.NMT.sln`` so you can open & build it with Visual Studio (>= Visual Studio 2015).

If it succeeds, you will get an executable file **`NiuTrans.NMT`** in the 'bin' directory.



## Usage

### Training

#### Commands

*Make sure compiling the program with CUDA because training on CPUs is not supported now.*

Step 1: Prepare the training data.

```bash
# Convert the BPE vocabulary
python3 tools/GetVocab.py \
  -raw $bpeVocab \
  -new $niutransVocab
```

Description:
* `raw` - Path of the BPE vocabulary.
* `new` - Path of the NiuTrans.NMT vocabulary to be saved.

```bash
# Binarize the training data
python3 tools/PrepareParallelData.py \ 
  -src $srcFile \
  -tgt $tgtFile \
  -sv $srcVocab \
  -tv $tgtVocab \
  -maxsrc 200 \
  -maxtgt 200 \
  -output $trainingFile 
```

Description:

* `src` - Path of the source language data. One sentence per line with tokens separated by spaces or tabs.
* `tgt` - Path of the target language data. The same format as the source language data.
* `sv` - Path of the source language vocabulary. Its first line is the vocabulary size and the first index, followed by a word and its index in each following line.
* `tv` - Path of the target language vocabulary. The same format as the source language vocabulary.
* `maxsrc` - The maximum length of a source sentence. Default: 200.
* `maxtgt` - The maximum length of a target sentence. Default: 200.
* `output` - Path of the training data to be saved. 



Step 2: Train the model

```bash
bin/NiuTrans.NMT \
  -dev 0 \
  -nepoch 50 \
  -model model.bin \
  -ncheckpoint 10 \
  -train train.data \
  -valid valid.data
```

Description:

* `dev` - Device id (>= 0 for GPUs). Default: 0.
* `model` - Path of the model to be saved.
* `train` - Path to the training file. The same format as the output file in step 1.
* `valid` - Path to the validation file. The same format as the output file in step 1.
* `wbatch` - Word batch size. Default: 4096.
* `sbatch` - Sentence batch size. Default: 32.
* `dropout` - Dropout rate for the model. Default: 0.3.
* `fnndrop` - Dropout rate for fnn layers. Default: 0.1.
* `attdrop` - Dropout rate for attention layers. Default: 0.1.
* `lrate`- Learning rate. Default: 0.0015.
* `minlr` - The minimum learning rate for training. Default: 1e-9.
* `warmupinitlr` - The initial learning rate for warm-up. Default: 1e-7.
* `weightdecay` - The weight decay factor. Default: 0.
* `nwarmup` - Step number of warm-up for training. Default: 8000.
* `adam` - Indicates whether Adam is used. Default: true.
* `adambeta1` - Hyper parameters of Adam. Default: 0.9.
* `adambeta2` - Hyper parameters of Adam. Default: 0.98.
* `adambeta` - Hyper parameters of Adam. Default: 1e-9.
* `labelsmoothing` - Label smoothing factor. Default: 0.1.
* `updatefreq` - Update the model every `updatefreq` step. Default: 1.
* `nepoch` - The maximum training epoch. Default: 50.
* `nstep` - The maximum traing step. Default: 100000.
* `ncheckpoint` - The maximum checkpoint to be saved. Default: 0.1.


#### Training Example

Refer to [this page for the training example.](./sample/train/)

### Translating

*Make sure compiling the program with CUDA and FP16 if you want to translate with FP16 on GPUs.*

#### Commands

```bash
bin/NiuTrans.NMT \
 -dev $deviceID \
 -input $inputFile \
 -model $modelPath \
 -wbatch $wordBatchSize \
 -sbatch $sentenceBatchSize \
 -beamsize $beamSize \
 -srcvocab $srcVocab \
 -tgtvocab $tgtVocab \
 -output $outputFile
```


Description:


* `model` - Path of the model.
* `sbatch` - Sentence batch size. Default: 32.
* `dev` - Device id (-1 for CPUs, and >= 0 for GPUs). Default: 0.
* `beamsize` - Size of the beam. 1 for the greedy search.
* `input` - Path of the input file. One sentence per line with tokens separated by spaces.
* `output` - Path of the output file to be saved. The same format as the input file.
* `srcvocab` - Path of the source language vocabulary. Its first line is the vocabulary size, followed by a word and its index in each following line.
* `tgtvocab` - Path of the target language vocabulary. The same format as the source language vocabulary.
* `fp16 (optional)` - Inference with FP16. This will not work if the model is stored in FP32. Default: false.
* `lenalpha` - The alpha parameter controls the length preference. Default: 0.6.
* `maxlenalpha` - Scalar of the input sequence (for the max number of search steps). Default: 1.2.



#### An Example

Refer to [this page for the translating example.](./sample/translate/)

## Low Precision Inference

NiuTrans.NMT supports inference with FP16 and INT8, you can convert the model to FP16 with our tools:

```bash
python3 tools/FormatConverter.py \
  -input $inputModel \
  -output $outputModel \ 
  -format $targetFormat
```

Description:

* `input` - Path of the raw model file.
* `output` - Path of the new model file.
* `format` - Target storage format, FP16 (Default) or FP32.

## Converting Models from Fairseq

The core implementation is framework agnostic, so we can easily convert models trained with other frameworks to a binary format for efficient inference. 

The following frameworks and models are currently supported:

|     | [fairseq (>=0.6.2)](https://github.com/pytorch/fairseq/tree/v0.6.2) |
| --- | :---: |
| Transformer ([Vaswani et al. 2017](https://arxiv.org/abs/1706.03762)) | ✓ |
| RPR attention ([Shaw et al. 2018](https://arxiv.org/abs/1803.02155)) | ✓ |
| Deep Transformer ([Wang et al. 2019](https://www.aclweb.org/anthology/P19-1176/)) | ✓ |

*Refer to [this page](https://fairseq.readthedocs.io/en/latest/getting_started.html#training-a-new-model) for the details about training models with fairseq.*

After training, you can convert the fairseq checkpoint and vocabulary with the following steps.

Step 1: Convert parameters of a single fairseq model
```bash
python3 tools/ModelConverter.py -i $fairseqCheckpoint -o $niutransModel
```
Description:

* `raw` - Path of the fairseq checkpoint, [refer to this for more details](https://fairseq.readthedocs.io/en/latest/).
* `new` - Path to save the converted model parameters. All parameters are stored in a binary format.
* `fp16 (optional)` - Save the parameters with 16-bit data type. Default: disabled.

Step 2: Convert the vocabulary:
```bash
python3 tools/VocabConverter.py -raw $fairseqVocabPath -new $niutransVocabPath
```
Description:

* `raw` - Path of the fairseq vocabulary, [refer to this for more details](https://fairseq.readthedocs.io/en/latest/).
* `new` - Path to save the converted vocabulary. Its first line is the vocabulary size, followed by a word and its index in each following line.

*You may need to convert both the source language vocabulary and the target language vocabulary if they are not shared.*

## A Model Zoo

We provide several pre-trained models to test the system.
All models and runnable systems are packaged into docker files so that one can easily reproduce our result.

Refer to [this page](./sample/translate) for more details.

## Papers

Here are the papers related to this project:

[Learning Deep Transformer Models for Machine Translation.](https://www.aclweb.org/anthology/P19-1176) Qiang Wang, Bei Li, Tong Xiao, Jingbo Zhu, Changliang Li, Derek F. Wong, Lidia S. Chao. 2019. Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics.

[The NiuTrans System for WNGT 2020 Efficiency Task.](https://arxiv.org/abs/2109.08008)  Chi Hu, Bei Li, Yinqiao Li, Ye Lin, Yanyang Li, Chenglong Wang, Tong Xiao, Jingbo Zhu. 2020. Proceedings of the Fourth Workshop on Neural Generation and Translation.

[The NiuTrans System for the WMT21 Efficiency Task.](https://arxiv.org/abs/2109.08003) Chenglong Wang, Chi Hu, Yongyu Mu, Zhongxiang Yan, Siming Wu, Minyi Hu, Hang Cao, Bei Li, Ye Lin, Tong Xiao, Jingbo Zhu. 2020. 


## Team Members

This project is maintained by a joint team from NiuTrans Research and NEU NLP Lab. Current team members are

*Chi Hu, Chenglong Wang, Siming Wu, Bei Li, Yinqiao Li, Ye Lin, Quan Du, Tong Xiao and Jingbo Zhu*

Feel free to contact huchinlp[at]gmail.com or niutrans[at]mail.neu.edu.cn if you have any questions.

