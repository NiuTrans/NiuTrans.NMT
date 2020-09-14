# NiuTrans.NMT

- [NiuTrans.NMT](#niutransnmt)
  - [Features](#features)
  - [Installation](#installation)
    - [Requirements](#requirements)
    - [Build from Source](#build-from-source)
      - [Configure with CMake](#configure-with-cmake)
      - [Configuration Example](#configuration-example)
      - [Compile on Different Systems](#compile-on-different-systems)
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
NiuTrans.NMT is a lightweight and efficient Transformer-based neural machine translation system. Its main features are:
* Few dependencies. It is implemented with pure C++, and all dependencies are optional.
* Fast decoding. It supports various decoding acceleration strategies, such as batch pruning and dynamic batch size.
* Advanced NMT models, such as [Deep Transformer](https://www.aclweb.org/anthology/P19-1176).
* Flexible running modes. The system can be run on various systems and devices (Linux vs. Windows, CPUs vs. GPUs, and FP32 vs. FP16, etc.).
* Framework agnostic. It supports various models trained with other tools, e.g., fairseq models.
* The code is simple and friendly to beginners.

## Installation

### Requirements
* OS: Linux or Windows

* [GCC/G++](https://gcc.gnu.org/) >=4.8.4 (on Linux)

* [VC++](https://www.microsoft.com/en-us/download/details.aspx?id=48145) >=2015 (on Windows)

* [CMake](https://cmake.org/download/) >= 2.8

* [CUDA](https://developer.nvidia.com/cuda-92-download-archive) >= 9.2, <= 10.2 (optional)

* [MKL](https://software.intel.com/content/www/us/en/develop/tools/math-kernel-library.html) latest version (optional)

* [OpenBLAS](https://github.com/xianyi/OpenBLAS) latest version (optional)


### Build from Source

#### Configure with CMake

The default configuration enables compiling for the pure CPU version:

```bash
git clone https://github.com/NiuTrans/NiuTrans.NMT.git
git clone https://github.com/NiuTrans/NiuTensor.git
mv NiuTrans.Tensor/source NiuTrans.NMT/source/niutensor
rm NiuTrans.NMT/source/niutensor/Main.cpp
rm -rf NiuTrans.NMT/source/niutensor/sample NiuTrans.NMT/source/niutensor/tensor/test
mkdir NiuTrans.NMT/build && cd NiuTrans.NMT/build
cmake ..
```

You can add compilation options to support accelerations with MKL, OpenBLAS, or CUDA.

*Please note that you can only select at most one of MKL or OpenBLAS.*

**Use CUDA (required for training)**

Add ``-DUSE_CUDA=ON`` and ``-DCUDA_TOOLKIT_ROOT_DIR=$CUDA_PATH`` to the CMake command, where ``$CUDA_PATH`` is the path of the CUDA toolkit.

You can also add ``-DUSE_FP16=ON`` to the CMake command to get half-precision supported.

**Use MKL (optional)**

Add ``-DUSE_MKL=ON`` and ``-DINTEL_ROOT=$MKL_PATH`` to the CMake command, where ``$MKL_PATH`` is the path of MKL.

**Use OpenBLAS (optional)**

Add ``-DUSE_OPENBLAS=ON`` and ``-DOPENBLAS_ROOT=$OPENBLAS_PATH`` to the CMake command, where ``$OPENBLAS_PATH`` is the path of OpenBLAS.


*Note that half-precision requires Pascal or newer architectures on GPUs.*

#### Configuration Example

We provide [several examples](./sample/compile/README.md) to build the project with different options. 

#### Compile on Different Systems

**Compile on Linux**

```bash
make -j && cd ..
```

**Compile on Windows**

Add ``-A 64`` to the CMake command.

It will generate a visual studio project on windows, i.e., ``NiuTrans.NMT.sln`` and you can open & build it with Visual Studio (>= Visual Studio 2015).

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
-src_vocab $srcVocab \
-tgt_vocab $tgtVocab \
-output $trainingFile 
```

Description:

* `src` - Path of the source language data. One sentence per line with tokens separated by spaces or tabs.
* `tgt` - Path of the target language data. The same format as the source language data.
* `sv` - Path of the source language vocabulary. Its first line is the vocabulary size and the first index, followed by a word and its index in each following line.
* `tv` - Path of the target language vocabulary. The same format as the source language vocabulary.
* `output` - Path of the training data to be saved. 



Step 2: Train the model

```bash
bin/NiuTrans.NMT \
-dev $deviceID \
-model $modelFile \
-train $trainingData \
-valid $validData 
```

Description:

* `dev` - Device id (>= 0 for GPUs). Default: 0.
* `model` - Path of the model to be saved.
* `train` - Path to the training file. The same format as the output file in step 1.
* `valid` - Path to the validation file. The same format as the output file in step 1.
* `wbatch` - Word batch size. Default: 4096.
* `sbatch` - Sentence batch size. Default: 8.
* `mt` - Indicates whether the model runs for machine translation. Default: true.
* `dropout` - Dropout rate for the model. Default: 0.3.
* `fnndrop` - Dropout rate for fnn layers. Default: 0.1.
* `attdrop` - Dropout rate for attention layers. Default: 0.1.
* `lrate`- Learning rate. Default: 0.0015.
* `lrbias` - The parameter that controls the maximum learning rate in training. Default: 0.
* `nepoch` - Training epoch number. Default: 50.
* `nstep` - Traing step number. Default: 100000.
* `nwarmup` - Step number of warm-up for training. Default: 8000.
* `adam` - Indicates whether Adam is used. Default: true.
* `adambeta1` - Hyper parameters of Adam. Default: 0.9.
* `adambeta2` - Hyper parameters of Adam. Default: 0.98.
* `adambeta` - Hyper parameters of Adam. Default: 1e-9.
* `shuffled` - Indicates whether the data file is shuffled for training. Default: true.
* `labelsmoothing` - Label smoothing factor. Default: 0.1.
* `nstepcheckpoint` - Number of steps after which we make a checkpoint. Default: -1.
* `epochcheckpoint` - Indicates whether we make a checkpoint after each training epoch. Default: true.
* `updatestep` - Number of batches that we collect for model update. Default: 1 (one can set > 1 for gradient accumulation).
* `sorted` - Indicates whether the sequence is sorted by length. Default: false.
* `bufsize` - Buffer size for the batch loader. Default: 50000.
* `doubledend` - Indicates whether we double the </s> symbol for the output of LM. Default: false.
* `smallbatch` - Indicates whether we use batchsize = max *  sc
       rather rather than batchsize = word-number, where max is the maximum
       length and sc is the sentence number. Default: true.
* `bigbatch` - Counterpart of "isSmallBatch". Default: false.
* `randbatch` - Randomize batches. Default: false.
* `bucketsize` - Bucket size for the batch loader. Default: wbatch * 10.



#### An Example

Refer to [this page for the training example.](./sample/train/)

### Translating

*Make sure compiling the program with CUDA and FP16 if you want to translate with FP16 on GPUs.*

#### Commands

```bash
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
* `maxlenalpha` - Scalar of the input sequence (for the max number of search steps). Default: 1.2.



#### An Example

Refer to [this page for the translating example.](./sample/translate/)

## Low Precision Inference

NiuTrans.NMT supports inference with FP16, you can convert the model to FP16 with our tools:

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

|     | [fairseq (0.6.2)](https://github.com/pytorch/fairseq/tree/v0.6.2) |
| --- | :---: |
| Transformer ([Vaswani et al. 2017](https://arxiv.org/abs/1706.03762)) | ✓ |
| RPR attention ([Shaw et al. 2018](https://arxiv.org/abs/1803.02155)) | ✓ |
| Deep Transformer ([Wang et al. 2019](https://www.aclweb.org/anthology/P19-1176/)) | ✓ |

*Refer to [this page](https://fairseq.readthedocs.io/en/latest/getting_started.html#training-a-new-model) for the details about training models with fairseq.*

After training, you can convert the fairseq models and vocabulary with the following steps.

Step 1: Convert parameters of a single fairseq model
```bash
python3 tools/ModelConverter.py -src $src -tgt $tgt
```
Description:

* `src` - Path of the fairseq checkpoint, [refer to this for more details](https://fairseq.readthedocs.io/en/latest/).
* `tgt` - Path to save the converted model parameters. All parameters are stored in a binary format.
* `fp16 (optional)` - Save the parameters with 16-bit data type. Default: disabled.

Step 2: Convert the vocabulary:
```bash
python3 tools/VocabConverter.py -src $fairseqVocabPath -tgt $newVocabPath
```
Description:

* `src` - Path of the fairseq vocabulary, [refer to this for more details](https://fairseq.readthedocs.io/en/latest/).
* `tgt` - Path to save the converted vocabulary. Its first line is the vocabulary size, followed by a word and its index in each following line.

*You may need to convert both the source language vocabulary and the target language vocabulary if they are not shared.*

## A Model Zoo

We provide several pre-trained models to test the system.
All models and runnable systems are packaged into docker files so that one can easily reproduce our result.

Refer to [this page](./sample/translate) for more details.

## Papers

Here are the papers related to this project:

[Learning Deep Transformer Models for Machine Translation.](https://www.aclweb.org/anthology/P19-1176) Qiang Wang, Bei Li, Tong Xiao, Jingbo Zhu, Changliang Li, Derek F. Wong, Lidia S. Chao. 2019. Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics.

[The NiuTrans System for WNGT 2020 Efficiency Task.](https://www.aclweb.org/anthology/2020.ngt-1.24)  Chi Hu, Bei Li, Yinqiao Li, Ye Lin, Yanyang Li, Chenglong Wang, Tong Xiao, Jingbo Zhu. 2020. Proceedings of the Fourth Workshop on Neural Generation and Translation.

## Team Members

This project is maintained by a joint team from NiuTrans Research and NEU NLP Lab. Current team members are

*Chi Hu, Bei Li, Yinqiao Li, Ye Lin, Quan Du, Tong Xiao and Jingbo Zhu*

Please contact niutrans@mail.neu.edu.cn if you have any questions.

