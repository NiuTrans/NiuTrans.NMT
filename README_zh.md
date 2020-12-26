# NiuTrans.NMT

- [NiuTrans.NMT](#niutransnmt)
  - [特色](#特色)
  - [更新说明](#更新说明)
  - [安装说明](#安装说明)
    - [要求](#要求)
    - [编译源代码](#编译源代码)
      - [配置Cmake](#配置cmake)
      - [编译示例](#编译示例)
      - [在Linux上编译](#在linux上编译)
      - [在Windows上编译](#在windows上编译)
  - [使用说明](#使用说明)
    - [训练](#训练)
      - [命令行](#命令行)
      - [示例](#示例)
    - [翻译](#翻译)
      - [命令行](#命令行-1)
      - [示例](#示例-1)
  - [低精度推断](#低精度推断)
  - [从Fairseq导出模型](#从fairseq导出模型)
  - [预训练模型](#预训练模型)
  - [相关论文](#相关论文)
  - [团队成员](#团队成员)

## 特色
NiuTrans.NMT是一个轻量级、高效的神经机器翻译项目，主要特色包括：
* 依赖少，由纯C++代码实现，所有的依赖项都是可选的
* 快速解码，融合了多种解码加速策略，例如batch裁剪和动态batch大小
* 支持多种先进的NMT模型，例如[深层Transformer](https://www.aclweb.org/anthology/P19-1176)
* 支持多种操作系统和设备，包括Linux/Windows，GPU/CPU
* 支持从其他框架导入模型权重
* 代码架构简洁易上手

## 更新说明
2020.12: 新增[DLCL](https://arxiv.org/abs/1906.01787)和[RPR Attention](https://arxiv.org/abs/1803.02155)模型训练功能。

2020.12: 大幅优化训练时显存占用，并显著提高了训练速度。

## 安装说明

### 要求
* 操作系统: Linux 或 Windows

* [GCC/G++](https://gcc.gnu.org/) >=4.8.4 (on Linux)

* [VC++](https://www.microsoft.com/en-us/download/details.aspx?id=48145) >=2015 (Windows)

* [CMake](https://cmake.org/download/) >= 2.8

* [CUDA](https://developer.nvidia.com/cuda-92-download-archive) >= 9.2, <= 10.1 (可选)

* [MKL](https://software.intel.com/content/www/us/en/develop/tools/math-kernel-library.html) 最新版 (可选)

* [OpenBLAS](https://github.com/xianyi/OpenBLAS) 最新版 (可选)


### 编译源代码

#### 配置Cmake

项目默认配置编译**纯CPU**版本。

```bash
# 下载代码
git clone https://github.com/NiuTrans/NiuTrans.NMT.git
git clone https://github.com/NiuTrans/NiuTensor.git
# 替换文件夹
mv NiuTensor/source NiuTrans.NMT/source/niutensor
rm NiuTrans.NMT/source/niutensor/Main.cpp
rm -rf NiuTrans.NMT/source/niutensor/sample NiuTrans.NMT/source/niutensor/tensor/test
mkdir NiuTrans.NMT/build && cd NiuTrans.NMT/build
# 运行CMake
cmake ..
```

您也可以通过添加cmake选项来在本项目中使用MKL/OpenBLAS/CUDA。

*注意：不能同时使用MKL与OpenBLAS。*

* 使用CUDA (可选)

  添加 ``-DUSE_CUDA=ON`` 和 ``-DCUDA_TOOLKIT_ROOT_DIR=$CUDA_PATH`` 到Cmake命令行, 其中 ``$CUDA_PATH`` 是CUDA的安装路径。

  您也可以添加 ``-DUSE_FP16=ON`` 以使用半精度计算.

* 使用MKL (可选)

  添加 ``-DUSE_MKL=ON`` 和 ``-DINTEL_ROOT=$MKL_PATH`` 到Cmake命令行, 其中 ``$MKL_PATH`` 是MKL的安装路径。

* 使用OpenBLAS (可选)

  添加 ``-DUSE_OPENBLAS=ON`` 和 ``-DOPENBLAS_ROOT=$OPENBLAS_PATH`` 到Cmake命令行, 其中 ``$OPENBLAS_PATH`` 是OpenBLAS的安装路径。


*注意：半精度计算需要Pascal或者更新版本的GPU设备。*

#### 编译示例

这里是一些[编译示例](./sample/compile/README.md)。

#### 在Linux上编译

在使用Cmake配置好编译选项后，指向make命令即可：

```bash
make -j && cd ..
```

#### 在Windows上编译

在使用Cmake配置好编译选项后，会在配置的文件夹下生成 **`NiuTrans.NMT.sln`**，用Visual Studio打开后右键项目->“设为启动项目”，然后进行编译即可。


## 使用说明

### 训练

#### 命令行

步骤 1: 准备训练数据

```bash
# Convert the BPE vocabulary
python3 tools/GetVocab.py \
  -raw $bpeVocab \
  -new $niutransVocab
```

参数说明:
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

参数说明:

* `src` - 源语数据路径，格式：每行一条句子，由空格或TAB分开。
* `tgt` - 目标语数据路径，格式：每行一条句子，由空格或TAB分开。
* `sv` - 源语词汇表路径，格式：首行为词汇表大小和起始符号，其余行是单词和对应的索引（数字）。
* `tv` - 目标语词汇表路径，格式：首行为词汇表大小和起始符号，其余行是单词和对应的索引（数字）。
* `output` - 输出的二进制文件。


步骤2：训练模型

```bash
bin/NiuTrans.NMT \
-dev $deviceID \
-model $modelFile \
-train $trainingData \
-valid $validData 
```

参数说明:

* `dev` - 设备ID，大于0为GPU设备，-1为CPU设备。
* `model` - 模型存储路径。
* `train` - 训练数据路径。
* `valid` - 校验数据路径。
* `wbatch` - 按词数组batch大小，默认：4096。
* `sbatch` - 按句子数组batch大小，默认：16。
* `mt` - 是否训练机器翻译模型，默认：是。
* `dropout` - 模型Dropout概率，默认：0.3。
* `fnndrop` - FNN层Dropout概率，默认：0.1。
* `attdrop` - 注意力层Dropout概率，默认：0.1。
* `lrate`- 初始化学习率，默认：0.0015。
* `nepoch` - 最大训练轮数，默认：50。
* `nstep` - 最大训练步数，默认：100000。
* `nwarmup` - warm-up步数，默认：8000。
* `adam` - 是否使用Adam优化器，默认：是。
* `adambeta1` - Adam的超参数beta1，默认：0.9。
* `adambeta2` - Adam的超参数beta2，默认：0.98。
* `adambeta` - Adam的超参数beta，默认：1e-9。
* `shuffled` - 是否随机化训练数据，默认：是。
* `labelsmoothing` - Label smoothing概率，默认：0.1。
* `nstepcheckpoint` - 多少步保存一次检查点，默认：-1（禁用）。
* `epochcheckpoint` - 是否每轮保存一次检查点，默认：是。
* `updatestep` - 多少步更新一次参数，默认：1，若大于1则执行梯度累积。



#### 示例

详见 [训练示例](./sample/train/)。

### 翻译

#### 命令行

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

参数说明:

* `model` - 模型存储路径。
* `sbatch` - batch中的句子数。
* `dev` - 设备ID，大于0为GPU设备，-1为CPU设备。
* `beamsize` - 束大小，若为1则执行贪心搜索。
* `test` - 输入文件路径，格式：每行一条句子，单词用空格分开。
* `output` - 输出文件路径，格式：每行一条句子，单词用空格分开。
* `srcvocab` - 源语词汇表路径，格式：首行为词汇表大小和起始符号，其余行是单词和对应的索引（数字）。
* `tgtvocab` - 源语词汇表路径，格式：首行为词汇表大小和起始符号，其余行是单词和对应的索引（数字）。
* `fp16 (optional)` - 是否使用FP16进行计算，默认：否。
* `lenalpha` - 长度惩罚因子，默认：0.6。
* `maxlenalpha` - 最大译文句长因子（源语长度倍数），默认：1.2。



#### 示例

详见 [翻译示例](./sample/translate/)。

## 低精度推断

NiuTrans.NMT支持FP16推断, 您可以通过下面的命令将模型转换为FP16格式：

```bash
python3 tools/FormatConverter.py \
  -input $inputModel \
  -output $outputModel \ 
  -format $targetFormat
```

参数说明:

* `input` - 原始模型路径。
* `output` - 目标模型路径。
* `format` - 目标模型格式，默认：FP16。

## 从Fairseq导出模型

本项目支持从其他框架中导入训练好的模型，目前支持的框架和模型有：

|     | [fairseq (0.6.2)](https://github.com/pytorch/fairseq/tree/v0.6.2) |
| --- | :---: |
| Transformer ([Vaswani et al. 2017](https://arxiv.org/abs/1706.03762)) | ✓ |
| RPR attention ([Shaw et al. 2018](https://arxiv.org/abs/1803.02155)) | ✓ |
| Deep Transformer ([Wang et al. 2019](https://www.aclweb.org/anthology/P19-1176/)) | ✓ |

您仅需对词表和模型权重进行转换：

步骤1: 从Fairseq中导出模型权重：

```bash
python3 tools/ModelConverter.py -src $src -tgt $tgt
```

参数说明:

* `src` - Fairseq模型路径。
* `tgt` - 目标模型路径。
* `fp16 (optional)` - 是否储存为FP16格式，默认：否。

步骤2: 从Fairseq中导出词汇表:

```bash
python3 tools/VocabConverter.py -src $fairseqVocabPath -tgt $newVocabPath
```

参数说明:

* `src` - Fairseq词汇表路径。
* `tgt` - 目标词汇表路径。

## 预训练模型

我们提供了一些预训练模型以供用户快速体验，详见[该页](./sample/translate)。

## 相关论文

下面的与本项目相关的论文：

[Learning Deep Transformer Models for Machine Translation.](https://www.aclweb.org/anthology/P19-1176) Qiang Wang, Bei Li, Tong Xiao, Jingbo Zhu, Changliang Li, Derek F. Wong, Lidia S. Chao. 2019. Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics.

[The NiuTrans System for WNGT 2020 Efficiency Task.](https://www.aclweb.org/anthology/2020.ngt-1.24)  Chi Hu, Bei Li, Yinqiao Li, Ye Lin, Yanyang Li, Chenglong Wang, Tong Xiao, Jingbo Zhu. 2020. Proceedings of the Fourth Workshop on Neural Generation and Translation.

## 团队成员

本项目由NiuTrans Research和东北大学自然语言处理实验室团队维护，目前成员有：

*胡驰，李北，李垠桥，林野，杜权，肖桐，朱靖波*

如有疑问请提issue，或者联系niutrans[at]mail.neu.edu.cn
