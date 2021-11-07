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
* 快速解码，融合了多种推断优化策略，例如FP16/INT8、计算图优化、高效显存管理机制
* 支持多种先进的NMT模型，例如[深层Transformer](https://www.aclweb.org/anthology/P19-1176)
* 支持多种操作系统和设备，包括Linux/Windows，GPU/CPU
* 支持从其他框架导入模型权重

## 更新说明
2021.11：发布我们提交至[WMT21效率评测](http://statmt.org/wmt21/efficiency-task.html)的版本，相较于上个版本在GPU上的推断速度加快3倍。

2020.12: 新增[DLCL](https://arxiv.org/abs/1906.01787)和[RPR Attention](https://arxiv.org/abs/1803.02155)模型训练功能。

2020.12: 大幅优化训练时显存占用，并显著提高了训练速度。

## 安装说明

### 要求
* 操作系统: Linux 或 Windows

* [GCC/G++](https://gcc.gnu.org/) >=4.8.5 (on Linux)

* [VC++](https://www.microsoft.com/en-us/download/details.aspx?id=48145) >=2015 (Windows)

* [CMake](https://cmake.org/download/) >= 2.8

* [CUDA](https://developer.nvidia.com/cuda-92-download-archive) >= 10.2 (可选)

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

  添加 ``-DUSE_CUDA=ON``、``-DCUDA_TOOLKIT_ROOT=$CUDA_PATH``和``DGPU_ARCH=$GPU_ARCH``到Cmake命令行, 其中 ``$CUDA_PATH`` 是CUDA的安装路径，``$GPU_ARCH``是GPU架构编号。

  支持的GPU架构编号如下：
  K：Kepler
  M：Maxwell
  P：Pascal
  V：Volta
  T：Turing
  A：Ampere
  
  您可以访问[英伟达官方文档](https://developer.nvidia.com/cuda-gpus#compute)来查看GPU架构编号详情。
  您也可以添加 ``-DUSE_HALF_PRECISION=ON`` 以使用半精度计算.

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
  -sv $srcVocab \
  -tv $tgtVocab \
  -maxsrc 200 \
  -maxtgt 200 \
  -output $trainingFile 
```

参数说明:

* `src` - 源语数据路径，格式：每行一条句子，由空格或TAB分开。
* `tgt` - 目标语数据路径，格式：每行一条句子，由空格或TAB分开。
* `sv` - 源语词汇表路径，格式：首行为词汇表大小和起始符号，其余行是单词和对应的索引（数字）。
* `tv` - 目标语词汇表路径，格式：首行为词汇表大小和起始符号，其余行是单词和对应的索引（数字）。
* `maxsrc` - 源语句子最大长度. 默认: 200.
* `maxtgt` - 目标语句子最大长度. 默认: 200.
* `output` - 输出的二进制文件路径。


步骤2：训练模型

```bash
bin/NiuTrans.NMT \
  -dev 0 \
  -nepoch 50 \
  -model model.bin \
  -ncheckpoint 10 \
  -train train.data \
  -valid valid.data
```

参数说明:

* `dev` - 设备ID，大于0为GPU设备，-1为CPU设备。
* `model` - 模型存储路径。
* `train` - 训练数据路径。
* `valid` - 校验数据路径。
* `wbatch` - 按词数组batch大小，默认：4096。
* `sbatch` - 按句子数组batch大小，默认：32。
* `dropout` - 模型Dropout概率，默认：0.3。
* `fnndrop` - FNN层Dropout概率，默认：0.1。
* `attdrop` - 注意力层Dropout概率，默认：0.1。
* `lrate`- 初始化学习率，默认：0.0015。
* `minlr` - 训练时最小学习率. 默认: 1e-9.
* `warmupinitlr` - 预热阶段初始化学习率. Default: 1e-7.
* `weightdecay` - 权重衰减因子. Default: 0.
* `nwarmup` - 预热步数，默认：8000。
* `adam` - 是否使用Adam优化器，默认：是。
* `adambeta1` - Adam的超参数beta1，默认：0.9。
* `adambeta2` - Adam的超参数beta2，默认：0.98。
* `adambeta` - Adam的超参数beta，默认：1e-9。
* `labelsmoothing` - Label smoothing概率，默认：0.1。
* `updatefreq` - 多少步更新一次参数，默认：1，若大于1则执行梯度累积。
* `nepoch` - 最大训练轮数，默认：50。
* `nstep` - 最大训练步数，默认：100000。
* `ncheckpoint` - 保存检查点的最大数量. 默认: 10.


#### 示例

详见 [训练示例](./sample/train/)。

### 翻译

#### 命令行

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

参数说明:

* `model` - 模型存储路径。
* `sbatch` - batch中的句子数。
* `dev` - 设备ID，大于0为GPU设备，-1为CPU设备。
* `beamsize` - 束大小，若为1则执行贪心搜索。
* `input` - 输入文件路径，格式：每行一条句子，单词用空格分开。
* `output` - 输出文件路径，格式：每行一条句子，单词用空格分开。
* `srcvocab` - 源语词汇表路径，格式：首行为词汇表大小和起始符号，其余行是单词和对应的索引（数字）。
* `tgtvocab` - 源语词汇表路径，格式：首行为词汇表大小和起始符号，其余行是单词和对应的索引（数字）。
* `fp16` - 是否使用FP16进行计算，默认：否。
* `lenalpha` - 长度惩罚因子，默认：0.6。
* `maxlenalpha` - 最大译文句长因子（源语长度倍数），默认：1.2。



#### 示例

详见 [翻译示例](./sample/translate/)。

## 低精度推断

NiuTrans.NMT支持FP16和INT8低精度推断, 您可以通过下面的命令将模型转换为FP16格式：

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

|     | [fairseq (>=0.6.2)](https://github.com/pytorch/fairseq/tree/v0.6.2) |
| --- | :---: |
| Transformer ([Vaswani et al. 2017](https://arxiv.org/abs/1706.03762)) | ✓ |
| RPR attention ([Shaw et al. 2018](https://arxiv.org/abs/1803.02155)) | ✓ |
| Deep Transformer ([Wang et al. 2019](https://www.aclweb.org/anthology/P19-1176/)) | ✓ |

您仅需对词表和模型权重进行转换：

步骤1: 从Fairseq中导出模型权重：

```bash
python3 tools/ModelConverter.py -raw $fairseqCheckpoint -new $niutransModel
```

参数说明:

* `raw` - Fairseq模型路径。
* `new` - 目标模型路径。
* `fp16 (optional)` - 是否储存为FP16格式，默认：否。

步骤2: 从Fairseq中导出词汇表:

```bash
python3 tools/VocabConverter.py -raw $fairseqVocabPath -new $newVocabPath
```

参数说明:

* `raw` - Fairseq词汇表路径。
* `new` - 目标词汇表路径。

## 预训练模型

我们提供了一些预训练模型以供用户快速体验，详见[该页](./sample/translate)。

## 相关论文

下面是与本项目相关的论文：

[The NiuTrans System for WNGT 2020 Efficiency Task.](https://arxiv.org/abs/2109.08008)  Chi Hu, Bei Li, Yinqiao Li, Ye Lin, Yanyang Li, Chenglong Wang, Tong Xiao, Jingbo Zhu. 2020. Proceedings of the Fourth Workshop on Neural Generation and Translation.

[The NiuTrans System for the WMT21 Efficiency Task.](https://arxiv.org/abs/2109.08003) Chenglong Wang, Chi Hu, Yongyu Mu, Zhongxiang Yan, Siming Wu, Minyi Hu, Hang Cao, Bei Li, Ye Lin, Tong Xiao, Jingbo Zhu. 2020. 

## 团队成员

本项目由NiuTrans Research和东北大学自然语言处理实验室团队维护，目前成员有：

*胡驰，王成龙，吴斯铭，李北，李垠桥，林野，杜权，肖桐，朱靖波*

如有疑问请提issue，或者联系niutrans[at]mail.neu.edu.cn
