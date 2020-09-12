# Compilation Example

Here is some compilation example for Linux with MKL, OpenBLAS, or CUDA supported. 

**Replace the path in your environment.**

## Download the code
```bash
git clone http://47.105.50.196/huchi/nmt.git
git clone http://47.105.50.196/NiuTrans/NiuTrans.Tensor.git --branch liyinqiao
mv NiuTrans.Tensor/source nmt/source/niutensor
rm nmt/source/niutensor/Main.cpp
rm -rf nmt/source/niutensor/sample nmt/source/niutensor/tensor/test
mkdir nmt/build && cd nmt/build
```

## Compile with CUDA supported


```bash
cmake -DUSE_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR='/home/nlplab/cuda9.2/' ..
make -j
```

## Compile with CUDA and FP16 supported


```bash
git clone https://github.com/NiuTrans/NiuTrans.NMT.git
mkdir nmt/build && cd nmt/build
cmake -DUSE_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR='/home/nlplab/cuda9.2/' -DUSE_FP16=ON ..
make -j
```

## Compile with MKL supported


```bash
git clone https://github.com/NiuTrans/NiuTrans.NMT.git
mkdir nmt/build && cd nmt/build
cmake -DUSE_MKL=ON -DINTEL_ROOT='/home/nlplab/intel/compilers_and_libraries_2020.2.254/linux' ..
make -j
```

## Compile with OpenBLAS supported


```bash
git clone https://github.com/NiuTrans/NiuTrans.NMT.git
mkdir nmt/build && cd nmt/build
cmake -DUSE_OPENBLAS=ON -DOPENBLAS_ROOT='/home/nlplab/openblas/' ..
make -j
```

