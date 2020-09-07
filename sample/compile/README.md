# Compilation Example

Here is some compilation example with MKL, OpenBLAS, or CUDA supported. Replace the path in your environment.

## Compile with CUDA supported


```bash
git clone https://github.com/NiuTrans/NiuTrans.NMT.git
mkdir nmt/build && cd nmt/build
cmake -DUSE_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR='/home/nlplab/cuda9.2/' ..
```

## Compile with CUDA and FP16 supported


```bash
git clone https://github.com/NiuTrans/NiuTrans.NMT.git
mkdir nmt/build && cd nmt/build
cmake -DUSE_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR='/home/nlplab/cuda9.2/' -DUSE_FP16=ON ..
```

## Compile with MKL supported


```bash
git clone https://github.com/NiuTrans/NiuTrans.NMT.git
mkdir nmt/build && cd nmt/build
cmake -DUSE_MKL=ON -DINTEL_ROOT='/home/nlplab/intel/compilers_and_libraries_2020.2.254/linux' ..
```

## Compile with OpenBLAS supported


```bash
git clone https://github.com/NiuTrans/NiuTrans.NMT.git
mkdir nmt/build && cd nmt/build
cmake -DUSE_OPENBLAS=ON -DOPENBLAS_ROOT='/home/nlplab/openblas/' ..
```

