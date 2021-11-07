# Compilation Example

Here are some compilation examples for Linux with MKL, OpenBLAS, or CUDA supported. 

**Replace the path in your environment.**


## Compile with CUDA supported

```bash
git clone https://github.com/NiuTrans/NiuTrans.NMT.git
git clone https://github.com/NiuTrans/NiuTensor.git
mv NiuTensor/source NiuTrans.NMT/source/niutensor
rm NiuTrans.NMT/source/niutensor/Main.cpp
rm -rf NiuTrans.NMT/source/niutensor/sample NiuTrans.NMT/source/niutensor/tensor/test
mkdir NiuTrans.NMT/build && cd NiuTrans.NMT/build
cmake -DUSE_CUDA=ON -DCUDA_TOOLKIT_ROOT='/home/huchi/cuda-10.2/' -DGPU_ARCH=V ..
make -j
```

You may modify `DGPU_ARCH` with other GPU architectures:

K：Kepler
M：Maxwell
P：Pascal
V：Volta
T：Turing
A：Ampere

See the [NVIDIA's official page](https://developer.nvidia.com/cuda-gpus#compute) for more details.

## Compile with CUDA and FP16 supported


```bash
git clone https://github.com/NiuTrans/NiuTrans.NMT.git
git clone https://github.com/NiuTrans/NiuTensor.git
mv NiuTensor/source NiuTrans.NMT/source/niutensor
rm NiuTrans.NMT/source/niutensor/Main.cpp
rm -rf NiuTrans.NMT/source/niutensor/sample NiuTrans.NMT/source/niutensor/tensor/test
mkdir NiuTrans.NMT/build && cd NiuTrans.NMT/build
cmake -DUSE_CUDA=ON -DCUDA_TOOLKIT_ROOT='/home/huchi/cuda-10.2/' -DGPU_ARCH=V -DUSE_HALF_PRECISION=ON ..
make -j
```

## Compile with MKL supported


```bash
git clone https://github.com/NiuTrans/NiuTrans.NMT.git
git clone https://github.com/NiuTrans/NiuTensor.git
mv NiuTensor/source NiuTrans.NMT/source/niutensor
rm NiuTrans.NMT/source/niutensor/Main.cpp
rm -rf NiuTrans.NMT/source/niutensor/sample NiuTrans.NMT/source/niutensor/tensor/test
mkdir NiuTrans.NMT/build && cd NiuTrans.NMT/build
cmake -DUSE_MKL=ON -DINTEL_ROOT='/home/nlplab/intel/compilers_and_libraries_2020.2.254/linux' ..
make -j
```

## Compile with OpenBLAS supported


```bash
git clone https://github.com/NiuTrans/NiuTrans.NMT.git
git clone https://github.com/NiuTrans/NiuTensor.git
mv NiuTensor/source NiuTrans.NMT/source/niutensor
rm NiuTrans.NMT/source/niutensor/Main.cpp
rm -rf NiuTrans.NMT/source/niutensor/sample NiuTrans.NMT/source/niutensor/tensor/test
mkdir NiuTrans.NMT/build && cd NiuTrans.NMT/build
cmake -DUSE_OPENBLAS=ON -DOPENBLAS_ROOT='/home/nlplab/openblas/' ..
make -j
```

