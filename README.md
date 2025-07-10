# CUDA black magic

find it in [Triton](https://github.com/triton-lang/triton/commit/ade3d49e624ac414cde33a4c982d656fc7e49605).

> fp8 is ~100 tflops faster when the kernel name has "cutlass"
in it.

You can reproduce it by running the command below — no need to build Triton. The only difference between [gluon_attention.ptx](./triton_cache/gluon_attention/attention_kernel.ptx) and [cutlass_gluon_attention.ptx](./triton_cache/cutlass_gluon_attention/attention_kernel.ptx) lies in their function names.

```bash
wget https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.8.93-archive.tar.xz
tar -xf cuda_nvcc-linux-x86_64-12.8.93-archive.tar.xz
```

```bash
git clone https://github.com/OpenMLIR/cuda-magic
cd cuda-magic
cuda_nvcc-linux-x86_64-12.8.93-archive/bin/ptxas -lineinfo -v --gpu-name=sm_100a triton_cache/gluon_attention/attention_kernel.ptx -o gluon_attention.cubin
cuda_nvcc-linux-x86_64-12.8.93-archive/bin/ptxas -lineinfo -v --gpu-name=sm_100a triton_cache/cutlass_gluon_attention/attention_kernel.ptx -o cutlass_gluon_attention.cubin
```

You can use `ls -lh` to check the sizes of different .cubin files.
