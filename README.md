# GPUprobe

Provides some eBPF utilities for tracking GPU actvity from user-space via
uprobes. 

Three utilies are currently available:

```
Usage: gpu_probe [OPTIONS]

Options:
      --memleak         Detects leaking calls to cudaMalloc from the CUDA runtime API
      --cudatrace       Maintains a histogram on frequencies of cuda kernel launches
      --bandwidth-util  Approximates bandwidth utilization of cudaMemcpy
  -h, --help            Print help
  -V, --version         Print version
```

## Memleak utility

The following is a sample output from the memleak utility displaying the 
allocations made by a program that does the following

- calls `cudaMalloc()` three times
- performs some computation some time
- calls `cudaFree()` for two of the input arrays, forgetting the third one

```
GPUprobe memleak utility
========================


total number of `cudaMalloc` calls: 0
0 bytes leaked from 0 cuda memory allocation(s)
========================

total number of `cudaMalloc` calls: 3
25165824 bytes leaked from 3 cuda memory allocation(s)
        0x769af1400000: 8388608 bytes
        0x769aec000000: 8388608 bytes
        0x769af0c00000: 8388608 bytes
========================

total number of `cudaMalloc` calls: 3
25165824 bytes leaked from 3 cuda memory allocation(s)
        0x769af1400000: 8388608 bytes
        0x769aec000000: 8388608 bytes
        0x769af0c00000: 8388608 bytes
========================

total number of `cudaMalloc` calls: 3
8388608 bytes leaked from 1 cuda memory allocation(s)
        0x769aec000000: 8388608 bytes
========================
```

## CudaTrace utility

In this sample output, a cuda program is executed that runs a 1000-iteration 
loop that launches two kernels per iteration.

```
GPUprobe cudatrace utility
========================


0 `cudaLaunchKernel` calls for 0 kernels
========================

1194 `cudaLaunchKernel` calls for 2 kernels
        0x626693de7b80: 597 launches
        0x626693de7c60: 597 launches
========================

2000 `cudaLaunchKernel` calls for 2 kernels
        0x626693de7b80: 1000 launches
        0x626693de7c60: 1000 launches
========================

2000 `cudaLaunchKernel` calls for 2 kernels
        0x626693de7b80: 1000 launches
        0x626693de7c60: 1000 launches
========================
```

## Bandwidth utilization utility

In this sample output, we infer the average bandwidth utilization of calls to
`cudaMemcpy`, a well as the direction of the transfers and their durations.

```
GPUprobe bandwidth_util utility
========================


Traced 1 cudaMemcpy calls
        H2D 3045740550.87548 bytes/sec for 0.00263 secs
========================

Traced 2 cudaMemcpy calls
        H2D 2981869117.56429 bytes/sec for 0.00268 secs
        D2H 3039108386.38160 bytes/sec for 0.00263 secs
========================
```

This is computed naively with

```
throughput = count / (end - start)
```
