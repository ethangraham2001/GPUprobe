# GPUprobe

Provides some eBPF utilities for tracking GPU actvity from user-space via
uprobes.

For now, only a memory leak utility is available *(similar to bcc-memleak)*.
It works by correlating calls to `cudaMalloc()` and `cudaFree()` to detect when
a memory leak has occured 

## Example output

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
