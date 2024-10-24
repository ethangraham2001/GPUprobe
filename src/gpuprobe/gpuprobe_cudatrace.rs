mod gpuprobe {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/bpf/gpuprobe.skel.rs"
    ));
}

use gpuprobe::*;

use std::error::Error;

use libbpf_rs::{MapCore, MapFlags};

use super::gpuprobe::GpuprobeLinks;
use super::{Gpuprobe, GpuprobeError};

const LIBCUDART_PATH: &str = "/usr/local/cuda/lib64/libcudart.so";

/// contains implementations for the cudatrace program
impl Gpuprobe {
    pub fn attach_cudatrace_uprobes(&mut self) -> Result<(), GpuprobeError> {
        let cuda_launch_kernel_uprobe_link = self
            .skel
            .progs
            .trace_cuda_launch_kernel
            .attach_uprobe(false, -1, LIBCUDART_PATH, 0x0000000000074440)
            .map_err(|_| GpuprobeError::AttachError)?;

        self.links = GpuprobeLinks {
            trace_cuda_malloc: None,
            trace_cuda_malloc_ret: None,
            trace_cuda_free: None,
            trace_cuda_free_ret: None,
            trace_cuda_launch_kernel: Some(cuda_launch_kernel_uprobe_link),
        };
        Ok(())
    }

    pub fn get_kernel_launch_frequencies(&self) -> Result<Vec<(u64, u64)>, GpuprobeError> {
        Ok(self
            .skel
            .maps
            .kernel_calls_hist
            .keys()
            .map(|addr| {
                let key: [u8; 8] = addr.try_into().expect("unable to convert addr");
                let call_count = self
                    .skel
                    .maps
                    .kernel_calls_hist
                    .lookup(&key, MapFlags::ANY)
                    .expect("lookup failed")
                    .unwrap_or(u64::to_ne_bytes(0u64).to_vec());
                let call_count: [u8; 8] = call_count.try_into().expect("unable to convert count");
                (u64::from_ne_bytes(key), u64::from_ne_bytes(call_count))
            })
            .collect())
    }
}
