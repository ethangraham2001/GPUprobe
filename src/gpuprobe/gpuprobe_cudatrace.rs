mod gpuprobe {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/bpf/gpuprobe.skel.rs"
    ));
}

use libbpf_rs::{MapCore, MapFlags};

use super::{Gpuprobe, GpuprobeError, DEFAULT_LINKS, LIBCUDART_PATH};

/// contains implementations for the cudatrace program
impl Gpuprobe {
    /// attaches uprobes for the cudatrace program, or returns an error on
    /// failure
    pub fn attach_cudatrace_uprobes(&mut self) -> Result<(), GpuprobeError> {
        let cuda_launch_kernel_uprobe_link = self
            .skel
            .progs
            .trace_cuda_launch_kernel
            .attach_uprobe(false, -1, LIBCUDART_PATH, 0x0000000000074440)
            .map_err(|_| GpuprobeError::AttachError)?;

        self.links.trace_cuda_launch_kernel = Some(cuda_launch_kernel_uprobe_link);
        Ok(())
    }

    /// returns the histogram of frequencies of CUDA kernels
    /// TODO: symbol resolution - right now if we launch the same program
    /// twice, we cannot recognize that the same kernel was launched due to
    /// ASLR and other factors. We would ideally like to resolve which kernel
    /// is being launched by looking at the relative addresses inside of the
    /// cuda binary.
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
