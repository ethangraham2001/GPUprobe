mod gpuprobe {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/bpf/gpuprobe.skel.rs"
    ));
}

use libbpf_rs::MapCore;

use super::{Gpuprobe, GpuprobeError, DEFAULT_LINKS, LIBCUDART_PATH};

impl Gpuprobe {
    /// attaches uprobes for the bandwidth util program, or returns an error on
    /// failure
    pub fn attach_bandwidth_util_uprobes(&mut self) -> Result<(), GpuprobeError> {
        let cuda_memcpy_uprobe_link = self
            .skel
            .progs
            .trace_cuda_memcpy
            .attach_uprobe(false, -1, LIBCUDART_PATH, 0x000000000006f150)
            .map_err(|_| GpuprobeError::AttachError)?;

        let cuda_memcpy_uretprobe_link = self
            .skel
            .progs
            .trace_cuda_memcpy_ret
            .attach_uprobe(true, -1, LIBCUDART_PATH, 0x000000000006f150)
            .map_err(|_| GpuprobeError::AttachError)?;

        let mut links = DEFAULT_LINKS;
        links.trace_cuda_memcpy = Some(cuda_memcpy_uprobe_link);
        links.trace_cuda_memcpy_ret = Some(cuda_memcpy_uretprobe_link);
        self.links = links;
        Ok(())
    }

    /// Copies all cudaMemcpy calls out of the queue and returns them as a Vec,
    /// or returns a GpuProbeError on failure
    pub fn consume_queue(&self) -> Result<Vec<CudaMemcpy>, GpuprobeError> {
        let mut output: Vec<CudaMemcpy> = Vec::new();
        let key: [u8; 0] = []; // key size must be zero for BPF_MAP_TYPE_QUEUE
                               // `lookup_and_delete` calls.

        while let Ok(opt) = self
            .skel
            .maps
            .successful_cuda_memcpy_q
            .lookup_and_delete(&key)
        {
            match opt {
                Some(bytes) => match CudaMemcpy::from_bytes(&bytes) {
                    Some(valid_instance) => output.push(valid_instance),
                    None => {
                        return Err(GpuprobeError::RuntimeError(
                            "alloc conversion failure".to_string(),
                        ))
                    }
                },
                None => {
                    return Ok(output);
                }
            }
        }

        Ok(output)
    }
}

pub struct CudaMemcpy {
    pub start_time: u64,
    pub end_time: u64,
    pub dst: *mut std::ffi::c_void,
    pub src: *mut std::ffi::c_void,
    pub count: u64,
    pub memcpy_kind: u32,
}

impl CudaMemcpy {
    /// Constructs a CudaMemcpy struct from a raw byte array and returns it, or
    /// None if the byte array is invalid.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < std::mem::size_of::<Self>() {
            return None;
        }
        // This is safe if:
        // 1. The byte array contains valid data for this struct
        // 2. The byte array is at least as large as the struct
        unsafe { Some(std::ptr::read_unaligned(bytes.as_ptr() as *const Self)) }
    }
}

impl std::fmt::Display for CudaMemcpy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{{")?;
        writeln!(f, "\tstart_time: {}", self.start_time)?;
        writeln!(f, "\tend_time: {}", self.end_time)?;
        writeln!(f, "\tdst: {:p}", self.dst)?;
        writeln!(f, "\tsrc: {:p}", self.dst)?;
        writeln!(f, "\tcount: {}", self.count)?;
        writeln!(f, "\tkind: {}", self.memcpy_kind)?;
        writeln!(f, "}}")
    }
}