use std::error::Error;

use libbpf_rs::{MapCore, MapFlags};

use super::uprobe_data::MemleakData;
use super::{Gpuprobe, GpuprobeError, LIBCUDART_PATH};

/// contains implementation for the memleak program
impl Gpuprobe {
    /// attaches uprobes for the memleak program, or returns an error on
    /// failure
    pub fn attach_memleak_uprobes(&mut self) -> Result<(), GpuprobeError> {
        let cuda_malloc_uprobe_link = self
            .skel
            .progs
            .trace_cuda_malloc
            .attach_uprobe(false, -1, LIBCUDART_PATH, 0x00000000000560c0)
            .map_err(|_| GpuprobeError::AttachError)?;

        let cuda_malloc_uretprobe_link = self
            .skel
            .progs
            .trace_cuda_malloc_ret
            .attach_uprobe(true, -1, LIBCUDART_PATH, 0x00000000000560c0)
            .map_err(|_| GpuprobeError::AttachError)?;

        let cuda_free_uprobe_link = self
            .skel
            .progs
            .trace_cuda_free
            .attach_uprobe(false, -1, LIBCUDART_PATH, 0x00000000000568c0)
            .map_err(|_| GpuprobeError::AttachError)?;

        let cuda_free_uretprobe_link = self
            .skel
            .progs
            .trace_cuda_free_ret
            .attach_uprobe(true, -1, LIBCUDART_PATH, 0x00000000000568c0)
            .map_err(|_| GpuprobeError::AttachError)?;

        self.links.trace_cuda_malloc = Some(cuda_malloc_uprobe_link);
        self.links.trace_cuda_malloc_ret = Some(cuda_malloc_uretprobe_link);
        self.links.trace_cuda_free = Some(cuda_free_uprobe_link);
        self.links.trace_cuda_free_ret = Some(cuda_free_uretprobe_link);
        Ok(())
    }

    /// converts an address to its given allocation information by performing
    /// a lookup in the map of successful allocations
    fn addr_to_allocation(&self, addr: Vec<u8>) -> Result<(u64, u64), GpuprobeError> {
        let addr_key: [u8; 8] = addr
            .try_into()
            .map_err(|_| GpuprobeError::RuntimeError("conversion error".to_string()))?;

        let size_bytes = self
            .skel
            .maps
            .successful_allocs
            .lookup(&addr_key, MapFlags::ANY)
            .map_err(|_| GpuprobeError::RuntimeError("map lookup error".to_string()))?
            .unwrap_or(u64::to_ne_bytes(0u64).to_vec());

        let size_bytes: [u8; 8] = size_bytes
            .try_into()
            .map_err(|_| GpuprobeError::RuntimeError("conversion error".to_string()))?;

        let size = u64::from_ne_bytes(size_bytes);

        Ok((u64::from_ne_bytes(addr_key), size))
    }

    /// returns a map of outsanding cuda memory allocations - i.e. ones that
    /// have not yet been freed
    pub fn collect_data_memleak(&mut self) -> Result<MemleakData, GpuprobeError> {
        let outstanding_allocs: Vec<(u64, u64)> = self
            .skel
            .maps
            .successful_allocs
            .keys()
            .map(|addr| {
                self.addr_to_allocation(addr)
                    .expect("failed to get allocation")
            })
            .filter(|(_, size)| size > &0)
            .collect();

        Ok(MemleakData { outstanding_allocs })
    }
}
