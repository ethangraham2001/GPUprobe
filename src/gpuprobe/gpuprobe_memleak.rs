use std::error::Error;

use libbpf_rs::{MapCore, MapFlags};

use super::uprobe_data::MemleakData;
use super::{Gpuprobe, GpuprobeError, LIBCUDART_PATH};
use std::collections::HashMap;

/// Wraps the state that is maintained by the memleak program
pub struct MemleakProgramData {
    allocations: HashMap<u32, MemleakEvent>,
}

impl MemleakProgramData {
    pub fn new() -> Self {
        return MemleakProgramData {
            allocations: HashMap::new(),
        };
    }
}

struct MemleakEvent {
    start: u64,
    end: u64,
    device_addr: u64,
    size: u64,
    pid: u32,
    ret: i32,
    event_type: i32,
}

impl MemleakEvent {
    /// Constructs a MemleakEvent struct from a raw byte array and returns it,
    /// or None if the byte array isn't correctly sized.
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

impl std::fmt::Display for MemleakEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{\n")?;

        if self.event_type == 0 {
            write!(f, "\ttype: {}\n", "cudaMalloc")?;
        } else if self.event_type == 1 {
            write!(f, "\ttype: {}\n", "cudaFree")?;
        }

        write!(f, "\tsize: {}\n", self.size)?;
        write!(f, "\tpid: {}\n", self.pid)?;
        write!(f, "\tstart: {}\n", self.start)?;
        write!(f, "\tend: {}\n", self.end)?;
        write!(f, "\tdev_addr: 0x{:x}\n", self.device_addr)?;
        write!(f, "\tret: {}\n", self.ret)?;

        write!(f, "}}")
    }
}

/// contains implementation for the memleak program
impl Gpuprobe {
    /// attaches uprobes for the memleak program, or returns an error on
    /// failure
    pub fn attach_memleak_uprobes(&mut self) -> Result<(), GpuprobeError> {
        let cuda_malloc_uprobe_link = self
            .skel
            .skel
            .progs
            .memleak_cuda_malloc
            .attach_uprobe(false, -1, LIBCUDART_PATH, 0x00000000000560c0)
            .map_err(|_| GpuprobeError::AttachError)?;

        let cuda_malloc_uretprobe_link = self
            .skel
            .skel
            .progs
            .memleak_cuda_malloc_ret
            .attach_uprobe(true, -1, LIBCUDART_PATH, 0x00000000000560c0)
            .map_err(|_| GpuprobeError::AttachError)?;

        let cuda_free_uprobe_link = self
            .skel
            .skel
            .progs
            .trace_cuda_free
            .attach_uprobe(false, -1, LIBCUDART_PATH, 0x00000000000568c0)
            .map_err(|_| GpuprobeError::AttachError)?;

        let cuda_free_uretprobe_link = self
            .skel
            .skel
            .progs
            .trace_cuda_free_ret
            .attach_uprobe(true, -1, LIBCUDART_PATH, 0x00000000000568c0)
            .map_err(|_| GpuprobeError::AttachError)?;

        self.links.links.memleak_cuda_malloc = Some(cuda_malloc_uprobe_link);
        self.links.links.memleak_cuda_malloc_ret = Some(cuda_malloc_uretprobe_link);
        self.links.links.trace_cuda_free = Some(cuda_free_uprobe_link);
        self.links.links.trace_cuda_free_ret = Some(cuda_free_uretprobe_link);
        Ok(())
    }

    fn update_memleak_prog_data(&mut self, data: MemleakEvent) -> Result<(), GpuprobeError> {
        if data.event_type == 0 {
        } else if data.event_type == 1 {
        } else {
            return Err(GpuprobeError::RuntimeError(
                    "invalid memleak event type".to_string()
            ));
        }
        Ok(())
    }

    pub fn consume_memleak_queue(&mut self) -> Result<(), GpuprobeError> {
        let key: [u8; 0] = []; // key size must be zero for BPF_MAP_TYPE_QUEUE
                               // `lookup_and_delete` calls.
        while let Ok(opt) = self
            .skel
            .skel
            .maps
            .memleak_events_queue
            .lookup_and_delete(&key)
        {
            let event_bytes = match opt {
                Some(b) => b,
                None => { return Ok(()); }
            };
            let event = match MemleakEvent::from_bytes(&event_bytes) {
                Some(e) => e,
                None => {
                    return Err(GpuprobeError::RuntimeError(
                        "unable to construct MemleakEvent from bytes".to_string(),
                    ));
                }
            };

            println!("{}", event);
        }
        Ok(())
    }

    /// returns a map of outsanding cuda memory allocations - i.e. ones that
    /// have not yet been freed
    pub fn collect_data_memleak(&mut self) -> Result<MemleakData, GpuprobeError> {
        self.consume_memleak_queue()?;
        //let outstanding_allocs: Vec<(u64, u64)> = self
        //    .skel
        //    .skel
        //    .maps
        //    .successful_allocs
        //    .keys()
        //    .map(|addr| {
        //        self.addr_to_allocation(addr)
        //            .expect("failed to get allocation")
        //    })
        //    .filter(|(_, size)| size > &0)
        //    .collect();

        // !!TODO!! this is just for compatibility
        Ok(MemleakData {
            outstanding_allocs: vec![],
        })
    }
}
