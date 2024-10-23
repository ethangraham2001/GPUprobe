use std::error::Error;
use std::mem::MaybeUninit;

use libbpf_rs::{
    skel::{OpenSkel, SkelBuilder},
    MapCore, MapFlags, OpenObject,
};

mod gpuprobe {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/bpf/gpuprobe.skel.rs"
    ));
}

use gpuprobe::*;

// TODO maybe consider using orobouros self-referential
pub struct GpuprobeMemleak {
    open_obj: Box<MaybeUninit<OpenObject>>,
    pub skel: GpuprobeSkel<'static>, // trust me bro
    links: GpuprobeLinks,
}

const LIBCUDART_PATH: &str = "/usr/local/cuda/lib64/libcudart.so";

impl GpuprobeMemleak {
    /// returns a new GpuprobeMemleak or an initialization error on failure
    pub fn new() -> Result<Self, GpuprobeMemleakError> {
        let skel_builder = GpuprobeSkelBuilder::default();
        let mut open_obj = Box::new(MaybeUninit::uninit());
        let open_obj_ptr = Box::as_mut(&mut open_obj) as *mut MaybeUninit<OpenObject>;
        let open_skel = unsafe {
            skel_builder
                .open(&mut *open_obj_ptr)
                .map_err(|_| GpuprobeMemleakError::OpenError)?
        };
        let skel = open_skel
            .load()
            .map_err(|_| GpuprobeMemleakError::LoadError)?;
        Ok(Self {
            open_obj,
            skel,
            links: GpuprobeLinks {
                trace_cuda_malloc: None,
                trace_cuda_malloc_ret: None,
                trace_cuda_free: None,
                trace_cuda_free_ret: None,
            },
        })
    }

    /// attaches probes or returns an initialization error on failure
    pub fn attach_uprobes(&mut self) -> Result<(), GpuprobeMemleakError> {
        let cuda_malloc_uprobe_link = self
            .skel
            .progs
            .trace_cuda_malloc
            .attach_uprobe(false, -1, LIBCUDART_PATH, 0x00000000000560c0)
            .map_err(|_| GpuprobeMemleakError::AttachError)?;

        let cuda_malloc_uretprobe_link = self
            .skel
            .progs
            .trace_cuda_malloc_ret
            .attach_uprobe(true, -1, LIBCUDART_PATH, 0x00000000000560c0)
            .map_err(|_| GpuprobeMemleakError::AttachError)?;

        let cuda_free_uprobe_link = self
            .skel
            .progs
            .trace_cuda_free
            .attach_uprobe(false, -1, LIBCUDART_PATH, 0x00000000000568c0)
            .map_err(|_| GpuprobeMemleakError::AttachError)?;

        let cuda_free_uretprobe_link = self
            .skel
            .progs
            .trace_cuda_free_ret
            .attach_uprobe(true, -1, LIBCUDART_PATH, 0x00000000000568c0)
            .map_err(|_| GpuprobeMemleakError::AttachError)?;

        self.links = GpuprobeLinks {
            trace_cuda_malloc: Some(cuda_malloc_uprobe_link),
            trace_cuda_malloc_ret: Some(cuda_malloc_uretprobe_link),
            trace_cuda_free: Some(cuda_free_uprobe_link),
            trace_cuda_free_ret: Some(cuda_free_uretprobe_link),
        };

        Ok(())
    }

    /// converts an address to its given allocation information by performing
    /// a lookup in the map of successful allocations
    fn addr_to_allocation(&self, addr: Vec<u8>) -> Result<(u64, u64), GpuprobeMemleakError> {
        let addr_key: [u8; 8] = addr
            .try_into()
            .map_err(|_| GpuprobeMemleakError::RuntimeError("conversion error".to_string()))?;

        let size_bytes = self
            .skel
            .maps
            .successful_allocs
            .lookup(&addr_key, MapFlags::ANY)
            .map_err(|_| GpuprobeMemleakError::RuntimeError("map lookup error".to_string()))?
            .unwrap_or(u64::to_ne_bytes(0u64).to_vec());

        let size_bytes: [u8; 8] = size_bytes
            .try_into()
            .map_err(|_| GpuprobeMemleakError::RuntimeError("conversion error".to_string()))?;

        let size = u64::from_ne_bytes(size_bytes);

        Ok((u64::from_ne_bytes(addr_key), size))
    }

    /// returns a map of outsanding cuda memory allocations - i.e. ones that
    /// have not yet been freed
    pub fn get_outstanding_allocs(&mut self) -> Result<Vec<(u64, u64)>, Box<dyn Error>> {
        let output: Vec<(u64, u64)> = self
            .skel
            .maps
            .successful_allocs
            .keys()
            .map(|addr| {
                self.addr_to_allocation(addr)
                    .expect("unable to convert alloc")
            })
            .filter(|(_, size)| size > &0)
            .collect();
        Ok(output)
    }
}

#[derive(Debug)]
pub enum GpuprobeMemleakError {
    OpenError,
    LoadError,
    AttachError,
    RuntimeError(String),
}

impl std::fmt::Display for GpuprobeMemleakError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuprobeMemleakError::OpenError => write!(f, "failed to open skeleton"),
            GpuprobeMemleakError::LoadError => write!(f, "failed to load skeleton"),
            GpuprobeMemleakError::AttachError => write!(f, "failed to attach skeleton"),
            GpuprobeMemleakError::RuntimeError(reason) => write!(f, "runtime error: {}", reason),
        }
    }
}
