pub mod gpuprobe_cudatrace;
pub mod gpuprobe_memleak;

use std::mem::MaybeUninit;

use libbpf_rs::{
    skel::{OpenSkel, SkelBuilder},
    OpenObject,
};

mod gpuprobe {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/bpf/gpuprobe.skel.rs"
    ));
}
use gpuprobe::*;

// TODO maybe consider using orobouros self-referential
pub struct Gpuprobe {
    open_obj: Box<MaybeUninit<OpenObject>>,
    pub skel: GpuprobeSkel<'static>, // trust me bro
    links: GpuprobeLinks,
}

impl Gpuprobe {
    /// returns a new Gpuprobe or an initialization error on failure
    pub fn new() -> Result<Self, GpuprobeError> {
        let skel_builder = GpuprobeSkelBuilder::default();
        let mut open_obj = Box::new(MaybeUninit::uninit());
        let open_obj_ptr = Box::as_mut(&mut open_obj) as *mut MaybeUninit<OpenObject>;
        let open_skel = unsafe {
            skel_builder
                .open(&mut *open_obj_ptr)
                .map_err(|_| GpuprobeError::OpenError)?
        };
        let skel = open_skel.load().map_err(|_| GpuprobeError::LoadError)?;
        Ok(Self {
            open_obj,
            skel,
            links: GpuprobeLinks {
                trace_cuda_malloc: None,
                trace_cuda_malloc_ret: None,
                trace_cuda_free: None,
                trace_cuda_free_ret: None,
                trace_cuda_launch_kernel: None,
            },
        })
    }
}

#[derive(Debug)]
pub enum GpuprobeError {
    OpenError,
    LoadError,
    AttachError,
    RuntimeError(String),
}

impl std::fmt::Display for GpuprobeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuprobeError::OpenError => write!(f, "failed to open skeleton"),
            GpuprobeError::LoadError => write!(f, "failed to load skeleton"),
            GpuprobeError::AttachError => write!(f, "failed to attach skeleton"),
            GpuprobeError::RuntimeError(reason) => write!(f, "runtime error: {}", reason),
        }
    }
}
