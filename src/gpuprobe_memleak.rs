use std::error::Error;
use std::mem::MaybeUninit;

use libbpf_rs::skel::OpenSkel;
use libbpf_rs::skel::SkelBuilder;
use libbpf_rs::OpenObject;

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

impl GpuprobeMemleak {
    /// returns a new GpuprobeMemleak or an initialization error on failure
    pub fn new() -> Result<Self, InitError> {
        let skel_builder = GpuprobeSkelBuilder::default();
        let mut open_obj = Box::new(MaybeUninit::uninit());
        let open_obj_ptr = Box::as_mut(&mut open_obj) as *mut MaybeUninit<OpenObject>;
        let open_skel = unsafe {
            skel_builder
                .open(&mut *open_obj_ptr)
                .map_err(|_| InitError::OpenError)?
        };
        let skel = open_skel.load().map_err(|_| InitError::LoadError)?;
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
    pub fn attach_uprobes(&mut self) -> Result<(), InitError> {
        let cuda_malloc_uprobe_link = self
            .skel
            .progs
            .trace_cuda_malloc
            .attach_uprobe(
                false,
                -1,
                "/usr/local/cuda/lib64/libcudart.so",
                0x00000000000560c0,
            )
            .map_err(|_| InitError::AttachError)?;

        let cuda_malloc_uretprobe_link = self
            .skel
            .progs
            .trace_cuda_malloc_ret
            .attach_uprobe(
                true,
                -1,
                "/usr/local/cuda/lib64/libcudart.so",
                0x00000000000560c0,
            )
            .map_err(|_| InitError::AttachError)?;

        let cuda_free_uprobe_link = self
            .skel
            .progs
            .trace_cuda_free
            .attach_uprobe(
                false,
                -1,
                "/usr/local/cuda/lib64/libcudart.so",
                0x00000000000568c0,
            )
            .map_err(|_| InitError::AttachError)?;

        let cuda_free_uretprobe_link = self
            .skel
            .progs
            .trace_cuda_free_ret
            .attach_uprobe(
                true,
                -1,
                "/usr/local/cuda/lib64/libcudart.so",
                0x00000000000568c0,
            )
            .map_err(|_| InitError::AttachError)?;

        self.links = GpuprobeLinks {
            trace_cuda_malloc: Some(cuda_malloc_uprobe_link),
            trace_cuda_malloc_ret: Some(cuda_malloc_uretprobe_link),
            trace_cuda_free: Some(cuda_free_uprobe_link),
            trace_cuda_free_ret: Some(cuda_free_uretprobe_link),
        };

        Ok(())
    }
}

#[derive(Debug)]
pub enum InitError {
    OpenError,
    LoadError,
    AttachError,
}

impl std::fmt::Display for InitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InitError::OpenError => write!(f, "failed to open skeleton"),
            InitError::LoadError => write!(f, "failed to load skeleton"),
            InitError::AttachError => write!(f, "failed to attach skeleton"),
        }
    }
}
