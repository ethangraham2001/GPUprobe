pub mod collector;
pub mod gpuprobe_bandwidth_util;
pub mod gpuprobe_cudatrace;
pub mod gpuprobe_memleak;
pub mod metrics;
pub mod uprobe_data;

use metrics::GpuprobeMetrics;
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

use self::metrics::AddrLabel;

const LIBCUDART_PATH: &str = "/usr/local/cuda/lib64/libcudart.so";

/// Gpuuprobe wraps the eBPF program state, provides an interface for
/// attaching relevant uprobes, and exporting their metrics.
///
/// !!TODO!! maybe consider using orobouros self-referential instead of the
/// static lifetime
pub struct Gpuprobe {
    open_obj: Box<MaybeUninit<OpenObject>>,
    skel: GpuprobeSkel<'static>, // trust me bro
    links: GpuprobeLinks,
    opts: Opts,
    pub metrics: GpuprobeMetrics,
}

pub struct Opts {
    pub memleak: bool,
    pub cudatrace: bool,
    pub bandwidth_util: bool,
}

const DEFAULT_LINKS: GpuprobeLinks = GpuprobeLinks {
    trace_cuda_malloc: None,
    trace_cuda_malloc_ret: None,
    trace_cuda_free: None,
    trace_cuda_free_ret: None,
    trace_cuda_launch_kernel: None,
    trace_cuda_memcpy: None,
    trace_cuda_memcpy_ret: None,
};

impl Gpuprobe {
    /// returns a new Gpuprobe, or an initialization error on failure
    pub fn new(opts: Opts) -> Result<Self, GpuprobeError> {
        let skel_builder = GpuprobeSkelBuilder::default();
        let mut open_obj = Box::new(MaybeUninit::uninit());
        let open_obj_ptr = Box::as_mut(&mut open_obj) as *mut MaybeUninit<OpenObject>;
        let open_skel = unsafe {
            skel_builder
                .open(&mut *open_obj_ptr)
                .map_err(|_| GpuprobeError::OpenError)?
        };
        let skel = open_skel.load().map_err(|_| GpuprobeError::LoadError)?;
        let metrics = GpuprobeMetrics::new()?;
        Ok(Self {
            open_obj,
            skel,
            links: DEFAULT_LINKS,
            opts,
            metrics,
        })
    }

    /// Collects metrics from the attached uprobes and updates the the
    /// GpuMetrics Prometheus metrics.
    pub fn collect_metrics_uprobes(&mut self) -> Result<(), GpuprobeError> {
        // updates memory leak stats
        if self.opts.memleak {
            let memleak_data = self.collect_data_memleak()?;

            self.metrics
                .num_mallocs
                .set(memleak_data.outstanding_allocs.len() as i64);
            for (addr, count) in memleak_data.outstanding_allocs {
                self.metrics
                    .memleaks
                    .get_or_create(&AddrLabel { addr })
                    .set(count as i64);
            }
        }
        // updates kernel launch stats
        if self.opts.cudatrace {
            let cudatrace_data = self.collect_data_cudatrace()?;
            for (addr, count) in cudatrace_data.kernel_frequencies_histogram {
                self.metrics
                    .kernel_launches
                    .get_or_create(&AddrLabel { addr })
                    .set(count as i64);
            }
        }

        // !!TODO update bandwidth statistics as well
        Ok(())
    }

    /// Attaches relevant uprobes as defined in `opts`.
    /// # Example:
    /// ```rust
    /// let opts = Opts {
    ///     memleak: true,
    ///     cudatrace: false,
    ///     bandwidth_util: true,
    /// }
    ///
    /// // attaches memleak and bandwidth util uprobes and uretprobes
    /// gpuprobe.attach_uprobes_from_opts(&opts).unwrap();
    ///
    /// ```
    pub fn attach_uprobes(&mut self) -> Result<(), GpuprobeError> {
        if self.opts.memleak {
            self.attach_memleak_uprobes()?;
        }
        if self.opts.cudatrace {
            self.attach_cudatrace_uprobes()?;
        }
        if self.opts.bandwidth_util {
            self.attach_bandwidth_util_uprobes()?;
        }

        Ok(())
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
