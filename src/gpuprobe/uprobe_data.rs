use super::gpuprobe_bandwidth_util::CudaMemcpy;

/// Defines the data collected by a uprobe / uretprobe
pub trait UprobeData: std::fmt::Display /* add prometheus */ {}

/// defines the data that is collected in a cycle of the memleak program
pub struct MemleakData {
    /// a Vec of `(addr, count)` where:
    ///     - `addr` is the virtual address (on-device) of the allocation
    ///     - `count` is the number of bytes associated to that allocation
    pub outstanding_allocs: Vec<(u64, u64)>,
}

impl UprobeData for MemleakData {}

/// defines the data that is collected in a cycle of the cudatrace program
pub struct CudaTraceData {
    /// a Vec of `(addr, count)` where:
    ///     - `addr` is the function pointer to the launched cuda kernel
    ///     - `count` is the number of times that that kernel was launched
    pub kernel_frequencies_histogram: Vec<(u64, u64)>,
}

impl UprobeData for CudaTraceData {}

/// defines the data that is collected in a cycle of the cudatrace program
pub struct BandwidthUtilData {
    /// a Vec of `CudaMemcpy` calls
    pub cuda_memcpys: Vec<CudaMemcpy>,
}

impl UprobeData for BandwidthUtilData {}

impl std::fmt::Display for MemleakData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let leaked_bytes = self
            .outstanding_allocs
            .iter()
            .fold(0u64, |total, (_, size)| total + size);

        writeln!(
            f,
            "{} bytes leaked from {} cuda memory allocation(s)",
            leaked_bytes,
            self.outstanding_allocs.len()
        )?;

        Ok(for (addr, size) in self.outstanding_allocs.iter() {
            writeln!(f, "\t0x{addr:x}: {size} bytes")?;
        })
    }
}

impl std::fmt::Display for CudaTraceData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kernel_launches = &self.kernel_frequencies_histogram;
        let num_launches = kernel_launches
            .iter()
            .fold(0u64, |total, (_, count)| total + count);

        writeln!(
            f,
            "{} `cudaLaunchKernel` calls for {} kernels",
            num_launches,
            kernel_launches.len()
        )?;
        kernel_launches.iter().for_each(|(addr, count)| {
            writeln!(f, "\t0x{addr:x}: {count} launches");
        });
        Ok(())
    }
}

impl std::fmt::Display for BandwidthUtilData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let calls = &self.cuda_memcpys;
        if calls.len() == 0 {
            return Ok(());
        }

        writeln!(f, "Traced {} cudaMemcpy calls", calls.len())?;
        calls.iter().for_each(|c| {
            let bandwidth_util = c.compute_bandwidth_util().unwrap_or(0.0);
            let delta = (c.end_time - c.start_time) as f64 / 1e9;
            writeln!(
                f,
                "\t{} {:.5} bytes/sec for {:.5} secs",
                c.kind_to_str(),
                bandwidth_util,
                delta
            );
        });
        Ok(())
    }
}
