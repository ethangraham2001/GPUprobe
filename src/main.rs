mod gpuprobe;

use clap::Parser;

#[derive(Parser)]
#[command(author, version, about, long_about = None, arg_required_else_help = true)]
struct Args {
    /// Detects leaking calls to cudaMalloc from the CUDA runtime API.
    #[arg(long, exclusive = false)]
    memleak: bool,

    /// Maintains a histogram on frequencies of cuda kernel launches.
    #[arg(long, exclusive = false)]
    cudatrace: bool,

    /// Approximates bandwidth utilization of cudaMemcpy.
    #[arg(long, exclusive = false)]
    bandwidth_util: bool,
}

fn main() -> Result<(), gpuprobe::GpuprobeError> {
    let args = Args::parse();
    let opts = gpuprobe::Opts {
        memleak: args.memleak,
        cudatrace: args.cudatrace,
        bandwidth_util: args.bandwidth_util,
    };
    let mut gpuprobe = gpuprobe::Gpuprobe::new(opts)?;
    gpuprobe.attach_uprobes()?;

    loop {
        gpuprobe.collect_data_uprobes()?;
        println!("========================\n");
        // !!TODO!! make this variable
        std::thread::sleep(std::time::Duration::from_secs(5));
    }
}
