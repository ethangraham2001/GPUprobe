mod gpuprobe;

use libbpf_rs::MapCore as _;
use libbpf_rs::MapFlags;

use clap::Parser;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Enable memory leak detection mode
    #[arg(long, exclusive = true)]
    memleak: bool,

    /// Enable CUDA trace mode
    #[arg(long, exclusive = true)]
    cudatrace: bool,
}

const USAGE: &str = r#"
GPUprobe usage:
    ./gpuprobe --memleak # runs the memleak utility
    ./gpuprobe --cudatrace # runs the cudatrace utility
"#;

const WELCOME_MEMLEAK_MSG: &str = r#"
GPUprobe memleak utility
========================

"#;

const WELCOME_CUDATRACE_MSG: &str = r#"
GPUprobe cudatrace utility
========================

"#;

fn usage() {
    print!("{USAGE}");
    std::process::exit(1);
}

fn main() -> Result<(), gpuprobe::GpuprobeError> {
    let args = Args::parse();

    match (args.memleak, args.cudatrace) {
        (true, false) => memleak_prog(),
        (false, true) => cudatrace_prog(),
        (_, _) => Ok(usage()),
    }
}

fn memleak_prog() -> Result<(), gpuprobe::GpuprobeError> {
    println!("{WELCOME_MEMLEAK_MSG}");

    let mut memleak = gpuprobe::Gpuprobe::new()?;
    memleak.attach_memleak_uprobes()?;

    let key0 = vec![0u8; 4];
    loop {
        let num_cuda_malloc_calls = memleak
            .skel
            .maps
            .num_cuda_malloc_calls
            .lookup(&key0, MapFlags::ANY)
            .map(|v| match v {
                None => {
                    panic!("nothing found")
                }
                Some(num_mallocs) => {
                    let array: [u8; 8] = num_mallocs.try_into().expect("unable to convert arr");
                    u64::from_ne_bytes(array)
                }
            })
            .expect("unable to perform map lookup");

        println!(
            "total number of `cudaMalloc` calls: {}",
            num_cuda_malloc_calls
        );

        let oustanding_allocs = memleak
            .get_outstanding_allocs()
            .expect("unable to get allocations");

        let leaked_bytes = oustanding_allocs
            .iter()
            .fold(0u64, |total, (_, size)| total + size);

        println!(
            "{} bytes leaked from {} cuda memory allocation(s)",
            leaked_bytes,
            oustanding_allocs.len()
        );

        oustanding_allocs.iter().for_each(|(addr, size)| {
            println!("\t0x{addr:x}: {size} bytes");
        });

        println!("========================\n");
        std::thread::sleep(std::time::Duration::from_secs(5));
    }
}

fn cudatrace_prog() -> Result<(), gpuprobe::GpuprobeError> {
    println!("{WELCOME_CUDATRACE_MSG}");

    let mut cudatrace = gpuprobe::Gpuprobe::new()?;
    cudatrace.attach_cudatrace_uprobes()?;

    loop {
        let kernel_launches = cudatrace.get_kernel_launch_frequencies()?;
        let num_launches = kernel_launches.iter()
            .fold(0u64, |total, (_, count)| total + count);

        println!("{} `cudaLaunchKernel` calls for {} kernels", num_launches, kernel_launches.len());
        kernel_launches.iter().for_each({|(addr, count)|
            println!("\t0x{addr:x}: {count} launches")
        });

        println!("========================\n");
        std::thread::sleep(std::time::Duration::from_secs(5));
    }
}
