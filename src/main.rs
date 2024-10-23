mod gpuprobe_memleak;

use libbpf_rs::MapCore as _;
use libbpf_rs::MapFlags;

mod gpuprobe {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/bpf/gpuprobe.skel.rs"
    ));
}

const WELCOME_MSG: &str = r#"
GPUprobe memleak utility
========================

"#;

fn main() -> Result<(), gpuprobe_memleak::GpuprobeMemleakError> {
    println!("{WELCOME_MSG}");

    let mut memleak = gpuprobe_memleak::GpuprobeMemleak::new()?;
    memleak.attach_uprobes()?;

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
