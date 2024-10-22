use std::error;
use std::mem::MaybeUninit;

use libbpf_rs::skel::OpenSkel;
use libbpf_rs::skel::SkelBuilder;

use libbpf_rs::MapCore as _;
use libbpf_rs::MapFlags;

mod gpuprobe {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/bpf/gpuprobe.skel.rs"
    ));
}

use gpuprobe::*;

fn main() {
    let skel_builder = GpuprobeSkelBuilder::default();
    let mut open_object = MaybeUninit::uninit();
    let open_skel = skel_builder
        .open(&mut open_object)
        .expect("unable to open skeleton");
    let skel = open_skel.load().expect("unable to load skeleton");

    let _malloc_uprobe_link = skel
        .progs
        .trace_cuda_malloc
        .attach_uprobe(
            false,
            -1,
            "/usr/local/cuda/lib64/libcudart.so",
            0x00000000000560c0,
        )
        .expect("unable to attach cuda uprobe");

    let _malloc_uretprobe_link = skel
        .progs
        .trace_cuda_malloc_ret
        .attach_uprobe(
            true,
            -1,
            "/usr/local/cuda/lib64/libcudart.so",
            0x00000000000560c0,
        )
        .expect("unable to attach cuda uretprobe");

    let _free_uprobe_link = skel
        .progs
        .trace_cuda_free
        .attach_uprobe(
            false,
            -1,
            "/usr/local/cuda/lib64/libcudart.so",
            0x00000000000568c0,
        )
        .expect("unable to attach cuda uretprobe");

    let _free_uretprobe_link = skel
        .progs
        .trace_cuda_free_ret
        .attach_uprobe(
            true,
            -1,
            "/usr/local/cuda/lib64/libcudart.so",
            0x00000000000568c0,
        )
        .expect("unable to attach cuda uretprobe");

    let key0 = vec![0u8; 4];
    loop {
        let num_cuda_malloc_calls = skel
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

        println!("total number of `cudaMalloc` calls: {}", num_cuda_malloc_calls);
        println!("outstanding allocations");

        skel.maps.successful_allocs.keys().for_each(|addr| {
            let addr_key: [u8; 8] = addr.try_into().expect("unexpected key size");
            let outstanding_allocs = skel
                .maps
                .successful_allocs
                .lookup(&addr_key, MapFlags::ANY)
                .expect("no value found");
            match outstanding_allocs {
                Some(count) => {
                    let count: [u8; 8] = count.try_into().expect("unable to convert result");
                    println!(
                        "0x{:8x} -> {} bytes",
                        u64::from_ne_bytes(addr_key),
                        u64::from_ne_bytes(count)
                    );
                }
                None => todo!(),
            }
        });

        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}
