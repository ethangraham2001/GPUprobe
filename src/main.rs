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
    let open_skel = skel_builder.open(&mut open_object).expect("failed to open skeleton");
    let skel = open_skel.load().expect("failed to load skeleton");
    let _link_enter = skel.progs.malloc_enter.attach_uprobe(false, -1, "/lib/x86_64-linux-gnu/libc.so.6", 0x00000000000ad640)
        .expect("unable to attach uprobe");

    let _link_exit = skel.progs.malloc_exit.attach_uprobe(true, -1, "/lib/x86_64-linux-gnu/libc.so.6", 0x00000000000ad640)
        .expect("unable to attach uprobe");

    let key0 = vec![0u8; 4];
    loop {
       let mallocs = skel.maps.num_mallocs.lookup(&key0, MapFlags::ANY)
            .map(|v| match v {
                None => { panic!("nothing there...") },
                Some(num_mallocs) =>  {
                    let array: [u8; 8] = num_mallocs.try_into().expect("oops");
                    u64::from_ne_bytes(array)
                }
            }).expect("unable to perform lookup");

        println!("num_mallocs={}", mallocs);
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}
