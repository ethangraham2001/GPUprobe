#include "vmlinux.h"
#include <bpf/bpf_core_read.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__type(key, u32);
	__type(value, u64);
	__uint(max_entries, 1);
} num_mallocs SEC(".maps");

/**
 * just keeps tracks of how many mallocs have been done
 */
SEC("uprobe")
int BPF_KPROBE(malloc_enter, size_t size)
{
	u32 key0 = 0;
	u64 *mallocs;

	char msg[] = "invoked program";
	bpf_trace_printk(msg, sizeof(msg));
	mallocs = (u64 *)bpf_map_lookup_elem(&num_mallocs, &key0);
	if (mallocs) {
		__sync_fetch_and_add(mallocs, 1);
	} else {
		char new_msg[] = "map lookup failed";
		bpf_trace_printk(new_msg, sizeof(new_msg));
	}

	return 0;
}

SEC("uprobe")
int BPF_KRETPROBE(malloc_exit, size_t size)
{
	bpf_printk("malloc exit");
	return 0;
}

char LICENSE[] SEC("license") = "GPL";
