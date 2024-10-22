#include "vmlinux.h"
#include <bpf/bpf_core_read.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__type(key, u32);
	__type(value, u64);
	__uint(max_entries, 1);
} num_cuda_malloc_calls SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_ARRAY);
	__type(key, u32);
	__type(value, u64);
	__uint(max_entries, 1);
} cuda_malloc_failures SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__type(key, u64);
	__type(value, size_t);
	__uint(max_entries, 10240);
} alive_allocs SEC(".maps");

void **ptr_addr;

SEC("uprobe/cudaMalloc")
int trace_cuda_malloc(struct pt_regs *ctx)
{
	u32 key0 = 0;
	u64 *num_mallocs;
	void **dev_ptr;
	size_t size;

	num_mallocs = bpf_map_lookup_elem(&num_cuda_malloc_calls, &key0);
	if (num_mallocs) {
		__sync_fetch_and_add(num_mallocs, 1);
	}

	dev_ptr = (void**)PT_REGS_PARM1(ctx);
	size = (size_t)PT_REGS_PARM2(ctx);

	if (bpf_map_update_elem(&alive_allocs, &dev_ptr, &size, 0b0)) {
		bpf_printk("unable to update hash map");
	}

	return 0;
}

SEC("uretprobe/cudaMalloc")
int trace_cuda_malloc_ret(struct pt_regs *ctx)
{
	u32 key0 = 0;
	u64 *num_mallocs;
	void **dev_ptr;
	void *allocated_ptr;
	size_t size;

	if (bpf_probe_read_user(&allocated_ptr, sizeof(allocated_ptr), \
				ptr_addr)) {
		bpf_printk("unable to read from dev_ptr");
		return 0;
	}

	if (allocated_ptr) {
		bpf_printk("allocated_ptr=%p", allocated_ptr);
	}

	return 0;
}

char LICENSE[] SEC("license") = "GPL";
