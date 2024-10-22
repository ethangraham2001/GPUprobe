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

/**
 * cuda memory allocations that have been launched
 */
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__type(key, u64);
	__type(value, size_t);
	__uint(max_entries, 10240);
} launched_allocs SEC(".maps");

/**
 * cuda memory allocations that have succeeded
 */
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__type(key, u64);
	__type(value, size_t);
	__uint(max_entries, 10240);
} successful_allocs SEC(".maps");

/**
 * The passed `dev_ptr` parameter can only be read from the inital uprobe. We
 * store it before execution so that we can read the virtual address of the
 * device in the uretprobe
 */
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__type(key, u32);
	__type(value, void**);
	__uint(max_entries, 10240);
} pid_to_dev_ptr SEC(".maps");

void **ptr_addr;

SEC("uprobe/cudaMalloc")
int trace_cuda_malloc(struct pt_regs *ctx)
{
	u32 key0 = 0;
	u64 *num_mallocs;
	u32 pid;
	void **dev_ptr;
	size_t size;

	num_mallocs = bpf_map_lookup_elem(&num_cuda_malloc_calls, &key0);
	if (num_mallocs) {
		__sync_fetch_and_add(num_mallocs, 1);
	}

	dev_ptr = (void**)PT_REGS_PARM1(ctx);
	size = (size_t)PT_REGS_PARM2(ctx);
	bpf_map_update_elem(&launched_allocs, &dev_ptr, &size, 0);

	pid = (u32)bpf_get_current_pid_tgid();
	bpf_map_update_elem(&pid_to_dev_ptr, &pid, &dev_ptr, 0);

	return 0;
}

SEC("uretprobe/cudaMalloc")
int trace_cuda_malloc_ret(struct pt_regs *ctx)
{
	int cuda_malloc_ret;
	u32 pid, key0 = 0;
	size_t *size, *num_failures;
	void *alloc_ptr;
	void **dev_ptr;
	void ***map_ptr;

	cuda_malloc_ret = (int)PT_REGS_RC(ctx);
	if (cuda_malloc_ret) {
		num_failures = bpf_map_lookup_elem(&cuda_malloc_failures, &key0);
		if (num_failures)
			__sync_fetch_and_add(num_failures, 1);
	}
	
	pid = (u32)bpf_get_current_pid_tgid();
	map_ptr = bpf_map_lookup_elem(&pid_to_dev_ptr, &pid);

	if (!map_ptr)
		return -1;

	dev_ptr = *map_ptr;
	if (bpf_probe_read_user(&alloc_ptr, sizeof(alloc_ptr), dev_ptr)) {
		return -1;
	}

	size = bpf_map_lookup_elem(&launched_allocs, &dev_ptr);
	if (!size) {
		return -1;
	}

	bpf_printk("host_ptr = 0x%p, device_ptr = 0x%p", dev_ptr, alloc_ptr);
	bpf_map_update_elem(&successful_allocs, &alloc_ptr, size, 0);

	return 0;
}

char LICENSE[] SEC("license") = "GPL";
