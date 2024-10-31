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
	__type(value, void **);
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

	dev_ptr = (void **)PT_REGS_PARM1(ctx);
	size = (size_t)PT_REGS_PARM2(ctx);
	bpf_map_update_elem(&launched_allocs, &dev_ptr, &size, 0);

	pid = (u32)bpf_get_current_pid_tgid();
	return bpf_map_update_elem(&pid_to_dev_ptr, &pid, &dev_ptr, 0);
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
		num_failures =
			bpf_map_lookup_elem(&cuda_malloc_failures, &key0);
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

	return bpf_map_update_elem(&successful_allocs, &alloc_ptr, size, 0);
}

SEC("uprobe/cudaFree")
int trace_cuda_free(struct pt_regs *ctx)
{
	bpf_printk("called cudaFree");
	u32 pid;
	void *dev_ptr;

	dev_ptr = (void **)PT_REGS_PARM1(ctx);
	pid = (u32)bpf_get_current_pid_tgid();

	if (bpf_map_update_elem(&pid_to_dev_ptr, &pid, &dev_ptr, 0)) {
		bpf_printk("failed to update dev_ptr cudaFree");
	}

	return 0;
}

SEC("uretprobe/cudaFree")
int trace_cuda_free_ret(struct pt_regs *ctx)
{
	int cuda_free_ret;
	u32 pid;
	void *dev_ptr;
	void **map_ptr;
	size_t zero = 0;

	cuda_free_ret = PT_REGS_RC(ctx);
	if (cuda_free_ret) {
		return -1;
	}

	pid = (u32)bpf_get_current_pid_tgid();
	map_ptr = bpf_map_lookup_elem(&pid_to_dev_ptr, &pid);

	if (!map_ptr) {
		return -1;
	}

	dev_ptr = *map_ptr;
	if (bpf_map_update_elem(&successful_allocs, &dev_ptr, &zero, 0)) {
		return -1;
	}

	return 0;
}

/**
 * maps a kernel function address to the number of times it has been called
 */
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__type(key, void*);
	__type(value, size_t);
	__uint(max_entries, 10240);
} kernel_calls_hist SEC(".maps");

SEC("uprobe/cudaKernelLaunch")
int trace_cuda_launch_kernel(struct pt_regs *ctx)
{
	void *func;
	u64 one = 1;
	size_t *num_launches;

	func = (void *)PT_REGS_PARM1(ctx);
	if (!(num_launches = bpf_map_lookup_elem(&kernel_calls_hist, &func))) {
		bpf_map_update_elem(&kernel_calls_hist, &func, &one, 0);
	} else {
		__sync_fetch_and_add(num_launches, 1);
	}
	return 0;
}


/**
 * redefinition of `enum cudaMemcpyKind` in driver_types.h.
 */
enum memcpy_kind {
	D2D = 0,	// device to device
	D2H = 1,	// device to host
	H2D = 2,	// host to device
	H2H = 3,	// host to host
	DEFAULT = 4, // inferred from pointer type at runtime
};

struct cuda_memcpy {
	__u64 start_time;
	__u64 end_time;
	void* dst;
	void* src;
	size_t count;
	enum memcpy_kind kind;
};

/**
 * Maps a pid to an information on an incomplete cudaMemcpy call. This is 
 * needed because we cannot access the input arguments inside of the uretprobe.
 */
struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__type(key, __u32);
	__type(value, struct cuda_memcpy);
	__uint(max_entries, 10240);
} pid_to_memcpy SEC(".maps");

/**
 * Queue of successful cudaMemcpy calls to be processed from userspace.
 */
struct {
	__uint(type, BPF_MAP_TYPE_QUEUE);
	__uint(key_size, 0);
	__uint(value_size, sizeof(struct cuda_memcpy));
	__uint(max_entries, 10240);
} successful_cuda_memcpy_q SEC(".maps");

/**
 * This function exhibits synchronous behavior in MOST cases as specified by
 * Nvidia documentation. It is under the assumption that this call is 
 * synchronous that we compute the average memory bandwidth of a transfer as:
 *		avg_throughput = count /  (end - start)
 */
SEC("uprobe/cudaMemcpy")
int trace_cuda_memcpy(struct pt_regs *ctx)
{
	void* dst = (void*) PT_REGS_PARM1(ctx);
	void *src = (void*) PT_REGS_PARM2(ctx);
	size_t count = PT_REGS_PARM3(ctx);
	enum memcpy_kind kind = PT_REGS_PARM4(ctx);
	__u32 pid = (__u32)bpf_get_current_pid_tgid();

	/* no host-side synchronization is performed in the D2D case - as a result,
	 * we cannot compute average throughput using information available from
	 * this uprobe. If the DEFAULT argument is passed, we cannot make any 
	 * assumption on the direction of the transfer */
	if (kind == D2D || kind == DEFAULT)
		return 0;

	struct cuda_memcpy in_progress_memcpy = {
		.start_time = bpf_ktime_get_ns(),
		.dst = dst,
		.src = src,
		.count = count,
		.kind = kind
	};

	if (bpf_map_update_elem(&pid_to_memcpy, &pid, &in_progress_memcpy, 0)) {
		return -1;
	}

	return 0;
}

SEC("uretprobe/cudaMemcpy")
int trace_cuda_memcpy_ret(struct pt_regs *ctx)
{
	__u32 ret = PT_REGS_RC(ctx);
	__u32 pid = (__u32)bpf_get_current_pid_tgid();
	struct cuda_memcpy *exited_memcpy;

	if (ret) {
		return -1;
	}

	exited_memcpy = (struct cuda_memcpy *) bpf_map_lookup_elem(&pid_to_memcpy, &pid);
	if (!exited_memcpy) {
		return -1;
	}

	if (bpf_map_delete_elem(&pid_to_memcpy, &pid)) {
		return -1;
	}

	exited_memcpy->end_time = bpf_ktime_get_ns();
	if (bpf_map_push_elem(&successful_cuda_memcpy_q, exited_memcpy, 0)) {
		return -1;
	}

	return 0;
}

char LICENSE[] SEC("license") = "GPL";
