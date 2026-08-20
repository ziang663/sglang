// flashboot — pinned_stage implementation. See pinned_stage.h.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE  // MAP_ANONYMOUS, MAP_NORESERVE
#endif
#include "flashboot/pinned_stage.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#include <fcntl.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/vfs.h>
#include <unistd.h>

#include <cuda_runtime.h>

#ifndef SYS_mbind
#if defined(__aarch64__)
#define SYS_mbind 235
#elif defined(__x86_64__)
#define SYS_mbind 237
#endif
#endif
#ifndef MPOL_BIND
#define MPOL_BIND 2
#endif
#ifndef MPOL_MF_MOVE
#define MPOL_MF_MOVE (1 << 1)
#endif

namespace flashboot {
namespace {

using clk = std::chrono::steady_clock;
double secs(clk::time_point a, clk::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}
constexpr size_t kAlign = 4096;
size_t round_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

void cuda_check(cudaError_t e, const char* what) {
  if (e != cudaSuccess)
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

// NUMA-bind [addr, addr+len) to `node` (mbind), best-effort. torch's pin_memory
// ignores set_mempolicy, so this is how alloc_pinned_on_node forces the socket.
bool bind_to_node(void* addr, size_t len, int node) {
#ifdef SYS_mbind
  if (node < 0 || node >= (int)(8 * sizeof(unsigned long))) return false;
  unsigned long mask = 1UL << node;
  long r = syscall(SYS_mbind, addr, len, MPOL_BIND, &mask, 8 * sizeof(mask),
                   MPOL_MF_MOVE);
  if (r != 0)
    fprintf(stderr, "[pinned_stage] WARN: mbind(node=%d): %s\n", node,
            strerror(errno));
  return r == 0;
#else
  (void)addr; (void)len; (void)node; return false;
#endif
}

}  // namespace

// ===================== host(pinned) -> GPU ======================
H2DStats copy_pinned_to_gpu(void* dst_gpu, void* src_host, size_t nbytes,
                            const H2DConfig& cfg) {
  H2DStats stats{};
  const size_t chunk = round_up(cfg.chunk_bytes, kAlign);
  const int S = std::max(1, cfg.streams);
  cuda_check(cudaSetDevice(cfg.device), "cudaSetDevice");

  std::vector<cudaStream_t> streams(S, nullptr);
  struct Cleanup {
    std::vector<cudaStream_t>& streams;
    ~Cleanup() { for (auto s : streams) if (s) cudaStreamDestroy(s); }
  } cleanup{streams};
  for (int i = 0; i < S; ++i)
    cuda_check(cudaStreamCreate(&streams[i]), "cudaStreamCreate");

  // chunk jobs
  struct Job { size_t off, len; };
  std::vector<Job> jobs;
  for (size_t off = 0; off < nbytes; off += chunk)
    jobs.push_back({off, std::min(chunk, nbytes - off)});

  std::atomic<size_t> next{0};
  std::vector<std::string> errs(S);
  auto worker = [&](int sid) {
    cudaStream_t st = streams[sid];
    try {
      cuda_check(cudaSetDevice(cfg.device), "cudaSetDevice(worker)");
      for (;;) {
        size_t k = next.fetch_add(1);
        if (k >= jobs.size()) break;
        char* h = (char*)src_host + jobs[k].off;
        char* d = (char*)dst_gpu + jobs[k].off;
        size_t n = jobs[k].len;
        // Register length rounded to a page (chunk offsets are page-aligned and
        // the region is page-padded, so this stays in-bounds). Copy only n.
        size_t reglen = round_up(n, kAlign);
        if (cfg.register_in_chunks)
          cuda_check(cudaHostRegister(h, reglen, cudaHostRegisterDefault),
                     "cudaHostRegister(chunk)");
        cuda_check(cudaMemcpyAsync(d, h, n, cudaMemcpyHostToDevice, st),
                   "cudaMemcpyAsync");
        cuda_check(cudaStreamSynchronize(st), "cudaStreamSynchronize");
        if (cfg.register_in_chunks) cudaHostUnregister(h);  // bound footprint
      }
    } catch (const std::exception& e) { errs[sid] = e.what(); }
  };

  auto t0 = clk::now();
  std::vector<std::thread> ths;
  for (int i = 0; i < S; ++i) ths.emplace_back(worker, i);
  for (auto& t : ths) t.join();
  cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
  auto t1 = clk::now();

  for (int i = 0; i < S; ++i)
    if (!errs[i].empty())
      throw std::runtime_error("h2d worker " + std::to_string(i) + ": " +
                               errs[i]);

  stats.total_bytes = nbytes;
  stats.wall_s = secs(t0, t1);
  stats.gbps = stats.wall_s > 0 ? nbytes / 1e9 / stats.wall_s : 0;
  stats.streams = S;
  stats.registered_in_chunks = cfg.register_in_chunks;
  return stats;
}

// ===================== NUMA / in-RAM helpers ==================
namespace {

// tmpfs/ramfs magic numbers (linux/magic.h).
constexpr long kTmpfsMagic = 0x01021994L;
constexpr long kRamfsMagic = 0x858458F6L;

// Process-wide mmap cache: a tmpfs part is mmap'd ONCE (MAP_PRIVATE, read-only) and
// kept for the process lifetime. munmap of the ~48GB region would tear down ~12M PTEs
// holding mmap_sem and, under N-rank concurrency, stalls the load critical path; the
// mapping is pure virtual address space sharing the resident tmpfs pages (zero RSS).
std::mutex g_mmap_mu;
std::unordered_map<std::string, std::pair<void*, size_t>> g_mmap_cache;

const char* mmap_file_cached(const std::string& path) {
  std::lock_guard<std::mutex> lk(g_mmap_mu);
  auto it = g_mmap_cache.find(path);
  if (it != g_mmap_cache.end()) return static_cast<const char*>(it->second.first);
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0)
    throw std::runtime_error("read_file_into: open " + path + ": " + strerror(errno));
  struct stat st {};
  if (fstat(fd, &st) != 0) { close(fd); throw std::runtime_error("read_file_into: fstat"); }
  size_t file_size = static_cast<size_t>(st.st_size);
  void* mapping = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (mapping == MAP_FAILED)
    throw std::runtime_error("read_file_into: mmap " + path + ": " + strerror(errno));
  g_mmap_cache[path] = {mapping, file_size};
  return static_cast<const char*>(mapping);
}

// CPUs of a NUMA node, parsed from /sys (no libnuma). e.g. "0-3,8-11".
std::vector<int> cpus_for_node(int node) {
  std::vector<int> cpus;
  char sysfs_path[128];
  snprintf(sysfs_path, sizeof(sysfs_path),
           "/sys/devices/system/node/node%d/cpulist", node);
  FILE* cpulist = fopen(sysfs_path, "r");
  if (!cpulist) return cpus;
  char line[8192] = {0};
  if (fgets(line, sizeof(line), cpulist)) {
    char* cursor = line;
    while (*cursor && *cursor != '\n') {
      int range_first = (int)strtol(cursor, &cursor, 10);
      int range_last = range_first;
      if (*cursor == '-') { ++cursor; range_last = (int)strtol(cursor, &cursor, 10); }
      for (int cpu = range_first; cpu <= range_last; ++cpu) cpus.push_back(cpu);
      if (*cursor == ',') ++cursor; else break;
    }
  }
  fclose(cpulist);
  return cpus;
}

void bind_thread_to_cpus(const std::vector<int>& cpus) {
  if (cpus.empty()) return;
  cpu_set_t set;
  CPU_ZERO(&set);
  for (int c : cpus)
    if (c >= 0 && c < CPU_SETSIZE) CPU_SET(c, &set);
  sched_setaffinity(0, sizeof(set), &set);  // best-effort
}

}  // namespace

bool path_in_ram(const std::string& path) {
  struct statfs fs_info {};
  if (statfs(path.c_str(), &fs_info) != 0) return false;
  long fs_type = static_cast<long>(fs_info.f_type);
  return fs_type == kTmpfsMagic || fs_type == kRamfsMagic;
}

void* alloc_pinned_on_node(size_t nbytes, int node) {
  size_t len = round_up(nbytes, kAlign);
  void* p = mmap(nullptr, len, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (p == MAP_FAILED)
    throw std::runtime_error(std::string("alloc_pinned_on_node: mmap: ") + strerror(errno));
  if (node >= 0) bind_to_node(p, len, node);
  memset(p, 0, len);  // first-touch so placement is realized before register pins it
  cudaError_t e = cudaHostRegister(p, len, cudaHostRegisterDefault);
  if (e != cudaSuccess) {
    munmap(p, len);
    throw std::runtime_error(std::string("alloc_pinned_on_node: cudaHostRegister: ") +
                             cudaGetErrorString(e));
  }
  return p;
}

void free_pinned_on_node(void* ptr, size_t nbytes) {
  if (!ptr) return;
  cudaHostUnregister(ptr);  // best-effort
  munmap(ptr, round_up(nbytes, kAlign));
}

double read_file_into(const std::string& path, size_t file_off, size_t len,
                      void* dst, size_t dst_off, int threads, size_t chunk,
                      int numa_node) {
  if (len == 0) return 0.0;
  if (threads < 1) threads = 1;
  if (chunk == 0) chunk = 16ull << 20;
  char* dst_base = static_cast<char*>(dst) + dst_off;
  bool src_in_ram = path_in_ram(path);
  const char* src_mapping = src_in_ram ? mmap_file_cached(path) : nullptr;
  std::vector<int> numa_cpus =
      (numa_node >= 0) ? cpus_for_node(numa_node) : std::vector<int>{};

  // The [file_off, file_off+len) region is cut into chunk-sized jobs pulled from a
  // shared counter, so however the filesystem stalls, no worker idles while jobs remain.
  struct ChunkJob { size_t region_offset, nbytes; };
  std::vector<ChunkJob> chunk_jobs;
  for (size_t region_offset = 0; region_offset < len; region_offset += chunk)
    chunk_jobs.push_back({region_offset, std::min(chunk, len - region_offset)});

  std::atomic<size_t> next_job_index{0};
  std::vector<std::string> worker_errors(threads);
  auto worker = [&](int worker_id) {
    if (!numa_cpus.empty()) bind_thread_to_cpus(numa_cpus);
    int fd = -1;
    if (!src_in_ram) {
      fd = open(path.c_str(), O_RDONLY);
      if (fd < 0) {
        worker_errors[worker_id] = std::string("open: ") + strerror(errno);
        return;
      }
    }
    try {
      for (;;) {
        size_t job_index = next_job_index.fetch_add(1);
        if (job_index >= chunk_jobs.size()) break;
        size_t region_offset = chunk_jobs[job_index].region_offset;
        size_t nbytes = chunk_jobs[job_index].nbytes;
        if (src_in_ram) {
          memcpy(dst_base + region_offset, src_mapping + file_off + region_offset, nbytes);
        } else {
          size_t bytes_read = 0;
          while (bytes_read < nbytes) {
            ssize_t nread = pread(fd, dst_base + region_offset + bytes_read,
                                  nbytes - bytes_read,
                                  file_off + region_offset + bytes_read);
            if (nread < 0) {
              if (errno == EINTR) continue;
              throw std::runtime_error(std::string("pread: ") + strerror(errno));
            }
            if (nread == 0) break;  // premature EOF: file shorter than requested
            bytes_read += static_cast<size_t>(nread);
          }
        }
      }
    } catch (const std::exception& e) { worker_errors[worker_id] = e.what(); }
    if (fd >= 0) close(fd);
  };

  auto started = clk::now();
  std::vector<std::thread> worker_threads;
  worker_threads.reserve(threads);
  for (int i = 0; i < threads; ++i) worker_threads.emplace_back(worker, i);
  for (auto& thread : worker_threads) thread.join();
  auto ended = clk::now();
  for (auto& error : worker_errors)
    if (!error.empty()) throw std::runtime_error("read_file_into: " + error);
  return secs(started, ended);
}

double stage_file_to_shm(const std::string& src, const std::string& dst, int threads,
                         size_t chunk, int numa_node) {
  int sfd = open(src.c_str(), O_RDONLY);
  if (sfd < 0)
    throw std::runtime_error("stage_file_to_shm: open src " + src + ": " + strerror(errno));
  struct stat st{};
  if (fstat(sfd, &st) != 0) {
    close(sfd);
    throw std::runtime_error("stage_file_to_shm: fstat " + src);
  }
  size_t size = static_cast<size_t>(st.st_size);
  close(sfd);
  if (size == 0) return 0.0;

  // Create the tmpfs destination at full size and map it, so the parallel pread lands the
  // GPFS bytes straight into the tmpfs pages (single copy). MAP_SHARED so writes persist.
  int dfd = open(dst.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
  if (dfd < 0)
    throw std::runtime_error("stage_file_to_shm: open dst " + dst + ": " + strerror(errno));
  if (ftruncate(dfd, static_cast<off_t>(size)) != 0) {
    close(dfd);
    throw std::runtime_error("stage_file_to_shm: ftruncate " + dst + ": " + strerror(errno));
  }
  void* map = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, dfd, 0);
  if (map == MAP_FAILED) {
    close(dfd);
    throw std::runtime_error("stage_file_to_shm: mmap " + dst + ": " + strerror(errno));
  }
  double dt;
  try {
    dt = read_file_into(src, /*file_off=*/0, size, map, /*dst_off=*/0, threads, chunk,
                             numa_node);
  } catch (...) {
    munmap(map, size);
    close(dfd);
    throw;
  }
  munmap(map, size);
  close(dfd);
  return dt;
}

}  // namespace flashboot
