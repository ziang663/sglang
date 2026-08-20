// flashboot raw-ibverbs RDMA-READ engine (the RdmaEndpoint class).
//
// Implements one-sided RDMA-READ pulls with RAW libibverbs (no NIXL/UCX, no NCCL):
// build RC QPs by hand, register GPU memory for GPUDirect (nv_peermem direct
// ibv_reg_mr on the GPU ptr — host memory registers with the same call), and READ a
// remote GPU arena into local memory. One RdmaEndpoint per GPU rank; one endpoint
// owns MULTIPLE RC QPs so a rank can hold a distinct reliable connection per peer —
// the rdma chain broadcast pairs a rank with its chain predecessor/successor, the
// rdma all-gather with its ring neighbours or with every other rank (see
// python/flashboot/rdma_collectives.py).
//
// This code is torch-FREE (only pybind11 headers + libibverbs/libcudart), but it is
// NOT a standalone extension: register_rdma() (bottom of this file) is called from
// csrc/python_bindings.cc inside PYBIND11_MODULE(_C), so the class ships as
// `flashboot._C.Endpoint` — one native .so, no separate _fbrdma.so. (It being
// torch-free just means adding it to _C only pulls in libibverbs, nothing else.)
//
// RC QP state machine (see connect()): INIT -> RTR (Ready To Receive: peer info filled,
// can respond to reads) -> RTS (Ready To Send: can now post RDMA ops). Both sides
// exchange their QP "business card" (qpn/psn/lid/gid/mtu) over the TCP control plane
// before connect(). The collectives exchange EVERY card in one rendezvous round
// (flashboot.control_plane.collective_rendezvous), so create_qp() runs once per
// expected connection BEFORE the rendezvous and connect() pairs the cards after —
// no per-link handshake round trips.
//
// Build: part of the _C CUDAExtension when FB_BUILD_RDMA resolves on — see setup.py.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <infiniband/verbs.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

namespace py = pybind11;

namespace {

void logf(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  fprintf(stderr, "[fbrdma] ");
  vfprintf(stderr, fmt, ap);
  fprintf(stderr, "\n");
  fflush(stderr);
  va_end(ap);
}


#define CHECK(cond, msg)                                                    \
  do {                                                                      \
    if (!(cond)) {                                                          \
      throw std::runtime_error(std::string("[fbrdma] ") + (msg) +           \
                               " (errno=" + std::to_string(errno) + ")");   \
    }                                                                       \
  } while (0)

// 16-byte IB GID rendered as 32 hex chars (for the TCP-exchanged QP info).
std::string gid_to_hex(const union ibv_gid& gid) {
  char buf[33];
  for (int i = 0; i < 16; ++i) sprintf(buf + 2 * i, "%02x", gid.raw[i]);
  buf[32] = 0;
  return std::string(buf);
}
union ibv_gid gid_from_hex(const std::string& hex) {
  union ibv_gid gid;
  memset(&gid, 0, sizeof(gid));
  for (int i = 0; i < 16 && 2 * i + 1 < (int)hex.size(); ++i) {
    unsigned byte_val;
    sscanf(hex.c_str() + 2 * i, "%02x", &byte_val);
    gid.raw[i] = (uint8_t)byte_val;
  }
  return gid;
}

// One RC endpoint: device + protection-domain + completion-queue + its QPs, plus the
// registered memory regions. One RdmaEndpoint per GPU rank. All QPs share the one
// completion queue; drive the endpoint from a single thread (post_read /
// wait_completions interleave their completions on that CQ).
class RdmaEndpoint {
 public:
  RdmaEndpoint(int gpu, const std::string& hca_name) : gpu_(gpu) {
    // Seed the PSN generator: lrand48 is deterministic when unseeded, giving every
    // process (and every reset cycle) the SAME packet sequence numbers. Random PSNs
    // keep stale packets from a torn-down pairing from validating against a new one.
    srand48((long)getpid() ^ ((long)time(nullptr) << 16) ^ (long)(uintptr_t)this);
    // Pick the HCA by name (caller passes the positional GPU->NIC map result — see
    // rdma_transport.infiniband_device_for_gpu).
    int num_devices = 0;
    struct ibv_device** device_list = ibv_get_device_list(&num_devices);
    CHECK(device_list && num_devices > 0, "ibv_get_device_list: no RDMA devices");
    struct ibv_device* device = nullptr;
    for (int i = 0; i < num_devices; ++i) {
      if (hca_name.empty() || hca_name == ibv_get_device_name(device_list[i])) {
        device = device_list[i];
        break;
      }
    }
    if (!device) device = device_list[0];
    dev_name_ = ibv_get_device_name(device);
    context_ = ibv_open_device(device);
    ibv_free_device_list(device_list);
    CHECK(context_, "ibv_open_device failed");

    protection_domain_ = ibv_alloc_pd(context_);
    CHECK(protection_domain_, "ibv_alloc_pd failed");
    // Shared across every QP of this endpoint: sized for the deepest posting pattern
    // (the all-gather's concurrent per-peer segment reads), far above any real burst.
    completion_queue_ = ibv_create_cq(context_, /*cqe=*/4096, nullptr, nullptr, 0);
    CHECK(completion_queue_, "ibv_create_cq failed");

    // Use port 1 (single-port HCAs). Query LID + link layer + active MTU.
    port_num_ = 1;
    struct ibv_port_attr port_attr;
    CHECK(ibv_query_port(context_, port_num_, &port_attr) == 0, "ibv_query_port failed");
    local_lid_ = port_attr.lid;
    active_mtu_ = port_attr.active_mtu;
    is_roce_ = (port_attr.link_layer == IBV_LINK_LAYER_ETHERNET);
    // GID: needed for RoCE (and harmless to carry for IB). Default index 3 is the usual
    // RoCEv2 entry; override via FB_RDMA_GID_INDEX. For IB we still query idx 0.
    gid_index_ = is_roce_
        ? (getenv("FB_RDMA_GID_INDEX") ? atoi(getenv("FB_RDMA_GID_INDEX")) : 3)
        : 0;
    CHECK(ibv_query_gid(context_, port_num_, gid_index_, &local_gid_) == 0,
          "ibv_query_gid failed");
    logf("open dev=%s port=%d lid=%u link=%s mtu=%d gid_idx=%d gid=%s",
         dev_name_.c_str(), port_num_, local_lid_, is_roce_ ? "ETH/RoCE" : "IB",
         (int)active_mtu_, gid_index_, gid_to_hex(local_gid_).c_str());
  }

  ~RdmaEndpoint() {
    release();
    if (completion_queue_) { ibv_destroy_cq(completion_queue_); completion_queue_ = nullptr; }
    if (protection_domain_) { ibv_dealloc_pd(protection_domain_); protection_domain_ = nullptr; }
    if (context_) { ibv_close_device(context_); context_ = nullptr; }
  }

  RdmaEndpoint(const RdmaEndpoint&) = delete;
  RdmaEndpoint& operator=(const RdmaEndpoint&) = delete;

  // Free the heavy/single-use resources: the QPs, the MRs, and the cudaMalloc'd bounce
  // buffer(s). Call when no peer will read this rank's memory anymore (a chain rank's
  // successor keeps READing our arena until its own pull completes — hold the endpoint
  // until the collective is done). Idempotent; the dtor also calls it.
  // (completion-queue / protection-domain / context are tiny and freed in the dtor.)
  void release() {
    for (auto* qp : queue_pairs_) if (qp) ibv_destroy_qp(qp);
    queue_pairs_.clear();
    local_psns_.clear();
    free_bounce();
  }

  // Free ONLY the cudaMalloc'd bounce buffer(s) + dereg their MRs, but KEEP the QPs.
  // A clone calls this right after its pull so a bounce is reclaimed BEFORE sglang
  // sizes the KV pool (it measures free GPU mem after load) — otherwise the bounce
  // permanently shrinks KV. We deliberately do NOT destroy QPs here: ibv_destroy_qp
  // right before DeepEP's cuda-graph capture (which also sets up IBGDA/nvshmem on the
  // NIC) regressed launch->health 63s->146s; idle QPs are ~KB and are torn down with
  // the RdmaEndpoint at server exit.
  void free_bounce() {
    for (auto* mr : mem_regions_) if (mr) ibv_dereg_mr(mr);
    mem_regions_.clear();
    for (auto ptr : bounce_ptrs_) if (ptr) cudaFree((void*)ptr);
    bounce_ptrs_.clear();
  }

  // Dedicated cudaMalloc buffer (a COMPLETE allocation the nv_peermem registration in
  // reg() can pin reliably — unlike a sub-range of torch's caching allocator). Freed
  // by free_bounce() / the destructor. Returns the device pointer.
  uintptr_t alloc(size_t nbytes) {
    py::gil_scoped_release gil_off;  // release the Python GIL during the blocking native
    // call (same pattern as the other _C backends: copy_d2d / stage_disk_to_pinned)
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, nbytes);
    CHECK(err == cudaSuccess, (std::string("cudaMalloc bounce failed: ") +
                               cudaGetErrorString(err)).c_str());
    bounce_ptrs_.push_back((uintptr_t)ptr);
    return (uintptr_t)ptr;
  }

  // Register a buffer for RDMA. Returns (local_key, remote_key). GPU pointers take the
  // GPUDirect nv_peermem path: ibv_reg_mr directly on the GPU pointer — works on plain
  // cudaMalloc arenas when the nvidia_peermem module is loaded, which is the case on
  // this H100 fabric. HOST pointers (the collectives' progress counters) register with
  // the very same call — ibv_reg_mr pins host memory natively.
  std::pair<uint32_t, uint32_t> reg(uintptr_t ptr, size_t len) {
    py::gil_scoped_release gil_off;  // release GIL: registering a 57GB MR takes ~1.8s and
    // would otherwise block the seed's post-load monitoredBarrier on the main thread.
    int access_flags =
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
    struct ibv_mr* mr =
        ibv_reg_mr(protection_domain_, (void*)ptr, len, access_flags);
    CHECK(mr, (std::string("[fbrdma] ibv_reg_mr failed errno=") +
               std::to_string(errno) +
               " — for a GPU pointer, is the nvidia_peermem module loaded for "
               "GPUDirect RDMA?").c_str());
    mem_regions_.push_back(mr);
    logf("reg ptr=0x%lx len=%.2fGB lkey=%u rkey=%u",
         (unsigned long)ptr, len / 1e9, mr->lkey, mr->rkey);
    return {mr->lkey, mr->rkey};
  }

  // Create a NEW RC QP and move it to INIT. Returns this end's QP "business card" for
  // the peer, including "qp" — the LOCAL index to pass to connect()/read()/post_read()
  // for this connection (the peer ignores it). Call once per expected connection; the
  // collectives pre-build every card BEFORE the rendezvous so one membership broadcast
  // finishes the whole control plane.
  py::dict create_qp() {
    struct ibv_qp_init_attr qp_init_attr;
    memset(&qp_init_attr, 0, sizeof(qp_init_attr));
    qp_init_attr.send_cq = completion_queue_;
    qp_init_attr.recv_cq = completion_queue_;
    qp_init_attr.qp_type = IBV_QPT_RC;            // Reliable Connection (TCP-like)
    qp_init_attr.cap.max_send_wr = 256;
    qp_init_attr.cap.max_recv_wr = 16;
    qp_init_attr.cap.max_send_sge = 1;
    qp_init_attr.cap.max_recv_sge = 1;
    struct ibv_qp* qp = ibv_create_qp(protection_domain_, &qp_init_attr);
    CHECK(qp, "ibv_create_qp failed");
    queue_pairs_.push_back(qp);
    local_psns_.push_back(0);
    int qp_index = (int)queue_pairs_.size() - 1;
    to_init(qp_index);
    return business_card(qp_index);
  }

  // Recycle a queue pair for a NEW peer: cycle RTS/ERR -> RESET -> INIT with a fresh
  // PSN. Verbs forbids changing the destination of an RC QP in RTS (only the alternate
  // path may change there); the reset cycle is the legal re-target path and is legal
  // from ANY state, so a half-connected QP from a failed session recycles too. The qpn
  // and every memory registration are untouched — a seed serves clone after clone
  // with ONE arena MR. Returns the refreshed business card.
  py::dict reset_qp(int qp = 0) {
    struct ibv_qp* queue_pair = qp_at(qp, "reset_qp");
    struct ibv_qp_attr qp_attr;
    memset(&qp_attr, 0, sizeof(qp_attr));
    qp_attr.qp_state = IBV_QPS_RESET;
    CHECK(ibv_modify_qp(queue_pair, &qp_attr, IBV_QP_STATE) == 0,
          "ibv_modify_qp -> RESET failed");
    to_init(qp);
    return business_card(qp);
  }

  // Move a QP INIT -> RTR (Ready To Receive) -> RTS (Ready To Send) using the PEER's
  // QP info (the r-prefixed / remote_* args). After RTS this end can post RDMA reads;
  // RTR alone already lets the peer read our memory, so a passive serving side runs
  // the same call.
  void connect(uint32_t remote_qpn, uint32_t remote_psn, uint32_t remote_lid,
               const std::string& remote_gid_hex, int remote_mtu, int qp = 0) {
    py::gil_scoped_release gil_off;  // release GIL during the blocking native call
    struct ibv_qp* queue_pair = qp_at(qp, "connect");
    enum ibv_mtu negotiated_mtu = (enum ibv_mtu)std::min((int)active_mtu_, remote_mtu);

    // INIT -> RTR: fill the peer's address + QP number so the path is established.
    struct ibv_qp_attr qp_attr;
    memset(&qp_attr, 0, sizeof(qp_attr));
    qp_attr.qp_state = IBV_QPS_RTR;
    qp_attr.path_mtu = negotiated_mtu;
    qp_attr.dest_qp_num = remote_qpn;
    qp_attr.rq_psn = remote_psn;
    qp_attr.max_dest_rd_atomic = 16;       // how many remote reads the peer may have in flight
    qp_attr.min_rnr_timer = 12;
    qp_attr.ah_attr.port_num = port_num_;
    qp_attr.ah_attr.sl = 0;
    if (is_roce_) {
      qp_attr.ah_attr.is_global = 1;       // RoCE: route by GID (needs the global header)
      qp_attr.ah_attr.grh.dgid = gid_from_hex(remote_gid_hex);
      qp_attr.ah_attr.grh.sgid_index = gid_index_;
      qp_attr.ah_attr.grh.hop_limit = 1;
      qp_attr.ah_attr.grh.traffic_class = 0;
    } else {
      qp_attr.ah_attr.is_global = 0;       // pure IB: route by LID
      qp_attr.ah_attr.dlid = (uint16_t)remote_lid;
      qp_attr.ah_attr.src_path_bits = 0;
    }
    CHECK(ibv_modify_qp(queue_pair, &qp_attr,
            IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
            IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER) == 0,
          "ibv_modify_qp -> RTR failed");

    // RTR -> RTS: set retry/timeout params + this end's start PSN; now we can post sends.
    memset(&qp_attr, 0, sizeof(qp_attr));
    qp_attr.qp_state = IBV_QPS_RTS;
    qp_attr.timeout = 14;
    qp_attr.retry_cnt = 7;
    qp_attr.rnr_retry = 7;
    qp_attr.sq_psn = local_psns_[qp];
    qp_attr.max_rd_atomic = 16;            // how many reads THIS end may have in flight
    CHECK(ibv_modify_qp(queue_pair, &qp_attr,
            IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
            IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC) == 0,
          "ibv_modify_qp -> RTS failed");
    logf("connect qp=%d -> RTS (peer qpn=%u lid=%u mtu=%d)", qp, remote_qpn,
         remote_lid, (int)negotiated_mtu);
  }

  // Post one signaled RDMA READ of the peer's [remote_addr, remote_addr+len) into
  // local_ptr (which must sit inside a reg'd MR identified by local_key) WITHOUT
  // waiting — collect it later with wait_completions(). Posting several reads (across
  // one or many QPs) before waiting keeps the wire saturated back-to-back: the chain
  // pulls a whole ready batch of chunks in one wave, the all-gather reads every
  // peer's segment concurrently.
  void post_read(uintptr_t local_ptr, uint32_t local_key, uintptr_t remote_addr,
                 uint32_t remote_key, size_t len, uint64_t wr_id, int qp = 0) {
    struct ibv_qp* queue_pair = qp_at(qp, "post_read");
    // Where the bytes land locally (a scatter-gather element = addr + length + local key).
    struct ibv_sge local_sge;
    memset(&local_sge, 0, sizeof(local_sge));
    local_sge.addr = local_ptr;
    local_sge.length = (uint32_t)len;
    local_sge.lkey = local_key;

    // The work request = "do one RDMA READ from remote_addr (using remote_key) into local_sge".
    struct ibv_send_wr read_wr, *bad_wr = nullptr;
    memset(&read_wr, 0, sizeof(read_wr));
    read_wr.wr_id = wr_id;
    read_wr.sg_list = &local_sge;
    read_wr.num_sge = 1;
    read_wr.opcode = IBV_WR_RDMA_READ;
    read_wr.send_flags = IBV_SEND_SIGNALED;       // post a completion to the CQ when done
    read_wr.wr.rdma.remote_addr = remote_addr;    // peer arena address
    read_wr.wr.rdma.rkey = remote_key;            // peer MR's remote key
    int post_rc = ibv_post_send(queue_pair, &read_wr, &bad_wr);
    CHECK(post_rc == 0, "ibv_post_send(RDMA_READ) failed");
  }

  // Block until `count` completions landed on the shared CQ (any QP), raising on the
  // first errored one. timeout_ms <= 0 => spin forever.
  void wait_completions(int count, int timeout_ms) {
    py::gil_scoped_release gil_off;  // release GIL during the blocking native call
    struct ibv_wc completions[16];
    long spins = 0;
    long max_spins = timeout_ms > 0 ? (long)timeout_ms * 100000L : -1;  // ~coarse
    int remaining = count;
    while (remaining > 0) {
      int batch = std::min(remaining, (int)(sizeof(completions) / sizeof(*completions)));
      int num_completions = ibv_poll_cq(completion_queue_, batch, completions);
      CHECK(num_completions >= 0, "ibv_poll_cq failed");
      for (int i = 0; i < num_completions; ++i) {
        CHECK(completions[i].status == IBV_WC_SUCCESS,
              (std::string("RDMA_READ completion error status=") +
               std::to_string(completions[i].status) + " (" +
               ibv_wc_status_str(completions[i].status) + ") wr_id=" +
               std::to_string(completions[i].wr_id)).c_str());
      }
      remaining -= num_completions;
      if (num_completions == 0 && max_spins > 0 && ++spins > max_spins)
        throw std::runtime_error("[fbrdma] timed out waiting for " +
                                 std::to_string(remaining) + " RDMA completions");
    }
  }

  // One-sided blocking RDMA READ: post + wait for its single completion. The simple
  // pull primitive (and the collectives' small progress-counter poll).
  void read(uintptr_t local_ptr, uint32_t local_key, uintptr_t remote_addr,
            uint32_t remote_key, size_t len, int timeout_ms, int qp = 0) {
    post_read(local_ptr, local_key, remote_addr, remote_key, len, /*wr_id=*/1, qp);
    wait_completions(1, timeout_ms);
  }

 private:
  struct ibv_qp* qp_at(int qp_index, const char* what) {
    if (qp_index < 0 || qp_index >= (int)queue_pairs_.size())
      throw std::runtime_error(std::string("[fbrdma] ") + what + ": qp index " +
                               std::to_string(qp_index) + " out of range (" +
                               std::to_string(queue_pairs_.size()) + " created)");
    return queue_pairs_[qp_index];
  }

  // RESET/fresh -> INIT: set port + access flags, roll a fresh random 24-bit PSN.
  void to_init(int qp_index) {
    struct ibv_qp_attr qp_attr;
    memset(&qp_attr, 0, sizeof(qp_attr));
    qp_attr.qp_state = IBV_QPS_INIT;
    qp_attr.pkey_index = 0;
    qp_attr.port_num = port_num_;
    qp_attr.qp_access_flags =
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
    CHECK(ibv_modify_qp(queue_pairs_[qp_index], &qp_attr,
            IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) == 0,
          "ibv_modify_qp -> INIT failed");
    local_psns_[qp_index] = (uint32_t)(lrand48() & 0xffffff);
  }

  py::dict business_card(int qp_index) {
    py::dict local_qp_info;
    local_qp_info["qp"] = qp_index;   // LOCAL index for connect()/read(); peers ignore it
    local_qp_info["qpn"] = queue_pairs_[qp_index]->qp_num;
    local_qp_info["psn"] = local_psns_[qp_index];
    local_qp_info["lid"] = (uint32_t)local_lid_;
    local_qp_info["gid"] = gid_to_hex(local_gid_);
    local_qp_info["gid_index"] = gid_index_;
    local_qp_info["mtu"] = (int)active_mtu_;
    local_qp_info["is_roce"] = is_roce_;
    logf("create_qp qp=%d qpn=%u psn=%u", qp_index,
         queue_pairs_[qp_index]->qp_num, local_psns_[qp_index]);
    return local_qp_info;
  }

  int gpu_;
  std::string dev_name_;
  struct ibv_context* context_ = nullptr;            // the opened HCA (network card)
  struct ibv_pd* protection_domain_ = nullptr;       // groups the QPs + MRs
  struct ibv_cq* completion_queue_ = nullptr;        // where finished ops are reported
  std::vector<struct ibv_qp*> queue_pairs_;          // one RC connection endpoint each
  std::vector<uint32_t> local_psns_;                 // per-QP start packet seq number
  std::vector<struct ibv_mr*> mem_regions_;          // registered (pinned) memory regions
  std::vector<uintptr_t> bounce_ptrs_;               // cudaMalloc'd bounce buffers
  uint8_t port_num_ = 1;
  uint16_t local_lid_ = 0;                           // this end's IB local id
  enum ibv_mtu active_mtu_ = IBV_MTU_1024;
  bool is_roce_ = false;
  int gid_index_ = 0;
  union ibv_gid local_gid_;                          // this end's global id (RoCE/IB)
};

}  // namespace

// Registered INTO the main `_C` pybind module (see python_bindings.cc) —
// NOT a standalone `_fbrdma` extension. The class becomes `flashboot._C.Endpoint`,
// so rdma is just another module inside flashboot (one native .so, no separate _fbrdma.so).
void register_rdma(py::module_& m) {
  // GIL handling matches the other _C backends (copy_d2d / stage_disk_to_pinned): each
  // method that does a long/blocking native call releases the GIL via a LOCAL
  // `py::gil_scoped_release` inside its body (alloc/reg/connect/wait_completions), NOT
  // via a py::call_guard<> on the .def(). reg of a 57GB MR (~1.8s) MUST release so it
  // doesn't block the seed's post-load monitoredBarrier on the main thread. create_qp
  // keeps the GIL (builds a py::dict); release/free_bounce/post_read keep it too (they
  // are fast).
  py::class_<RdmaEndpoint>(m, "Endpoint")
      .def(py::init<int, const std::string&>(), py::arg("gpu"), py::arg("hca"))
      .def("alloc", &RdmaEndpoint::alloc, py::arg("n"))
      .def("reg", &RdmaEndpoint::reg, py::arg("ptr"), py::arg("len"))
      .def("create_qp", &RdmaEndpoint::create_qp)
      .def("reset_qp", &RdmaEndpoint::reset_qp, py::arg("qp") = 0)
      .def("connect", &RdmaEndpoint::connect, py::arg("rqpn"), py::arg("rpsn"),
           py::arg("rlid"), py::arg("rgid"), py::arg("rmtu"), py::arg("qp") = 0)
      .def("post_read", &RdmaEndpoint::post_read, py::arg("local_ptr"),
           py::arg("lkey"), py::arg("remote_addr"), py::arg("rkey"), py::arg("len"),
           py::arg("wr_id") = 1, py::arg("qp") = 0)
      .def("wait_completions", &RdmaEndpoint::wait_completions, py::arg("count"),
           py::arg("timeout_ms") = 600000)
      .def("read", &RdmaEndpoint::read, py::arg("local_ptr"), py::arg("lkey"),
           py::arg("remote_addr"), py::arg("rkey"), py::arg("len"),
           py::arg("timeout_ms") = 600000, py::arg("qp") = 0)
      .def("release", &RdmaEndpoint::release)
      .def("free_bounce", &RdmaEndpoint::free_bounce);
}
