// flashboot — IMEX / NVLink-fabric preflight (ported from the GB300-verified main
// branch src/imex_check.cc).
//
// Fabric handle export/import (the "fabric" clone transport) only works when:
//   1. the CUDA driver is >= 12.4 (CU_MEM_HANDLE_TYPE_FABRIC exists),
//   2. the device reports CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
//   3. the `nvidia-caps-imex-channels` device class exists (/proc/devices), and
//   4. at least one /dev/nvidia-caps-imex-channels/channel* is R/W accessible to the
//      running user — exporter and importer must share the same channel (the
//      nvidia-imex daemon manages the domain).
//
// A misconfigured system surfaces at import time as an opaque CUDA_ERROR_NOT_PERMITTED;
// this check turns that into an actionable message BEFORE any arena is allocated, and
// lets transport_select's `auto` mode prefer fabric only where it can actually work
// (GB300 NVL72 racks) while non-IMEX systems resolve to ipc (same node).
//
// Like peer_arena_import.cu this file is torch-free and NOT a
// standalone extension: register_imex_check() is called from csrc/python_bindings.cc
// inside PYBIND11_MODULE(_C), so the probe ships as `flashboot._C.check_imex`.

#include <dirent.h>
#include <unistd.h>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <cuda.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

bool proc_devices_has_imex() {
  std::ifstream f("/proc/devices");
  if (!f) return false;
  std::string line;
  while (std::getline(f, line)) {
    if (line.find("nvidia-caps-imex-channels") != std::string::npos) return true;
  }
  return false;
}

// Channel files the current user can open for read/write.
std::vector<std::string> accessible_channels() {
  std::vector<std::string> out;
  const char* dir = "/dev/nvidia-caps-imex-channels";
  DIR* d = opendir(dir);
  if (!d) return out;
  struct dirent* e;
  while ((e = readdir(d)) != nullptr) {
    std::string name = e->d_name;
    if (name.rfind("channel", 0) != 0) continue;  // must start with "channel"
    std::string path = std::string(dir) + "/" + name;
    if (access(path.c_str(), R_OK | W_OK) == 0) out.push_back(path);
  }
  closedir(d);
  return out;
}

struct ImexStatus {
  bool ok = false;                    // all required conditions met
  int driver_version = 0;             // e.g. 12040 for 12.4
  bool fabric_supported = false;      // device reports HANDLE_TYPE_FABRIC support
  bool imex_device_present = false;   // nvidia-caps-imex-channels in /proc/devices
  std::vector<std::string> channels;  // accessible channel files
  std::string detail;                 // human-readable summary / first failure
};

// Run all checks for `device`. Never throws; inspect .ok / .detail.
ImexStatus check_imex(int device) {
  ImexStatus s;
  std::ostringstream msg;

  // 1) CUDA driver version (need >= 12.4 == 12040 for CU_MEM_HANDLE_TYPE_FABRIC).
  if (cuInit(0) != CUDA_SUCCESS) {
    s.detail = "cuInit failed: NVIDIA driver not available";
    return s;
  }
  cuDriverGetVersion(&s.driver_version);
  const bool driver_ok = s.driver_version >= 12040;

  // 2) Device fabric-handle support.
  int supported = 0;
  CUdevice dev;
  if (cuDeviceGet(&dev, device) == CUDA_SUCCESS) {
    cuDeviceGetAttribute(&supported,
                         CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, dev);
  }
  s.fabric_supported = supported != 0;

  // 3) IMEX device class + accessible channels.
  s.imex_device_present = proc_devices_has_imex();
  s.channels = accessible_channels();

  s.ok = driver_ok && s.fabric_supported && s.imex_device_present &&
         !s.channels.empty();

  if (s.ok) {
    msg << "OK: driver " << s.driver_version << ", fabric supported, "
        << s.channels.size() << " IMEX channel(s) accessible";
  } else {
    msg << "fabric sharing NOT ready:";
    if (!driver_ok)
      msg << "\n  - CUDA driver " << s.driver_version << " < 12040 (need >= 12.4)";
    if (!s.fabric_supported)
      msg << "\n  - device " << device
          << " does not report HANDLE_TYPE_FABRIC support";
    if (!s.imex_device_present)
      msg << "\n  - 'nvidia-caps-imex-channels' missing from /proc/devices "
             "(IMEX driver capability not present)";
    if (s.channels.empty())
      msg << "\n  - no accessible channel in /dev/nvidia-caps-imex-channels/ "
             "(start the nvidia-imex daemon and ensure this user can open a "
             "channel; exporter and importer must share the same channel)";
  }
  s.detail = msg.str();
  return s;
}

}  // namespace

// Registered INTO the main `_C` pybind module (see python_bindings.cc).
void register_imex_check(py::module_& m) {
  m.def(
      "check_imex",
      [](int device) {
        ImexStatus s = check_imex(device);
        py::dict out;
        out["ok"] = s.ok;
        out["driver_version"] = s.driver_version;
        out["fabric_supported"] = s.fabric_supported;
        out["imex_device_present"] = s.imex_device_present;
        out["channels"] = s.channels;
        out["detail"] = s.detail;
        return out;
      },
      py::arg("device") = 0,
      "Preflight the NVLink-fabric / IMEX requirements of the fabric transport for "
      "`device`. Never raises; returns a dict with 'ok' and a human-readable "
      "'detail'. ok=True means a fabric handle exported here can be imported by any "
      "GPU sharing the IMEX domain (e.g. a GB300 NVL72 rack).");
}
