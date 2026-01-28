#ifndef VX_OCL_DIAG_H
#define VX_OCL_DIAG_H

#include <CL/cl.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#define VX_CL_DIAG_PREFIX "[opencl-diag] "

static inline const char* vx_cl_errstr(cl_int err) {
  switch (err) {
    case CL_SUCCESS: return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
    case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
    case CL_PROFILING_INFO_NOT_AVAILABLE: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case CL_MEM_COPY_OVERLAP: return "CL_MEM_COPY_OVERLAP";
    case CL_IMAGE_FORMAT_MISMATCH: return "CL_IMAGE_FORMAT_MISMATCH";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
    case CL_MAP_FAILURE: return "CL_MAP_FAILURE";
#ifdef CL_MISALIGNED_SUB_BUFFER_OFFSET
    case CL_MISALIGNED_SUB_BUFFER_OFFSET: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
#endif
#ifdef CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST
    case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
#endif
#ifdef CL_COMPILE_PROGRAM_FAILURE
    case CL_COMPILE_PROGRAM_FAILURE: return "CL_COMPILE_PROGRAM_FAILURE";
#endif
#ifdef CL_LINKER_NOT_AVAILABLE
    case CL_LINKER_NOT_AVAILABLE: return "CL_LINKER_NOT_AVAILABLE";
#endif
#ifdef CL_LINK_PROGRAM_FAILURE
    case CL_LINK_PROGRAM_FAILURE: return "CL_LINK_PROGRAM_FAILURE";
#endif
#ifdef CL_DEVICE_PARTITION_FAILED
    case CL_DEVICE_PARTITION_FAILED: return "CL_DEVICE_PARTITION_FAILED";
#endif
#ifdef CL_KERNEL_ARG_INFO_NOT_AVAILABLE
    case CL_KERNEL_ARG_INFO_NOT_AVAILABLE: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
#endif
    case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
    case CL_INVALID_DEVICE_TYPE: return "CL_INVALID_DEVICE_TYPE";
    case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
    case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
    case CL_INVALID_QUEUE_PROPERTIES: return "CL_INVALID_QUEUE_PROPERTIES";
    case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_HOST_PTR: return "CL_INVALID_HOST_PTR";
    case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case CL_INVALID_IMAGE_SIZE: return "CL_INVALID_IMAGE_SIZE";
    case CL_INVALID_SAMPLER: return "CL_INVALID_SAMPLER";
    case CL_INVALID_BINARY: return "CL_INVALID_BINARY";
    case CL_INVALID_BUILD_OPTIONS: return "CL_INVALID_BUILD_OPTIONS";
    case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
    case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_KERNEL_DEFINITION: return "CL_INVALID_KERNEL_DEFINITION";
    case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
    case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
    case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
    case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
    case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
    case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_GLOBAL_OFFSET: return "CL_INVALID_GLOBAL_OFFSET";
    case CL_INVALID_EVENT_WAIT_LIST: return "CL_INVALID_EVENT_WAIT_LIST";
    case CL_INVALID_EVENT: return "CL_INVALID_EVENT";
    case CL_INVALID_OPERATION: return "CL_INVALID_OPERATION";
    case CL_INVALID_GL_OBJECT: return "CL_INVALID_GL_OBJECT";
    case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_MIP_LEVEL: return "CL_INVALID_MIP_LEVEL";
    case CL_INVALID_GLOBAL_WORK_SIZE: return "CL_INVALID_GLOBAL_WORK_SIZE";
#ifdef CL_INVALID_PROPERTY
    case CL_INVALID_PROPERTY: return "CL_INVALID_PROPERTY";
#endif
#ifdef CL_INVALID_IMAGE_DESCRIPTOR
    case CL_INVALID_IMAGE_DESCRIPTOR: return "CL_INVALID_IMAGE_DESCRIPTOR";
#endif
#ifdef CL_INVALID_COMPILER_OPTIONS
    case CL_INVALID_COMPILER_OPTIONS: return "CL_INVALID_COMPILER_OPTIONS";
#endif
#ifdef CL_INVALID_LINKER_OPTIONS
    case CL_INVALID_LINKER_OPTIONS: return "CL_INVALID_LINKER_OPTIONS";
#endif
#ifdef CL_INVALID_DEVICE_PARTITION_COUNT
    case CL_INVALID_DEVICE_PARTITION_COUNT: return "CL_INVALID_DEVICE_PARTITION_COUNT";
#endif
    default: return "CL_UNKNOWN_ERROR";
  }
}

static inline int vx_cl_diag_enabled(void) {
  const char* v = getenv("VX_OPENCL_DIAG");
  return (v && v[0] && strcmp(v, "0") != 0);
}

static inline void vx_cl_diag_printf(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  fprintf(stderr, "%s", VX_CL_DIAG_PREFIX);
  vfprintf(stderr, fmt, ap);
  fprintf(stderr, "\n");
  va_end(ap);
}

static inline void vx_cl_diag_print_block(const char* text) {
  if (!text || !text[0])
    return;
  const char* line = text;
  while (*line) {
    const char* nl = strchr(line, '\n');
    if (!nl) {
      fprintf(stderr, "%s%s\n", VX_CL_DIAG_PREFIX, line);
      break;
    }
    fprintf(stderr, "%s%.*s\n", VX_CL_DIAG_PREFIX, (int)(nl - line), line);
    line = nl + 1;
  }
}

static inline size_t vx_cl_size_product(cl_uint work_dim, const size_t* sizes) {
  if (!sizes || work_dim == 0)
    return 0;
  size_t prod = 1;
  for (cl_uint i = 0; i < work_dim; ++i) {
    if (sizes[i] == 0)
      return 0;
    prod *= sizes[i];
  }
  return prod;
}

static inline void vx_cl_print_work_sizes(cl_uint work_dim, const size_t* gws, const size_t* lws) {
  fprintf(stderr, VX_CL_DIAG_PREFIX "OpenCL work sizes: dims=%u", (unsigned)work_dim);
  if (gws) {
    fprintf(stderr, " gws=[");
    for (cl_uint i = 0; i < work_dim; ++i) {
      fprintf(stderr, "%zu%s", gws[i], (i + 1 == work_dim) ? "" : ",");
    }
    fprintf(stderr, "] (product=%zu)", vx_cl_size_product(work_dim, gws));
  }
  if (lws) {
    fprintf(stderr, " lws=[");
    for (cl_uint i = 0; i < work_dim; ++i) {
      fprintf(stderr, "%zu%s", lws[i], (i + 1 == work_dim) ? "" : ",");
    }
    fprintf(stderr, "] (product=%zu)", vx_cl_size_product(work_dim, lws));
  }
  fprintf(stderr, "\n");
}

static inline void vx_cl_print_device_limits(cl_device_id dev) {
  char name[256] = {0};
  char vendor[256] = {0};
  char driver[256] = {0};
  char version[256] = {0};
  cl_uint compute_units = 0;
  cl_uint clock_mhz = 0;
  size_t max_wg = 0;
  size_t max_wi[3] = {0, 0, 0};
  cl_ulong local_mem = 0;
  cl_ulong max_alloc = 0;
  cl_ulong global_mem = 0;

  clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name), name, NULL);
  clGetDeviceInfo(dev, CL_DEVICE_VENDOR, sizeof(vendor), vendor, NULL);
  clGetDeviceInfo(dev, CL_DRIVER_VERSION, sizeof(driver), driver, NULL);
  clGetDeviceInfo(dev, CL_DEVICE_VERSION, sizeof(version), version, NULL);
  clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);
  clGetDeviceInfo(dev, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clock_mhz), &clock_mhz, NULL);
  clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_wg), &max_wg, NULL);
  clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_wi), max_wi, NULL);
  clGetDeviceInfo(dev, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem), &local_mem, NULL);
  clGetDeviceInfo(dev, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_alloc), &max_alloc, NULL);
  clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem), &global_mem, NULL);

  fprintf(stderr, VX_CL_DIAG_PREFIX "OpenCL device: %s | %s\n", name, vendor);
  fprintf(stderr, VX_CL_DIAG_PREFIX "  driver=%s version=%s\n", driver, version);
  fprintf(stderr, VX_CL_DIAG_PREFIX "  compute_units=%u clock_mhz=%u\n", compute_units, clock_mhz);
  fprintf(stderr, VX_CL_DIAG_PREFIX "  max_wg=%zu max_wi=[%zu,%zu,%zu]\n", max_wg, max_wi[0], max_wi[1], max_wi[2]);
  fprintf(stderr, VX_CL_DIAG_PREFIX "  local_mem=%lu max_alloc=%lu global_mem=%lu\n",
          (unsigned long)local_mem, (unsigned long)max_alloc, (unsigned long)global_mem);
}

static inline void vx_cl_print_kernel_limits(cl_kernel kernel, cl_device_id dev) {
  if (!kernel || !dev) {
    fprintf(stderr, VX_CL_DIAG_PREFIX "OpenCL kernel limits: <unavailable>\n");
    return;
  }
  size_t kernel_wg = 0;
  size_t compile_wg[3] = {0, 0, 0};
  size_t pref_multiple = 0;
  cl_ulong local_mem = 0;
  cl_ulong private_mem = 0;
  clGetKernelWorkGroupInfo(kernel, dev, CL_KERNEL_WORK_GROUP_SIZE,
                           sizeof(kernel_wg), &kernel_wg, NULL);
  clGetKernelWorkGroupInfo(kernel, dev, CL_KERNEL_COMPILE_WORK_GROUP_SIZE,
                           sizeof(compile_wg), compile_wg, NULL);
  clGetKernelWorkGroupInfo(kernel, dev, CL_KERNEL_LOCAL_MEM_SIZE,
                           sizeof(local_mem), &local_mem, NULL);
  clGetKernelWorkGroupInfo(kernel, dev, CL_KERNEL_PRIVATE_MEM_SIZE,
                           sizeof(private_mem), &private_mem, NULL);
  clGetKernelWorkGroupInfo(kernel, dev, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                           sizeof(pref_multiple), &pref_multiple, NULL);

  fprintf(stderr,
          VX_CL_DIAG_PREFIX "OpenCL kernel limits: kernel_wg=%zu pref_wg_multiple=%zu "
          "local_mem=%lu private_mem=%lu compile_wg=[%zu,%zu,%zu]\n",
          kernel_wg, pref_multiple,
          (unsigned long)local_mem, (unsigned long)private_mem,
          compile_wg[0], compile_wg[1], compile_wg[2]);
}

static inline void vx_cl_report_wg_violations(cl_uint work_dim, const size_t* gws, const size_t* lws,
                                              cl_device_id dev, cl_kernel kernel) {
  if (!lws) {
    fprintf(stderr, VX_CL_DIAG_PREFIX "OpenCL work-group check: local size is NULL (implementation-selected)\n");
    return;
  }

  size_t max_wg = 0;
  size_t max_wi[3] = {0, 0, 0};
  cl_ulong local_mem = 0;
  cl_ulong max_alloc = 0;
  clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_wg), &max_wg, NULL);
  clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_wi), max_wi, NULL);
  clGetDeviceInfo(dev, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem), &local_mem, NULL);
  clGetDeviceInfo(dev, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_alloc), &max_alloc, NULL);

  size_t kernel_wg = 0;
  size_t compile_wg[3] = {0, 0, 0};
  size_t pref_multiple = 0;
  cl_ulong kernel_local_mem = 0;
  cl_ulong kernel_private_mem = 0;
  if (kernel) {
    clGetKernelWorkGroupInfo(kernel, dev, CL_KERNEL_WORK_GROUP_SIZE,
                             sizeof(kernel_wg), &kernel_wg, NULL);
    clGetKernelWorkGroupInfo(kernel, dev, CL_KERNEL_COMPILE_WORK_GROUP_SIZE,
                             sizeof(compile_wg), compile_wg, NULL);
    clGetKernelWorkGroupInfo(kernel, dev, CL_KERNEL_LOCAL_MEM_SIZE,
                             sizeof(kernel_local_mem), &kernel_local_mem, NULL);
    clGetKernelWorkGroupInfo(kernel, dev, CL_KERNEL_PRIVATE_MEM_SIZE,
                             sizeof(kernel_private_mem), &kernel_private_mem, NULL);
    clGetKernelWorkGroupInfo(kernel, dev, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                             sizeof(pref_multiple), &pref_multiple, NULL);
  }

  size_t lprod = vx_cl_size_product(work_dim, lws);
  int violations = 0;
  fprintf(stderr, VX_CL_DIAG_PREFIX "OpenCL work-group check:\n");
  if (max_wg && lprod > max_wg) {
    fprintf(stderr, VX_CL_DIAG_PREFIX "  - local product %zu exceeds device max_wg=%zu\n", lprod, max_wg);
    violations = 1;
  }
  if (kernel_wg && lprod > kernel_wg) {
    fprintf(stderr, VX_CL_DIAG_PREFIX "  - local product %zu exceeds kernel_wg=%zu\n", lprod, kernel_wg);
    violations = 1;
  }
  for (cl_uint i = 0; i < work_dim && i < 3; ++i) {
    if (max_wi[i] && lws[i] > max_wi[i]) {
      fprintf(stderr, VX_CL_DIAG_PREFIX "  - lws[%u]=%zu exceeds device max_wi[%u]=%zu\n",
              (unsigned)i, lws[i], (unsigned)i, max_wi[i]);
      violations = 1;
    }
    if (compile_wg[i] && lws[i] != compile_wg[i]) {
      fprintf(stderr, VX_CL_DIAG_PREFIX "  - lws[%u]=%zu does not match kernel compile_wg[%u]=%zu\n",
              (unsigned)i, lws[i], (unsigned)i, compile_wg[i]);
      violations = 1;
    }
  }
  if (kernel_local_mem && local_mem && kernel_local_mem > local_mem) {
    fprintf(stderr, VX_CL_DIAG_PREFIX "  - kernel local_mem=%lu exceeds device local_mem=%lu\n",
            (unsigned long)kernel_local_mem, (unsigned long)local_mem);
    violations = 1;
  }
  if (pref_multiple && lprod && (lprod % pref_multiple) != 0) {
    fprintf(stderr, VX_CL_DIAG_PREFIX "  - local product %zu is not multiple of preferred %zu\n",
            lprod, pref_multiple);
  }
  if (!violations && gws) {
    size_t gprod = vx_cl_size_product(work_dim, gws);
    (void)gprod;
  }
  if (!violations) {
    fprintf(stderr, VX_CL_DIAG_PREFIX "  - no obvious limit violations detected\n");
  }
}

static inline void vx_cl_print_build_log(cl_program program, cl_device_id dev) {
  size_t log_size = 0;
  cl_int err = clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
  if (err != CL_SUCCESS || log_size == 0) {
    fprintf(stderr, VX_CL_DIAG_PREFIX "OpenCL build log: <unavailable> (err=%d)\n", (int)err);
    return;
  }
  char* log = (char*)malloc(log_size + 1);
  if (!log) {
    fprintf(stderr, VX_CL_DIAG_PREFIX "OpenCL build log: <alloc failed>\n");
    return;
  }
  memset(log, 0, log_size + 1);
  clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
  vx_cl_diag_printf("OpenCL build log:");
  vx_cl_diag_print_block(log);
  free(log);
}

static inline void vx_cl_print_build_log_all(cl_program program) {
  cl_uint num_devices = 0;
  cl_int err = clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES,
                                sizeof(num_devices), &num_devices, NULL);
  if (err != CL_SUCCESS || num_devices == 0) {
    fprintf(stderr, VX_CL_DIAG_PREFIX "OpenCL build log: <no devices> (err=%d)\n", (int)err);
    return;
  }
  cl_device_id* devices = (cl_device_id*)calloc(num_devices, sizeof(cl_device_id));
  if (!devices) {
    fprintf(stderr, VX_CL_DIAG_PREFIX "OpenCL build log: <alloc failed>\n");
    return;
  }
  err = clGetProgramInfo(program, CL_PROGRAM_DEVICES,
                         sizeof(cl_device_id) * num_devices, devices, NULL);
  if (err != CL_SUCCESS) {
    fprintf(stderr, VX_CL_DIAG_PREFIX "OpenCL build log: <device list failed> (err=%d)\n", (int)err);
    free(devices);
    return;
  }
  for (cl_uint i = 0; i < num_devices; ++i) {
    vx_cl_print_build_log(program, devices[i]);
  }
  free(devices);
}

static inline void vx_cl_report_error(const char* api, cl_int err, const char* file, int line) {
  fprintf(stderr, VX_CL_DIAG_PREFIX "OpenCL error: %s -> %s (%d) at %s:%d\n",
          api, vx_cl_errstr(err), (int)err, file, line);
}

static inline void vx_cl_report_build_error(const char* api, cl_int err,
                                            cl_program program, cl_device_id dev,
                                            const char* file, int line) {
  vx_cl_report_error(api, err, file, line);
  vx_cl_print_build_log_all(program);
  if (vx_cl_diag_enabled()) {
    vx_cl_print_device_limits(dev);
  }
}

static inline void vx_cl_report_enqueue_error(const char* api, cl_int err,
                                              cl_device_id dev, cl_kernel kernel,
                                              cl_uint work_dim,
                                              const size_t* gws, const size_t* lws,
                                              const char* file, int line) {
  vx_cl_report_error(api, err, file, line);
  vx_cl_print_work_sizes(work_dim, gws, lws);
  if (dev) {
    vx_cl_print_device_limits(dev);
  }
  if (kernel && dev) {
    vx_cl_print_kernel_limits(kernel, dev);
  }
  if (dev) {
    vx_cl_report_wg_violations(work_dim, gws, lws, dev, kernel);
  }
}

#endif /* VX_OCL_DIAG_H */
