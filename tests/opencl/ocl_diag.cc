#include <CL/cl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ocl_diag.h"

static void diag_line(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  fprintf(stderr, "[opencl-diag] ");
  vfprintf(stderr, fmt, ap);
  fprintf(stderr, "\n");
  va_end(ap);
}

static void print_platform_info(cl_platform_id platform) {
  char name[256] = {0};
  char vendor[256] = {0};
  char version[256] = {0};
  clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(name), name, NULL);
  clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, sizeof(vendor), vendor, NULL);
  clGetPlatformInfo(platform, CL_PLATFORM_VERSION, sizeof(version), version, NULL);
  diag_line("OpenCL platform: %s | %s | %s", name, vendor, version);
}

static void print_device_summary(cl_device_id dev) {
  char name[256] = {0};
  clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(name), name, NULL);
  size_t max_wg = 0;
  size_t max_wi[3] = {0, 0, 0};
  cl_ulong local_mem = 0;
  cl_ulong max_alloc = 0;
  clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_wg), &max_wg, NULL);
  clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_wi), max_wi, NULL);
  clGetDeviceInfo(dev, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem), &local_mem, NULL);
  clGetDeviceInfo(dev, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_alloc), &max_alloc, NULL);
  diag_line("OpenCL device: %s max_wg=%zu max_wi=[%zu,%zu,%zu] local_mem=%lu max_alloc=%lu",
            name, max_wg, max_wi[0], max_wi[1], max_wi[2],
            (unsigned long)local_mem, (unsigned long)max_alloc);
}

int main(int argc, char** argv) {
  int full = 0;
  if (argc > 1 && (strcmp(argv[1], "--full") == 0)) {
    full = 1;
  }

  cl_uint num_platforms = 0;
  cl_int err = clGetPlatformIDs(0, NULL, &num_platforms);
  if (err != CL_SUCCESS || num_platforms == 0) {
    diag_line("OpenCL diag: no platforms (err=%d)", (int)err);
    return 1;
  }

  cl_platform_id* platforms = (cl_platform_id*)calloc(num_platforms, sizeof(cl_platform_id));
  if (!platforms) {
    diag_line("OpenCL diag: out of memory");
    return 1;
  }
  clGetPlatformIDs(num_platforms, platforms, NULL);

  for (cl_uint p = 0; p < num_platforms; ++p) {
    print_platform_info(platforms[p]);
    cl_uint num_devices = 0;
    clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
    if (num_devices == 0) {
      diag_line("OpenCL diag: no devices on platform %u", p);
      continue;
    }
    cl_device_id* devices = (cl_device_id*)calloc(num_devices, sizeof(cl_device_id));
    if (!devices) {
      diag_line("OpenCL diag: out of memory");
      free(platforms);
      return 1;
    }
    clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);

    for (cl_uint d = 0; d < num_devices; ++d) {
      print_device_summary(devices[d]);
      if (full) {
        vx_cl_print_device_limits(devices[d]);
      }
      if (!full) {
        break; // summary uses first device only
      }
    }
    free(devices);
    if (!full) {
      break; // summary uses first platform only
    }
  }

  free(platforms);
  return 0;
}
