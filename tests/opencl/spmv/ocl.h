#ifndef __OCLH__
#define __OCLH__

#ifdef __cplusplus
extern "C" {
#endif

#include "../ocl_diag.h"

typedef struct {
	cl_uint major;
	cl_uint minor;
	cl_uint multiProcessorCount;
} OpenCLDeviceProp;

void clMemSet(cl_command_queue, cl_mem, int, size_t);
char* readFile(const char*);

#define CHECK_ERROR(errorMessage)           \
  if (clStatus != CL_SUCCESS) {             \
    vx_cl_report_error((errorMessage),      \
                       clStatus,            \
                       __FILE__, __LINE__); \
    exit(1);                                \
  }

#define CHECK_ENQUEUE_ERROR(errorMessage, dev, kernel, dim, gws, lws)   \
  if (clStatus != CL_SUCCESS) {                                        \
    vx_cl_report_enqueue_error((errorMessage), clStatus, (dev),        \
                               (kernel), (dim), (gws), (lws),          \
                               __FILE__, __LINE__);                   \
    exit(1);                                                           \
  }

#ifdef __cplusplus
}
#endif

#endif
