#ifndef __OCLH__
#define __OCLH__

#ifdef __cplusplus
extern "C" {
#endif

#include "../ocl_diag.h"

typedef struct {
	cl_platform_id clPlatform;
	cl_context_properties clCps[3];
	cl_device_id clDevice;
	cl_context clContext;
	cl_command_queue clCommandQueue;
	cl_program clProgram;
	cl_kernel clKernel;
} OpenCL_Param;


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

char* readFile(char*);

#ifdef __cplusplus
}
#endif

#endif
