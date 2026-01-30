#include <CL/opencl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "video_utils.h"
#include "../ocl_diag.h"

#define CL_CHECK(_expr)                                                \
   do {                                                                \
     cl_int _err = _expr;                                              \
     if (_err == CL_SUCCESS)                                           \
       break;                                                          \
     vx_cl_report_error(#_expr, _err, __FILE__, __LINE__);             \
     cleanup();                                                        \
     return -1;                                                        \
   } while (0)

#define CL_CHECK2(_expr)                                               \
   ({                                                                  \
     cl_int _err = CL_INVALID_VALUE;                                   \
     decltype(_expr) _ret = _expr;                                     \
     if (_err != CL_SUCCESS) {                                         \
       vx_cl_report_error(#_expr, _err, __FILE__, __LINE__);           \
       cleanup();                                                      \
       return -1;                                                      \
     }                                                                 \
     _ret;                                                             \
   })

#define CL_BUILD_CHECK(_prog, _dev, _expr)                              \
   do {                                                                 \
     cl_int _err = (_expr);                                             \
     if (_err != CL_SUCCESS) {                                          \
       vx_cl_report_build_error(#_expr, _err, (_prog), (_dev),          \
                                __FILE__, __LINE__);                   \
       cleanup();                                                      \
       return -1;                                                      \
     }                                                                  \
   } while (0)

#define CL_ENQUEUE_CHECK(_expr, _dev, _kernel, _dim, _gws, _lws)         \
   do {                                                                  \
     cl_int _err = (_expr);                                              \
     if (_err != CL_SUCCESS) {                                           \
       vx_cl_report_enqueue_error(#_expr, _err, (_dev), (_kernel),       \
                                  (_dim), (_gws), (_lws),                \
                                  __FILE__, __LINE__);                  \
       cleanup();                                                        \
       return -1;                                                        \
     }                                                                   \
   } while (0)

static cl_device_id device_id = NULL;
static cl_context context = NULL;
static cl_command_queue commandQueue = NULL;
static cl_program program = NULL;
static cl_kernel kernel = NULL;
static cl_mem y_mem = NULL;
static cl_mem uv_mem = NULL;
static cl_mem out_mem = NULL;
static uint8_t* kernel_bin = NULL;

static void cleanup() {
    if (commandQueue) clReleaseCommandQueue(commandQueue);
    if (kernel) clReleaseKernel(kernel);
    if (program) clReleaseProgram(program);
    if (y_mem) clReleaseMemObject(y_mem);
    if (uv_mem) clReleaseMemObject(uv_mem);
    if (out_mem) clReleaseMemObject(out_mem);
    if (context) clReleaseContext(context);
    if (device_id) clReleaseDevice(device_id);
    if (kernel_bin) free(kernel_bin);
}

static int read_kernel_file(const char* filename, uint8_t** data, size_t* size) {
    if (!filename || !data || !size) return -1;
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel %s\n", filename);
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    rewind(fp);
    *data = (uint8_t*)malloc(fsize);
    *size = fread(*data, 1, fsize, fp);
    fclose(fp);
    return 0;
}

static inline uint8_t clamp_u8(int v) {
    if (v < 0) return 0;
    if (v > 255) return 255;
    return (uint8_t)v;
}

static void cpu_yuv420_to_rgba(const uint8_t* y_plane, const uint8_t* uv_plane,
                               uint8_t* out_rgba, int width, int height,
                               int out_stride_bytes) {
    int out_stride_pixels = out_stride_bytes / 4;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int y_idx = y * width + x;
            int uvx = x >> 1;
            int uvy = y >> 1;
            int uv_idx = (uvy * (width >> 1) + uvx) * 2;
            int Y = (int)y_plane[y_idx];
            int U = (int)uv_plane[uv_idx + 0] - 128;
            int V = (int)uv_plane[uv_idx + 1] - 128;
            int R = Y + (1436 * V) / 1024;
            int G = Y - (352 * U + 731 * V) / 1024;
            int B = Y + (1814 * U) / 1024;
            int idx = (y * out_stride_pixels + x) * 4;
            out_rgba[idx + 0] = clamp_u8(R);
            out_rgba[idx + 1] = clamp_u8(G);
            out_rgba[idx + 2] = clamp_u8(B);
            out_rgba[idx + 3] = 255;
        }
    }
}

static void generate_yuv420(std::vector<uint8_t>& y, std::vector<uint8_t>& uv,
                             int width, int height, int seed) {
    y.resize(width * height);
    uv.resize((width / 2) * (height / 2) * 2);
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            int idx = j * width + i;
            y[idx] = (uint8_t)((i * 255) / (width - 1) ^ (seed * 13));
        }
    }
    for (int j = 0; j < height / 2; ++j) {
        for (int i = 0; i < width / 2; ++i) {
            int idx = (j * (width / 2) + i) * 2;
            uv[idx + 0] = (uint8_t)(128 + ((i * 31 + seed) % 64) - 32);
            uv[idx + 1] = (uint8_t)(128 + ((j * 29 + seed) % 64) - 32);
        }
    }
}

int main(int argc, char** argv) {
    vx::video::Args args;
    std::vector<std::string> extras;
    std::string err;

    if (!vx::video::utils_self_test(err)) {
        fprintf(stderr, "video_utils self-test failed: %s\n", err.c_str());
        return -1;
    }

    if (!vx::video::parse_common_args(argc, argv, args, extras, err)) {
        fprintf(stderr, "Argument error: %s\n", err.c_str());
        vx::video::print_common_usage(argv[0]);
        return -1;
    }
    for (const auto& e : extras) {
        if (e == "-help" || e == "--help") {
            vx::video::print_common_usage(argv[0]);
            printf("  (YUV420 requires even width/height)\n");
            return 0;
        }
    }

    int width = args.width;
    int height = args.height;
    if ((width & 1) || (height & 1)) {
        fprintf(stderr, "Width/height must be even for YUV420\n");
        return -1;
    }

    int out_stride_bytes = width * 4;
    size_t out_bytes = static_cast<size_t>(out_stride_bytes) * height;

    printf("YUV420->RGBA: %dx%d seed=%d\n", width, height, args.seed);

    std::vector<uint8_t> y_plane;
    std::vector<uint8_t> uv_plane;
    generate_yuv420(y_plane, uv_plane, width, height, args.seed);

    cl_platform_id platform_id;
    size_t kernel_size = 0;

    CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
    CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

    context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL, &_err));
    commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

    if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size)) {
        cleanup();
        return -1;
    }
    program = CL_CHECK2(clCreateProgramWithSource(context, 1, (const char**)&kernel_bin, &kernel_size, &_err));
    CL_BUILD_CHECK(program, device_id, clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));

    kernel = CL_CHECK2(clCreateKernel(program, "yuv420_to_rgba", &_err));

    y_mem = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     y_plane.size(), y_plane.data(), &_err));
    uv_mem = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      uv_plane.size(), uv_plane.data(), &_err));
    out_mem = CL_CHECK2(clCreateBuffer(context, CL_MEM_WRITE_ONLY, out_bytes, NULL, &_err));

    int out_stride_pixels = out_stride_bytes / 4;
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &y_mem));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &uv_mem));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &out_mem));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int), &width));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int), &height));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int), &out_stride_pixels));

    size_t gws[2] = {static_cast<size_t>(width), static_cast<size_t>(height)};
    CL_ENQUEUE_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, gws, NULL, 0, NULL, NULL),
                     device_id, kernel, 2, gws, NULL);

    std::vector<uint8_t> out(out_bytes);
    CL_CHECK(clEnqueueReadBuffer(commandQueue, out_mem, CL_TRUE, 0, out_bytes, out.data(), 0, NULL, NULL));

    std::vector<uint8_t> ref(out_bytes);
    cpu_yuv420_to_rgba(y_plane.data(), uv_plane.data(), ref.data(), width, height, out_stride_bytes);

    auto cmp = vx::video::compare_rgba(ref.data(), out_stride_bytes, out.data(), out_stride_bytes,
                                       width, height, 0);

    bool want_dump = args.dump || !cmp.ok;
    if (want_dump) {
        if (!vx::video::ensure_dir(args.outdir)) {
            fprintf(stderr, "Failed to create outdir: %s\n", args.outdir.c_str());
        } else {
            std::string out_ppm = vx::video::make_output_path(args, "output.ppm");
            if (!vx::video::write_ppm_rgba(out_ppm, width, height, out.data(), out_stride_bytes)) {
                fprintf(stderr, "Failed to write %s\n", out_ppm.c_str());
            } else {
                printf("Wrote %s\n", out_ppm.c_str());
            }
        }
    }

    if (!cmp.ok) {
        fprintf(stderr, "FAILED: mismatch at (%d,%d), max_err=%d\n", cmp.first_x, cmp.first_y, cmp.max_err);
        cleanup();
        return -1;
    }

    printf("PASSED! max_err=%d\n", cmp.max_err);
    cleanup();
    return 0;
}
