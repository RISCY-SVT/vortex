#include <CL/opencl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>

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
static cl_kernel k_blur = NULL;
static cl_kernel k_sobel = NULL;
static cl_mem in_mem = NULL;
static cl_mem out_blur = NULL;
static cl_mem out_sobel = NULL;
static uint8_t* kernel_bin = NULL;

static void cleanup() {
    if (commandQueue) clReleaseCommandQueue(commandQueue);
    if (k_blur) clReleaseKernel(k_blur);
    if (k_sobel) clReleaseKernel(k_sobel);
    if (program) clReleaseProgram(program);
    if (in_mem) clReleaseMemObject(in_mem);
    if (out_blur) clReleaseMemObject(out_blur);
    if (out_sobel) clReleaseMemObject(out_sobel);
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

static inline int clamp_int(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static inline int luma_at(const uint8_t* src, int x, int y, int w, int h, int stride) {
    x = clamp_int(x, 0, w - 1);
    y = clamp_int(y, 0, h - 1);
    const uint8_t* p = src + (y * stride + x) * 4;
    return (77 * p[0] + 150 * p[1] + 29 * p[2]) >> 8;
}

static void generate_input(std::vector<uint8_t>& out, int width, int height, int seed) {
    out.resize(static_cast<size_t>(width) * height * 4);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            uint8_t r = (width <= 1) ? 0 : (uint8_t)((x * 255) / (width - 1));
            uint8_t g = (height <= 1) ? 0 : (uint8_t)((y * 255) / (height - 1));
            uint8_t b = (uint8_t)(((x * 7 + y * 13 + seed) * 3) & 0xFF);
            int idx = (y * width + x) * 4;
            out[idx + 0] = r;
            out[idx + 1] = g;
            out[idx + 2] = b;
            out[idx + 3] = 255;
        }
    }
}

static void cpu_blur(const uint8_t* src, uint8_t* dst, int w, int h) {
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int s00 = luma_at(src, x - 1, y - 1, w, h, w);
            int s01 = luma_at(src, x,     y - 1, w, h, w);
            int s02 = luma_at(src, x + 1, y - 1, w, h, w);
            int s10 = luma_at(src, x - 1, y,     w, h, w);
            int s11 = luma_at(src, x,     y,     w, h, w);
            int s12 = luma_at(src, x + 1, y,     w, h, w);
            int s20 = luma_at(src, x - 1, y + 1, w, h, w);
            int s21 = luma_at(src, x,     y + 1, w, h, w);
            int s22 = luma_at(src, x + 1, y + 1, w, h, w);
            int sum = s00 + 2 * s01 + s02 + 2 * s10 + 4 * s11 + 2 * s12 + s20 + 2 * s21 + s22;
            int v = sum >> 4;
            v = clamp_int(v, 0, 255);
            int idx = (y * w + x) * 4;
            dst[idx + 0] = dst[idx + 1] = dst[idx + 2] = (uint8_t)v;
            dst[idx + 3] = 255;
        }
    }
}

static void cpu_sobel(const uint8_t* src, uint8_t* dst, int w, int h) {
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int s00 = luma_at(src, x - 1, y - 1, w, h, w);
            int s01 = luma_at(src, x,     y - 1, w, h, w);
            int s02 = luma_at(src, x + 1, y - 1, w, h, w);
            int s10 = luma_at(src, x - 1, y,     w, h, w);
            int s12 = luma_at(src, x + 1, y,     w, h, w);
            int s20 = luma_at(src, x - 1, y + 1, w, h, w);
            int s21 = luma_at(src, x,     y + 1, w, h, w);
            int s22 = luma_at(src, x + 1, y + 1, w, h, w);
            int gx = -s00 + s02 - 2 * s10 + 2 * s12 - s20 + s22;
            int gy = -s00 - 2 * s01 - s02 + s20 + 2 * s21 + s22;
            int mag = std::abs(gx) + std::abs(gy);
            mag = clamp_int(mag, 0, 255);
            int idx = (y * w + x) * 4;
            dst[idx + 0] = dst[idx + 1] = dst[idx + 2] = (uint8_t)mag;
            dst[idx + 3] = 255;
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

    int w = args.width;
    int h = args.height;
    printf("Convolution: %dx%d\n", w, h);

    size_t bytes = static_cast<size_t>(w) * h * 4;
    std::vector<uint8_t> in;
    generate_input(in, w, h, args.seed);

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

    k_blur = CL_CHECK2(clCreateKernel(program, "gaussian_blur", &_err));
    k_sobel = CL_CHECK2(clCreateKernel(program, "sobel_edge", &_err));

    in_mem = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, in.data(), &_err));
    out_blur = CL_CHECK2(clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, &_err));
    out_sobel = CL_CHECK2(clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, &_err));

    int stride = w;
    CL_CHECK(clSetKernelArg(k_blur, 0, sizeof(cl_mem), &in_mem));
    CL_CHECK(clSetKernelArg(k_blur, 1, sizeof(cl_mem), &out_blur));
    CL_CHECK(clSetKernelArg(k_blur, 2, sizeof(int), &w));
    CL_CHECK(clSetKernelArg(k_blur, 3, sizeof(int), &h));
    CL_CHECK(clSetKernelArg(k_blur, 4, sizeof(int), &stride));

    CL_CHECK(clSetKernelArg(k_sobel, 0, sizeof(cl_mem), &in_mem));
    CL_CHECK(clSetKernelArg(k_sobel, 1, sizeof(cl_mem), &out_sobel));
    CL_CHECK(clSetKernelArg(k_sobel, 2, sizeof(int), &w));
    CL_CHECK(clSetKernelArg(k_sobel, 3, sizeof(int), &h));
    CL_CHECK(clSetKernelArg(k_sobel, 4, sizeof(int), &stride));

    size_t gws[2] = {static_cast<size_t>(w), static_cast<size_t>(h)};
    CL_ENQUEUE_CHECK(clEnqueueNDRangeKernel(commandQueue, k_blur, 2, NULL, gws, NULL, 0, NULL, NULL),
                     device_id, k_blur, 2, gws, NULL);
    CL_ENQUEUE_CHECK(clEnqueueNDRangeKernel(commandQueue, k_sobel, 2, NULL, gws, NULL, 0, NULL, NULL),
                     device_id, k_sobel, 2, gws, NULL);

    std::vector<uint8_t> outB(bytes);
    std::vector<uint8_t> outS(bytes);
    CL_CHECK(clEnqueueReadBuffer(commandQueue, out_blur, CL_TRUE, 0, bytes, outB.data(), 0, NULL, NULL));
    CL_CHECK(clEnqueueReadBuffer(commandQueue, out_sobel, CL_TRUE, 0, bytes, outS.data(), 0, NULL, NULL));

    std::vector<uint8_t> refB(bytes);
    std::vector<uint8_t> refS(bytes);
    cpu_blur(in.data(), refB.data(), w, h);
    cpu_sobel(in.data(), refS.data(), w, h);

    auto cmpB = vx::video::compare_rgba(refB.data(), w * 4, outB.data(), w * 4, w, h, 0);
    auto cmpS = vx::video::compare_rgba(refS.data(), w * 4, outS.data(), w * 4, w, h, 0);

    bool want_dump = args.dump || !cmpB.ok || !cmpS.ok;
    if (want_dump) {
        if (!vx::video::ensure_dir(args.outdir)) {
            fprintf(stderr, "Failed to create outdir: %s\n", args.outdir.c_str());
        } else {
            std::string out_blur = vx::video::make_output_path(args, "output_blur.ppm");
            std::string out_sobel = vx::video::make_output_path(args, "output_sobel.ppm");
            if (!vx::video::write_ppm_rgba(out_blur, w, h, outB.data(), w * 4)) {
                fprintf(stderr, "Failed to write %s\n", out_blur.c_str());
            }
            if (!vx::video::write_ppm_rgba(out_sobel, w, h, outS.data(), w * 4)) {
                fprintf(stderr, "Failed to write %s\n", out_sobel.c_str());
            }
        }
    }

    if (!cmpB.ok) {
        fprintf(stderr, "FAILED (blur): mismatch at (%d,%d), max_err=%d\n", cmpB.first_x, cmpB.first_y, cmpB.max_err);
        cleanup();
        return -1;
    }
    if (!cmpS.ok) {
        fprintf(stderr, "FAILED (sobel): mismatch at (%d,%d), max_err=%d\n", cmpS.first_x, cmpS.first_y, cmpS.max_err);
        cleanup();
        return -1;
    }

    printf("PASSED! max_err_blur=%d max_err_sobel=%d\n", cmpB.max_err, cmpS.max_err);
    cleanup();
    return 0;
}
