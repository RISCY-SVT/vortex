#include <CL/opencl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
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
static cl_kernel k_nearest = NULL;
static cl_kernel k_bilinear = NULL;
static cl_mem in_mem = NULL;
static cl_mem out_nearest = NULL;
static cl_mem out_bilinear = NULL;
static uint8_t* kernel_bin = NULL;

static void cleanup() {
    if (commandQueue) clReleaseCommandQueue(commandQueue);
    if (k_nearest) clReleaseKernel(k_nearest);
    if (k_bilinear) clReleaseKernel(k_bilinear);
    if (program) clReleaseProgram(program);
    if (in_mem) clReleaseMemObject(in_mem);
    if (out_nearest) clReleaseMemObject(out_nearest);
    if (out_bilinear) clReleaseMemObject(out_bilinear);
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

static void generate_input(std::vector<uint8_t>& out, int width, int height, int stride_bytes, int seed) {
    if (stride_bytes == 0) stride_bytes = width * 4;
    out.resize(static_cast<size_t>(stride_bytes) * height);
    for (int y = 0; y < height; ++y) {
        uint8_t* row = out.data() + y * stride_bytes;
        for (int x = 0; x < width; ++x) {
            uint8_t r = (width <= 1) ? 0 : (uint8_t)((x * 255) / (width - 1));
            uint8_t g = (height <= 1) ? 0 : (uint8_t)((y * 255) / (height - 1));
            uint8_t b = (uint8_t)(((x + y + seed) * 13) & 0xFF);
            if (((x + seed) % 11) == 0) { r = 255; g = 32; b = 32; }
            row[x * 4 + 0] = r;
            row[x * 4 + 1] = g;
            row[x * 4 + 2] = b;
            row[x * 4 + 3] = 255;
        }
    }
}

static void cpu_scale_nearest(const uint8_t* src, int in_w, int in_h, int in_stride,
                              uint8_t* dst, int out_w, int out_h, int out_stride) {
    for (int y = 0; y < out_h; ++y) {
        for (int x = 0; x < out_w; ++x) {
            float fx = (out_w > 1) ? ((float)x * (float)(in_w - 1) / (float)(out_w - 1)) : 0.0f;
            float fy = (out_h > 1) ? ((float)y * (float)(in_h - 1) / (float)(out_h - 1)) : 0.0f;
            int ix = (int)(fx + 0.5f);
            int iy = (int)(fy + 0.5f);
            if (ix < 0) {
                ix = 0;
            }
            if (ix >= in_w) {
                ix = in_w - 1;
            }
            if (iy < 0) {
                iy = 0;
            }
            if (iy >= in_h) {
                iy = in_h - 1;
            }
            const uint8_t* p = src + (iy * in_stride + ix) * 4;
            uint8_t* q = dst + (y * out_stride + x) * 4;
            q[0] = p[0]; q[1] = p[1]; q[2] = p[2]; q[3] = p[3];
        }
    }
}

static void cpu_scale_bilinear(const uint8_t* src, int in_w, int in_h, int in_stride,
                               uint8_t* dst, int out_w, int out_h, int out_stride) {
    for (int y = 0; y < out_h; ++y) {
        for (int x = 0; x < out_w; ++x) {
            float fx = (out_w > 1) ? ((float)x * (float)(in_w - 1) / (float)(out_w - 1)) : 0.0f;
            float fy = (out_h > 1) ? ((float)y * (float)(in_h - 1) / (float)(out_h - 1)) : 0.0f;
            int x0 = (int)floorf(fx);
            int y0 = (int)floorf(fy);
            int x1 = x0 + 1;
            if (x1 >= in_w) x1 = in_w - 1;
            int y1 = y0 + 1;
            if (y1 >= in_h) y1 = in_h - 1;
            float tx = fx - (float)x0;
            float ty = fy - (float)y0;

            const uint8_t* c00 = src + (y0 * in_stride + x0) * 4;
            const uint8_t* c10 = src + (y0 * in_stride + x1) * 4;
            const uint8_t* c01 = src + (y1 * in_stride + x0) * 4;
            const uint8_t* c11 = src + (y1 * in_stride + x1) * 4;

            for (int c = 0; c < 4; ++c) {
                float top = c00[c] + (c10[c] - c00[c]) * tx;
                float bot = c01[c] + (c11[c] - c01[c]) * tx;
                float val = top + (bot - top) * ty;
                int iv = (int)(val + 0.5f);
                if (iv < 0) {
                    iv = 0;
                }
                if (iv > 255) {
                    iv = 255;
                }
                dst[(y * out_stride + x) * 4 + c] = (uint8_t)iv;
            }
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
            printf("  -ow/-oh set output size (default 2x)\n");
            return 0;
        }
    }

    int in_w = args.width;
    int in_h = args.height;
    int in_stride = in_w; // pixels
    int out_w = args.out_width ? args.out_width : in_w * 2;
    int out_h = args.out_height ? args.out_height : in_h * 2;
    out_w = std::max(1, out_w);
    out_h = std::max(1, out_h);

    printf("Scaler: in=%dx%d out=%dx%d\n", in_w, in_h, out_w, out_h);

    size_t in_bytes = static_cast<size_t>(in_w) * in_h * 4;
    size_t out_bytes = static_cast<size_t>(out_w) * out_h * 4;
    size_t out2_bytes = static_cast<size_t>(out_w) * out_h * 4;

    std::vector<uint8_t> in;
    generate_input(in, in_w, in_h, in_w * 4, args.seed);

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

    k_nearest = CL_CHECK2(clCreateKernel(program, "scale_nearest", &_err));
    k_bilinear = CL_CHECK2(clCreateKernel(program, "scale_bilinear", &_err));

    in_mem = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, in_bytes, in.data(), &_err));
    out_nearest = CL_CHECK2(clCreateBuffer(context, CL_MEM_WRITE_ONLY, out_bytes, NULL, &_err));
    out_bilinear = CL_CHECK2(clCreateBuffer(context, CL_MEM_WRITE_ONLY, out2_bytes, NULL, &_err));

    int in_stride_pixels = in_stride;
    int out_stride_pixels = out_w;
    int out2_stride_pixels = out_w;

    CL_CHECK(clSetKernelArg(k_nearest, 0, sizeof(cl_mem), &in_mem));
    CL_CHECK(clSetKernelArg(k_nearest, 1, sizeof(cl_mem), &out_nearest));
    CL_CHECK(clSetKernelArg(k_nearest, 2, sizeof(int), &in_w));
    CL_CHECK(clSetKernelArg(k_nearest, 3, sizeof(int), &in_h));
    CL_CHECK(clSetKernelArg(k_nearest, 4, sizeof(int), &in_stride_pixels));
    CL_CHECK(clSetKernelArg(k_nearest, 5, sizeof(int), &out_w));
    CL_CHECK(clSetKernelArg(k_nearest, 6, sizeof(int), &out_h));
    CL_CHECK(clSetKernelArg(k_nearest, 7, sizeof(int), &out_stride_pixels));

    size_t gws_near[2] = {static_cast<size_t>(out_w), static_cast<size_t>(out_h)};
    CL_ENQUEUE_CHECK(clEnqueueNDRangeKernel(commandQueue, k_nearest, 2, NULL, gws_near, NULL, 0, NULL, NULL),
                     device_id, k_nearest, 2, gws_near, NULL);

    CL_CHECK(clSetKernelArg(k_bilinear, 0, sizeof(cl_mem), &in_mem));
    CL_CHECK(clSetKernelArg(k_bilinear, 1, sizeof(cl_mem), &out_bilinear));
    CL_CHECK(clSetKernelArg(k_bilinear, 2, sizeof(int), &in_w));
    CL_CHECK(clSetKernelArg(k_bilinear, 3, sizeof(int), &in_h));
    CL_CHECK(clSetKernelArg(k_bilinear, 4, sizeof(int), &in_stride_pixels));
    CL_CHECK(clSetKernelArg(k_bilinear, 5, sizeof(int), &out_w));
    CL_CHECK(clSetKernelArg(k_bilinear, 6, sizeof(int), &out_h));
    CL_CHECK(clSetKernelArg(k_bilinear, 7, sizeof(int), &out2_stride_pixels));

    size_t gws_bil[2] = {static_cast<size_t>(out_w), static_cast<size_t>(out_h)};
    CL_ENQUEUE_CHECK(clEnqueueNDRangeKernel(commandQueue, k_bilinear, 2, NULL, gws_bil, NULL, 0, NULL, NULL),
                     device_id, k_bilinear, 2, gws_bil, NULL);

    std::vector<uint8_t> outN(out_bytes);
    std::vector<uint8_t> outB(out2_bytes);
    CL_CHECK(clEnqueueReadBuffer(commandQueue, out_nearest, CL_TRUE, 0, out_bytes, outN.data(), 0, NULL, NULL));
    CL_CHECK(clEnqueueReadBuffer(commandQueue, out_bilinear, CL_TRUE, 0, out2_bytes, outB.data(), 0, NULL, NULL));

    std::vector<uint8_t> refN(out_bytes);
    std::vector<uint8_t> refB(out2_bytes);
    cpu_scale_nearest(in.data(), in_w, in_h, in_stride, refN.data(), out_w, out_h, out_stride_pixels);
    cpu_scale_bilinear(in.data(), in_w, in_h, in_stride, refB.data(), out_w, out_h, out2_stride_pixels);

    auto cmpN = vx::video::compare_rgba(refN.data(), out_w * 4, outN.data(), out_w * 4, out_w, out_h, 1);
    auto cmpB = vx::video::compare_rgba(refB.data(), out_w * 4, outB.data(), out_w * 4, out_w, out_h, 1);

    bool want_dump = args.dump || !cmpN.ok || !cmpB.ok;
    if (want_dump) {
        if (!vx::video::ensure_dir(args.outdir)) {
            fprintf(stderr, "Failed to create outdir: %s\n", args.outdir.c_str());
        } else {
            std::string out_near = vx::video::make_output_path(args, "output_nearest.ppm");
            std::string out_bil = vx::video::make_output_path(args, "output_bilinear.ppm");
            if (!vx::video::write_ppm_rgba(out_near, out_w, out_h, outN.data(), out_w * 4)) {
                fprintf(stderr, "Failed to write %s\n", out_near.c_str());
            }
            if (!vx::video::write_ppm_rgba(out_bil, out_w, out_h, outB.data(), out_w * 4)) {
                fprintf(stderr, "Failed to write %s\n", out_bil.c_str());
            }
        }
    }

    if (!cmpN.ok) {
        fprintf(stderr, "FAILED (nearest): mismatch at (%d,%d), max_err=%d\n", cmpN.first_x, cmpN.first_y, cmpN.max_err);
        cleanup();
        return -1;
    }
    if (!cmpB.ok) {
        fprintf(stderr, "FAILED (bilinear): mismatch at (%d,%d), max_err=%d\n", cmpB.first_x, cmpB.first_y, cmpB.max_err);
        cleanup();
        return -1;
    }

    printf("PASSED! max_err_nearest=%d max_err_bilinear=%d\n", cmpN.max_err, cmpB.max_err);
    cleanup();
    return 0;
}
