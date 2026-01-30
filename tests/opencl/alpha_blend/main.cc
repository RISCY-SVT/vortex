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
static cl_kernel kernel = NULL;
static cl_mem bg_mem = NULL;
static cl_mem fg_mem = NULL;
static cl_mem out_mem = NULL;
static uint8_t* kernel_bin = NULL;

static void cleanup() {
    if (commandQueue) clReleaseCommandQueue(commandQueue);
    if (kernel) clReleaseKernel(kernel);
    if (program) clReleaseProgram(program);
    if (bg_mem) clReleaseMemObject(bg_mem);
    if (fg_mem) clReleaseMemObject(fg_mem);
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

static void generate_bg(std::vector<uint8_t>& out, int w, int h) {
    out.resize(static_cast<size_t>(w) * h * 4);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            uint8_t r = (uint8_t)((x * 255) / (w - 1));
            uint8_t g = (uint8_t)((y * 255) / (h - 1));
            uint8_t b = 32;
            int idx = (y * w + x) * 4;
            out[idx + 0] = r;
            out[idx + 1] = g;
            out[idx + 2] = b;
            out[idx + 3] = 255;
        }
    }
}

static void generate_fg(std::vector<uint8_t>& out, int w, int h) {
    out.resize(static_cast<size_t>(w) * h * 4);
    int cx = w / 2;
    int cy = h / 2;
    int rad = (w < h ? w : h) / 3;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int dx = x - cx;
            int dy = y - cy;
            int dist2 = dx * dx + dy * dy;
            int idx = (y * w + x) * 4;
            if (dist2 <= rad * rad) {
                int a = 255 - (dist2 * 255) / (rad * rad);
                out[idx + 0] = 255;
                out[idx + 1] = 64;
                out[idx + 2] = 32;
                out[idx + 3] = (uint8_t)a;
            } else {
                out[idx + 0] = 0;
                out[idx + 1] = 0;
                out[idx + 2] = 0;
                out[idx + 3] = 0;
            }
        }
    }
}

static void cpu_blend(const uint8_t* bg, const uint8_t* fg, uint8_t* out, int w, int h) {
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int idx = (y * w + x) * 4;
            int a = fg[idx + 3];
            int inv = 255 - a;
            out[idx + 0] = (uint8_t)((fg[idx + 0] * a + bg[idx + 0] * inv + 127) / 255);
            out[idx + 1] = (uint8_t)((fg[idx + 1] * a + bg[idx + 1] * inv + 127) / 255);
            out[idx + 2] = (uint8_t)((fg[idx + 2] * a + bg[idx + 2] * inv + 127) / 255);
            out[idx + 3] = 255;
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
    printf("Alpha blend: %dx%d\n", w, h);

    size_t bytes = static_cast<size_t>(w) * h * 4;
    std::vector<uint8_t> bg;
    std::vector<uint8_t> fg;
    generate_bg(bg, w, h);
    generate_fg(fg, w, h);

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

    kernel = CL_CHECK2(clCreateKernel(program, "alpha_blend", &_err));

    bg_mem = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, bg.data(), &_err));
    fg_mem = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, fg.data(), &_err));
    out_mem = CL_CHECK2(clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, &_err));

    int stride = w;
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &bg_mem));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &fg_mem));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &out_mem));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int), &w));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int), &h));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int), &stride));

    size_t gws[2] = {static_cast<size_t>(w), static_cast<size_t>(h)};
    CL_ENQUEUE_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, gws, NULL, 0, NULL, NULL),
                     device_id, kernel, 2, gws, NULL);

    std::vector<uint8_t> out(bytes);
    CL_CHECK(clEnqueueReadBuffer(commandQueue, out_mem, CL_TRUE, 0, bytes, out.data(), 0, NULL, NULL));

    std::vector<uint8_t> ref(bytes);
    cpu_blend(bg.data(), fg.data(), ref.data(), w, h);

    auto cmp = vx::video::compare_rgba(ref.data(), w * 4, out.data(), w * 4, w, h, 0);

    bool want_dump = args.dump || !cmp.ok;
    if (want_dump) {
        if (!vx::video::ensure_dir(args.outdir)) {
            fprintf(stderr, "Failed to create outdir: %s\n", args.outdir.c_str());
        } else {
            std::string out_blend = vx::video::make_output_path(args, "output_blend.ppm");
            if (!vx::video::write_ppm_rgba(out_blend, w, h, out.data(), w * 4)) {
                fprintf(stderr, "Failed to write %s\n", out_blend.c_str());
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
