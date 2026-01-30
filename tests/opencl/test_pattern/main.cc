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
static cl_mem out_mem = NULL;
static uint8_t* kernel_bin = NULL;

static void cleanup() {
    if (commandQueue) clReleaseCommandQueue(commandQueue);
    if (kernel) clReleaseKernel(kernel);
    if (program) clReleaseProgram(program);
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

static void cpu_pattern(std::vector<uint8_t>& out, int width, int height,
                        int stride_bytes, int mode, int seed) {
    if (stride_bytes == 0) stride_bytes = width * 4;
    out.resize(static_cast<size_t>(stride_bytes) * height);
    for (int y = 0; y < height; ++y) {
        uint8_t* row = out.data() + y * stride_bytes;
        for (int x = 0; x < width; ++x) {
            uint8_t r=0, g=0, b=0, a=255;
            if (mode == 0) {
                int bar = (x * 8) / width;
                switch (bar) {
                case 0: r=255; g=255; b=255; break;
                case 1: r=255; g=255; b=0; break;
                case 2: r=0; g=255; b=255; break;
                case 3: r=0; g=255; b=0; break;
                case 4: r=255; g=0; b=255; break;
                case 5: r=255; g=0; b=0; break;
                case 6: r=0; g=0; b=255; break;
                default: r=0; g=0; b=0; break;
                }
            } else if (mode == 1) {
                int block = 8;
                int v = ((x / block) ^ (y / block)) & 1;
                if (v) { r=220; g=220; b=220; } else { r=30; g=30; b=30; }
            } else {
                r = (width <= 1) ? 0 : (x * 255) / (width - 1);
                g = (height <= 1) ? 0 : (y * 255) / (height - 1);
                b = (width + height <= 2) ? 0 : ((x + y) * 255) / (width + height - 2);
                int cx = width / 2;
                int cy = height / 2;
                int dx = x - cx;
                int dy = y - cy;
                int rad = (width < height ? width : height) / 4;
                if (dx * dx + dy * dy <= rad * rad) {
                    r = 255; g = 64; b = 32;
                }
                if (((x + seed) % 17) == 0) {
                    r = 32; g = 64; b = 255;
                }
            }
            row[x * 4 + 0] = r;
            row[x * 4 + 1] = g;
            row[x * 4 + 2] = b;
            row[x * 4 + 3] = a;
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
            printf("  -mode 0|1|2           pattern mode (bars/checker/gradient)\n");
            return 0;
        }
    }

    int width = args.width;
    int height = args.height;
    int stride_bytes = args.stride ? args.stride : width * 4;
    int stride_pixels = stride_bytes / 4;

    printf("Pattern: %dx%d stride=%d mode=%d seed=%d\n", width, height, stride_bytes, args.mode, args.seed);

    size_t out_bytes = static_cast<size_t>(stride_bytes) * height;

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

    kernel = CL_CHECK2(clCreateKernel(program, "test_pattern", &_err));

    out_mem = CL_CHECK2(clCreateBuffer(context, CL_MEM_WRITE_ONLY, out_bytes, NULL, &_err));

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &out_mem));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(int), &width));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(int), &height));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int), &stride_pixels));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int), &args.mode));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int), &args.seed));

    size_t gws[2] = {static_cast<size_t>(width), static_cast<size_t>(height)};
    CL_ENQUEUE_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, gws, NULL, 0, NULL, NULL),
                     device_id, kernel, 2, gws, NULL);

    std::vector<uint8_t> out(out_bytes);
    CL_CHECK(clEnqueueReadBuffer(commandQueue, out_mem, CL_TRUE, 0, out_bytes, out.data(), 0, NULL, NULL));

    std::vector<uint8_t> ref;
    cpu_pattern(ref, width, height, stride_bytes, args.mode, args.seed);

    auto cmp = vx::video::compare_rgba(ref.data(), stride_bytes, out.data(), stride_bytes,
                                       width, height, 0);

    bool want_dump = args.dump || !cmp.ok;
    if (want_dump) {
        if (!vx::video::ensure_dir(args.outdir)) {
            fprintf(stderr, "Failed to create outdir: %s\n", args.outdir.c_str());
        } else {
            std::string out_ppm = vx::video::make_output_path(args, "output.ppm");
            if (!vx::video::write_ppm_rgba(out_ppm, width, height, out.data(), stride_bytes)) {
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
