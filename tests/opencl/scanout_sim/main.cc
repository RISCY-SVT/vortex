#include <CL/opencl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

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
static cl_kernel k_fill_rgba = NULL;
static cl_kernel k_fill_rgb565 = NULL;
static cl_mem buf_rgba = NULL;
static cl_mem buf_rgba_b = NULL;
static cl_mem buf_rgb565 = NULL;
static uint8_t* kernel_bin = NULL;

static void cleanup() {
    if (commandQueue) clReleaseCommandQueue(commandQueue);
    if (k_fill_rgba) clReleaseKernel(k_fill_rgba);
    if (k_fill_rgb565) clReleaseKernel(k_fill_rgb565);
    if (program) clReleaseProgram(program);
    if (buf_rgba) clReleaseMemObject(buf_rgba);
    if (buf_rgba_b) clReleaseMemObject(buf_rgba_b);
    if (buf_rgb565) clReleaseMemObject(buf_rgb565);
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

static void pattern_rgba(std::vector<uint8_t>& out, int w, int h, int stride_bytes, int mode, int seed) {
    if (stride_bytes == 0) stride_bytes = w * 4;
    out.resize(static_cast<size_t>(stride_bytes) * h);
    for (int y = 0; y < h; ++y) {
        uint8_t* row = out.data() + y * stride_bytes;
        for (int x = 0; x < w; ++x) {
            uint8_t r=0, g=0, b=0, a=255;
            if (mode == 0) {
                int bar = (x * 8) / w;
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
                r = (w <= 1) ? 0 : (x * 255) / (w - 1);
                g = (h <= 1) ? 0 : (y * 255) / (h - 1);
                b = (w + h <= 2) ? 0 : ((x + y) * 255) / (w + h - 2);
                if (((x + seed) % 17) == 0) { r=32; g=64; b=255; }
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

    int w = args.width;
    int h = args.height;
    int stride_bytes = args.stride ? args.stride : (int)vx::video::align_up(w * 4, 64);
    int stride_pixels = stride_bytes / 4;
    if (stride_bytes <= w * 4) {
        fprintf(stderr, "Stride padding required: stride_bytes=%d row_bytes=%d\n", stride_bytes, w * 4);
        return -1;
    }

    int rect_x = args.rect_w ? args.rect_x : w / 4;
    int rect_y = args.rect_h ? args.rect_y : h / 4;
    int rect_w = args.rect_w ? args.rect_w : w / 2;
    int rect_h = args.rect_h ? args.rect_h : h / 2;

    printf("Scanout: %dx%d stride_bytes=%d rect=%d,%d %dx%d\n", w, h, stride_bytes, rect_x, rect_y, rect_w, rect_h);

    size_t rgba_bytes = static_cast<size_t>(stride_bytes) * h;

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

    k_fill_rgba = CL_CHECK2(clCreateKernel(program, "fill_rgba", &_err));
    k_fill_rgb565 = CL_CHECK2(clCreateKernel(program, "fill_rgb565", &_err));

    // stride/pitch test (RGBA)
    buf_rgba = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, rgba_bytes, NULL, &_err));

    int mode0 = 0;
    int seed0 = 1;
    int use_rect = 0;
    CL_CHECK(clSetKernelArg(k_fill_rgba, 0, sizeof(cl_mem), &buf_rgba));
    CL_CHECK(clSetKernelArg(k_fill_rgba, 1, sizeof(int), &w));
    CL_CHECK(clSetKernelArg(k_fill_rgba, 2, sizeof(int), &h));
    CL_CHECK(clSetKernelArg(k_fill_rgba, 3, sizeof(int), &stride_pixels));
    CL_CHECK(clSetKernelArg(k_fill_rgba, 4, sizeof(int), &mode0));
    CL_CHECK(clSetKernelArg(k_fill_rgba, 5, sizeof(int), &seed0));
    CL_CHECK(clSetKernelArg(k_fill_rgba, 6, sizeof(int), &use_rect));
    CL_CHECK(clSetKernelArg(k_fill_rgba, 7, sizeof(int), &rect_x));
    CL_CHECK(clSetKernelArg(k_fill_rgba, 8, sizeof(int), &rect_y));
    CL_CHECK(clSetKernelArg(k_fill_rgba, 9, sizeof(int), &rect_w));
    CL_CHECK(clSetKernelArg(k_fill_rgba, 10, sizeof(int), &rect_h));

    size_t gws[2] = {static_cast<size_t>(w), static_cast<size_t>(h)};
    CL_ENQUEUE_CHECK(clEnqueueNDRangeKernel(commandQueue, k_fill_rgba, 2, NULL, gws, NULL, 0, NULL, NULL),
                     device_id, k_fill_rgba, 2, gws, NULL);

    std::vector<uint8_t> out_rgba(rgba_bytes);
    CL_CHECK(clEnqueueReadBuffer(commandQueue, buf_rgba, CL_TRUE, 0, rgba_bytes, out_rgba.data(), 0, NULL, NULL));

    std::vector<uint8_t> ref_rgba;
    pattern_rgba(ref_rgba, w, h, stride_bytes, mode0, seed0);

    auto cmp_stride = vx::video::compare_rgba(ref_rgba.data(), stride_bytes, out_rgba.data(), stride_bytes, w, h, 0);
    bool want_dump_stride = args.dump || !cmp_stride.ok;
    if (want_dump_stride) {
        if (!vx::video::ensure_dir(args.outdir)) {
            fprintf(stderr, "Failed to create outdir: %s\n", args.outdir.c_str());
        } else {
            std::string out_stride = vx::video::make_output_path(args, "output_stride.ppm");
            if (!vx::video::write_ppm_rgba(out_stride, w, h, out_rgba.data(), stride_bytes)) {
                fprintf(stderr, "Failed to write %s\n", out_stride.c_str());
            }
        }
    }

    if (!cmp_stride.ok) {
        fprintf(stderr, "FAILED (stride): mismatch at (%d,%d) max_err=%d\n", cmp_stride.first_x, cmp_stride.first_y, cmp_stride.max_err);
        cleanup();
        return -1;
    }

    // pixel format test (RGB565)
    int stride565_bytes = (int)vx::video::align_up(w * 2, 64);
    int stride565_pixels = stride565_bytes / 2;
    size_t rgb565_bytes = static_cast<size_t>(stride565_bytes) * h;

    buf_rgb565 = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, rgb565_bytes, NULL, &_err));
    int mode1 = 2;
    int seed1 = 3;
    CL_CHECK(clSetKernelArg(k_fill_rgb565, 0, sizeof(cl_mem), &buf_rgb565));
    CL_CHECK(clSetKernelArg(k_fill_rgb565, 1, sizeof(int), &w));
    CL_CHECK(clSetKernelArg(k_fill_rgb565, 2, sizeof(int), &h));
    CL_CHECK(clSetKernelArg(k_fill_rgb565, 3, sizeof(int), &stride565_pixels));
    CL_CHECK(clSetKernelArg(k_fill_rgb565, 4, sizeof(int), &mode1));
    CL_CHECK(clSetKernelArg(k_fill_rgb565, 5, sizeof(int), &seed1));

    CL_ENQUEUE_CHECK(clEnqueueNDRangeKernel(commandQueue, k_fill_rgb565, 2, NULL, gws, NULL, 0, NULL, NULL),
                     device_id, k_fill_rgb565, 2, gws, NULL);

    std::vector<uint16_t> out565(rgb565_bytes / 2);
    CL_CHECK(clEnqueueReadBuffer(commandQueue, buf_rgb565, CL_TRUE, 0, rgb565_bytes, out565.data(), 0, NULL, NULL));

    // CPU reference for RGB565
    std::vector<uint8_t> ref_rgba565;
    pattern_rgba(ref_rgba565, w, h, w * 4, mode1, seed1);
    std::vector<uint16_t> ref565(stride565_pixels * h);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const uint8_t* p = ref_rgba565.data() + (y * w + x) * 4;
            ref565[y * stride565_pixels + x] = vx::video::pack_rgb565(p[0], p[1], p[2]);
        }
    }

    std::vector<uint8_t> out_rgb;
    std::vector<uint8_t> ref_rgb;
    vx::video::rgb565_to_rgb(out565.data(), w, h, stride565_pixels, out_rgb);
    vx::video::rgb565_to_rgb(ref565.data(), w, h, stride565_pixels, ref_rgb);

    auto cmp_fmt = vx::video::compare_rgb(ref_rgb.data(), w * 3, out_rgb.data(), w * 3, w, h, 1);
    bool want_dump_fmt = args.dump || !cmp_fmt.ok;
    if (want_dump_fmt) {
        if (!vx::video::ensure_dir(args.outdir)) {
            fprintf(stderr, "Failed to create outdir: %s\n", args.outdir.c_str());
        } else {
            std::string out_fmt = vx::video::make_output_path(args, "output_fmt_rgb565.ppm");
            if (!vx::video::write_ppm_rgb(out_fmt, w, h, out_rgb.data(), w * 3)) {
                fprintf(stderr, "Failed to write %s\n", out_fmt.c_str());
            }
        }
    }

    if (!cmp_fmt.ok) {
        fprintf(stderr, "FAILED (rgb565): mismatch at (%d,%d) max_err=%d\n", cmp_fmt.first_x, cmp_fmt.first_y, cmp_fmt.max_err);
        cleanup();
        return -1;
    }

    // double-buffering test
    buf_rgba_b = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, rgba_bytes, NULL, &_err));
    int mode_a = 0;
    int seed_a = 5;
    int mode_b = 1;
    int seed_b = 7;

    CL_CHECK(clSetKernelArg(k_fill_rgba, 0, sizeof(cl_mem), &buf_rgba));
    CL_CHECK(clSetKernelArg(k_fill_rgba, 1, sizeof(int), &w));
    CL_CHECK(clSetKernelArg(k_fill_rgba, 2, sizeof(int), &h));
    CL_CHECK(clSetKernelArg(k_fill_rgba, 3, sizeof(int), &stride_pixels));
    CL_CHECK(clSetKernelArg(k_fill_rgba, 4, sizeof(int), &mode_a));
    CL_CHECK(clSetKernelArg(k_fill_rgba, 5, sizeof(int), &seed_a));
    CL_CHECK(clSetKernelArg(k_fill_rgba, 6, sizeof(int), &use_rect));
    CL_CHECK(clSetKernelArg(k_fill_rgba, 7, sizeof(int), &rect_x));
    CL_CHECK(clSetKernelArg(k_fill_rgba, 8, sizeof(int), &rect_y));
    CL_CHECK(clSetKernelArg(k_fill_rgba, 9, sizeof(int), &rect_w));
    CL_CHECK(clSetKernelArg(k_fill_rgba, 10, sizeof(int), &rect_h));

    CL_ENQUEUE_CHECK(clEnqueueNDRangeKernel(commandQueue, k_fill_rgba, 2, NULL, gws, NULL, 0, NULL, NULL),
                     device_id, k_fill_rgba, 2, gws, NULL);

    CL_CHECK(clSetKernelArg(k_fill_rgba, 0, sizeof(cl_mem), &buf_rgba_b));
    CL_CHECK(clSetKernelArg(k_fill_rgba, 4, sizeof(int), &mode_b));
    CL_CHECK(clSetKernelArg(k_fill_rgba, 5, sizeof(int), &seed_b));

    CL_ENQUEUE_CHECK(clEnqueueNDRangeKernel(commandQueue, k_fill_rgba, 2, NULL, gws, NULL, 0, NULL, NULL),
                     device_id, k_fill_rgba, 2, gws, NULL);

    std::vector<uint8_t> out_a(rgba_bytes);
    std::vector<uint8_t> out_b(rgba_bytes);
    CL_CHECK(clEnqueueReadBuffer(commandQueue, buf_rgba, CL_TRUE, 0, rgba_bytes, out_a.data(), 0, NULL, NULL));
    CL_CHECK(clEnqueueReadBuffer(commandQueue, buf_rgba_b, CL_TRUE, 0, rgba_bytes, out_b.data(), 0, NULL, NULL));

    std::vector<uint8_t> ref_a;
    std::vector<uint8_t> ref_b;
    pattern_rgba(ref_a, w, h, stride_bytes, mode_a, seed_a);
    pattern_rgba(ref_b, w, h, stride_bytes, mode_b, seed_b);

    auto cmp_a = vx::video::compare_rgba(ref_a.data(), stride_bytes, out_a.data(), stride_bytes, w, h, 0);
    auto cmp_b = vx::video::compare_rgba(ref_b.data(), stride_bytes, out_b.data(), stride_bytes, w, h, 0);

    bool want_dump_db = args.dump || !cmp_a.ok || !cmp_b.ok;
    if (want_dump_db) {
        if (!vx::video::ensure_dir(args.outdir)) {
            fprintf(stderr, "Failed to create outdir: %s\n", args.outdir.c_str());
        } else {
            std::string out_f0 = vx::video::make_output_path(args, "output_frame0.ppm");
            std::string out_f1 = vx::video::make_output_path(args, "output_frame1.ppm");
            if (!vx::video::write_ppm_rgba(out_f0, w, h, out_a.data(), stride_bytes)) {
                fprintf(stderr, "Failed to write %s\n", out_f0.c_str());
            }
            if (!vx::video::write_ppm_rgba(out_f1, w, h, out_b.data(), stride_bytes)) {
                fprintf(stderr, "Failed to write %s\n", out_f1.c_str());
            }
        }
    }

    if (!cmp_a.ok || !cmp_b.ok) {
        fprintf(stderr, "FAILED (double buffer): mismatch frame0=%d frame1=%d\n", cmp_a.ok ? 0 : 1, cmp_b.ok ? 0 : 1);
        cleanup();
        return -1;
    }

    // partial update test
    int mode_base = 0;
    int mode_patch = 2;
    int seed_base = 11;
    int seed_patch = 13;
    use_rect = 0;
    CL_CHECK(clSetKernelArg(k_fill_rgba, 0, sizeof(cl_mem), &buf_rgba));
    CL_CHECK(clSetKernelArg(k_fill_rgba, 4, sizeof(int), &mode_base));
    CL_CHECK(clSetKernelArg(k_fill_rgba, 5, sizeof(int), &seed_base));
    CL_CHECK(clSetKernelArg(k_fill_rgba, 6, sizeof(int), &use_rect));
    CL_ENQUEUE_CHECK(clEnqueueNDRangeKernel(commandQueue, k_fill_rgba, 2, NULL, gws, NULL, 0, NULL, NULL),
                     device_id, k_fill_rgba, 2, gws, NULL);

    use_rect = 1;
    CL_CHECK(clSetKernelArg(k_fill_rgba, 4, sizeof(int), &mode_patch));
    CL_CHECK(clSetKernelArg(k_fill_rgba, 5, sizeof(int), &seed_patch));
    CL_CHECK(clSetKernelArg(k_fill_rgba, 6, sizeof(int), &use_rect));
    CL_CHECK(clSetKernelArg(k_fill_rgba, 7, sizeof(int), &rect_x));
    CL_CHECK(clSetKernelArg(k_fill_rgba, 8, sizeof(int), &rect_y));
    CL_CHECK(clSetKernelArg(k_fill_rgba, 9, sizeof(int), &rect_w));
    CL_CHECK(clSetKernelArg(k_fill_rgba, 10, sizeof(int), &rect_h));
    CL_ENQUEUE_CHECK(clEnqueueNDRangeKernel(commandQueue, k_fill_rgba, 2, NULL, gws, NULL, 0, NULL, NULL),
                     device_id, k_fill_rgba, 2, gws, NULL);

    std::vector<uint8_t> out_partial(rgba_bytes);
    CL_CHECK(clEnqueueReadBuffer(commandQueue, buf_rgba, CL_TRUE, 0, rgba_bytes, out_partial.data(), 0, NULL, NULL));

    std::vector<uint8_t> ref_partial;
    pattern_rgba(ref_partial, w, h, stride_bytes, mode_base, seed_base);
    // apply patch rect on CPU
    for (int y = rect_y; y < rect_y + rect_h; ++y) {
        for (int x = rect_x; x < rect_x + rect_w; ++x) {
            uint8_t r=0, g=0, b=0, a=255;
            int mode = mode_patch;
            if (mode == 0) {
                int bar = (x * 8) / w;
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
                r = (w <= 1) ? 0 : (x * 255) / (w - 1);
                g = (h <= 1) ? 0 : (y * 255) / (h - 1);
                b = (w + h <= 2) ? 0 : ((x + y) * 255) / (w + h - 2);
                if (((x + seed_patch) % 17) == 0) { r=32; g=64; b=255; }
            }
            uint8_t* row = ref_partial.data() + y * stride_bytes;
            row[x * 4 + 0] = r;
            row[x * 4 + 1] = g;
            row[x * 4 + 2] = b;
            row[x * 4 + 3] = a;
        }
    }

    auto cmp_partial = vx::video::compare_rgba(ref_partial.data(), stride_bytes, out_partial.data(), stride_bytes, w, h, 0);
    bool want_dump_partial = args.dump || !cmp_partial.ok;
    if (want_dump_partial) {
        if (!vx::video::ensure_dir(args.outdir)) {
            fprintf(stderr, "Failed to create outdir: %s\n", args.outdir.c_str());
        } else {
            std::string out_partial_ppm = vx::video::make_output_path(args, "output_partial.ppm");
            if (!vx::video::write_ppm_rgba(out_partial_ppm, w, h, out_partial.data(), stride_bytes)) {
                fprintf(stderr, "Failed to write %s\n", out_partial_ppm.c_str());
            }
        }
    }

    if (!cmp_partial.ok) {
        fprintf(stderr, "FAILED (partial): mismatch at (%d,%d) max_err=%d\n", cmp_partial.first_x, cmp_partial.first_y, cmp_partial.max_err);
        cleanup();
        return -1;
    }

    printf("PASSED! scanout tests ok\n");
    cleanup();
    return 0;
}
