// Copyright © 2026
// Shared utilities for video-like OpenCL tests (SimX-friendly)
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vx {
namespace video {

enum PixelFormat {
    PIXEL_RGBA8888 = 0,
    PIXEL_RGB565   = 1,
    PIXEL_BGRA8888 = 2,
};

struct Args {
    int width = 64;
    int height = 64;
    int stride = 0;      // bytes, 0 = tight
    int mode = 0;
    int seed = 1;
    int out_width = 0;   // 0 = same as width
    int out_height = 0;  // 0 = same as height
    int fmt = PIXEL_RGBA8888;
    int rect_x = 0;
    int rect_y = 0;
    int rect_w = 0;
    int rect_h = 0;
    bool dump = false;
    std::string outdir = "artifacts";
    std::string prefix;
};

struct CompareResult {
    bool ok = true;
    int first_x = -1;
    int first_y = -1;
    int max_err = 0;
};

// alignment helpers
size_t align_up(size_t v, size_t align);

// clamp helper
int clamp_int(int v, int lo, int hi);

// parse shared arguments, returns unconsumed args in extras
bool parse_common_args(int argc, char** argv, Args& args, std::vector<std::string>& extras, std::string& err);
void print_common_usage(const char* app);

// minimal self-test (CPU) for utilities
bool utils_self_test(std::string& err);

// filesystem helpers
bool ensure_dir(const std::string& path);
std::string make_output_path(const Args& args, const std::string& filename);

// PPM writers (P6)
bool write_ppm_rgb(const std::string& path, int width, int height,
                   const uint8_t* rgb, int stride_bytes);
bool write_ppm_rgba(const std::string& path, int width, int height,
                    const uint8_t* rgba, int stride_bytes);
bool write_ppm_bgra(const std::string& path, int width, int height,
                    const uint8_t* bgra, int stride_bytes);
bool write_ppm_rgb565(const std::string& path, int width, int height,
                      const uint16_t* rgb565, int stride_pixels);

// conversions
void rgba_to_rgb(const uint8_t* rgba, int width, int height, int stride_bytes,
                 std::vector<uint8_t>& out_rgb);
void bgra_to_rgb(const uint8_t* bgra, int width, int height, int stride_bytes,
                 std::vector<uint8_t>& out_rgb);
void rgb565_to_rgb(const uint16_t* rgb565, int width, int height, int stride_pixels,
                   std::vector<uint8_t>& out_rgb);

// comparisons
CompareResult compare_rgb(const uint8_t* a, int stride_a,
                          const uint8_t* b, int stride_b,
                          int width, int height, int tolerance);
CompareResult compare_rgba(const uint8_t* a, int stride_a,
                           const uint8_t* b, int stride_b,
                           int width, int height, int tolerance);

// packing helpers
uint32_t pack_rgba8888(uint8_t r, uint8_t g, uint8_t b, uint8_t a);
uint32_t pack_bgra8888(uint8_t r, uint8_t g, uint8_t b, uint8_t a);
uint16_t pack_rgb565(uint8_t r, uint8_t g, uint8_t b);

} // namespace video
} // namespace vx
