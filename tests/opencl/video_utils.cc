// Copyright © 2026
#include "video_utils.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>

namespace vx {
namespace video {

size_t align_up(size_t v, size_t align) {
    if (align == 0) return v;
    return (v + align - 1) / align * align;
}

int clamp_int(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static bool parse_int(const char* s, int& out) {
    if (!s) return false;
    char* end = nullptr;
    long v = std::strtol(s, &end, 0);
    if (!end || *end != '\0') return false;
    out = static_cast<int>(v);
    return true;
}

static bool parse_rect(const std::string& s, Args& args) {
    int x=0, y=0, w=0, h=0;
    if (std::sscanf(s.c_str(), "%d,%d,%d,%d", &x, &y, &w, &h) != 4) {
        return false;
    }
    args.rect_x = x;
    args.rect_y = y;
    args.rect_w = w;
    args.rect_h = h;
    return true;
}

bool parse_common_args(int argc, char** argv, Args& args, std::vector<std::string>& extras, std::string& err) {
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (!a) continue;
        auto take_val = [&](int& dst) -> bool {
            if (i + 1 >= argc) return false;
            return parse_int(argv[++i], dst);
        };
        if (0 == std::strcmp(a, "-w") || 0 == std::strcmp(a, "--width")) {
            if (!take_val(args.width)) { err = "invalid -w"; return false; }
        } else if (0 == std::strcmp(a, "-h") || 0 == std::strcmp(a, "--height")) {
            if (!take_val(args.height)) { err = "invalid -h"; return false; }
        } else if (0 == std::strcmp(a, "-stride") || 0 == std::strcmp(a, "--stride")) {
            if (!take_val(args.stride)) { err = "invalid -stride"; return false; }
        } else if (0 == std::strcmp(a, "-mode") || 0 == std::strcmp(a, "--mode")) {
            if (!take_val(args.mode)) { err = "invalid -mode"; return false; }
        } else if (0 == std::strcmp(a, "-seed") || 0 == std::strcmp(a, "--seed")) {
            if (!take_val(args.seed)) { err = "invalid -seed"; return false; }
        } else if (0 == std::strcmp(a, "-ow") || 0 == std::strcmp(a, "--outw")) {
            if (!take_val(args.out_width)) { err = "invalid -ow"; return false; }
        } else if (0 == std::strcmp(a, "-oh") || 0 == std::strcmp(a, "--outh")) {
            if (!take_val(args.out_height)) { err = "invalid -oh"; return false; }
        } else if (0 == std::strcmp(a, "-fmt") || 0 == std::strcmp(a, "--format")) {
            if (!take_val(args.fmt)) { err = "invalid -fmt"; return false; }
        } else if (0 == std::strcmp(a, "-rect") || 0 == std::strcmp(a, "--rect")) {
            if (i + 1 >= argc) { err = "invalid -rect"; return false; }
            std::string s = argv[++i];
            if (!parse_rect(s, args)) { err = "invalid -rect"; return false; }
        } else if (0 == std::strcmp(a, "--dump")) {
            args.dump = true;
        } else if (0 == std::strcmp(a, "--outdir")) {
            if (i + 1 >= argc) { err = "invalid --outdir"; return false; }
            args.outdir = argv[++i];
        } else if (0 == std::strcmp(a, "--prefix")) {
            if (i + 1 >= argc) { err = "invalid --prefix"; return false; }
            args.prefix = argv[++i];
        } else if (0 == std::strcmp(a, "-help") || 0 == std::strcmp(a, "--help")) {
            extras.emplace_back(a);
        } else {
            extras.emplace_back(a);
        }
    }
    if (args.out_width == 0) args.out_width = args.width;
    if (args.out_height == 0) args.out_height = args.height;
    return true;
}

void print_common_usage(const char* app) {
    std::printf("Usage: %s [options]\n", app);
    std::printf("  -w, --width <n>       input width (default 64)\n");
    std::printf("  -h, --height <n>      input height (default 64)\n");
    std::printf("  -stride <bytes>       input stride in bytes (0=tight)\n");
    std::printf("  -mode <n>             mode selector (test-specific)\n");
    std::printf("  -seed <n>             pattern seed (default 1)\n");
    std::printf("  -ow, --outw <n>        output width (default = input)\n");
    std::printf("  -oh, --outh <n>        output height (default = input)\n");
    std::printf("  -fmt <n>              pixel format selector (test-specific)\n");
    std::printf("  -rect x,y,w,h          rectangle (test-specific)\n");
    std::printf("  --dump                write PPM outputs (on PASS)\n");
    std::printf("  --outdir <path>        output directory for PPM (default: artifacts)\n");
    std::printf("  --prefix <string>      filename prefix for outputs\n");
}

bool ensure_dir(const std::string& path) {
    if (path.empty() || path == ".") return true;
    std::string cur;
    size_t start = 0;
    if (path[0] == '/') {
        cur = "/";
        start = 1;
    }
    while (start <= path.size()) {
        size_t end = path.find('/', start);
        if (end == std::string::npos) end = path.size();
        std::string part = path.substr(start, end - start);
        if (!part.empty()) {
            if (!cur.empty() && cur.back() != '/') {
                cur.push_back('/');
            }
            cur += part;
            struct stat st;
            if (stat(cur.c_str(), &st) == 0) {
                if (!S_ISDIR(st.st_mode)) return false;
            } else {
                if (mkdir(cur.c_str(), 0755) != 0) return false;
            }
        }
        if (end == path.size()) break;
        start = end + 1;
    }
    return true;
}

std::string make_output_path(const Args& args, const std::string& filename) {
    std::string base = filename;
    if (!args.prefix.empty()) {
        base = args.prefix + "_" + base;
    }
    if (args.outdir.empty() || args.outdir == ".") {
        return base;
    }
    return args.outdir + "/" + base;
}

static bool write_ppm_header(std::ofstream& ofs, int width, int height) {
    if (!ofs.is_open()) return false;
    ofs << "P6\n" << width << " " << height << "\n255\n";
    return static_cast<bool>(ofs);
}

bool write_ppm_rgb(const std::string& path, int width, int height,
                   const uint8_t* rgb, int stride_bytes) {
    if (!rgb || width <= 0 || height <= 0) return false;
    if (stride_bytes == 0) stride_bytes = width * 3;
    std::ofstream ofs(path, std::ios::binary);
    if (!write_ppm_header(ofs, width, height)) return false;
    for (int y = 0; y < height; ++y) {
        ofs.write(reinterpret_cast<const char*>(rgb + y * stride_bytes), width * 3);
        if (!ofs.good()) return false;
    }
    return true;
}

void rgba_to_rgb(const uint8_t* rgba, int width, int height, int stride_bytes,
                 std::vector<uint8_t>& out_rgb) {
    out_rgb.resize(static_cast<size_t>(width) * height * 3);
    if (stride_bytes == 0) stride_bytes = width * 4;
    size_t idx = 0;
    for (int y = 0; y < height; ++y) {
        const uint8_t* row = rgba + y * stride_bytes;
        for (int x = 0; x < width; ++x) {
            out_rgb[idx++] = row[x * 4 + 0];
            out_rgb[idx++] = row[x * 4 + 1];
            out_rgb[idx++] = row[x * 4 + 2];
        }
    }
}

void bgra_to_rgb(const uint8_t* bgra, int width, int height, int stride_bytes,
                 std::vector<uint8_t>& out_rgb) {
    out_rgb.resize(static_cast<size_t>(width) * height * 3);
    if (stride_bytes == 0) stride_bytes = width * 4;
    size_t idx = 0;
    for (int y = 0; y < height; ++y) {
        const uint8_t* row = bgra + y * stride_bytes;
        for (int x = 0; x < width; ++x) {
            out_rgb[idx++] = row[x * 4 + 2];
            out_rgb[idx++] = row[x * 4 + 1];
            out_rgb[idx++] = row[x * 4 + 0];
        }
    }
}

void rgb565_to_rgb(const uint16_t* rgb565, int width, int height, int stride_pixels,
                   std::vector<uint8_t>& out_rgb) {
    out_rgb.resize(static_cast<size_t>(width) * height * 3);
    if (stride_pixels == 0) stride_pixels = width;
    size_t idx = 0;
    for (int y = 0; y < height; ++y) {
        const uint16_t* row = rgb565 + y * stride_pixels;
        for (int x = 0; x < width; ++x) {
            uint16_t v = row[x];
            uint8_t r = static_cast<uint8_t>(((v >> 11) & 0x1F) * 255 / 31);
            uint8_t g = static_cast<uint8_t>(((v >> 5) & 0x3F) * 255 / 63);
            uint8_t b = static_cast<uint8_t>((v & 0x1F) * 255 / 31);
            out_rgb[idx++] = r;
            out_rgb[idx++] = g;
            out_rgb[idx++] = b;
        }
    }
}

bool write_ppm_rgba(const std::string& path, int width, int height,
                    const uint8_t* rgba, int stride_bytes) {
    std::vector<uint8_t> rgb;
    rgba_to_rgb(rgba, width, height, stride_bytes, rgb);
    return write_ppm_rgb(path, width, height, rgb.data(), width * 3);
}

bool write_ppm_bgra(const std::string& path, int width, int height,
                    const uint8_t* bgra, int stride_bytes) {
    std::vector<uint8_t> rgb;
    bgra_to_rgb(bgra, width, height, stride_bytes, rgb);
    return write_ppm_rgb(path, width, height, rgb.data(), width * 3);
}

bool write_ppm_rgb565(const std::string& path, int width, int height,
                      const uint16_t* rgb565, int stride_pixels) {
    std::vector<uint8_t> rgb;
    rgb565_to_rgb(rgb565, width, height, stride_pixels, rgb);
    return write_ppm_rgb(path, width, height, rgb.data(), width * 3);
}

CompareResult compare_rgb(const uint8_t* a, int stride_a,
                          const uint8_t* b, int stride_b,
                          int width, int height, int tolerance) {
    CompareResult res;
    if (stride_a == 0) stride_a = width * 3;
    if (stride_b == 0) stride_b = width * 3;
    for (int y = 0; y < height; ++y) {
        const uint8_t* ra = a + y * stride_a;
        const uint8_t* rb = b + y * stride_b;
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < 3; ++c) {
                int diff = std::abs(int(ra[x * 3 + c]) - int(rb[x * 3 + c]));
                if (diff > res.max_err) res.max_err = diff;
                if (diff > tolerance && res.ok) {
                    res.ok = false;
                    res.first_x = x;
                    res.first_y = y;
                }
            }
        }
    }
    return res;
}

CompareResult compare_rgba(const uint8_t* a, int stride_a,
                           const uint8_t* b, int stride_b,
                           int width, int height, int tolerance) {
    CompareResult res;
    if (stride_a == 0) stride_a = width * 4;
    if (stride_b == 0) stride_b = width * 4;
    for (int y = 0; y < height; ++y) {
        const uint8_t* ra = a + y * stride_a;
        const uint8_t* rb = b + y * stride_b;
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < 4; ++c) {
                int diff = std::abs(int(ra[x * 4 + c]) - int(rb[x * 4 + c]));
                if (diff > res.max_err) res.max_err = diff;
                if (diff > tolerance && res.ok) {
                    res.ok = false;
                    res.first_x = x;
                    res.first_y = y;
                }
            }
        }
    }
    return res;
}

uint32_t pack_rgba8888(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    return (uint32_t(r) << 0) | (uint32_t(g) << 8) | (uint32_t(b) << 16) | (uint32_t(a) << 24);
}

uint32_t pack_bgra8888(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    return (uint32_t(b) << 0) | (uint32_t(g) << 8) | (uint32_t(r) << 16) | (uint32_t(a) << 24);
}

uint16_t pack_rgb565(uint8_t r, uint8_t g, uint8_t b) {
    uint16_t R = (r * 31 + 127) / 255;
    uint16_t G = (g * 63 + 127) / 255;
    uint16_t B = (b * 31 + 127) / 255;
    return static_cast<uint16_t>((R << 11) | (G << 5) | B);
}

bool utils_self_test(std::string& err) {
    // align/clamp quick checks
    if (align_up(5, 4) != 8) { err = "align_up failed"; return false; }
    if (clamp_int(5, 0, 3) != 3) { err = "clamp_int failed"; return false; }
    // ppm conversion checks (2x1)
    uint8_t rgba[8] = {255, 0, 0, 255, 0, 255, 0, 255};
    std::vector<uint8_t> rgb;
    rgba_to_rgb(rgba, 2, 1, 0, rgb);
    if (rgb.size() != 6 || rgb[0] != 255 || rgb[1] != 0 || rgb[2] != 0 || rgb[3] != 0 || rgb[4] != 255) {
        err = "rgba_to_rgb failed"; return false;
    }
    return true;
}

} // namespace video
} // namespace vx
