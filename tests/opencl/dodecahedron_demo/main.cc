#include <CL/opencl.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <string>
#include <vector>
#include <set>

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

struct Vec3 { float x, y, z; };
struct Vec2 { float x, y; };


static cl_device_id device_id = NULL;
static cl_context context = NULL;
static cl_command_queue commandQueue = NULL;
static cl_program program = NULL;
static cl_kernel kernel = NULL;
static cl_mem verts_buf = NULL;
static cl_mem faces_buf = NULL;
static cl_mem face_order_buf = NULL;
static cl_mem face_colors_buf = NULL;
static cl_mem face_origin_buf = NULL;
static cl_mem face_u_buf = NULL;
static cl_mem face_v_buf = NULL;
static cl_mem face_n_buf = NULL;
static cl_mem hole_radii_buf = NULL;
static cl_mem out_buf = NULL;
static uint8_t* kernel_bin = NULL;

static void cleanup() {
    if (commandQueue) clReleaseCommandQueue(commandQueue);
    if (kernel) clReleaseKernel(kernel);
    if (program) clReleaseProgram(program);
    if (verts_buf) clReleaseMemObject(verts_buf);
    if (faces_buf) clReleaseMemObject(faces_buf);
    if (face_order_buf) clReleaseMemObject(face_order_buf);
    if (face_colors_buf) clReleaseMemObject(face_colors_buf);
    if (face_origin_buf) clReleaseMemObject(face_origin_buf);
    if (face_u_buf) clReleaseMemObject(face_u_buf);
    if (face_v_buf) clReleaseMemObject(face_v_buf);
    if (face_n_buf) clReleaseMemObject(face_n_buf);
    if (hole_radii_buf) clReleaseMemObject(hole_radii_buf);
    if (out_buf) clReleaseMemObject(out_buf);
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

static Vec3 rotate_y(const Vec3& v, float ang) {
    float c = cosf(ang);
    float s = sinf(ang);
    return {v.x * c + v.z * s, v.y, -v.x * s + v.z * c};
}

static Vec3 rotate_z(const Vec3& v, float ang) {
    float c = cosf(ang);
    float s = sinf(ang);
    return {v.x * c - v.y * s, v.x * s + v.y * c, v.z};
}

static Vec3 rotate_x(const Vec3& v, float ang) {
    float c = cosf(ang);
    float s = sinf(ang);
    return {v.x, v.y * c - v.z * s, v.y * s + v.z * c};
}

static Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

static float dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static Vec3 normalize(const Vec3& v) {
    float len = sqrtf(dot(v, v));
    if (len < 1e-6f) return {0, 0, 1};
    return {v.x / len, v.y / len, v.z / len};
}

static Vec3 reflect_vec(const Vec3& v, const Vec3& n) {
    float d = dot(v, n);
    return {v.x - 2.0f * d * n.x, v.y - 2.0f * d * n.y, v.z - 2.0f * d * n.z};
}

static void blend_pixel(std::vector<uint8_t>& img, int width, int height, int x, int y,
                        float r, float g, float b, float a) {
    if (x < 0 || y < 0 || x >= width || y >= height) return;
    if (a <= 0.0f) return;
    if (a > 1.0f) a = 1.0f;
    size_t idx = (static_cast<size_t>(y) * width + x) * 4;
    float dr = img[idx + 0] / 255.0f;
    float dg = img[idx + 1] / 255.0f;
    float db = img[idx + 2] / 255.0f;
    float out_r = r * a + dr * (1.0f - a);
    float out_g = g * a + dg * (1.0f - a);
    float out_b = b * a + db * (1.0f - a);
    img[idx + 0] = (uint8_t)vx::video::clamp_int((int)(out_r * 255.0f + 0.5f), 0, 255);
    img[idx + 1] = (uint8_t)vx::video::clamp_int((int)(out_g * 255.0f + 0.5f), 0, 255);
    img[idx + 2] = (uint8_t)vx::video::clamp_int((int)(out_b * 255.0f + 0.5f), 0, 255);
    img[idx + 3] = 255;
}

static void overlay_knobs(std::vector<uint8_t>& img, int width, int height,
                          const std::vector<cl_float4>& verts,
                          const std::vector<uint8_t>& visible,
                          float radius,
                          const Vec3& light1,
                          const Vec3& light2,
                          const Vec3& view_dir) {
    if (radius <= 0.0f) return;
    float aa = 1.0f;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            Vec2 p{(float)x + 0.5f, (float)y + 0.5f};
            float max_alpha = 0.0f;
            float best_dx = 0.0f;
            float best_dy = 0.0f;
            for (size_t i = 0; i < verts.size(); ++i) {
                if (i < visible.size() && !visible[i]) continue;
                float dx = p.x - verts[i].s[0];
                float dy = p.y - verts[i].s[1];
                float dist = sqrtf(dx * dx + dy * dy);
                float a = (radius + aa - dist) / aa;
                if (a > max_alpha) {
                    max_alpha = a;
                    best_dx = dx;
                    best_dy = dy;
                }
            }
            if (max_alpha > 0.0f) {
                if (max_alpha > 1.0f) max_alpha = 1.0f;
                float r2 = radius * radius;
                float dz2 = fmaxf(r2 - (best_dx * best_dx + best_dy * best_dy), 0.0f);
                float dz = sqrtf(dz2);
                Vec3 n = normalize({best_dx / fmaxf(radius, 1e-6f),
                                    best_dy / fmaxf(radius, 1e-6f),
                                    dz / fmaxf(radius, 1e-6f)});
                float diff = fmaxf(0.0f, dot(n, light1));
                float diff2 = 0.35f * fmaxf(0.0f, dot(n, light2));
                Vec3 refl = reflect_vec({-light1.x, -light1.y, -light1.z}, n);
                float spec = powf(fmaxf(0.0f, dot(refl, view_dir)), 50.0f);
                float ambient = 0.25f;
                float kd = 0.75f;
                float ks = 0.35f;
                float shade = ambient + kd * (diff + diff2);
                Vec3 knob = {0.12f, 0.22f, 0.50f};
                Vec3 rgb = {knob.x * shade + ks * spec,
                            knob.y * shade + ks * spec,
                            knob.z * shade + ks * spec};
                blend_pixel(img, width, height, x, y, rgb.x, rgb.y, rgb.z, max_alpha);
            }
        }
    }
}

static Vec3 hsv_to_rgb(float h, float s, float v) {
    float r = v, g = v, b = v;
    if (s <= 0.0f) return {r, g, b};
    h = fmodf(h, 1.0f) * 6.0f;
    int i = (int)floorf(h);
    float f = h - i;
    float p = v * (1.0f - s);
    float q = v * (1.0f - s * f);
    float t = v * (1.0f - s * (1.0f - f));
    switch (i) {
        case 0: r = v; g = t; b = p; break;
        case 1: r = q; g = v; b = p; break;
        case 2: r = p; g = v; b = t; break;
        case 3: r = p; g = q; b = v; break;
        case 4: r = t; g = p; b = v; break;
        default: r = v; g = p; b = q; break;
    }
    return {r, g, b};
}

static bool parse_int(const std::string& s, int& out) {
    char* end = nullptr;
    long v = std::strtol(s.c_str(), &end, 0);
    if (!end || *end != '\0') return false;
    out = (int)v;
    return true;
}

static bool parse_float(const std::string& s, float& out) {
    char* end = nullptr;
    float v = std::strtof(s.c_str(), &end);
    if (!end || *end != '\0') return false;
    out = v;
    return true;
}

static float clampf(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

int main(int argc, char** argv) {
    vx::video::Args args;
    args.width = 640;
    args.height = 640;

    float alpha = 0.55f;
    int wire = 1;
    float wire_thickness = 1.5f;
    int holes = 1;
    float hole_scale = 0.35f;
    int knobs = 1;
    float knob_radius = 3.5f;
    float knob_diam_frac = -1.0f;
    int rings = 0;
    int shadow = 0;
    float shadow_strength = 0.22f;

    float yaw_deg = 45.0f;
    float pitch_deg = 35.264f;
    float roll_deg = 0.0f;
    float zoom = 0.90f;
    std::string palette = "face_hues";
    std::string style = "iso_old";

    bool set_alpha = false;
    bool set_wire = false;
    bool set_wire_thickness = false;
    bool set_holes = false;
    bool set_hole_scale = false;
    bool set_knobs = false;
    bool set_knob_radius = false;
    bool set_knob_diam_frac = false;
    bool set_rings = false;
    bool set_shadow = false;
    bool set_yaw = false;
    bool set_pitch = false;
    bool set_roll = false;
    bool set_zoom = false;
    bool set_palette = false;

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

    for (size_t i = 0; i < extras.size(); ++i) {
        const std::string& e = extras[i];
        auto next = [&](std::string& val) -> bool {
            if (i + 1 >= extras.size()) return false;
            val = extras[++i];
            return true;
        };
        if (e == "-help" || e == "--help") {
            vx::video::print_common_usage(argv[0]);
            printf("  --alpha <f>          face alpha (0..1, default 0.55)\n");
            printf("  --wire <0|1>         enable wireframe (default 1)\n");
            printf("  --wire-thickness <f> wire thickness in pixels (default 1.5)\n");
            printf("  --holes <0|1>        enable face holes (default 1)\n");
            printf("  --hole-scale <f>     hole radius scale (default 0.35)\n");
            printf("  --knobs <0|1>        enable vertex knobs (default 1)\n");
            printf("  --knob-radius <f>    knob radius in pixels (default 3.5)\n");
            printf("  --knob-diam-frac <f> knob diameter as fraction of edge length\n");
            printf("  --rings <0|1>        enable concentric rings (default 0)\n");
            printf("  --shadow <0|1>       enable soft shadow (default 0)\n");
            printf("  --yaw <deg>          camera yaw (deg)\n");
            printf("  --pitch <deg>        camera pitch (deg)\n");
            printf("  --roll <deg>         camera roll (deg)\n");
            printf("  --zoom <f>           zoom scale (default 0.90)\n");
            printf("  --style <name>       iso_old | reference_blue\n");
            printf("  --palette <name>     mono_blue | face_hues\n");
            return 0;
        } else if (e == "--alpha") {
            std::string v; if (!next(v) || !parse_float(v, alpha)) { fprintf(stderr, "invalid --alpha\n"); return -1; }
            set_alpha = true;
        } else if (e == "--wire") {
            std::string v; if (!next(v) || !parse_int(v, wire)) { fprintf(stderr, "invalid --wire\n"); return -1; }
            set_wire = true;
        } else if (e == "--wire-thickness") {
            std::string v; if (!next(v) || !parse_float(v, wire_thickness)) { fprintf(stderr, "invalid --wire-thickness\n"); return -1; }
            set_wire_thickness = true;
        } else if (e == "--holes") {
            std::string v; if (!next(v) || !parse_int(v, holes)) { fprintf(stderr, "invalid --holes\n"); return -1; }
            set_holes = true;
        } else if (e == "--hole-scale") {
            std::string v; if (!next(v) || !parse_float(v, hole_scale)) { fprintf(stderr, "invalid --hole-scale\n"); return -1; }
            set_hole_scale = true;
        } else if (e == "--knobs") {
            std::string v; if (!next(v) || !parse_int(v, knobs)) { fprintf(stderr, "invalid --knobs\n"); return -1; }
            set_knobs = true;
        } else if (e == "--knob-radius") {
            std::string v; if (!next(v) || !parse_float(v, knob_radius)) { fprintf(stderr, "invalid --knob-radius\n"); return -1; }
            set_knob_radius = true;
        } else if (e == "--knob-diam-frac") {
            std::string v; if (!next(v) || !parse_float(v, knob_diam_frac)) { fprintf(stderr, "invalid --knob-diam-frac\n"); return -1; }
            set_knob_diam_frac = true;
        } else if (e == "--rings") {
            std::string v; if (!next(v) || !parse_int(v, rings)) { fprintf(stderr, "invalid --rings\n"); return -1; }
            set_rings = true;
        } else if (e == "--shadow") {
            std::string v; if (!next(v) || !parse_int(v, shadow)) { fprintf(stderr, "invalid --shadow\n"); return -1; }
            set_shadow = true;
        } else if (e == "--yaw") {
            std::string v; if (!next(v) || !parse_float(v, yaw_deg)) { fprintf(stderr, "invalid --yaw\n"); return -1; }
            set_yaw = true;
        } else if (e == "--pitch") {
            std::string v; if (!next(v) || !parse_float(v, pitch_deg)) { fprintf(stderr, "invalid --pitch\n"); return -1; }
            set_pitch = true;
        } else if (e == "--roll") {
            std::string v; if (!next(v) || !parse_float(v, roll_deg)) { fprintf(stderr, "invalid --roll\n"); return -1; }
            set_roll = true;
        } else if (e == "--zoom") {
            std::string v; if (!next(v) || !parse_float(v, zoom)) { fprintf(stderr, "invalid --zoom\n"); return -1; }
            set_zoom = true;
        } else if (e == "--style") {
            std::string v; if (!next(v)) { fprintf(stderr, "invalid --style\n"); return -1; }
            style = v;
        } else if (e == "--palette") {
            std::string v; if (!next(v)) { fprintf(stderr, "invalid --palette\n"); return -1; }
            palette = v;
            set_palette = true;
        }
    }

    if (style == "reference_blue") {
        if (!set_alpha) alpha = 1.0f;
        if (!set_palette) palette = "mono_blue";
        if (!set_wire) wire = 1;
        if (!set_wire_thickness) wire_thickness = 1.0f;
        if (!set_holes) holes = 1;
        if (!set_knobs) knobs = 1;
        if (!set_rings) rings = 1;
        if (!set_shadow) shadow = 1;
        if (!set_yaw) yaw_deg = 28.0f;
        if (!set_pitch) pitch_deg = 20.0f;
        if (!set_roll) roll_deg = -10.0f;
        if (!set_zoom) zoom = 0.98f;
        if (!set_hole_scale) hole_scale = 0.33f;
        if (!set_knob_radius && !set_knob_diam_frac) knob_diam_frac = 0.20f;
        if (!set_knob_radius && knob_diam_frac <= 0.0f) knob_radius = 4.2f;
    }

    int width = args.width > 0 ? args.width : 640;
    int height = args.height > 0 ? args.height : 640;

    alpha = clampf(alpha, 0.0f, 1.0f);
    wire_thickness = fmaxf(0.5f, wire_thickness);
    hole_scale = clampf(hole_scale, 0.0f, 0.8f);
    if (set_knob_diam_frac || knob_diam_frac > 0.0f) {
        knob_diam_frac = fmaxf(0.0f, knob_diam_frac);
    }
    knob_radius = fmaxf(0.0f, knob_radius);
    zoom = clampf(zoom, 0.2f, 2.0f);

    printf("Dodecahedron demo: %dx%d style=%s alpha=%.2f wire=%d holes=%d knobs=%d\n",
           width, height, style.c_str(), alpha, wire, holes, knobs);

    const float phi = 1.618f;
    const float invphi = 0.618f;

    Vec3 verts[20] = {
        {0, invphi, phi}, {0, -invphi, phi}, {0, -invphi, -phi}, {0, invphi, -phi},
        {phi, 0, invphi}, {-phi, 0, invphi}, {-phi, 0, -invphi}, {phi, 0, -invphi},
        {invphi, phi, 0}, {-invphi, phi, 0}, {-invphi, -phi, 0}, {invphi, -phi, 0},
        {1, 1, 1}, {-1, 1, 1}, {-1, -1, 1}, {1, -1, 1},
        {1, -1, -1}, {1, 1, -1}, {-1, 1, -1}, {-1, -1, -1}
    };

    int faces[12][5] = {
        {0, 1, 15, 4, 12},
        {0, 12, 8, 9, 13},
        {0, 13, 5, 14, 1},
        {1, 14, 10, 11, 15},
        {2, 3, 17, 7, 16},
        {2, 16, 11, 10, 19},
        {2, 19, 6, 18, 3},
        {18, 9, 8, 17, 3},
        {15, 11, 16, 7, 4},
        {4, 7, 17, 8, 12},
        {13, 9, 18, 6, 5},
        {5, 6, 19, 10, 14}
    };

    const float yaw = yaw_deg * 3.14159265f / 180.0f;
    const float pitch = pitch_deg * 3.14159265f / 180.0f;
    const float roll = roll_deg * 3.14159265f / 180.0f;

    std::vector<Vec3> cam(20);
    for (int i = 0; i < 20; ++i) {
        Vec3 v = rotate_y(verts[i], yaw);
        v = rotate_x(v, pitch);
        v = rotate_z(v, roll);
        cam[i] = v;
    }

    // compute bounds in camera xy
    float minx = cam[0].x, maxx = cam[0].x;
    float miny = cam[0].y, maxy = cam[0].y;
    for (int i = 1; i < 20; ++i) {
        minx = fminf(minx, cam[i].x); maxx = fmaxf(maxx, cam[i].x);
        miny = fminf(miny, cam[i].y); maxy = fmaxf(maxy, cam[i].y);
    }
    float spanx = maxx - minx;
    float spany = maxy - miny;
    float scale = zoom * (float)std::min(width, height) / fmaxf(spanx, spany);
    float cx = 0.5f * (minx + maxx);
    float cy = 0.5f * (miny + maxy);

    std::vector<cl_float4> vbuf(20);
    for (int i = 0; i < 20; ++i) {
        float sx = (cam[i].x - cx) * scale + width * 0.5f;
        float sy = (-(cam[i].y - cy)) * scale + height * 0.5f;
        vbuf[i] = {sx, sy, cam[i].z, 0.0f};
    }

    float edge_len_px = 0.0f;
    if (knobs && !set_knob_radius && knob_diam_frac > 0.0f) {
        std::set<int> edge_keys;
        std::vector<float> edge_lens;
        edge_lens.reserve(30);
        for (int f = 0; f < 12; ++f) {
            for (int k = 0; k < 5; ++k) {
                int a = faces[f][k];
                int b = faces[f][(k + 1) % 5];
                int lo = (a < b) ? a : b;
                int hi = (a < b) ? b : a;
                int key = (lo << 8) | hi;
                if (edge_keys.insert(key).second) {
                    float dx = vbuf[lo].s[0] - vbuf[hi].s[0];
                    float dy = vbuf[lo].s[1] - vbuf[hi].s[1];
                    edge_lens.push_back(sqrtf(dx * dx + dy * dy));
                }
            }
        }
        if (!edge_lens.empty()) {
            std::sort(edge_lens.begin(), edge_lens.end());
            edge_len_px = edge_lens[edge_lens.size() / 2];
            float knob_diam_px = knob_diam_frac * edge_len_px;
            knob_radius = 0.5f * knob_diam_px;
        }
    }

    if (knobs) {
        if (set_knob_radius) {
            printf("Knob sizing: radius_px=%.2f (override)\n", knob_radius);
        } else if (knob_diam_frac > 0.0f && edge_len_px > 0.0f) {
            printf("Knob sizing: edge_len_px=%.2f knob_diam_frac=%.3f radius_px=%.2f\n",
                   edge_len_px, knob_diam_frac, knob_radius);
        } else {
            printf("Knob sizing: radius_px=%.2f\n", knob_radius);
        }
    }

    // face colors, basis, hole radii (in face plane)
    std::vector<cl_float4> face_colors(12);
    std::vector<cl_float4> face_origin(12);
    std::vector<cl_float4> face_u(12);
    std::vector<cl_float4> face_v(12);
    std::vector<cl_float4> face_n(12);
    std::vector<float> hole_radii(12);
    std::vector<float> face_depths(12);

    Vec3 light1 = normalize({0.6f, 0.7f, 1.0f});
    Vec3 light2 = normalize({-0.2f, 0.2f, 0.8f});
    Vec3 view_dir = normalize({0.0f, 0.0f, 1.0f});
    Vec3 base_blue = {0.16f, 0.32f, 0.72f};

    for (int f = 0; f < 12; ++f) {
        int i0 = faces[f][0];
        int i1 = faces[f][1];
        int i2 = faces[f][2];

        Vec3 a = cam[i0];
        Vec3 b = cam[i1];
        Vec3 c = cam[i2];
        Vec3 n = normalize(cross({b.x - a.x, b.y - a.y, b.z - a.z}, {c.x - a.x, c.y - a.y, c.z - a.z}));
        face_n[f] = {n.x, n.y, n.z, 0.0f};

        float diff = fmaxf(0.0f, dot(n, light1)) + 0.35f * fmaxf(0.0f, dot(n, light2));
        Vec3 refl = reflect_vec({-light1.x, -light1.y, -light1.z}, n);
        float spec = powf(fmaxf(0.0f, dot(refl, view_dir)), 60.0f);
        float ambient = 0.25f;
        float kd = 0.75f;
        float ks = 0.28f;
        float shade = ambient + kd * diff;

        Vec3 base = base_blue;
        if (palette == "face_hues") {
            float hue = (float)f / 12.0f;
            base = hsv_to_rgb(hue, 0.55f, 0.95f);
        } else {
            float var = 0.96f + 0.06f * sinf((float)f * 1.17f + 0.4f);
            base = {base_blue.x * var, base_blue.y * var, base_blue.z * var};
        }
        Vec3 rgb = {base.x * shade + ks * spec, base.y * shade + ks * spec, base.z * shade + ks * spec};
        face_colors[f] = {rgb.x, rgb.y, rgb.z, alpha};

        Vec3 o = {0.0f, 0.0f, 0.0f};
        float cz2 = 0.0f;
        for (int k = 0; k < 5; ++k) {
            Vec3 vv = cam[faces[f][k]];
            o.x += vv.x; o.y += vv.y; o.z += vv.z;
            cz2 += vv.z;
        }
        o.x /= 5.0f; o.y /= 5.0f; o.z /= 5.0f;
        cz2 /= 5.0f;
        face_origin[f] = {o.x, o.y, o.z, 0.0f};
        face_depths[f] = cz2;

        Vec3 e0 = {b.x - a.x, b.y - a.y, b.z - a.z};
        Vec3 uvec = normalize(e0);
        Vec3 vvec = normalize(cross(n, uvec));
        face_u[f] = {uvec.x, uvec.y, uvec.z, 0.0f};
        face_v[f] = {vvec.x, vvec.y, vvec.z, 0.0f};

        float avg = 0.0f;
        for (int k = 0; k < 5; ++k) {
            Vec3 vv = cam[faces[f][k]];
            Vec3 d = {vv.x - o.x, vv.y - o.y, vv.z - o.z};
            float ru = dot(d, uvec);
            float rv = dot(d, vvec);
            avg += sqrtf(ru * ru + rv * rv);
        }
        avg /= 5.0f;
        float jitter = 0.92f + 0.08f * sinf((float)f * 2.17f + 0.7f);
        hole_radii[f] = hole_scale * avg * jitter;
    }

    // vertex visibility (approximate depth test for knobs)
    std::vector<Vec3> vert_normals(20, {0.0f, 0.0f, 0.0f});
    for (int f = 0; f < 12; ++f) {
        Vec3 n = {face_n[f].s[0], face_n[f].s[1], face_n[f].s[2]};
        for (int k = 0; k < 5; ++k) {
            int idx = faces[f][k];
            vert_normals[idx].x += n.x;
            vert_normals[idx].y += n.y;
            vert_normals[idx].z += n.z;
        }
    }
    std::vector<uint8_t> vert_visible(20, 0);
    for (int i = 0; i < 20; ++i) {
        Vec3 n = normalize(vert_normals[i]);
        vert_visible[i] = (dot(n, view_dir) > 0.0f) ? 1 : 0;
    }

    // flatten faces
    std::vector<int> face_idx(12 * 5);
    for (int f = 0; f < 12; ++f) {
        for (int k = 0; k < 5; ++k) {
            face_idx[f * 5 + k] = faces[f][k];
        }
    }

    std::vector<int> face_order(12);
    for (int i = 0; i < 12; ++i) face_order[i] = i;
    std::stable_sort(face_order.begin(), face_order.end(), [&](int a, int b) {
        return face_depths[a] < face_depths[b]; // far to near
    });

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

    kernel = CL_CHECK2(clCreateKernel(program, "render", &_err));

    size_t out_bytes = static_cast<size_t>(width) * height * 4;
    out_buf = CL_CHECK2(clCreateBuffer(context, CL_MEM_WRITE_ONLY, out_bytes, NULL, &_err));

    verts_buf = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         sizeof(cl_float4) * vbuf.size(), vbuf.data(), &_err));
    faces_buf = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         sizeof(int) * face_idx.size(), face_idx.data(), &_err));
    face_order_buf = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                             sizeof(int) * face_order.size(), face_order.data(), &_err));
    face_colors_buf = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                               sizeof(cl_float4) * face_colors.size(), face_colors.data(), &_err));
    face_origin_buf = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                               sizeof(cl_float4) * face_origin.size(), face_origin.data(), &_err));
    face_u_buf = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          sizeof(cl_float4) * face_u.size(), face_u.data(), &_err));
    face_v_buf = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          sizeof(cl_float4) * face_v.size(), face_v.data(), &_err));
    face_n_buf = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          sizeof(cl_float4) * face_n.size(), face_n.data(), &_err));
    hole_radii_buf = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                              sizeof(float) * hole_radii.size(), hole_radii.data(), &_err));
    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &verts_buf));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &faces_buf));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &face_order_buf));
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), &face_colors_buf));
    CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), &face_origin_buf));
    CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_mem), &face_u_buf));
    CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_mem), &face_v_buf));
    CL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_mem), &face_n_buf));
    CL_CHECK(clSetKernelArg(kernel, 8, sizeof(cl_mem), &hole_radii_buf));
    CL_CHECK(clSetKernelArg(kernel, 9, sizeof(int), &width));
    CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int), &height));
    CL_CHECK(clSetKernelArg(kernel, 11, sizeof(int), &holes));
    CL_CHECK(clSetKernelArg(kernel, 12, sizeof(int), &rings));
    CL_CHECK(clSetKernelArg(kernel, 13, sizeof(int), &wire));
    CL_CHECK(clSetKernelArg(kernel, 14, sizeof(float), &wire_thickness));

    cl_float2 cam_center = {cx, cy};
    float inv_scale = (scale != 0.0f) ? (1.0f / scale) : 1.0f;
    CL_CHECK(clSetKernelArg(kernel, 15, sizeof(cl_float2), &cam_center));
    CL_CHECK(clSetKernelArg(kernel, 16, sizeof(float), &inv_scale));
    CL_CHECK(clSetKernelArg(kernel, 17, sizeof(int), &shadow));

    float shadow_cx = width * 0.5f;
    float shadow_cy = height * 0.62f;
    float shadow_rx = (maxx - minx) * scale * 0.55f;
    float shadow_ry = (maxy - miny) * scale * 0.18f;
    cl_float2 shadow_center = {shadow_cx, shadow_cy};
    cl_float2 shadow_radii = {shadow_rx, shadow_ry};
    CL_CHECK(clSetKernelArg(kernel, 18, sizeof(cl_float2), &shadow_center));
    CL_CHECK(clSetKernelArg(kernel, 19, sizeof(cl_float2), &shadow_radii));
    CL_CHECK(clSetKernelArg(kernel, 20, sizeof(float), &shadow_strength));
    CL_CHECK(clSetKernelArg(kernel, 21, sizeof(cl_mem), &out_buf));

    size_t gws[2] = {static_cast<size_t>(width), static_cast<size_t>(height)};
    CL_ENQUEUE_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, gws, NULL, 0, NULL, NULL),
                     device_id, kernel, 2, gws, NULL);

    std::vector<uint8_t> out(out_bytes);
    CL_CHECK(clEnqueueReadBuffer(commandQueue, out_buf, CL_TRUE, 0, out_bytes, out.data(), 0, NULL, NULL));

    if (knobs) {
        overlay_knobs(out, width, height, vbuf, vert_visible, knob_radius, light1, light2, view_dir);
    }

    bool want_dump = args.dump;
    if (want_dump) {
        if (!vx::video::ensure_dir(args.outdir)) {
            fprintf(stderr, "Failed to create outdir: %s\n", args.outdir.c_str());
        } else {
            std::string out_ppm = vx::video::make_output_path(args, "output.ppm");
            if (!vx::video::write_ppm_rgba(out_ppm, width, height, out.data(), width * 4)) {
                fprintf(stderr, "Failed to write %s\n", out_ppm.c_str());
            } else {
                printf("Wrote %s\n", out_ppm.c_str());
            }
        }
    }

    printf("PASSED! faces=12\n");
    cleanup();
    return 0;
}
