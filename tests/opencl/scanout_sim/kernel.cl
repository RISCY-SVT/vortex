inline int clamp_int(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

inline uchar4 pattern_color(int x, int y, int w, int h, int mode, int seed) {
    uchar4 c = (uchar4)(0, 0, 0, 255);
    if (mode == 0) {
        int bar = (x * 8) / w;
        switch (bar) {
        case 0: c = (uchar4)(255, 255, 255, 255); break;
        case 1: c = (uchar4)(255, 255,   0, 255); break;
        case 2: c = (uchar4)(  0, 255, 255, 255); break;
        case 3: c = (uchar4)(  0, 255,   0, 255); break;
        case 4: c = (uchar4)(255,   0, 255, 255); break;
        case 5: c = (uchar4)(255,   0,   0, 255); break;
        case 6: c = (uchar4)(  0,   0, 255, 255); break;
        default: c = (uchar4)(0, 0, 0, 255); break;
        }
    } else if (mode == 1) {
        int block = 8;
        int v = ((x / block) ^ (y / block)) & 1;
        c = v ? (uchar4)(220, 220, 220, 255) : (uchar4)(30, 30, 30, 255);
    } else {
        int r = (w <= 1) ? 0 : (x * 255) / (w - 1);
        int g = (h <= 1) ? 0 : (y * 255) / (h - 1);
        int b = (w + h <= 2) ? 0 : ((x + y) * 255) / (w + h - 2);
        c = (uchar4)(r, g, b, 255);
        if (((x + seed) % 17) == 0) c = (uchar4)(32, 64, 255, 255);
    }
    return c;
}

inline ushort pack_rgb565(uchar4 c) {
    int r = (c.x * 31 + 127) / 255;
    int g = (c.y * 63 + 127) / 255;
    int b = (c.z * 31 + 127) / 255;
    return (ushort)((r << 11) | (g << 5) | b);
}

__kernel void fill_rgba(__global uchar4* out,
                        int width, int height, int stride,
                        int mode, int seed,
                        int use_rect, int rx, int ry, int rw, int rh) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;
    if (use_rect) {
        if (x < rx || y < ry || x >= (rx + rw) || y >= (ry + rh)) return;
    }
    int idx = y * stride + x;
    out[idx] = pattern_color(x, y, width, height, mode, seed);
}

__kernel void fill_rgb565(__global ushort* out,
                          int width, int height, int stride,
                          int mode, int seed) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;
    int idx = y * stride + x;
    out[idx] = pack_rgb565(pattern_color(x, y, width, height, mode, seed));
}
