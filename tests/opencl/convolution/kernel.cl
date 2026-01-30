inline int clamp_int(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

inline int luma_at(__global const uchar4* src, int x, int y, int w, int h, int stride) {
    x = clamp_int(x, 0, w - 1);
    y = clamp_int(y, 0, h - 1);
    uchar4 c = src[y * stride + x];
    return (77 * c.x + 150 * c.y + 29 * c.z) >> 8;
}

__kernel void gaussian_blur(__global const uchar4* src,
                            __global uchar4* dst,
                            int w, int h, int stride) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= w || y >= h) return;

    int s00 = luma_at(src, x - 1, y - 1, w, h, stride);
    int s01 = luma_at(src, x,     y - 1, w, h, stride);
    int s02 = luma_at(src, x + 1, y - 1, w, h, stride);
    int s10 = luma_at(src, x - 1, y,     w, h, stride);
    int s11 = luma_at(src, x,     y,     w, h, stride);
    int s12 = luma_at(src, x + 1, y,     w, h, stride);
    int s20 = luma_at(src, x - 1, y + 1, w, h, stride);
    int s21 = luma_at(src, x,     y + 1, w, h, stride);
    int s22 = luma_at(src, x + 1, y + 1, w, h, stride);

    int sum = s00 + 2 * s01 + s02 + 2 * s10 + 4 * s11 + 2 * s12 + s20 + 2 * s21 + s22;
    int v = sum >> 4;
    if (v < 0) v = 0; if (v > 255) v = 255;
    dst[y * stride + x] = (uchar4)(v, v, v, 255);
}

__kernel void sobel_edge(__global const uchar4* src,
                         __global uchar4* dst,
                         int w, int h, int stride) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= w || y >= h) return;

    int s00 = luma_at(src, x - 1, y - 1, w, h, stride);
    int s01 = luma_at(src, x,     y - 1, w, h, stride);
    int s02 = luma_at(src, x + 1, y - 1, w, h, stride);
    int s10 = luma_at(src, x - 1, y,     w, h, stride);
    int s12 = luma_at(src, x + 1, y,     w, h, stride);
    int s20 = luma_at(src, x - 1, y + 1, w, h, stride);
    int s21 = luma_at(src, x,     y + 1, w, h, stride);
    int s22 = luma_at(src, x + 1, y + 1, w, h, stride);

    int gx = -s00 + s02 - 2 * s10 + 2 * s12 - s20 + s22;
    int gy = -s00 - 2 * s01 - s02 + s20 + 2 * s21 + s22;
    int mag = abs(gx) + abs(gy);
    if (mag > 255) mag = 255;
    dst[y * stride + x] = (uchar4)(mag, mag, mag, 255);
}
