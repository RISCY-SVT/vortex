inline uchar clamp_u8(int v) {
    if (v < 0) return (uchar)0;
    if (v > 255) return (uchar)255;
    return (uchar)v;
}

__kernel void alpha_blend(__global const uchar4* bg,
                          __global const uchar4* fg,
                          __global uchar4* out,
                          int width, int height, int stride) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;
    int idx = y * stride + x;
    uchar4 b = bg[idx];
    uchar4 f = fg[idx];
    int a = f.w; // 0..255
    int inv = 255 - a;
    int r = (f.x * a + b.x * inv + 127) / 255;
    int g = (f.y * a + b.y * inv + 127) / 255;
    int bb = (f.z * a + b.z * inv + 127) / 255;
    out[idx] = (uchar4)(clamp_u8(r), clamp_u8(g), clamp_u8(bb), (uchar)255);
}
