inline uchar clamp_u8(int v) {
    if (v < 0) return (uchar)0;
    if (v > 255) return (uchar)255;
    return (uchar)v;
}

__kernel void scale_nearest(__global const uchar4* src,
                            __global uchar4* dst,
                            int in_w, int in_h, int in_stride,
                            int out_w, int out_h, int out_stride) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= out_w || y >= out_h) return;

    float fx = (out_w > 1) ? ((float)x * (float)(in_w - 1) / (float)(out_w - 1)) : 0.0f;
    float fy = (out_h > 1) ? ((float)y * (float)(in_h - 1) / (float)(out_h - 1)) : 0.0f;
    int ix = (int)(fx + 0.5f);
    int iy = (int)(fy + 0.5f);
    if (ix < 0) ix = 0; if (ix >= in_w) ix = in_w - 1;
    if (iy < 0) iy = 0; if (iy >= in_h) iy = in_h - 1;

    dst[y * out_stride + x] = src[iy * in_stride + ix];
}

__kernel void scale_bilinear(__global const uchar4* src,
                             __global uchar4* dst,
                             int in_w, int in_h, int in_stride,
                             int out_w, int out_h, int out_stride) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= out_w || y >= out_h) return;

    float fx = (out_w > 1) ? ((float)x * (float)(in_w - 1) / (float)(out_w - 1)) : 0.0f;
    float fy = (out_h > 1) ? ((float)y * (float)(in_h - 1) / (float)(out_h - 1)) : 0.0f;

    int x0 = (int)floor(fx);
    int y0 = (int)floor(fy);
    int x1 = x0 + 1; if (x1 >= in_w) x1 = in_w - 1;
    int y1 = y0 + 1; if (y1 >= in_h) y1 = in_h - 1;
    float tx = fx - (float)x0;
    float ty = fy - (float)y0;

    uchar4 c00 = src[y0 * in_stride + x0];
    uchar4 c10 = src[y0 * in_stride + x1];
    uchar4 c01 = src[y1 * in_stride + x0];
    uchar4 c11 = src[y1 * in_stride + x1];

    float4 a = convert_float4(c00);
    float4 b = convert_float4(c10);
    float4 c = convert_float4(c01);
    float4 d = convert_float4(c11);

    float4 top = a + (b - a) * tx;
    float4 bot = c + (d - c) * tx;
    float4 val = top + (bot - top) * ty;

    int r = (int)(val.x + 0.5f);
    int g = (int)(val.y + 0.5f);
    int bch = (int)(val.z + 0.5f);
    int aout = (int)(val.w + 0.5f);

    dst[y * out_stride + x] = (uchar4)(clamp_u8(r), clamp_u8(g), clamp_u8(bch), clamp_u8(aout));
}
