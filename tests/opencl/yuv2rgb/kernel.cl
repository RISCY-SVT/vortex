inline uchar clamp_u8(int v) {
    if (v < 0) return (uchar)0;
    if (v > 255) return (uchar)255;
    return (uchar)v;
}

__kernel void yuv420_to_rgba(__global const uchar* y_plane,
                            __global const uchar* uv_plane,
                            __global uchar4* out,
                            int width, int height, int out_stride_pixels) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    int y_idx = y * width + x;
    int uvx = x >> 1;
    int uvy = y >> 1;
    int uv_idx = (uvy * (width >> 1) + uvx) * 2;

    int Y = (int)y_plane[y_idx];
    int U = (int)uv_plane[uv_idx + 0] - 128;
    int V = (int)uv_plane[uv_idx + 1] - 128;

    // integer BT.601 full-range approximation
    int R = Y + (1436 * V) / 1024;
    int G = Y - (352 * U + 731 * V) / 1024;
    int B = Y + (1814 * U) / 1024;

    int idx = y * out_stride_pixels + x;
    out[idx] = (uchar4)(clamp_u8(R), clamp_u8(G), clamp_u8(B), (uchar)255);
}
