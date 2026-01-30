__kernel void test_pattern(__global uchar4* out, int width, int height,
                           int stride_pixels, int mode, int seed) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;
    int idx = y * stride_pixels + x;

    uchar4 c = (uchar4)(0, 0, 0, 255);

    if (mode == 0) {
        // Color bars
        int bar = (x * 8) / width;
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
        // Checkerboard
        int block = 8;
        int v = ((x / block) ^ (y / block)) & 1;
        c = v ? (uchar4)(220, 220, 220, 255) : (uchar4)(30, 30, 30, 255);
    } else {
        // Gradient + overlay shapes
        int r = (width <= 1) ? 0 : (x * 255) / (width - 1);
        int g = (height <= 1) ? 0 : (y * 255) / (height - 1);
        int b = (width + height <= 2) ? 0 : ((x + y) * 255) / (width + height - 2);
        c = (uchar4)(r, g, b, 255);
        int cx = width / 2;
        int cy = height / 2;
        int dx = x - cx;
        int dy = y - cy;
        int rad = (width < height ? width : height) / 4;
        if (dx * dx + dy * dy <= rad * rad) {
            c = (uchar4)(255, 64, 32, 255);
        }
        // small seeded stripe
        if (((x + seed) % 17) == 0) {
            c = (uchar4)(32, 64, 255, 255);
        }
    }

    out[idx] = c;
}
