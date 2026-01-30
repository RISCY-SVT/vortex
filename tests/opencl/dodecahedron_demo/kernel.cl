// Roman dodecahedron demo renderer (opaque style)

#define NUM_FACES 12
#define NUM_VERTS 20

static inline int inside_bary(float2 p, float2 a, float2 b, float2 c, float* u, float* v, float* w) {
    float2 v0 = b - a;
    float2 v1 = c - a;
    float2 v2 = p - a;
    float d00 = dot(v0, v0);
    float d01 = dot(v0, v1);
    float d11 = dot(v1, v1);
    float d20 = dot(v2, v0);
    float d21 = dot(v2, v1);
    float denom = d00 * d11 - d01 * d01;
    if (fabs(denom) < 1e-8f) return 0;
    float inv = 1.0f / denom;
    float vv = (d11 * d20 - d01 * d21) * inv;
    float ww = (d00 * d21 - d01 * d20) * inv;
    float uu = 1.0f - vv - ww;
    *u = uu; *v = vv; *w = ww;
    return (uu >= 0.0f && vv >= 0.0f && ww >= 0.0f);
}

static inline float dist_to_seg(float2 p, float2 a, float2 b) {
    float2 ab = b - a;
    float2 ap = p - a;
    float denom = dot(ab, ab);
    float t = (denom > 1e-6f) ? (dot(ap, ab) / denom) : 0.0f;
    t = clamp(t, 0.0f, 1.0f);
    float2 c = a + t * ab;
    float2 d = p - c;
    return sqrt(dot(d, d));
}

__kernel void render(
    __global const float4* verts,         // x,y screen; z cam
    __global const int* faces,            // 12*5 indices
    __global const int* face_order,       // 12 indices (back-to-front)
    __global const float4* face_colors,   // rgb + alpha
    __global const float4* face_origin,   // cam xyz
    __global const float4* face_u,        // cam xyz
    __global const float4* face_v,        // cam xyz
    __global const float4* face_n,        // cam xyz
    __global const float* hole_radii,     // cam units
    int width,
    int height,
    int holes,
    int rings,
    int wire,
    float wire_thickness,
    float2 cam_center,
    float inv_scale,
    int shadow,
    float2 shadow_center,
    float2 shadow_radii,
    float shadow_strength,
    __global uchar4* out)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    float2 p = (float2)((float)x + 0.5f, (float)y + 0.5f);

    // background gradient
    float u = (width > 1) ? ((float)x / (float)(width - 1)) : 0.0f;
    float v = (height > 1) ? ((float)y / (float)(height - 1)) : 0.0f;
    float3 bg0 = (float3)(0.96f, 0.97f, 0.99f);
    float3 bg1 = (float3)(0.90f, 0.92f, 0.96f);
    float t = 0.55f * v + 0.45f * u;
    float3 color = bg0 * (1.0f - t) + bg1 * t;

    // soft shadow
    if (shadow) {
        float dx = (p.x - shadow_center.x) / fmax(shadow_radii.x, 1.0f);
        float dy = (p.y - shadow_center.y) / fmax(shadow_radii.y, 1.0f);
        float d2 = dx * dx + dy * dy;
        if (d2 < 1.0f) {
            float fall = (1.0f - d2);
            float a = shadow_strength * fall * fall;
            float3 sh = (float3)(0.0f, 0.0f, 0.0f);
            color = sh * a + color * (1.0f - a);
        }
    }

    for (int fi = 0; fi < NUM_FACES; ++fi) {
        int f = face_order[fi];
        int i0 = faces[f * 5 + 0];
        int i1 = faces[f * 5 + 1];
        int i2 = faces[f * 5 + 2];
        int i3 = faces[f * 5 + 3];
        int i4 = faces[f * 5 + 4];

        float4 v0 = verts[i0];
        float4 v1 = verts[i1];
        float4 v2 = verts[i2];
        float4 v3 = verts[i3];
        float4 v4 = verts[i4];

        float uu, vv, ww;
        int inside = 0;
        float2 a = v0.xy;
        float2 b = v1.xy;
        float2 c = v2.xy;
        if (inside_bary(p, a, b, c, &uu, &vv, &ww)) {
            inside = 1;
        } else {
            b = v2.xy; c = v3.xy;
            if (inside_bary(p, a, b, c, &uu, &vv, &ww)) {
                inside = 1;
            } else {
                b = v3.xy; c = v4.xy;
                if (inside_bary(p, a, b, c, &uu, &vv, &ww)) {
                    inside = 1;
                }
            }
        }

        if (inside) {
            float4 fc = face_colors[f];
            float3 face_rgb = fc.xyz;
            float a_face = clamp(fc.w, 0.0f, 1.0f);

            // subtle edge darkening (bevel)
            if (wire && wire_thickness > 0.0f) {
                float min_dist = 1e9f;
                min_dist = fmin(min_dist, dist_to_seg(p, v0.xy, v1.xy));
                min_dist = fmin(min_dist, dist_to_seg(p, v1.xy, v2.xy));
                min_dist = fmin(min_dist, dist_to_seg(p, v2.xy, v3.xy));
                min_dist = fmin(min_dist, dist_to_seg(p, v3.xy, v4.xy));
                min_dist = fmin(min_dist, dist_to_seg(p, v4.xy, v0.xy));
                float edge_alpha = clamp((wire_thickness - min_dist) / wire_thickness, 0.0f, 1.0f);
                if (edge_alpha > 0.0f) {
                    face_rgb *= (1.0f - 0.35f * edge_alpha);
                }
            }

            if (holes) {
                float2 cam_xy = (float2)((p.x - (float)width * 0.5f) * inv_scale + cam_center.x,
                                         (-(p.y - (float)height * 0.5f)) * inv_scale + cam_center.y);
                float3 o = face_origin[f].xyz;
                float3 n = face_n[f].xyz;
                float3 uvec = face_u[f].xyz;
                float3 vvec = face_v[f].xyz;
                float z = o.z;
                float nz = n.z;
                if (fabs(nz) > 1e-6f) {
                    float dz = (n.x * (cam_xy.x - o.x) + n.y * (cam_xy.y - o.y));
                    z = o.z - dz / nz;
                }
                float3 d = (float3)(cam_xy.x - o.x, cam_xy.y - o.y, z - o.z);
                float ru = dot(d, uvec);
                float rv = dot(d, vvec);
                float r = sqrt(ru * ru + rv * rv);
                float hr = hole_radii[f];
                if (r < hr) {
                    float tt = clamp(r / fmax(hr, 1e-6f), 0.0f, 1.0f);
                    float3 c0 = (float3)(0.04f, 0.05f, 0.07f);
                    float3 c1 = (float3)(0.08f, 0.10f, 0.14f);
                    float3 cavity = c0 * (1.0f - tt * tt) + c1 * (tt * tt);
                    face_rgb = cavity;
                    a_face = 1.0f;
                } else if (rings) {
                    float rr = r / fmax(hr, 1e-6f);
                    float ring_width = 0.03f;
                    float ring_dark = 0.18f;
                    float d1 = fabs(rr - 1.05f);
                    float d2 = fabs(rr - 1.18f);
                    float d3 = fabs(rr - 1.32f);
                    float ring = 0.0f;
                    if (d1 < ring_width) ring = fmax(ring, 1.0f - d1 / ring_width);
                    if (d2 < ring_width) ring = fmax(ring, 1.0f - d2 / ring_width);
                    if (d3 < ring_width) ring = fmax(ring, 1.0f - d3 / ring_width);
                    if (ring > 0.0f) {
                        face_rgb *= (1.0f - ring_dark * ring);
                    }
                }
            }

            color = face_rgb * a_face + color * (1.0f - a_face);
        }
    }

    uchar4 out_px;
    out_px.x = (uchar)clamp(color.x * 255.0f, 0.0f, 255.0f);
    out_px.y = (uchar)clamp(color.y * 255.0f, 0.0f, 255.0f);
    out_px.z = (uchar)clamp(color.z * 255.0f, 0.0f, 255.0f);
    out_px.w = 255;
    out[y * width + x] = out_px;
}
