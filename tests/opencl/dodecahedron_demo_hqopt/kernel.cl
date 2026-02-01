// Roman dodecahedron demo renderer (opaque style, HQ/fast modes)

#define NUM_FACES 12
#define NUM_VERTS 20

static inline float edge_fn(float2 a, float2 b, float2 p) {
    return (p.x - a.x) * (b.y - a.y) - (p.y - a.y) * (b.x - a.x);
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
    __global const float4* face_bbox,     // minx,miny,maxx,maxy
    __global const float* hole_radii,     // cam units
    int width,
    int height,
    int holes,
    int rings,
    int wire,
    float wire_thickness,
    int quality,
    int debug_disable_bbox,
    float2 cam_center,
    float inv_scale,
    int shadow,
    float2 shadow_center,
    float2 shadow_radii,
    float shadow_strength,
    int debug_counters,
    __global uchar* bbox_mask,
    __global uchar* inside_mask,
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

    int bbox_hit = 0;
    int inside_hit = 0;
    for (int fi = 0; fi < NUM_FACES; ++fi) {
        int f = face_order[fi];
        float4 bb = face_bbox[f];
        if (!debug_disable_bbox) {
            if (p.x < bb.x || p.x > bb.z || p.y < bb.y || p.y > bb.w) {
                continue;
            }
            bbox_hit = 1;
        } else {
            bbox_hit = 1;
        }
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

        float2 s0 = v0.xy;
        float2 s1 = v1.xy;
        float2 s2 = v2.xy;
        float2 s3 = v3.xy;
        float2 s4 = v4.xy;
        float eps = 1e-4f;
        float e0 = edge_fn(s0, s1, p);
        float e1 = edge_fn(s1, s2, p);
        float e2 = edge_fn(s2, s3, p);
        float e3 = edge_fn(s3, s4, p);
        float e4 = edge_fn(s4, s0, p);
        if (!((e0 >= -eps && e1 >= -eps && e2 >= -eps && e3 >= -eps && e4 >= -eps) ||
              (e0 <= eps && e1 <= eps && e2 <= eps && e3 <= eps && e4 <= eps))) {
            continue;
        }

        {
            inside_hit = 1;
            float4 fc = face_colors[f];
            float3 face_rgb = fc.xyz;
            float a_face = clamp(fc.w, 0.0f, 1.0f);

            // subtle edge darkening (bevel)
            if (quality && wire && wire_thickness > 0.0f) {
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
                float hr = hole_radii[f];
                float r2 = ru * ru + rv * rv;
                float hr2 = hr * hr;
                if (r2 < hr2) {
                    float r = sqrt(r2);
                    float tt = clamp(r / fmax(hr, 1e-6f), 0.0f, 1.0f);
                    float3 c0 = (float3)(0.04f, 0.05f, 0.07f);
                    float3 c1 = (float3)(0.08f, 0.10f, 0.14f);
                    float3 cavity = c0 * (1.0f - tt * tt) + c1 * (tt * tt);
                    face_rgb = cavity;
                    a_face = 1.0f;
                } else if (quality && rings) {
                    float ring_width = 0.03f * hr;
                    float ring_dark = 0.18f;
                    float r1 = 1.05f * hr;
                    float r2c = 1.18f * hr;
                    float r3 = 1.32f * hr;
                    float r1min2 = (r1 - ring_width) * (r1 - ring_width);
                    float r1max2 = (r1 + ring_width) * (r1 + ring_width);
                    float r2min2 = (r2c - ring_width) * (r2c - ring_width);
                    float r2max2 = (r2c + ring_width) * (r2c + ring_width);
                    float r3min2 = (r3 - ring_width) * (r3 - ring_width);
                    float r3max2 = (r3 + ring_width) * (r3 + ring_width);
                    if ((r2 >= r1min2 && r2 <= r1max2) ||
                        (r2 >= r2min2 && r2 <= r2max2) ||
                        (r2 >= r3min2 && r2 <= r3max2)) {
                        float r = sqrt(r2);
                        float d1 = fabs(r - r1);
                        float d2 = fabs(r - r2c);
                        float d3 = fabs(r - r3);
                        float ring = 0.0f;
                        if (d1 < ring_width) ring = fmax(ring, 1.0f - d1 / ring_width);
                        if (d2 < ring_width) ring = fmax(ring, 1.0f - d2 / ring_width);
                        if (d3 < ring_width) ring = fmax(ring, 1.0f - d3 / ring_width);
                        if (ring > 0.0f) {
                            face_rgb *= (1.0f - ring_dark * ring);
                        }
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

    if (debug_counters) {
        int idx = y * width + x;
        bbox_mask[idx] = (uchar)(bbox_hit ? 1 : 0);
        inside_mask[idx] = (uchar)(inside_hit ? 1 : 0);
    }
}
