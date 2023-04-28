// number of kernels per cell
__constant int NumKernelsPerCell = 64;

float2 gabor(float lsq, float frequency, float bandwidth, float phase) {
    // gaussian
    float g = exp(-M_PI * (bandwidth * bandwidth) * lsq);
    // sinewave
    float c = native_cos(2.0 * M_PI * frequency + phase);
    float s = native_sin(2.0 * M_PI * frequency + phase);
    return (float2)(c * g, s * g);
}

float2 gabor2(float2 uv, float frequency, float angle, float bandwidth, float phase) {
    float2 dir = (float2)(native_cos(angle), native_sin(angle));
    return gabor(dot(uv, uv), frequency * dot(uv, dir), bandwidth, phase);
}

float2 gabor3(float3 uvw, float frequency, float3 dir, float bandwidth, float phase) {
    return gabor(dot(uvw, uvw), frequency * dot(uvw, dir), bandwidth, phase);
}

// from https://cas.ee.ic.ac.uk/people/dt10/research/rngs-gpu-mwc64x.html
uint MWC64X(uint2 *state) {
    enum {A=4294883355U};
    uint x=(*state).x, c=(*state).y;  // Unpack the state
    uint res=x^c;                     // Calculate the result
    uint hi=mul_hi(x,A);              // Step the RNG
    x=x*A+c;
    c=hi+(x<c);
    *state=(uint2)(x,c);              // Pack the state back up
    return res;                       // Return the next result
}

float random_01(uint2 *state) {
    return convert_float(MWC64X(state)) / 4294967296.0; // 2^32 = 4294967296
}

float2 sampleCell2(float2 ij, float2 luv, float frequency, float bandwidth, float kr, float factor_angle_spread, int seed) {
    // init random number generator
    uint2 state = (uint2)(seed + ij.x, seed + ij.y);
    for (int i = 0; i < 3; i++) random_01(&state);
    float2 v = (float2)(0.0, 0.0);
    float cellsz = 2.0 * kr;
    int N = NumKernelsPerCell;
    int nIter = 0;
    while (nIter < N) {
        float rx = random_01(&state);
        float ry = random_01(&state);
        float2 ctr = (float2)(rx, ry);
        float w = (2.0 * random_01(&state) - 1.0);
        float2 d = (luv - ctr) * cellsz;
        float ra = M_PI * (random_01(&state) - 0.5) * factor_angle_spread;  // angular spread
        float phase = random_01(&state) * 2.0 * M_PI;
        float2 k = gabor2(d, frequency, ra, bandwidth, phase);
        v += w * k;
        nIter++;
    }
    return v;
}

float2 sampleCell3(float3 ijk, float3 luvw, float frequency, float bandwidth, float kr, float factor_angle_spread, int seed) {
    // init random number generator
    uint2 state = (uint2)(seed + ijk.x + ijk.z, seed + ijk.y + ijk.z);
    for (int i = 0; i < 3; i++) random_01(&state);
    float2 v = (float2)(0.0, 0.0);
    float3 cellsz = 2.0 * kr;
    int N = NumKernelsPerCell;
    int nIter = 0;
    while (nIter < N) {
        float rx = random_01(&state);
        float ry = random_01(&state);
        float rz = random_01(&state);
        float3 ctr = (float3)(rx, ry, rz);
        float w = (2.0 * random_01(&state) - 1.0);
        float3 d = (luvw - ctr) * cellsz;
        // generate kernel direction
        // first option with single factor parameter
        float theta = 2.0 * M_PI * random_01(&state);
        float phi = acos(1.0 - 2.0 * random_01(&state) * factor_angle_spread * 0.5);
        float3 dir = (float3) (
                         native_sin(phi) * native_cos(theta),
                         native_sin(phi) * native_sin(theta),
                         native_cos(phi)
                     );
        float phase = random_01(&state) * 2.0 * M_PI;
        float2 k = gabor3(d, frequency, dir, bandwidth, phase);
        v += w * k;
        nIter++;
    }
    return v;
}

float kernelRadius(float ka, float truncate) {
    return sqrt(-log(truncate) / M_PI) / ka;
}

float2 makeNoise2(float2 uv, float frequency, float bandwidth, float factor_angle_spread, int make_periodic, float truncate, int seed) {
    // sparse convolution
    float kr = kernelRadius(bandwidth, truncate);
    float csz = 2.0 * kr;
    int ncsz = convert_int(round(1.0 / csz));
    // 1.0 / csz => integer
    float2 _ij = uv / csz;
    int2 ij = convert_int2(floor(_ij));
    float2 fij = _ij - convert_float2(ij);
    // sample four cells
    int2 nd = (int2)(0, 0);
    nd.x = (fij.x > 0.5) ? 1 : -1;
    nd.y = (fij.y > 0.5) ? 1 : -1;
    float2 v = (float2)(0.0, 0.0);
    for (int j = 0; j < 2; j++) {
        for (int i = 0; i < 2; i++) {
            int2 nij = (int2)(i, j);
            int2 mij = ij + nij * nd;
            if (make_periodic) {
                mij.x = (mij.x + ncsz) % ncsz;
                mij.y = (mij.y + ncsz) % ncsz;
            }
            v += sampleCell2(convert_float2(mij), fij - convert_float2(nij * nd), frequency, bandwidth, kr, factor_angle_spread, seed);
        }
    }
    return v;
}

float2 makeNoise3(float3 uvw, float frequency, float bandwidth, float factor_angle_spread, int make_periodic, float truncate, int seed) {
    // sparse convolution
    float kr = kernelRadius(bandwidth, truncate);
    float csz = 2.0 * kr;
    float3 cellsz = csz;
    int ncsz = round(1.0 / csz);
    // 1.0 / cellsz => integer
    float3 _ijk = uvw / cellsz;
    int3 ijk = convert_int3(floor(_ijk));
    float3 fijk = _ijk - convert_float3(ijk);
    // sample eight cells
    int3 nd = (int3)(0, 0, 0);
    nd.x = (fijk.x > 0.5) ? 1 : -1;
    nd.y = (fijk.y > 0.5) ? 1 : -1;
    nd.z = (fijk.z > 0.5) ? 1 : -1;
    float2 v = (float2)(0.0, 0.0);
    for (int k = 0; k < 2; k++) {
        for (int j = 0; j < 2; j++) {
            for (int i = 0; i < 2; i++) {
                int3 nijk = (int3)(i, j, k);
                int3 mijk = (int3)(ijk.x, ijk.y, ijk.z) + nijk * nd;
                if (make_periodic) {
                    mijk.x = (mijk.x + ncsz) % ncsz;
                    mijk.y = (mijk.y + ncsz) % ncsz;
                    mijk.z = (mijk.z + ncsz) % ncsz;
                }
                v += sampleCell3(convert_float3(mijk), fijk - convert_float3(nijk * nd), frequency, bandwidth, kr, factor_angle_spread, seed);
            }
        }
    }
    // done
    return v;
}

__kernel void phasor2D(__global float *result_r, float frequency,
                       float bandwidth, float phasor_density,
                       float factor_angle_spread, int make_periodic,
                       float truncate, int seed) {
    int n1 = get_global_id(0);
    int n2 = get_global_id(1);
    int N1 = get_global_size(0);
    int N2 = get_global_size(1);
    float2 uv = convert_float2((int2)(n1, n2)) / convert_float(N1);  // NOTE: expects positive coords
    float2 v = makeNoise2(uv, frequency, bandwidth, factor_angle_spread, make_periodic, truncate, seed);
    // float result = v.x; // gabor
    // float result = native_sin(atan2(v.y,v.x)); // phasor
    float result = (0.5 + 0.5 * atan2(v.y, v.x) / M_PI) > phasor_density ? 0.0 : 1.0;  // thesholded
    // row-major ordering (numpy array)
    result_r[n2 + N2 * n1] = result;
}

__kernel void phasor3D(__global float *result_r, float frequency,
                       float bandwidth, float phasor_density,
                       float factor_angle_spread, int make_periodic,
                       float truncate, int seed) {
    int n1 = get_global_id(0);
    int n2 = get_global_id(1);
    int n3 = get_global_id(2);
    int N1 = get_global_size(0);
    int N2 = get_global_size(1);
    int N3 = get_global_size(2);
    float3 uvw =
        convert_float3((int3)(n1, n2, n3)) / convert_float(N1);  // NOTE: expects positive coords
    float2 v = makeNoise3(uvw, frequency, bandwidth, factor_angle_spread, make_periodic, truncate, seed);
    // float result = v.x; // gabor
    // float result = native_sin(atan2(v.y,v.x)); // phasor
    float result = (0.5 + 0.5 * atan2(v.y, v.x) / M_PI) > phasor_density ? 0.0 : 1.0;  // thesholded
    // row-major ordering (numpy array)
    result_r[n3 + N3 * (n2 + N2 * n1)] = result;
}
