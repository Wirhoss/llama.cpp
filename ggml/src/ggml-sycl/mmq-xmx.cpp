#include "mmq-xmx.hpp"

#include "common.hpp"
#include "convert.hpp"

#include <array>
#include <atomic>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace syclex = sycl::ext::oneapi::experimental;

namespace {

constexpr int xmx_tile_m = 2;
constexpr int xmx_tile_n = 16;
constexpr int xmx_tile_k = 32;
constexpr int xmx_q5_q6_min_m = 2048;

static constexpr const char * xmx_opencl_source = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable

inline int8 load_b_frag(__global const char * values, int kg, uint lane) {
    const int base = kg * 32 * 16 + lane * 4;
    int8 result;
    result.s0 = as_int(vload4(0, values + base + 0 * 16 * 4));
    result.s1 = as_int(vload4(0, values + base + 1 * 16 * 4));
    result.s2 = as_int(vload4(0, values + base + 2 * 16 * 4));
    result.s3 = as_int(vload4(0, values + base + 3 * 16 * 4));
    result.s4 = as_int(vload4(0, values + base + 4 * 16 * 4));
    result.s5 = as_int(vload4(0, values + base + 5 * 16 * 4));
    result.s6 = as_int(vload4(0, values + base + 6 * 16 * 4));
    result.s7 = as_int(vload4(0, values + base + 7 * 16 * 4));
    return result;
}

inline uchar qk_scale(__global const uchar * values, int group) {
    return group < 4
        ? values[group] & 63
        : (values[group + 4] & 15) | ((values[group - 4] >> 6) << 4);
}

inline uchar qk_minimum(__global const uchar * values, int group) {
    return group < 4
        ? values[group + 4] & 63
        : (values[group + 4] >> 4) | ((values[group] >> 6) << 4);
}

inline ushort load_q4_pair(__global const uchar * qs, int group, uint lane) {
    const int element = 2 * lane;
    uchar q0 = qs[(group / 2) * 32 + element];
    uchar q1 = qs[(group / 2) * 32 + element + 1];
    if ((group & 1) == 0) {
        q0 &= 15;
        q1 &= 15;
    } else {
        q0 >>= 4;
        q1 >>= 4;
    }
    return as_ushort((uchar2)(q0, q1));
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void opencl_q4k_q8_m2n16(
        __global const uchar * weights,
        __global const char * activation_values,
        __global const half * activation_d,
        __global const half * activation_s,
        __global float * output,
        int active_n,
        int m,
        int groups,
        int blocks,
        int total_weight_blocks,
        int reordered) {
    const uint lane = get_sub_group_local_id();
    const int row0 = get_group_id(0) * 2;
    float2 sum = (float2)(0.0f);

    for (int kg = 0; kg < groups; ++kg) {
        const int block_index = kg / 8;
        const int group = kg & 7;
        const int weight_index0 = row0 * blocks + block_index;
        const int weight_index1 = (row0 + 1) * blocks + block_index;

        __global const uchar * qs0;
        __global const uchar * scales0;
        __global const half * dm0;
        __global const uchar * qs1;
        __global const uchar * scales1;
        __global const half * dm1;
        if (reordered) {
            qs0 = weights + weight_index0 * 128;
            scales0 = weights + total_weight_blocks * 128 + weight_index0 * 12;
            dm0 = (__global const half *)(weights + total_weight_blocks * 140 + weight_index0 * 4);
            qs1 = weights + weight_index1 * 128;
            scales1 = weights + total_weight_blocks * 128 + weight_index1 * 12;
            dm1 = (__global const half *)(weights + total_weight_blocks * 140 + weight_index1 * 4);
        } else {
            __global const uchar * weight0 = weights + weight_index0 * 144;
            __global const uchar * weight1 = weights + weight_index1 * 144;
            dm0 = (__global const half *)weight0;
            scales0 = weight0 + 4;
            qs0 = weight0 + 16;
            dm1 = (__global const half *)weight1;
            scales1 = weight1 + 4;
            qs1 = weight1 + 16;
        }

        const ushort2 a_frag = (ushort2)(
            load_q4_pair(qs0, group, lane),
            load_q4_pair(qs1, group, lane));
        const int8 b_frag = load_b_frag(activation_values, kg, lane);
        const int2 dot = intel_sub_group_u8_i8_matrix_mad_k32(a_frag, b_frag, (int2)(0));
        const float d = convert_float(activation_d[kg * 16 + lane]);
        const float s = convert_float(activation_s[kg * 16 + lane]);
        sum.s0 += convert_float(dm0[0]) * convert_float(qk_scale(scales0, group)) * d * convert_float(dot.s0)
                - convert_float(dm0[1]) * convert_float(qk_minimum(scales0, group)) * s;
        sum.s1 += convert_float(dm1[0]) * convert_float(qk_scale(scales1, group)) * d * convert_float(dot.s1)
                - convert_float(dm1[1]) * convert_float(qk_minimum(scales1, group)) * s;
    }

    if ((int)lane < active_n) {
        output[lane * m + row0] = sum.s0;
        output[lane * m + row0 + 1] = sum.s1;
    }
}

inline short load_q6_pair(
        __global const uchar * ql,
        __global const uchar * qh,
        int group,
        uint lane) {
    const int index = group * 32 + 2 * lane;
    const int chunk = index >> 7;
    const int position = index & 127;
    const int quadrant = position >> 5;
    const int element = position & 31;
    const int ql_index = chunk * 64 + (quadrant & 1) * 32 + element;
    const int qh_index = chunk * 32 + element;
    const int ql_shift = (quadrant >> 1) * 4;
    const int qh_shift = quadrant * 2;
    const uchar ql0 = (ql[ql_index] >> ql_shift) & 15;
    const uchar ql1 = (ql[ql_index + 1] >> ql_shift) & 15;
    const uchar qh0 = (qh[qh_index] >> qh_shift) & 3;
    const uchar qh1 = (qh[qh_index + 1] >> qh_shift) & 3;
    return as_short((char2)((char)(ql0 | (qh0 << 4)) - 32,
                            (char)(ql1 | (qh1 << 4)) - 32));
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void opencl_q6k_q8_m2n16(
        __global const uchar * weights,
        __global const char * activation_values,
        __global const half * activation_d,
        __global float * output,
        int active_n,
        int m,
        int groups,
        int blocks,
        int total_weight_blocks,
        int reordered) {
    const uint lane = get_sub_group_local_id();
    const int row0 = get_group_id(0) * 2;
    float2 sum = (float2)(0.0f);

    for (int kg = 0; kg < groups; ++kg) {
        const int block_index = kg / 8;
        const int group = kg & 7;
        const int weight_index0 = row0 * blocks + block_index;
        const int weight_index1 = (row0 + 1) * blocks + block_index;

        __global const uchar * ql0;
        __global const uchar * qh0;
        __global const char * scales0;
        __global const half * wd0;
        __global const uchar * ql1;
        __global const uchar * qh1;
        __global const char * scales1;
        __global const half * wd1;
        if (reordered) {
            ql0 = weights + weight_index0 * 128;
            qh0 = weights + total_weight_blocks * 128 + weight_index0 * 64;
            scales0 = (__global const char *)(weights + total_weight_blocks * 192 + weight_index0 * 16);
            wd0 = (__global const half *)(weights + total_weight_blocks * 208 + weight_index0 * 2);
            ql1 = weights + weight_index1 * 128;
            qh1 = weights + total_weight_blocks * 128 + weight_index1 * 64;
            scales1 = (__global const char *)(weights + total_weight_blocks * 192 + weight_index1 * 16);
            wd1 = (__global const half *)(weights + total_weight_blocks * 208 + weight_index1 * 2);
        } else {
            __global const uchar * weight0 = weights + weight_index0 * 210;
            __global const uchar * weight1 = weights + weight_index1 * 210;
            ql0 = weight0;
            qh0 = weight0 + 128;
            scales0 = (__global const char *)(weight0 + 192);
            wd0 = (__global const half *)(weight0 + 208);
            ql1 = weight1;
            qh1 = weight1 + 128;
            scales1 = (__global const char *)(weight1 + 192);
            wd1 = (__global const half *)(weight1 + 208);
        }

        const short pair0 = load_q6_pair(ql0, qh0, group, lane);
        const short pair1 = load_q6_pair(ql1, qh1, group, lane);

        // Q6_K has two independently scaled K=16 halves in every K=32 group,
        // while this device exposes an integer DPAS with K=32. Treat each
        // (output row, K=16 half) as a separate DPAS row, then combine the two
        // DPAS results with their Q6_K scales below. This obtains both halves
        // in one instruction instead of issuing two narrower DPAS operations.
        //
        // Cost: every logical DPAS row contains 16 values and 16 zeros, so half
        // of the systolic A operand is intentionally idle. Keep this explicit:
        // the two-pass alternative is benchmarked by the probes and this layout
        // must not be copied to other quant types without measuring both forms.
        const short4 a_frag = lane < 8
            ? (short4)(pair0, 0, pair1, 0)
            : (short4)(0, pair0, 0, pair1);
        const int8 b_frag = load_b_frag(activation_values, kg, lane);
        const int4 dot = intel_sub_group_i8_i8_matrix_mad_k32(a_frag, b_frag, (int4)(0));

        const float activation_scale = convert_float(activation_d[kg * 16 + lane]);
        sum.s0 += convert_float(*wd0) * activation_scale *
            (convert_float(scales0[2 * group]) * convert_float(dot.s0) +
             convert_float(scales0[2 * group + 1]) * convert_float(dot.s1));
        sum.s1 += convert_float(*wd1) * activation_scale *
            (convert_float(scales1[2 * group]) * convert_float(dot.s2) +
             convert_float(scales1[2 * group + 1]) * convert_float(dot.s3));
    }

    if ((int)lane < active_n) {
        output[lane * m + row0] = sum.s0;
        output[lane * m + row0 + 1] = sum.s1;
    }
}

inline int8 load_b_frag_n64(__global const char * values, int kg, int column0, uint lane) {
    const int base = kg * 32 * 64 + column0 * 4 + lane * 4;
    int8 result;
    result.s0 = as_int(vload4(0, values + base + 0 * 64 * 4));
    result.s1 = as_int(vload4(0, values + base + 1 * 64 * 4));
    result.s2 = as_int(vload4(0, values + base + 2 * 64 * 4));
    result.s3 = as_int(vload4(0, values + base + 3 * 64 * 4));
    result.s4 = as_int(vload4(0, values + base + 4 * 64 * 4));
    result.s5 = as_int(vload4(0, values + base + 5 * 64 * 4));
    result.s6 = as_int(vload4(0, values + base + 6 * 64 * 4));
    result.s7 = as_int(vload4(0, values + base + 7 * 64 * 4));
    return result;
}

// Exact Q8 N=64 fallback for shapes that do not satisfy the native GEMM's
// M alignment. The FP16-DPAS GEMM is the primary N=64 route.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void opencl_q6k_q8_m4n64(
        __global const uchar * weights,
        __global const char * activation_values,
        __global const half * activation_d,
        __global float * output,
        int m,
        int groups,
        int blocks,
        int total_weight_blocks,
        int reordered) {
    const uint lane = get_sub_group_local_id();
    const int row_tiles = m / 4;
    const int subgroup = get_group_id(0);
    const int row0 = (subgroup % row_tiles) * 4;
    const int column0 = (subgroup / row_tiles) * 64;
    float4 sum[4] = {
        (float4)(0.0f), (float4)(0.0f), (float4)(0.0f), (float4)(0.0f)
    };

    for (int kg = 0; kg < groups; ++kg) {
        const int block_index = kg / 8;
        const int group = kg & 7;
        __global const uchar * ql[4];
        __global const uchar * qh[4];
        __global const char * scales[4];
        __global const half * wd[4];
        #pragma unroll
        for (int row = 0; row < 4; ++row) {
            const int weight_index = (row0 + row) * blocks + block_index;
            if (reordered) {
                ql[row] = weights + weight_index * 128;
                qh[row] = weights + total_weight_blocks * 128 + weight_index * 64;
                scales[row] = (__global const char *)(weights + total_weight_blocks * 192 + weight_index * 16);
                wd[row] = (__global const half *)(weights + total_weight_blocks * 208 + weight_index * 2);
            } else {
                __global const uchar * weight = weights + weight_index * 210;
                ql[row] = weight;
                qh[row] = weight + 128;
                scales[row] = (__global const char *)(weight + 192);
                wd[row] = (__global const half *)(weight + 208);
            }
        }

        const short pair0 = load_q6_pair(ql[0], qh[0], group, lane);
        const short pair1 = load_q6_pair(ql[1], qh[1], group, lane);
        const short pair2 = load_q6_pair(ql[2], qh[2], group, lane);
        const short pair3 = load_q6_pair(ql[3], qh[3], group, lane);
        // Physical DPAS rows represent scaled K=16 halves. Four output rows
        // therefore occupy the full M=8 operand, still with half-zero rows.
        const short8 a_frag = lane < 8
            ? (short8)(pair0, 0, pair1, 0, pair2, 0, pair3, 0)
            : (short8)(0, pair0, 0, pair1, 0, pair2, 0, pair3);
        const float8 ws = (float8)(
            convert_float(*wd[0]) * convert_float(scales[0][2 * group]),
            convert_float(*wd[0]) * convert_float(scales[0][2 * group + 1]),
            convert_float(*wd[1]) * convert_float(scales[1][2 * group]),
            convert_float(*wd[1]) * convert_float(scales[1][2 * group + 1]),
            convert_float(*wd[2]) * convert_float(scales[2][2 * group]),
            convert_float(*wd[2]) * convert_float(scales[2][2 * group + 1]),
            convert_float(*wd[3]) * convert_float(scales[3][2 * group]),
            convert_float(*wd[3]) * convert_float(scales[3][2 * group + 1]));

        #pragma unroll
        for (int tile = 0; tile < 4; ++tile) {
            const int tile_column = column0 + tile * 16;
            const int8 b_frag = load_b_frag_n64(activation_values, kg, tile_column, lane);
            const int8 dot = intel_sub_group_i8_i8_matrix_mad_k32(a_frag, b_frag, (int8)(0));
            const float activation_scale = convert_float(activation_d[kg * 64 + tile_column + lane]);
            sum[tile].s0 += activation_scale * (ws.s0 * convert_float(dot.s0) + ws.s1 * convert_float(dot.s1));
            sum[tile].s1 += activation_scale * (ws.s2 * convert_float(dot.s2) + ws.s3 * convert_float(dot.s3));
            sum[tile].s2 += activation_scale * (ws.s4 * convert_float(dot.s4) + ws.s5 * convert_float(dot.s5));
            sum[tile].s3 += activation_scale * (ws.s6 * convert_float(dot.s6) + ws.s7 * convert_float(dot.s7));
        }
    }

    #pragma unroll
    for (int tile = 0; tile < 4; ++tile) {
        const int column = column0 + tile * 16 + lane;
        output[column * m + row0] = sum[tile].s0;
        output[column * m + row0 + 1] = sum[tile].s1;
        output[column * m + row0 + 2] = sum[tile].s2;
        output[column * m + row0 + 3] = sum[tile].s3;
    }
}

inline ushort load_q5_pair(
        __global const uchar * qs,
        __global const uchar * qh,
        int group,
        uint lane) {
    const int element = 2 * lane;
    uchar q0 = qs[(group / 2) * 32 + element];
    uchar q1 = qs[(group / 2) * 32 + element + 1];
    if ((group & 1) == 0) {
        q0 &= 15;
        q1 &= 15;
    } else {
        q0 >>= 4;
        q1 >>= 4;
    }
    q0 |= ((qh[element] >> group) & 1) << 4;
    q1 |= ((qh[element + 1] >> group) & 1) << 4;
    return as_ushort((uchar2)(q0, q1));
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void opencl_q5k_q8_m2n16(
        __global const uchar * weights,
        __global const char * activation_values,
        __global const half * activation_d,
        __global const half * activation_s,
        __global float * output,
        int active_n,
        int m,
        int groups,
        int blocks,
        int total_weight_blocks,
        int reordered) {
    const uint lane = get_sub_group_local_id();
    const int row0 = get_group_id(0) * 2;
    float2 sum = (float2)(0.0f);

    for (int kg = 0; kg < groups; ++kg) {
        const int block_index = kg / 8;
        const int group = kg & 7;
        const int weight_index0 = row0 * blocks + block_index;
        const int weight_index1 = (row0 + 1) * blocks + block_index;

        __global const uchar * qs0;
        __global const uchar * qh0;
        __global const uchar * scales0;
        __global const half * dm0;
        __global const uchar * qs1;
        __global const uchar * qh1;
        __global const uchar * scales1;
        __global const half * dm1;
        if (reordered) {
            qs0 = weights + weight_index0 * 128;
            qh0 = weights + total_weight_blocks * 128 + weight_index0 * 32;
            scales0 = weights + total_weight_blocks * 160 + weight_index0 * 12;
            dm0 = (__global const half *)(weights + total_weight_blocks * 172 + weight_index0 * 4);
            qs1 = weights + weight_index1 * 128;
            qh1 = weights + total_weight_blocks * 128 + weight_index1 * 32;
            scales1 = weights + total_weight_blocks * 160 + weight_index1 * 12;
            dm1 = (__global const half *)(weights + total_weight_blocks * 172 + weight_index1 * 4);
        } else {
            __global const uchar * weight0 = weights + weight_index0 * 176;
            __global const uchar * weight1 = weights + weight_index1 * 176;
            dm0 = (__global const half *)weight0;
            scales0 = weight0 + 4;
            qh0 = weight0 + 16;
            qs0 = weight0 + 48;
            dm1 = (__global const half *)weight1;
            scales1 = weight1 + 4;
            qh1 = weight1 + 16;
            qs1 = weight1 + 48;
        }

        const ushort2 a_frag = (ushort2)(
            load_q5_pair(qs0, qh0, group, lane),
            load_q5_pair(qs1, qh1, group, lane));
        const int8 b_frag = load_b_frag(activation_values, kg, lane);
        const int2 dot = intel_sub_group_u8_i8_matrix_mad_k32(a_frag, b_frag, (int2)(0));
        const float d = convert_float(activation_d[kg * 16 + lane]);
        const float s = convert_float(activation_s[kg * 16 + lane]);
        sum.s0 += convert_float(dm0[0]) * convert_float(qk_scale(scales0, group)) * d * convert_float(dot.s0)
                - convert_float(dm0[1]) * convert_float(qk_minimum(scales0, group)) * s;
        sum.s1 += convert_float(dm1[0]) * convert_float(qk_scale(scales1, group)) * d * convert_float(dot.s1)
                - convert_float(dm1[1]) * convert_float(qk_minimum(scales1, group)) * s;
    }

    if ((int)lane < active_n) {
        output[lane * m + row0] = sum.s0;
        output[lane * m + row0 + 1] = sum.s1;
    }
}


)CLC";

// Q4_K GEMM-native in-place layout (same 144 bytes/block as Q4_K):
// [row-tile/block quant tiles][row-tile/block transposed scales][d planes].
// Each row tile contains eight output rows. This lets each OpenCL vector load
// fetch one value for all eight DPAS rows without expanding the tensor.
static constexpr const char * xmx_gemm_opencl_source = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable

inline int8 xmx_gemm_load_b(__global const half * b, int col, int kk, int k, int n) {
    return col < n ? vload8(0, (__global const int *)(b + col*k + kk)) : (int8)(0);
}
inline void xmx_gemm_store(__global float * c, float8 v, int col, int row, int m, int n) {
    if (col >= n) return;
    c[col*m+row+0]=v.s0; c[col*m+row+1]=v.s1; c[col*m+row+2]=v.s2; c[col*m+row+3]=v.s3;
    c[col*m+row+4]=v.s4; c[col*m+row+5]=v.s5; c[col*m+row+6]=v.s6; c[col*m+row+7]=v.s7;
}
inline uchar8 qk_gemm_scale(__global const uchar * s, int tb, int group) {
    const int base=tb*96;
    if(group<4) return vload8(0,s+base+group*8)&(uchar8)(63);
    return (vload8(0,s+base+(group+4)*8)&(uchar8)(15))|((vload8(0,s+base+(group-4)*8)>>6)<<4);
}
inline uchar8 qk_gemm_min(__global const uchar * s, int tb, int group) {
    const int base=tb*96;
    if(group<4) return vload8(0,s+base+(group+4)*8)&(uchar8)(63);
    return (vload8(0,s+base+(group+4)*8)>>4)|((vload8(0,s+base+group*8)>>6)<<4);
}
inline void q4_gemm_decode(
        __global const uchar * qs, __global const uchar * scales, __global const half * dm,
        int tb, int group, int lane, short8 * low, short8 * high) {
    const uchar8 p=vload8(0,qs+tb*1024+(group*16+lane)*8);
    const float8 d=convert_float8(vload8(0,dm+tb*16));
    const float8 dmin=convert_float8(vload8(0,dm+tb*16+8));
    const float8 scale=convert_float8(qk_gemm_scale(scales,tb,group));
    const float8 off=dmin*convert_float8(qk_gemm_min(scales,tb,group));
    *low=as_short8(convert_half8_rte(d*scale*convert_float8(p&(uchar8)(15))-off));
    *high=as_short8(convert_half8_rte(d*scale*convert_float8(p>>4)-off));
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void opencl_q4k_to_f16(
        __global const uchar * weights, __global half * output, int m, int k, int blocks) {
    const int lane=get_sub_group_local_id();
    const int index=get_group_id(0), row_tile=index/blocks, block=index%blocks;
    const int total_tb=(m/8)*blocks;
    __global const uchar *qs=weights;
    __global const uchar *scales=weights+total_tb*1024;
    __global const half *dm=(__global const half *)(weights+total_tb*1120);
    #pragma unroll
    for(int group=0;group<8;++group){
        short8 low,high;q4_gemm_decode(qs,scales,dm,index,group,lane,&low,&high);
        #pragma unroll
        for(int r=0;r<8;++r){
            output[(row_tile*8+r)*k+block*256+group*32+lane]=as_half(low[r]);
            output[(row_tile*8+r)*k+block*256+group*32+lane+16]=as_half(high[r]);
        }
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void opencl_q4k_f16_gemm_m32n64(
        __global const uchar * weights, __global const half * activations,
        __global float * output, int m, int n, int k, int blocks) {
    const int lane=get_sub_group_local_id(), sg=get_sub_group_id();
    const int tiles_n=(n+63)/64, wg=get_group_id(0);
    const int row=(wg/tiles_n)*32+(sg>>1)*8;
    const int col=(wg%tiles_n)*64+(sg&1)*32+lane;
    const int row_tile=row/8;
    const int total_tb=(m/8)*blocks;
    __global const uchar *qs=weights;
    __global const uchar *scales=weights+total_tb*1024;
    __global const half *dm=(__global const half *)(weights+total_tb*1120);
    float8 c0=(float8)(0.0f),c1=(float8)(0.0f);
    for(int block=0;block<blocks;++block){
        #pragma unroll
        for(int group=0;group<8;++group){
            const int kk=block*256+group*32;
            short8 a0,a1;q4_gemm_decode(qs,scales,dm,row_tile*blocks+block,group,lane,&a0,&a1);
            c0=intel_sub_group_f16_f16_matrix_mad_k16(a0,xmx_gemm_load_b(activations,col,kk,k,n),c0);
            c0=intel_sub_group_f16_f16_matrix_mad_k16(a1,xmx_gemm_load_b(activations,col,kk+16,k,n),c0);
            c1=intel_sub_group_f16_f16_matrix_mad_k16(a0,xmx_gemm_load_b(activations,col+16,kk,k,n),c1);
            c1=intel_sub_group_f16_f16_matrix_mad_k16(a1,xmx_gemm_load_b(activations,col+16,kk+16,k,n),c1);
        }
    }
    xmx_gemm_store(output,c0,col,row,m,n);xmx_gemm_store(output,c1,col+16,row,m,n);
}

inline uchar8 q5_gemm_value(
        __global const uchar *qs,__global const uchar *qh,int tb,int group,int lane,int high){
    const uchar8 p=vload8(0,qs+tb*1024+(group*16+lane)*8);
    const int position=lane+(high?16:0);
    const uchar8 hb=(vload8(0,qh+tb*256+position*8)>>group)&(uchar8)(1);
    return (high?p>>4:p&(uchar8)(15))|(hb<<4);
}
inline void q5_gemm_decode(
        __global const uchar *qs,__global const uchar *qh,__global const uchar *scales,
        __global const half *dm,int tb,int group,int lane,short8 *low,short8 *high){
    const float8 d=convert_float8(vload8(0,dm+tb*16));
    const float8 dmin=convert_float8(vload8(0,dm+tb*16+8));
    const float8 scale=convert_float8(qk_gemm_scale(scales,tb,group));
    const float8 off=dmin*convert_float8(qk_gemm_min(scales,tb,group));
    *low=as_short8(convert_half8_rte(d*scale*convert_float8(q5_gemm_value(qs,qh,tb,group,lane,0))-off));
    *high=as_short8(convert_half8_rte(d*scale*convert_float8(q5_gemm_value(qs,qh,tb,group,lane,1))-off));
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void opencl_q5k_to_f16(
        __global const uchar *weights,__global half *output,int m,int k,int blocks){
    const int lane=get_sub_group_local_id(),index=get_group_id(0);
    const int row_tile=index/blocks,block=index%blocks,total_tb=(m/8)*blocks;
    __global const uchar *qs=weights,*qh=weights+total_tb*1024;
    __global const uchar *scales=weights+total_tb*1280;
    __global const half *dm=(__global const half *)(weights+total_tb*1376);
    #pragma unroll
    for(int group=0;group<8;++group){short8 low,high;
        q5_gemm_decode(qs,qh,scales,dm,index,group,lane,&low,&high);
        #pragma unroll
        for(int r=0;r<8;++r){
            output[(row_tile*8+r)*k+block*256+group*32+lane]=as_half(low[r]);
            output[(row_tile*8+r)*k+block*256+group*32+lane+16]=as_half(high[r]);
        }
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void opencl_q5k_f16_gemm_m32n64(
        __global const uchar *weights,__global const half *activations,
        __global float *output,int m,int n,int k,int blocks){
    const int lane=get_sub_group_local_id(),sg=get_sub_group_id();
    const int tiles_n=(n+63)/64,wg=get_group_id(0);
    const int row=(wg/tiles_n)*32+(sg>>1)*8;
    const int col=(wg%tiles_n)*64+(sg&1)*32+lane,row_tile=row/8;
    const int total_tb=(m/8)*blocks;
    __global const uchar *qs=weights,*qh=weights+total_tb*1024;
    __global const uchar *scales=weights+total_tb*1280;
    __global const half *dm=(__global const half *)(weights+total_tb*1376);
    float8 c0=(float8)(0.0f),c1=(float8)(0.0f);
    for(int block=0;block<blocks;++block){
        #pragma unroll
        for(int group=0;group<8;++group){const int kk=block*256+group*32;short8 a0,a1;
            q5_gemm_decode(qs,qh,scales,dm,row_tile*blocks+block,group,lane,&a0,&a1);
            c0=intel_sub_group_f16_f16_matrix_mad_k16(a0,xmx_gemm_load_b(activations,col,kk,k,n),c0);
            c0=intel_sub_group_f16_f16_matrix_mad_k16(a1,xmx_gemm_load_b(activations,col,kk+16,k,n),c0);
            c1=intel_sub_group_f16_f16_matrix_mad_k16(a0,xmx_gemm_load_b(activations,col+16,kk,k,n),c1);
            c1=intel_sub_group_f16_f16_matrix_mad_k16(a1,xmx_gemm_load_b(activations,col+16,kk+16,k,n),c1);
        }
    }
    xmx_gemm_store(output,c0,col,row,m,n);xmx_gemm_store(output,c1,col+16,row,m,n);
}

inline char8 q6_gemm_value(
        __global const uchar *ql,__global const uchar *qh,int tb,int group,int position){
    const int index=group*32+position,chunk=index>>7,pos=index&127;
    const int quadrant=pos>>5,element=pos&31;
    const int ql_index=chunk*64+(quadrant&1)*32+element;
    const int qh_index=chunk*32+element;
    const uchar8 low=(vload8(0,ql+tb*1024+ql_index*8)>>(quadrant>>1)*4)&(uchar8)(15);
    const uchar8 high=(vload8(0,qh+tb*512+qh_index*8)>>(quadrant*2))&(uchar8)(3);
    return convert_char8(low|(high<<4))-(char8)(32);
}
inline void q6_gemm_decode(
        __global const uchar *ql,__global const uchar *qh,__global const char *scales,
        __global const half *d,int tb,int group,int lane,short8 *low,short8 *high){
    const float8 wd=convert_float8(vload8(0,d+tb*8));
    const float8 s0=convert_float8(vload8(0,scales+tb*128+(2*group)*8));
    const float8 s1=convert_float8(vload8(0,scales+tb*128+(2*group+1)*8));
    *low=as_short8(convert_half8_rte(wd*s0*convert_float8(q6_gemm_value(ql,qh,tb,group,lane))));
    *high=as_short8(convert_half8_rte(wd*s1*convert_float8(q6_gemm_value(ql,qh,tb,group,lane+16))));
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void opencl_q6k_to_f16(
        __global const uchar *weights,__global half *output,int m,int k,int blocks){
    const int lane=get_sub_group_local_id(),index=get_group_id(0);
    const int row_tile=index/blocks,block=index%blocks,total_tb=(m/8)*blocks;
    __global const uchar *ql=weights,*qh=weights+total_tb*1024;
    __global const char *scales=(__global const char *)(weights+total_tb*1536);
    __global const half *d=(__global const half *)(weights+total_tb*1664);
    #pragma unroll
    for(int group=0;group<8;++group){short8 low,high;
        q6_gemm_decode(ql,qh,scales,d,index,group,lane,&low,&high);
        #pragma unroll
        for(int r=0;r<8;++r){
            output[(row_tile*8+r)*k+block*256+group*32+lane]=as_half(low[r]);
            output[(row_tile*8+r)*k+block*256+group*32+lane+16]=as_half(high[r]);
        }
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(128,1,1)))
__kernel void opencl_q6k_f16_gemm_m32n64(
        __global const uchar *weights,__global const half *activations,
        __global float *output,int m,int n,int k,int blocks){
    const int lane=get_sub_group_local_id(),sg=get_sub_group_id();
    const int tiles_n=(n+63)/64,wg=get_group_id(0);
    const int row=(wg/tiles_n)*32+(sg>>1)*8;
    const int col=(wg%tiles_n)*64+(sg&1)*32+lane,row_tile=row/8;
    const int total_tb=(m/8)*blocks;
    __global const uchar *ql=weights,*qh=weights+total_tb*1024;
    __global const char *scales=(__global const char *)(weights+total_tb*1536);
    __global const half *d=(__global const half *)(weights+total_tb*1664);
    float8 c0=(float8)(0.0f),c1=(float8)(0.0f);
    for(int block=0;block<blocks;++block){
        #pragma unroll
        for(int group=0;group<8;++group){
            const int kk=block*256+group*32;short8 a0,a1;
            q6_gemm_decode(ql,qh,scales,d,row_tile*blocks+block,group,lane,&a0,&a1);
            c0=intel_sub_group_f16_f16_matrix_mad_k16(a0,xmx_gemm_load_b(activations,col,kk,k,n),c0);
            c0=intel_sub_group_f16_f16_matrix_mad_k16(a1,xmx_gemm_load_b(activations,col,kk+16,k,n),c0);
            c1=intel_sub_group_f16_f16_matrix_mad_k16(a0,xmx_gemm_load_b(activations,col+16,kk,k,n),c1);
            c1=intel_sub_group_f16_f16_matrix_mad_k16(a1,xmx_gemm_load_b(activations,col+16,kk+16,k,n),c1);
        }
    }
    xmx_gemm_store(output,c0,col,row,m,n);xmx_gemm_store(output,c1,col+16,row,m,n);
}
)CLC";

using xmx_executable_bundle = sycl::kernel_bundle<sycl::bundle_state::executable>;

struct xmx_opencl_kernel_entry {
    std::optional<sycl::context> context;
    std::optional<xmx_executable_bundle> bundle;
    std::optional<sycl::kernel> kernel_q4_small;
    std::optional<sycl::kernel> kernel_q5_small;
    std::optional<sycl::kernel> kernel_q6_small;
    std::optional<sycl::kernel> kernel_q6_n64;
    bool failed = false;
};

std::mutex xmx_opencl_kernel_mutex;
std::array<xmx_opencl_kernel_entry, GGML_SYCL_MAX_DEVICES> xmx_opencl_kernels;
std::array<std::atomic<bool>, GGML_SYCL_MAX_DEVICES> xmx_opencl_kernel_ready = {};

struct xmx_gemm_kernel_entry {
    std::optional<sycl::context> context;
    std::optional<xmx_executable_bundle> bundle;
    std::optional<sycl::kernel> kernel;
    std::optional<sycl::kernel> q5_kernel;
    std::optional<sycl::kernel> q6_kernel;
    std::optional<sycl::kernel> dequant_kernel;
    std::optional<sycl::kernel> q5_dequant_kernel;
    std::optional<sycl::kernel> q6_dequant_kernel;
    bool failed = false;
};
std::mutex xmx_gemm_kernel_mutex;
std::array<xmx_gemm_kernel_entry, GGML_SYCL_MAX_DEVICES> xmx_gemm_kernels;
std::array<std::atomic<bool>, GGML_SYCL_MAX_DEVICES> xmx_gemm_kernel_ready = {};

bool xmx_ensure_opencl_kernels(ggml_backend_sycl_context & ctx) {
    // All backend instances use dpct's per-device default queue/context. Most
    // processes initialize one backend, but routers and tests may initialize
    // the same device more than once. Avoid even taking the initialization
    // mutex after the immutable cache entry has been published.
    if (xmx_opencl_kernel_ready[ctx.device].load(std::memory_order_acquire)) {
        return true;
    }

    queue_ptr queue = ctx.stream();
    const sycl::context context = queue->get_context();
    sycl::device device = queue->get_device();
    std::lock_guard<std::mutex> lock(xmx_opencl_kernel_mutex);
    xmx_opencl_kernel_entry & entry = xmx_opencl_kernels[ctx.device];

    if (entry.context && *entry.context != context) {
        xmx_opencl_kernel_ready[ctx.device].store(false, std::memory_order_release);
        entry = {};
    }
    if (entry.kernel_q4_small && entry.kernel_q5_small && entry.kernel_q6_small && entry.kernel_q6_n64) {
        xmx_opencl_kernel_ready[ctx.device].store(true, std::memory_order_release);
        return true;
    }
    if (entry.failed || !device.ext_oneapi_can_build(syclex::source_language::opencl)) {
        entry.failed = true;
        return false;
    }

    try {
        auto source_bundle = syclex::create_kernel_bundle_from_source(
            context, syclex::source_language::opencl, std::string(xmx_opencl_source));
        std::string build_log;
        auto executable_bundle = syclex::build(
            source_bundle, std::vector<sycl::device>{device},
            syclex::properties{syclex::save_log{&build_log}});
        if (!build_log.empty()) {
            GGML_LOG_DEBUG("SYCL XMX Q4_K/Q5_K/Q6_K OpenCL build log: %s\n", build_log.c_str());
        }
        entry.context = context;
        entry.bundle = std::move(executable_bundle);
        entry.kernel_q4_small = entry.bundle->ext_oneapi_get_kernel("opencl_q4k_q8_m2n16");
        entry.kernel_q5_small = entry.bundle->ext_oneapi_get_kernel("opencl_q5k_q8_m2n16");
        entry.kernel_q6_small = entry.bundle->ext_oneapi_get_kernel("opencl_q6k_q8_m2n16");
        entry.kernel_q6_n64 = entry.bundle->ext_oneapi_get_kernel("opencl_q6k_q8_m4n64");
        xmx_opencl_kernel_ready[ctx.device].store(true, std::memory_order_release);
        GGML_LOG_INFO("SYCL XMX Q4_K/Q5_K/Q6_K OpenCL kernels compiled for %s\n",
                      device.get_info<sycl::info::device::name>().c_str());
        return true;
    } catch (const sycl::exception & exception) {
        entry.failed = true;
        GGML_LOG_WARN("SYCL XMX Q4_K/Q5_K/Q6_K OpenCL kernels unavailable, using fallback: %s\n", exception.what());
        return false;
    }
}

bool xmx_ensure_gemm_kernels(ggml_backend_sycl_context & ctx) {
    if (xmx_gemm_kernel_ready[ctx.device].load(std::memory_order_acquire)) return true;
    queue_ptr queue=ctx.stream(); const sycl::context context=queue->get_context(); sycl::device device=queue->get_device();
    std::lock_guard<std::mutex> lock(xmx_gemm_kernel_mutex);
    auto &entry=xmx_gemm_kernels[ctx.device];
    if(entry.context&&*entry.context!=context){xmx_gemm_kernel_ready[ctx.device].store(false,std::memory_order_release);entry={};}
    if(entry.kernel){xmx_gemm_kernel_ready[ctx.device].store(true,std::memory_order_release);return true;}
    if(entry.failed||!device.ext_oneapi_can_build(syclex::source_language::opencl)){entry.failed=true;return false;}
    try{
        auto sb=syclex::create_kernel_bundle_from_source(context,syclex::source_language::opencl,std::string(xmx_gemm_opencl_source));
        std::string log;auto eb=syclex::build(sb,std::vector<sycl::device>{device},syclex::properties{syclex::save_log{&log}});
        if(!log.empty())GGML_LOG_DEBUG("SYCL XMX Q4_K/Q5_K/Q6_K GEMM OpenCL build log: %s\n",log.c_str());
        entry.context=context;entry.bundle=std::move(eb);entry.kernel=entry.bundle->ext_oneapi_get_kernel("opencl_q4k_f16_gemm_m32n64");
        entry.q5_kernel=entry.bundle->ext_oneapi_get_kernel("opencl_q5k_f16_gemm_m32n64");
        entry.q6_kernel=entry.bundle->ext_oneapi_get_kernel("opencl_q6k_f16_gemm_m32n64");
        entry.dequant_kernel=entry.bundle->ext_oneapi_get_kernel("opencl_q4k_to_f16");
        entry.q5_dequant_kernel=entry.bundle->ext_oneapi_get_kernel("opencl_q5k_to_f16");
        entry.q6_dequant_kernel=entry.bundle->ext_oneapi_get_kernel("opencl_q6k_to_f16");
        xmx_gemm_kernel_ready[ctx.device].store(true,std::memory_order_release);
        GGML_LOG_INFO("SYCL XMX Q4_K/Q5_K/Q6_K GEMM OpenCL kernels compiled for %s\n",device.get_info<sycl::info::device::name>().c_str());return true;
    }catch(const sycl::exception&e){entry.failed=true;GGML_LOG_WARN("SYCL XMX Q4_K/Q5_K/Q6_K GEMM unavailable, using fallback: %s\n",e.what());return false;}
}

bool xmx_opencl_kernels_are_ready(int device) {
    return xmx_opencl_kernel_ready[device].load(std::memory_order_acquire);
}

const xmx_opencl_kernel_entry & xmx_get_opencl_entry(ggml_backend_sycl_context & ctx) {
    GGML_ASSERT(xmx_opencl_kernels_are_ready(ctx.device));
    return xmx_opencl_kernels[ctx.device];
}

const sycl::kernel & xmx_get_q4_K_kernel(ggml_backend_sycl_context & ctx) {
    const xmx_opencl_kernel_entry & entry = xmx_get_opencl_entry(ctx);
    GGML_ASSERT(entry.kernel_q4_small.has_value());
    return *entry.kernel_q4_small;
}

const sycl::kernel & xmx_get_q5_K_kernel(ggml_backend_sycl_context & ctx) {
    const xmx_opencl_kernel_entry & entry = xmx_get_opencl_entry(ctx);
    GGML_ASSERT(entry.kernel_q5_small.has_value());
    return *entry.kernel_q5_small;
}

const sycl::kernel & xmx_get_q6_K_kernel(ggml_backend_sycl_context & ctx, bool n64) {
    // Backend initialization publishes the immutable entry with a release
    // store. Avoid taking the compilation mutex on every matmul.
    const xmx_opencl_kernel_entry & entry = xmx_get_opencl_entry(ctx);
    GGML_ASSERT(entry.kernel_q6_small.has_value() && entry.kernel_q6_n64.has_value());
    return n64 ? *entry.kernel_q6_n64 : *entry.kernel_q6_small;
}

template<bool store_sum>
void xmx_quantize_q8_vnni(
        queue_ptr queue,
        const float * input,
        int8_t * values,
        sycl::half * d_values,
        sycl::half * s_values,
        int active_n,
        int k,
        int groups,
        int n_stride) {
    constexpr int workgroup_size = 16;
    const size_t values_per_group = xmx_tile_k * static_cast<size_t>(n_stride);
    const sycl::nd_range<1> range(
        sycl::range<1>(groups * n_stride * workgroup_size),
        sycl::range<1>(workgroup_size));

    queue->parallel_for(range, [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
        const int block = item.get_group(0);
        const int column = block / groups;
        const int kg = block % groups;
        const int lane = item.get_local_linear_id();
        const int kk = lane * 2;

        float value0 = 0.0f;
        float value1 = 0.0f;
        if (column < active_n) {
            value0 = input[column * k + kg * xmx_tile_k + kk];
            value1 = input[column * k + kg * xmx_tile_k + kk + 1];
        }

        float amax = sycl::fmax(sycl::fabs(value0), sycl::fabs(value1));
        amax = sycl::reduce_over_group(item.get_sub_group(), amax, sycl::maximum<float>());
        const float d = amax / 127.0f;
        const float inverse_d = d == 0.0f ? 0.0f : 1.0f / d;
        const int8_t quant0 = static_cast<int8_t>(sycl::round(value0 * inverse_d));
        const int8_t quant1 = static_cast<int8_t>(sycl::round(value1 * inverse_d));
        const size_t vnni_index = kg * values_per_group +
            (kk / 4) * static_cast<size_t>(n_stride) * 4 + column * 4 + kk % 4;
        values[vnni_index] = quant0;
        values[vnni_index + 1] = quant1;

        int quant_sum = 0;
        if constexpr (store_sum) {
            quant_sum = static_cast<int>(quant0) + static_cast<int>(quant1);
            quant_sum = sycl::reduce_over_group(item.get_sub_group(), quant_sum, sycl::plus<int>());
        }
        if (lane == 0) {
            d_values[kg * n_stride + column] = sycl::half(d);
            if constexpr (store_sum) {
                s_values[kg * n_stride + column] = sycl::half(d * quant_sum);
            }
        }
    });
}

} // namespace

void ggml_sycl_xmx_init(ggml_backend_sycl_context & ctx) {
    const gpu_arch arch = ggml_sycl_info().devices[ctx.device].hw_info.arch;
    if (arch == gpu_arch::intel_gpu_bmg_g31) {
        (void) xmx_ensure_opencl_kernels(ctx);
        (void) xmx_ensure_gemm_kernels(ctx);
    }
}

bool ggml_sycl_can_use_mul_mat_q4_K_gemm_xmx(
        const ggml_backend_sycl_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * dst) {
    const auto * extra=static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
    return ggml_sycl_info().devices[ctx.device].hw_info.arch==gpu_arch::intel_gpu_bmg_g31 &&
           src0->type==GGML_TYPE_Q4_K && src1->type==GGML_TYPE_F32 && dst->type==GGML_TYPE_F32 &&
           extra && extra->optimized_feature.q4_xmx && ggml_is_contiguous(src0) && ggml_is_contiguous(src1) &&
           ggml_is_contiguous(dst) && src0->ne[0]%QK_K==0 && src0->ne[1]%32==0 &&
           src1->ne[1]>=1 && src1->ne[1]<=128 && src0->ne[2]==1 && src0->ne[3]==1 &&
           src1->ne[2]==1 && src1->ne[3]==1 && dst->ne[2]==1 && dst->ne[3]==1 &&
           xmx_gemm_kernel_ready[ctx.device].load(std::memory_order_acquire);
}

void ggml_sycl_mul_mat_q4_K_gemm_xmx(
        ggml_backend_sycl_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {
    GGML_ASSERT(ggml_sycl_can_use_mul_mat_q4_K_gemm_xmx(ctx,src0,src1,dst));
    const int m=src0->ne[1],n=src1->ne[1],k=src0->ne[0],blocks=k/QK_K;
    ggml_sycl_pool_alloc<sycl::half> activation(ctx.pool(),static_cast<size_t>(n)*k);
    const to_fp16_sycl_t to_fp16=ggml_get_to_fp16_sycl(src1->type,dst);
    to_fp16(src1->data,activation.get(),static_cast<int64_t>(n)*k,ctx.stream());
    const size_t workgroups=static_cast<size_t>(m/32)*((n+63)/64);
    const sycl::kernel &kernel=*xmx_gemm_kernels[ctx.device].kernel;
    ctx.stream()->submit([&](sycl::handler&h){h.set_arg(0,src0->data);h.set_arg(1,activation.get());h.set_arg(2,dst->data);
        h.set_arg(3,m);h.set_arg(4,n);h.set_arg(5,k);h.set_arg(6,blocks);
        h.parallel_for(sycl::nd_range<1>{sycl::range<1>{workgroups*128},sycl::range<1>{128}},kernel);});
}

void ggml_sycl_dequantize_q4_K_xmx_to_f16(
        ggml_backend_sycl_context &ctx,const void *weights,void *output,int m,int k){
    GGML_ASSERT(weights&&output&&m%8==0&&k%QK_K==0);
    const int blocks=k/QK_K;const size_t groups=static_cast<size_t>(m/8)*blocks;
    const sycl::kernel &kernel=*xmx_gemm_kernels[ctx.device].dequant_kernel;
    ctx.stream()->submit([&](sycl::handler&h){h.set_arg(0,weights);h.set_arg(1,output);h.set_arg(2,m);h.set_arg(3,k);h.set_arg(4,blocks);
        h.parallel_for(sycl::nd_range<1>{sycl::range<1>{groups*16},sycl::range<1>{16}},kernel);});
}

bool ggml_sycl_xmx_gemm_is_ready(const ggml_backend_sycl_context &ctx){
    return xmx_gemm_kernel_ready[ctx.device].load(std::memory_order_acquire);
}

bool ggml_sycl_can_use_mul_mat_q5_K_gemm_xmx(
        const ggml_backend_sycl_context &ctx,const ggml_tensor *src0,
        const ggml_tensor *src1,const ggml_tensor *dst){
    const auto *extra=static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
    return ggml_sycl_info().devices[ctx.device].hw_info.arch==gpu_arch::intel_gpu_bmg_g31&&
        src0->type==GGML_TYPE_Q5_K&&src1->type==GGML_TYPE_F32&&dst->type==GGML_TYPE_F32&&extra&&
        ggml_is_contiguous(src0)&&ggml_is_contiguous(src1)&&
        ggml_is_contiguous(dst)&&src0->ne[0]%QK_K==0&&src0->ne[1]>=xmx_q5_q6_min_m&&
        src0->ne[1]%32==0&&src1->ne[1]>=17&&src1->ne[1]<=64&&
        src0->ne[2]==1&&src0->ne[3]==1&&src1->ne[2]==1&&src1->ne[3]==1&&
        dst->ne[2]==1&&dst->ne[3]==1&&ggml_sycl_xmx_gemm_is_ready(ctx);
}

void ggml_sycl_mul_mat_q5_K_gemm_xmx(
        ggml_backend_sycl_context &ctx,const ggml_tensor *src0,
        const ggml_tensor *src1,ggml_tensor *dst){
    GGML_ASSERT(ggml_sycl_can_use_mul_mat_q5_K_gemm_xmx(ctx,src0,src1,dst));
    const int m=src0->ne[1],n=src1->ne[1],k=src0->ne[0],blocks=k/QK_K;
    ggml_sycl_pool_alloc<sycl::half> activation(ctx.pool(),static_cast<size_t>(n)*k);
    const to_fp16_sycl_t to_fp16=ggml_get_to_fp16_sycl(src1->type,dst);
    to_fp16(src1->data,activation.get(),static_cast<int64_t>(n)*k,ctx.stream());
    const size_t workgroups=static_cast<size_t>(m/32)*((n+63)/64);
    const sycl::kernel &kernel=*xmx_gemm_kernels[ctx.device].q5_kernel;
    ctx.stream()->submit([&](sycl::handler&h){h.set_arg(0,src0->data);h.set_arg(1,activation.get());h.set_arg(2,dst->data);
        h.set_arg(3,m);h.set_arg(4,n);h.set_arg(5,k);h.set_arg(6,blocks);
        h.parallel_for(sycl::nd_range<1>{sycl::range<1>{workgroups*128},sycl::range<1>{128}},kernel);});
}

void ggml_sycl_dequantize_q5_K_xmx_to_f16(
        ggml_backend_sycl_context &ctx,const void *weights,void *output,int m,int k){
    GGML_ASSERT(weights&&output&&m%8==0&&k%QK_K==0);
    const int blocks=k/QK_K;const size_t groups=static_cast<size_t>(m/8)*blocks;
    const sycl::kernel &kernel=*xmx_gemm_kernels[ctx.device].q5_dequant_kernel;
    ctx.stream()->submit([&](sycl::handler&h){h.set_arg(0,weights);h.set_arg(1,output);h.set_arg(2,m);h.set_arg(3,k);h.set_arg(4,blocks);
        h.parallel_for(sycl::nd_range<1>{sycl::range<1>{groups*16},sycl::range<1>{16}},kernel);});
}

bool ggml_sycl_can_use_mul_mat_q6_K_gemm_xmx(
        const ggml_backend_sycl_context &ctx,const ggml_tensor *src0,
        const ggml_tensor *src1,const ggml_tensor *dst){
    const auto *extra=static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
    return ggml_sycl_info().devices[ctx.device].hw_info.arch==gpu_arch::intel_gpu_bmg_g31 &&
        src0->type==GGML_TYPE_Q6_K&&src1->type==GGML_TYPE_F32&&dst->type==GGML_TYPE_F32&&extra&&
        ggml_is_contiguous(src0)&&ggml_is_contiguous(src1)&&
        ggml_is_contiguous(dst)&&src0->ne[0]%QK_K==0&&src0->ne[1]>=xmx_q5_q6_min_m&&
        src0->ne[1]%32==0&&src1->ne[1]>=17&&src1->ne[1]<=64&&
        src0->ne[2]==1&&src0->ne[3]==1&&src1->ne[2]==1&&src1->ne[3]==1&&
        dst->ne[2]==1&&dst->ne[3]==1&&ggml_sycl_xmx_gemm_is_ready(ctx);
}

void ggml_sycl_mul_mat_q6_K_gemm_xmx(
        ggml_backend_sycl_context &ctx,const ggml_tensor *src0,
        const ggml_tensor *src1,ggml_tensor *dst){
    GGML_ASSERT(ggml_sycl_can_use_mul_mat_q6_K_gemm_xmx(ctx,src0,src1,dst));
    const int m=src0->ne[1],n=src1->ne[1],k=src0->ne[0],blocks=k/QK_K;
    ggml_sycl_pool_alloc<sycl::half> activation(ctx.pool(),static_cast<size_t>(n)*k);
    const to_fp16_sycl_t to_fp16=ggml_get_to_fp16_sycl(src1->type,dst);
    to_fp16(src1->data,activation.get(),static_cast<int64_t>(n)*k,ctx.stream());
    const size_t workgroups=static_cast<size_t>(m/32)*((n+63)/64);
    const sycl::kernel &kernel=*xmx_gemm_kernels[ctx.device].q6_kernel;
    ctx.stream()->submit([&](sycl::handler&h){h.set_arg(0,src0->data);h.set_arg(1,activation.get());h.set_arg(2,dst->data);
        h.set_arg(3,m);h.set_arg(4,n);h.set_arg(5,k);h.set_arg(6,blocks);
        h.parallel_for(sycl::nd_range<1>{sycl::range<1>{workgroups*128},sycl::range<1>{128}},kernel);});
}

void ggml_sycl_dequantize_q6_K_xmx_to_f16(
        ggml_backend_sycl_context &ctx,const void *weights,void *output,int m,int k){
    GGML_ASSERT(weights&&output&&m%8==0&&k%QK_K==0);
    const int blocks=k/QK_K;const size_t groups=static_cast<size_t>(m/8)*blocks;
    const sycl::kernel &kernel=*xmx_gemm_kernels[ctx.device].q6_dequant_kernel;
    ctx.stream()->submit([&](sycl::handler&h){h.set_arg(0,weights);h.set_arg(1,output);h.set_arg(2,m);h.set_arg(3,k);h.set_arg(4,blocks);
        h.parallel_for(sycl::nd_range<1>{sycl::range<1>{groups*16},sycl::range<1>{16}},kernel);});
}

bool ggml_sycl_can_use_mul_mat_q4_K_xmx(
        const ggml_backend_sycl_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * dst) {
    const auto * extra = static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
    if (extra && extra->optimized_feature.q4_xmx) {
        return false;
    }
    const gpu_arch arch = ggml_sycl_info().devices[ctx.device].hw_info.arch;
    if (arch != gpu_arch::intel_gpu_bmg_g31) {
        return false;
    }

    return src0->type == GGML_TYPE_Q4_K && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst) &&
           src0->ne[0] == src1->ne[0] && src0->ne[0] % QK_K == 0 &&
           src0->ne[1] % xmx_tile_m == 0 &&
           src1->ne[1] >= 9 && src1->ne[1] <= xmx_tile_n &&
           src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 &&
           dst->ne[0] == src0->ne[1] && dst->ne[1] == src1->ne[1] && dst->ne[2] == 1 && dst->ne[3] == 1 &&
           xmx_opencl_kernels_are_ready(ctx.device);
}

bool ggml_sycl_can_use_mul_mat_q5_K_xmx(
        ggml_backend_sycl_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * dst) {
    const auto * extra = static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
    if (extra && extra->optimized_feature.q5_xmx) {
        return false;
    }
    const gpu_arch arch = ggml_sycl_info().devices[ctx.device].hw_info.arch;
    if (arch != gpu_arch::intel_gpu_bmg_g31) {
        return false;
    }

    return src0->type == GGML_TYPE_Q5_K && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst) &&
           src0->ne[0] == src1->ne[0] && src0->ne[0] % QK_K == 0 &&
           src0->ne[1] >= xmx_q5_q6_min_m && src0->ne[1] % xmx_tile_m == 0 &&
           src1->ne[1] >= 9 && src1->ne[1] <= xmx_tile_n &&
           src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 &&
           dst->ne[0] == src0->ne[1] && dst->ne[1] == src1->ne[1] && dst->ne[2] == 1 && dst->ne[3] == 1 &&
           xmx_opencl_kernels_are_ready(ctx.device);
}

bool ggml_sycl_can_use_mul_mat_q6_K_xmx(
        ggml_backend_sycl_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        const ggml_tensor * dst) {
    const auto * extra = static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
    if (extra && extra->optimized_feature.q6_xmx) {
        return false;
    }
    // The M crossover and runtime OpenCL kernel are validated only on BMG G31/B70.
    // Keep other Battlemage variants on MMQ until they have device-specific data.
    const gpu_arch arch = ggml_sycl_info().devices[ctx.device].hw_info.arch;
    if (arch != gpu_arch::intel_gpu_bmg_g31) {
        return false;
    }

    const int64_t n = src1->ne[1];
    const int64_t m_alignment = n == 64 ? 4 : xmx_tile_m;
    const bool n_supported = (n >= 9 && n <= xmx_tile_n) || n == 64;
    const bool shape_supported =
        src0->type == GGML_TYPE_Q6_K && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst) &&
        src0->ne[0] == src1->ne[0] && src0->ne[0] % QK_K == 0 &&
        src0->ne[1] >= xmx_q5_q6_min_m && src0->ne[1] % m_alignment == 0 && n_supported &&
        src0->ne[2] == 1 && src0->ne[3] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 &&
        dst->ne[0] == src0->ne[1] && dst->ne[1] == src1->ne[1] && dst->ne[2] == 1 && dst->ne[3] == 1;
    return shape_supported && xmx_opencl_kernels_are_ready(ctx.device);
}

void ggml_sycl_mul_mat_q5_K_xmx(
        ggml_backend_sycl_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {
    GGML_ASSERT(ggml_sycl_can_use_mul_mat_q5_K_xmx(ctx, src0, src1, dst));

    const int m = src0->ne[1];
    const int active_n = src1->ne[1];
    const int k = src0->ne[0];
    const int groups = k / xmx_tile_k;
    const int blocks = k / QK_K;
    constexpr int n_stride = xmx_tile_n;

    constexpr size_t values_per_group = xmx_tile_k * n_stride;
    const size_t values_size = groups * values_per_group;
    const size_t metadata_count = groups * n_stride;
    const size_t buffer_size = values_size + 2 * metadata_count * sizeof(sycl::half);

    ggml_sycl_pool_alloc<uint8_t> activation_alloc(ctx.pool(), buffer_size);
    uint8_t * activation_buffer = activation_alloc.get();
    int8_t * activation_values = reinterpret_cast<int8_t *>(activation_buffer);
    sycl::half * activation_d = reinterpret_cast<sycl::half *>(activation_buffer + values_size);
    sycl::half * activation_s = activation_d + metadata_count;

    const float * activation_input = static_cast<const float *>(src1->data);
    const uint8_t * weight_bytes = static_cast<const uint8_t *>(src0->data);
    const ggml_tensor_extra_gpu * weight_extra = static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
    const int weights_reordered = weight_extra && weight_extra->optimized_feature.reorder ? 1 : 0;
    const int total_weight_blocks = m * blocks;
    float * output = static_cast<float *>(dst->data);
    queue_ptr queue = ctx.stream();

    xmx_quantize_q8_vnni<true>(
        queue, activation_input, activation_values, activation_d, activation_s,
        active_n, k, groups, n_stride);

    const sycl::kernel & kernel = xmx_get_q5_K_kernel(ctx);
    queue->submit([&](sycl::handler & cgh) {
        cgh.set_arg(0, weight_bytes); cgh.set_arg(1, activation_values); cgh.set_arg(2, activation_d);
        cgh.set_arg(3, activation_s); cgh.set_arg(4, output); cgh.set_arg(5, active_n);
        cgh.set_arg(6, m); cgh.set_arg(7, groups); cgh.set_arg(8, blocks);
        cgh.set_arg(9, total_weight_blocks); cgh.set_arg(10, weights_reordered);
        cgh.parallel_for(sycl::nd_range<1>{sycl::range<1>{static_cast<size_t>(m / xmx_tile_m) * xmx_tile_n},
                                           sycl::range<1>{xmx_tile_n}}, kernel);
    });
}

void ggml_sycl_mul_mat_q6_K_xmx(
        ggml_backend_sycl_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_Q6_K && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    const int m = src0->ne[1];
    const int active_n = src1->ne[1];
    const int k = src0->ne[0];
    const int groups = k / xmx_tile_k;
    const int blocks = k / QK_K;
    const bool n64 = active_n == 64;
    const int n_stride = n64 ? 64 : xmx_tile_n;

    const size_t values_per_group = xmx_tile_k * static_cast<size_t>(n_stride);
    const size_t values_size = groups * values_per_group;
    const size_t metadata_count = groups * static_cast<size_t>(n_stride);
    const size_t buffer_size = values_size + metadata_count * sizeof(sycl::half);

    ggml_sycl_pool_alloc<uint8_t> activation_alloc(ctx.pool(), buffer_size);
    uint8_t * activation_buffer = activation_alloc.get();
    int8_t * activation_values = reinterpret_cast<int8_t *>(activation_buffer);
    sycl::half * activation_d = reinterpret_cast<sycl::half *>(activation_buffer + values_size);

    const float * activation_input = static_cast<const float *>(src1->data);
    const uint8_t * weight_bytes = static_cast<const uint8_t *>(src0->data);
    const ggml_tensor_extra_gpu * weight_extra = static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
    const int weights_reordered = weight_extra && weight_extra->optimized_feature.reorder ? 1 : 0;
    const int total_weight_blocks = m * blocks;
    float * output = static_cast<float *>(dst->data);
    queue_ptr queue = ctx.stream();

    xmx_quantize_q8_vnni<false>(
        queue, activation_input, activation_values, activation_d, nullptr,
        active_n, k, groups, n_stride);

    const sycl::kernel & kernel = xmx_get_q6_K_kernel(ctx, n64);
    queue->submit([&](sycl::handler & cgh) {
        cgh.set_arg(0, weight_bytes);
        cgh.set_arg(1, activation_values);
        cgh.set_arg(2, activation_d);
        cgh.set_arg(3, output);
        int arg = 4;
        if (!n64) {
            cgh.set_arg(arg++, active_n);
        }
        cgh.set_arg(arg++, m);
        cgh.set_arg(arg++, groups);
        cgh.set_arg(arg++, blocks);
        cgh.set_arg(arg++, total_weight_blocks);
        cgh.set_arg(arg++, weights_reordered);
        const size_t subgroups = n64 ? static_cast<size_t>(m / 4) : static_cast<size_t>(m / xmx_tile_m);
        cgh.parallel_for(
            sycl::nd_range<1>{sycl::range<1>{subgroups * xmx_tile_n}, sycl::range<1>{xmx_tile_n}},
            kernel);
    });
}

void ggml_sycl_mul_mat_q4_K_xmx(
        ggml_backend_sycl_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {
    GGML_ASSERT(ggml_sycl_can_use_mul_mat_q4_K_xmx(ctx, src0, src1, dst));

    const int m = src0->ne[1];
    const int active_n = src1->ne[1];
    const int k = src0->ne[0];
    const int groups = k / xmx_tile_k;
    const int blocks = k / QK_K;
    const ggml_tensor_extra_gpu * weight_extra = static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
    const bool weights_q4_xmx = weight_extra && weight_extra->optimized_feature.q4_xmx;
    const float * activation_input = static_cast<const float *>(src1->data);
    const uint8_t * weight_bytes = static_cast<const uint8_t *>(src0->data);
    float * output = static_cast<float *>(dst->data);
    queue_ptr queue = ctx.stream();
    constexpr size_t values_per_group = xmx_tile_k * xmx_tile_n;
    const size_t values_size = groups * values_per_group;
    const size_t metadata_count = groups * xmx_tile_n;
    const size_t buffer_size = values_size + 2 * metadata_count * sizeof(sycl::half);

    ggml_sycl_pool_alloc<uint8_t> activation_alloc(ctx.pool(), buffer_size);
    uint8_t * activation_buffer = activation_alloc.get();
    int8_t * activation_values = reinterpret_cast<int8_t *>(activation_buffer);
    sycl::half * activation_d = reinterpret_cast<sycl::half *>(activation_buffer + values_size);
    sycl::half * activation_s = activation_d + metadata_count;

    const bool weights_reordered = weight_extra && weight_extra->optimized_feature.reorder;
    const int total_weight_blocks = m * blocks;

    xmx_quantize_q8_vnni<true>(
        queue, activation_input, activation_values, activation_d, activation_s,
        active_n, k, groups, xmx_tile_n);

    GGML_ASSERT(!weights_q4_xmx);
    const sycl::kernel & kernel = xmx_get_q4_K_kernel(ctx);
    queue->submit([&](sycl::handler & cgh) {
        cgh.set_arg(0, weight_bytes); cgh.set_arg(1, activation_values); cgh.set_arg(2, activation_d);
        cgh.set_arg(3, activation_s); cgh.set_arg(4, output); cgh.set_arg(5, active_n);
        cgh.set_arg(6, m); cgh.set_arg(7, groups); cgh.set_arg(8, blocks);
        cgh.set_arg(9, total_weight_blocks); cgh.set_arg(10, weights_reordered ? 1 : 0);
        cgh.parallel_for(sycl::nd_range<1>{sycl::range<1>{static_cast<size_t>(m / xmx_tile_m) * xmx_tile_n},
                                           sycl::range<1>{xmx_tile_n}}, kernel);
    });
}
