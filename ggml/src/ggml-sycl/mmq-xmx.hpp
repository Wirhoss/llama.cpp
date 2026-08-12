#pragma once

#include "ggml.h"

struct ggml_backend_sycl_context;

// Compile runtime OpenCL kernels while the backend is initialized. This keeps
// capability predicates side-effect free and avoids a first-matmul JIT stall.
void ggml_sycl_xmx_init(ggml_backend_sycl_context & ctx);

bool ggml_sycl_can_use_mul_mat_q4_K_xmx(
    const ggml_backend_sycl_context & ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    const ggml_tensor * dst);

void ggml_sycl_mul_mat_q4_K_xmx(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    ggml_tensor * dst);

bool ggml_sycl_can_use_mul_mat_q4_K_gemm_xmx(
    const ggml_backend_sycl_context & ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    const ggml_tensor * dst);

void ggml_sycl_mul_mat_q4_K_gemm_xmx(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    ggml_tensor * dst);

void ggml_sycl_dequantize_q4_K_xmx_to_f16(
    ggml_backend_sycl_context & ctx,
    const void * weights,
    void * output,
    int m,
    int k);

bool ggml_sycl_xmx_gemm_is_ready(const ggml_backend_sycl_context & ctx);

bool ggml_sycl_can_use_mul_mat_q6_K_gemm_xmx(
    const ggml_backend_sycl_context & ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    const ggml_tensor * dst);

void ggml_sycl_mul_mat_q6_K_gemm_xmx(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    ggml_tensor * dst);

void ggml_sycl_dequantize_q6_K_xmx_to_f16(
    ggml_backend_sycl_context & ctx,
    const void * weights,
    void * output,
    int m,
    int k);

bool ggml_sycl_can_use_mul_mat_q5_K_gemm_xmx(
    const ggml_backend_sycl_context & ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    const ggml_tensor * dst);

void ggml_sycl_mul_mat_q5_K_gemm_xmx(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    ggml_tensor * dst);

void ggml_sycl_dequantize_q5_K_xmx_to_f16(
    ggml_backend_sycl_context & ctx,
    const void * weights,
    void * output,
    int m,
    int k);

bool ggml_sycl_can_use_mul_mat_q5_K_xmx(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    const ggml_tensor * dst);

void ggml_sycl_mul_mat_q5_K_xmx(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    ggml_tensor * dst);

bool ggml_sycl_can_use_mul_mat_q6_K_xmx(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    const ggml_tensor * dst);

void ggml_sycl_mul_mat_q6_K_xmx(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    ggml_tensor * dst);
