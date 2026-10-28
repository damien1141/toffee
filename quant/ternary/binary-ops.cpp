#include "binary-ops.h"
#include "ggml-quants.h"
#include <vector>

#if defined(GGML_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>

using vDSP_fn_t = void (*)(const float *, vDSP_Stride, const float *, vDSP_Stride, float *, vDSP_Stride, vDSP_Length);
#endif

static inline float op_add(float a, float b) {
    return a + b;
}

static inline float op_sub(float a, float b) {
    return a - b;
}

static inline float op_mul(float a, float b) {
    return a * b;
}

static inline float op_div(float a, float b) {
    return a / b;
}

template <float (*op)(float, float), typename src0_t, typename src1_t, typename dst_t>
static inline void vec_binary_op_contiguous(const int64_t n, dst_t * z, const src0_t * x, const src1_t * y) {
    constexpr auto src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr auto src1_to_f32 = type_conversion_table<src1_t>::to_f32;
    constexpr auto f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    for (int i = 0; i < n; i++) {
        z[i] = f32_to_dst(op(src0_to_f32(x[i]), src1_to_f32(y[i])));
    }
}

template <float (*op)(float, float), typename src0_t, typename src1_t, typename dst_t>
static inline void vec_binary_op_non_contiguous(const int64_t n, const int64_t ne10, const int64_t nb10, dst_t * z, const src0_t * x, const src1_t * y) {
    constexpr auto src0_to_f32 = type_conversion_table<src0_t>::to_f32;
    constexpr auto src1_to_f32 = type_conversion_table<src1_t>::to_f32;
    constexpr auto f32_to_dst  = type_conversion_table<dst_t >::from_f32;

    for (int i = 0; i < n; i++) {
        int i10 = i % ne10;
        const src1_t * y_ptr = (const src1_t *)((const char *)y + i10*nb10);
        z[i] = f32_to_dst(op(src0_to_f32(x[i]), src1_to_f32(*y_ptr)));
    }
}

template <float (*op)(float, float), typename src0_t, typename src1_t, typename dst_t>
static void apply_binary_op(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT( nb0 == sizeof(dst_t));
    GGML_ASSERT(nb00 == sizeof(src0_t));

    const auto [ir0, ir1] = get_thread_range(params, src0);
    const bool is_src1_contiguous_rows = ggml_is_contiguous_rows(src1);

#ifdef GGML_USE_ACCELERATE
    vDSP_fn_t vDSP_op = nullptr;
    // TODO - avoid the f32-only check using type 'trait' lookup tables and row-based src-to-float conversion functions
    if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        if (op == op_add) {
            vDSP_op = vDSP_vadd;
        } else if (op == op_sub) {
            vDSP_op = vDSP_vsub;
        } else if (op == op_mul) {
            vDSP_op = vDSP_vmul;
        } else if (op == op_div) {
            vDSP_op = vDSP_vdiv;
        }
    }
#endif

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i03 = ir/(ne02*ne01);
        const int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        const int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        const int64_t i13 = i03 % ne13;
        const int64_t i12 = i02 % ne12;
        const int64_t i11 = i01 % ne11;

        dst_t        * dst_ptr  = (dst_t  *)       ((char *)       dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
        const src0_t * src0_ptr = (const src0_t *) ((const char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);
        const src1_t * src1_ptr = (const src1_t *) ((const char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11);

        if (is_src1_contiguous_rows) {
            // src1 is broadcastable across src0 and dst in i1, i2, i3
            const int64_t nr0 = ne00 / ne10;

            for (int64_t r = 0; r < nr0; ++r) {
#ifdef GGML_USE_ACCELERATE
                if constexpr (std::is_same_v<src0_t, float> && std::is_same_v<src1_t, float> && std::is_same_v<dst_t, float>) {
                    if (vDSP_op != nullptr) {
                        vDSP_op(src1_ptr, 1, src0_ptr + r*ne10, 1, dst_ptr + r*ne10, 1, ne10);
                        continue;
                    }
                }
#endif
                vec_binary_op_contiguous<op>(ne10, dst_ptr + r*ne10, src0_ptr + r*ne10, src1_ptr);
            }
        } else {
            vec_binary_op_non_contiguous<op>(ne0, ne10, nb10, dst_ptr, src0_ptr, src1_ptr);
        }
    }
}

// F32 op Quantized -> F32 helper. Dequantizes src1 (PQ2_0 or PTQ1_0) row
// by row into a thread-local F32 scratch buffer, then runs the existing
// F32+F32+F32 vec_binary_op on the dequantized row. This is a fallback
// path that lets the CPU backend handle binary ops (add/sub/mul/div)
// whose second operand happens to be a ternary-quantized tensor; the
// CUDA backend already has dedicated PQ2_0 kernels via the
// dequant_pq2_0.comp shader so this only matters when the scheduler
// spills the op onto the CPU.
template <float (*op)(float, float), ggml_type SRC1_TYPE>
static void apply_binary_op_f32_quant_f32(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == SRC1_TYPE);
    // dequantize_row_pq2_0 / dequantize_row_ptq1_0 assume the input
    // blocks are contiguous. Our quantizer always emits contiguous
    // PQ2_0/PTQ1_0 tensors, so this assert should hold in practice;
    // if it ever fires, the fix is a dequant-with-stride helper in
    // ggml-quants.c (not something we can paper over here).
    GGML_ASSERT(ggml_is_contiguous_rows(src1) && "PQ2_0/PTQ1_0 src1 must be contiguous-rows for the CPU binary-op fallback");

    GGML_TENSOR_BINARY_OP_LOCALS

    // ne10 = number of *logical* elements in src1's innermost dim. The
    // dequantize_row_* functions take a logical element count and write
    // that many floats into the output buffer (they figure out the block
    // math from the input pointer type internally).
    const int64_t row_elems = ne10;
    std::vector<float> f32_row(row_elems);

    const auto [ir0, ir1] = get_thread_range(params, src0);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t i03 = ir/(ne02*ne01);
        const int64_t i02 = (ir - i03*ne02*ne01)/ne01;
        const int64_t i01 = (ir - i03*ne02*ne01 - i02*ne01);

        const int64_t i13 = i03 % ne13;
        const int64_t i12 = i02 % ne12;
        const int64_t i11 = i01 % ne11;

        float              * dst_ptr  = (float *)       ((char *)       dst->data  + i03*nb3  + i02*nb2  + i01*nb1 );
        const float        * src0_ptr = (const float *) ((const char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01);
        const void         * src1_row = (const char *)  src1->data + i13*nb13 + i12*nb12 + i11*nb11;

        // Dequantize the src1 row at (i11, i12, i13) into the F32 scratch.
        if constexpr (SRC1_TYPE == GGML_TYPE_PQ2_0) {
            dequantize_row_pq2_0((const block_pq2_0 *) src1_row, f32_row.data(), row_elems);
        } else if constexpr (SRC1_TYPE == GGML_TYPE_PTQ1_0) {
            dequantize_row_ptq1_0((const block_ptq1_0 *) src1_row, f32_row.data(), row_elems);
        }

        const float * src1_ptr = f32_row.data();

        // src1 is broadcastable across src0 and dst in i1, i2, i3.
        // We always take the contiguous path here because our dequantized
        // f32_row buffer is itself contiguous (sizeof(float) stride).
        const int64_t nr0 = ne00 / ne10;
        for (int64_t r = 0; r < nr0; ++r) {
            vec_binary_op_contiguous<op, float, float, float>(ne10, dst_ptr + r*ne10, src0_ptr + r*ne10, src1_ptr);
        }
    }
}

// TODO: Use the 'traits' lookup table (for type conversion fns), instead of a mass of 'if' conditions with long templates
template <float (*op)(float, float)>
static void binary_op(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    /*  */ if (src0->type == GGML_TYPE_F32  && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) { // all f32
        apply_binary_op<op, float, float, float>(params, dst);
    } else if (src0->type == GGML_TYPE_F16  && src1->type == GGML_TYPE_F16  && dst->type == GGML_TYPE_F16) { // all f16
        apply_binary_op<op, ggml_fp16_t, ggml_fp16_t, ggml_fp16_t>(params, dst);
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) { // all bf16
        apply_binary_op<op, ggml_bf16_t, ggml_bf16_t, ggml_bf16_t>(params, dst);
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_BF16) {
        apply_binary_op<op, ggml_bf16_t, float, ggml_bf16_t>(params, dst);
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) {
        apply_binary_op<op, ggml_bf16_t, float, float>(params, dst);
    } else if (src0->type == GGML_TYPE_F16  && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F16) {
        apply_binary_op<op, ggml_fp16_t, float, ggml_fp16_t>(params, dst);
    } else if (src0->type == GGML_TYPE_F16  && src1->type == GGML_TYPE_F32  && dst->type == GGML_TYPE_F32) {
        apply_binary_op<op, ggml_fp16_t, float, float>(params, dst);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_PQ2_0 && dst->type == GGML_TYPE_F32) {
        // F32 op PQ2_0 -> F32, dequantize src1 row-by-row on the fly.
        apply_binary_op_f32_quant_f32<op, GGML_TYPE_PQ2_0>(params, dst);
    } else if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_PTQ1_0 && dst->type == GGML_TYPE_F32) {
        // F32 op PTQ1_0 -> F32, dequantize src1 row-by-row on the fly.
        apply_binary_op_f32_quant_f32<op, GGML_TYPE_PTQ1_0>(params, dst);
    } else {
        GGML_ABORT("%s: unsupported types: dst: %s, src0: %s, src1: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type), ggml_type_name(src1->type));
    }
}

void ggml_compute_forward_add_non_quantized(const ggml_compute_params * params, ggml_tensor * dst) {
    binary_op<op_add>(params, dst);
}

void ggml_compute_forward_sub(const ggml_compute_params * params, ggml_tensor * dst) {
    binary_op<op_sub>(params, dst);
}

void ggml_compute_forward_mul(const ggml_compute_params * params, ggml_tensor * dst) {
    binary_op<op_mul>(params, dst);
}

void ggml_compute_forward_div(const ggml_compute_params * params, ggml_tensor * dst) {
    binary_op<op_div>(params, dst);
}
