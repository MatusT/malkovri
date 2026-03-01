// =============================================================================
// test_expressions.wgsl
// Exercises every branch in evaluate_expression / its sub-evaluators.
// =============================================================================

// ---------------------------------------------------------------------------
// Expression::Constant  (module-level const)
// ---------------------------------------------------------------------------
const C_F32:   f32        = 2.5;
const C_I32:   i32        = -10i;
const C_U32:   u32        = 42u;
const C_VEC3F: vec3<f32>  = vec3<f32>(1.0, 2.0, 3.0);

// ---------------------------------------------------------------------------
// Expression::Override  (SCALE has an init → evaluates to 1.0)
// ---------------------------------------------------------------------------
override SCALE: f32 = 1.0;

// ---------------------------------------------------------------------------
// Expression::GlobalVariable + Expression::ArrayLength
// ---------------------------------------------------------------------------
@group(0) @binding(0) var<storage, read_write> buf:     array<f32>;
@group(0) @binding(1) var<storage, read_write> out_buf: array<u32>;

// ---------------------------------------------------------------------------
// Helper functions  (for Expression::CallResult)
// ---------------------------------------------------------------------------
fn mul_add_f(a: f32, b: f32, c: f32) -> f32 {
    return a * b + c;
}

fn clamp_i(v: i32, lo: i32, hi: i32) -> i32 {
    return clamp(v, lo, hi);
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {

    // -----------------------------------------------------------------------
    // Expression::FunctionArgument  (@builtin → evaluates to U32x3)
    // -----------------------------------------------------------------------
    let idx: u32 = gid.x;

    // -----------------------------------------------------------------------
    // Expression::Literal
    // -----------------------------------------------------------------------
    let lit_f: f32 = 1.5f;
    let lit_i: i32 = -3i;
    let lit_u: u32 = 99u;

    // -----------------------------------------------------------------------
    // Expression::LocalVariable + Expression::Load
    // (every var declaration creates a LocalVariable; reading it emits a Load)
    // -----------------------------------------------------------------------
    var lv_f32: f32       = 3.14;
    var lv_i32: i32       = -5i;
    var lv_u32: u32       = 7u;
    var lv_v2f: vec2<f32> = vec2<f32>(1.0, 2.0);
    var lv_v3f: vec3<f32> = vec3<f32>(0.5, 1.5, 2.5);
    var lv_v4f: vec4<f32> = vec4<f32>(1.0, 2.0, 3.0, 4.0);
    var lv_v2i: vec2<i32> = vec2<i32>(1i, -1i);
    var lv_v3i: vec3<i32> = vec3<i32>(1i, 2i, 3i);
    var lv_v4i: vec4<i32> = vec4<i32>(1i, 2i, 3i, 4i);
    var lv_v2u: vec2<u32> = vec2<u32>(1u, 2u);
    var lv_v3u: vec3<u32> = vec3<u32>(1u, 2u, 3u);
    var lv_v4u: vec4<u32> = vec4<u32>(1u, 2u, 3u, 4u);

    let load_f: f32 = lv_f32;   // Load from local var
    let load_i: i32 = lv_i32;
    let load_u: u32 = lv_u32;

    // -----------------------------------------------------------------------
    // Expression::ZeroValue  (explicit zero-constructors)
    // -----------------------------------------------------------------------
    var zv_f32: f32       = f32();
    var zv_i32: i32       = i32();
    var zv_u32: u32       = u32();
    var zv_v3f: vec3<f32> = vec3<f32>();

    // -----------------------------------------------------------------------
    // Expression::Compose  (vector built from multiple components)
    // -----------------------------------------------------------------------
    let comp_v2f: vec2<f32> = vec2<f32>(lv_f32, lit_f);
    let comp_v3i: vec3<i32> = vec3<i32>(lv_i32, lit_i, 0i);
    let comp_v4u: vec4<u32> = vec4<u32>(lv_u32, lit_u, 0u, 1u);

    // -----------------------------------------------------------------------
    // Expression::Splat  (broadcast scalar to all lanes)
    // -----------------------------------------------------------------------
    let splat_f3: vec3<f32> = vec3(lv_f32);   // → F32x3
    let splat_i2: vec2<i32> = vec2(lv_i32);   // → I32x2
    let splat_u4: vec4<u32> = vec4(lv_u32);   // → U32x4

    // -----------------------------------------------------------------------
    // Expression::Swizzle
    // -----------------------------------------------------------------------
    let sw_xy:   vec2<f32> = lv_v3f.xy;
    let sw_zyx:  vec3<f32> = lv_v3f.zyx;
    let sw_wzyx: vec4<f32> = lv_v4f.wzyx;
    let sw_xi:   i32       = lv_v2i.x;
    let sw_yxu:  vec2<u32> = lv_v2u.yx;

    // -----------------------------------------------------------------------
    // Expression::AccessIndex  (compile-time constant component index)
    // -----------------------------------------------------------------------
    let aci_x:  f32 = lv_v3f.x;    // index 0
    let aci_y:  f32 = lv_v3f.y;    // index 1
    let aci_z:  f32 = lv_v3f.z;    // index 2
    let aci_i0: i32 = comp_v3i.x;
    let aci_u2: u32 = comp_v4u.z;

    // -----------------------------------------------------------------------
    // Expression::Access  (runtime dynamic index)
    // -----------------------------------------------------------------------
    let dyn_i:   u32 = idx % 3u;
    let acc_vf:  f32 = lv_v3f[dyn_i];
    let acc_vi:  i32 = lv_v3i[dyn_i];
    let acc_vu:  u32 = lv_v3u[dyn_i];

    // -----------------------------------------------------------------------
    // Expression::Binary — I32 scalar
    // -----------------------------------------------------------------------
    let bi_i_add: i32  = lv_i32 + 10i;
    let bi_i_sub: i32  = lv_i32 - 3i;
    let bi_i_mul: i32  = lv_i32 * 2i;
    let bi_i_div: i32  = lv_i32 / 2i;
    let bi_i_mod: i32  = lv_i32 % 3i;
    let bi_i_and: i32  = lv_i32 & 0xFF;
    let bi_i_or:  i32  = lv_i32 | 0x01;
    let bi_i_xor: i32  = lv_i32 ^ 0x0F;
    let bi_i_shl: i32  = lv_i32 << 1u;
    let bi_i_shr: i32  = lv_i32 >> 1u;
    let bi_i_eq:  bool = lv_i32 == 0i;
    let bi_i_ne:  bool = lv_i32 != 0i;
    let bi_i_lt:  bool = lv_i32 < 0i;
    let bi_i_le:  bool = lv_i32 <= 0i;
    let bi_i_gt:  bool = lv_i32 > 0i;
    let bi_i_ge:  bool = lv_i32 >= 0i;

    // Expression::Binary — U32 scalar
    let bi_u_add: u32  = lv_u32 + 1u;
    let bi_u_sub: u32  = lv_u32 - 1u;
    let bi_u_mul: u32  = lv_u32 * 2u;
    let bi_u_div: u32  = lv_u32 / 2u;
    let bi_u_mod: u32  = lv_u32 % 3u;
    let bi_u_and: u32  = lv_u32 & 0xFFu;
    let bi_u_or:  u32  = lv_u32 | 0x01u;
    let bi_u_xor: u32  = lv_u32 ^ 0x0Fu;
    let bi_u_shl: u32  = lv_u32 << 2u;
    let bi_u_shr: u32  = lv_u32 >> 2u;
    let bi_u_eq:  bool = lv_u32 == 7u;
    let bi_u_ne:  bool = lv_u32 != 7u;
    let bi_u_lt:  bool = lv_u32 < 10u;
    let bi_u_le:  bool = lv_u32 <= 7u;
    let bi_u_gt:  bool = lv_u32 > 5u;
    let bi_u_ge:  bool = lv_u32 >= 7u;

    // Expression::Binary — F32 scalar
    let bi_f_add: f32  = lv_f32 + 1.0;
    let bi_f_sub: f32  = lv_f32 - 1.0;
    let bi_f_mul: f32  = lv_f32 * 2.0;
    let bi_f_div: f32  = lv_f32 / 2.0;
    let bi_f_mod: f32  = lv_f32 % 1.0;
    let bi_f_eq:  bool = lv_f32 == 3.14;
    let bi_f_ne:  bool = lv_f32 != 0.0;
    let bi_f_lt:  bool = lv_f32 < 4.0;
    let bi_f_le:  bool = lv_f32 <= 3.14;
    let bi_f_gt:  bool = lv_f32 > 0.0;
    let bi_f_ge:  bool = lv_f32 >= 3.0;

    // Expression::Binary — Bool logical (LogicalAnd / LogicalOr)
    let bi_land: bool = (lv_u32 > 0u) && (lv_f32 > 0.0);
    let bi_lor:  bool = (lv_u32 == 0u) || (lv_f32 > 0.0);

    // Expression::Binary — F32 vector OP scalar
    let bvs_f_add: vec3<f32> = lv_v3f + 1.0;
    let bvs_f_sub: vec2<f32> = lv_v2f - 0.5;
    let bvs_f_mul: vec3<f32> = lv_v3f * 2.0;
    let bvs_f_div: vec4<f32> = lv_v4f / 2.0;
    let bvs_f_mod: vec2<f32> = lv_v2f % 0.5;

    // Expression::Binary — F32 scalar OP vector
    let bsv_f_add: vec3<f32> = 1.0 + lv_v3f;
    let bsv_f_sub: vec2<f32> = 1.0 - lv_v2f;
    let bsv_f_mul: vec3<f32> = 2.0 * lv_v3f;

    // Expression::Binary — I32 vector OP scalar
    let bvs_i_add: vec3<i32> = lv_v3i + 10i;
    let bvs_i_sub: vec2<i32> = lv_v2i - 1i;
    let bvs_i_mul: vec4<i32> = lv_v4i * 2i;
    let bvs_i_div: vec3<i32> = lv_v3i / 2i;
    let bvs_i_mod: vec2<i32> = lv_v2i % 3i;
    let bvs_i_and: vec3<i32> = lv_v3i & vec3(0xFFi);   // naga: bitwise vec×scalar unsupported; use splat
    let bvs_i_or:  vec2<i32> = lv_v2i | vec2(0x01i);
    let bvs_i_xor: vec4<i32> = lv_v4i ^ vec4(0x0Fi);
    let bvs_i_shl: vec2<i32> = lv_v2i << vec2(1u);   // naga: shift vec×scalar unsupported; use splat
    let bvs_i_shr: vec3<i32> = lv_v3i >> vec3(1u);

    // Expression::Binary — I32 scalar OP vector
    let bsv_i_sub: vec3<i32> = 100i - lv_v3i;

    // Expression::Binary — U32 vector OP scalar
    let bvs_u_add: vec4<u32> = lv_v4u + 1u;
    let bvs_u_sub: vec3<u32> = lv_v3u - 1u;
    let bvs_u_mul: vec3<u32> = lv_v3u * 3u;
    let bvs_u_div: vec4<u32> = lv_v4u / 2u;
    let bvs_u_mod: vec2<u32> = lv_v2u % 3u;
    let bvs_u_and: vec4<u32> = lv_v4u & vec4(0xFFu);   // naga: bitwise vec×scalar unsupported; use splat
    let bvs_u_or:  vec2<u32> = lv_v2u | vec2(0x80u);
    let bvs_u_xor: vec3<u32> = lv_v3u ^ vec3(0x0Fu);
    let bvs_u_shl: vec3<u32> = lv_v3u << vec3(2u);   // naga: shift vec×scalar unsupported; use splat
    let bvs_u_shr: vec4<u32> = lv_v4u >> vec4(1u);

    // Expression::Binary — U32 scalar OP vector
    let bsv_u_add: vec3<u32> = 10u + lv_v3u;

    // Expression::Binary — F32 vector OP F32 vector
    let bvv_f_add: vec3<f32>  = lv_v3f + splat_f3;
    let bvv_f_sub: vec2<f32>  = lv_v2f - sw_xy;
    let bvv_f_mul: vec2<f32>  = lv_v2f * lv_v2f;
    let bvv_f_div: vec4<f32>  = lv_v4f / lv_v4f;
    let bvv_f_mod: vec3<f32>  = lv_v3f % splat_f3;
    let bvv_f_eq:  vec3<bool> = lv_v3f == splat_f3;
    let bvv_f_ne:  vec3<bool> = lv_v3f != splat_f3;
    let bvv_f_lt:  vec3<bool> = lv_v3f < splat_f3;
    let bvv_f_le:  vec3<bool> = lv_v3f <= splat_f3;
    let bvv_f_gt:  vec3<bool> = lv_v3f > splat_f3;
    let bvv_f_ge:  vec3<bool> = lv_v3f >= splat_f3;

    // Expression::Binary — I32 vector OP I32 vector
    let bvv_i_add: vec3<i32>  = lv_v3i + comp_v3i;
    let bvv_i_sub: vec4<i32>  = lv_v4i - lv_v4i;
    let bvv_i_mul: vec3<i32>  = lv_v3i * comp_v3i;
    let bvv_i_div: vec3<i32>  = lv_v3i / comp_v3i;
    let bvv_i_mod: vec3<i32>  = lv_v3i % comp_v3i;
    let bvv_i_and: vec4<i32>  = lv_v4i & lv_v4i;
    let bvv_i_or:  vec2<i32>  = lv_v2i | lv_v2i;
    let bvv_i_xor: vec3<i32>  = lv_v3i ^ comp_v3i;
    let bvv_i_eq:  vec2<bool> = lv_v2i == lv_v2i;
    let bvv_i_ne:  vec3<bool> = lv_v3i != comp_v3i;
    let bvv_i_lt:  vec3<bool> = lv_v3i < comp_v3i;
    let bvv_i_le:  vec4<bool> = lv_v4i <= lv_v4i;
    let bvv_i_gt:  vec3<bool> = lv_v3i > comp_v3i;
    let bvv_i_ge:  vec4<bool> = lv_v4i >= lv_v4i;

    // Expression::Binary — I32 vector shift by U32 vector
    let bvv_i_shl: vec2<i32> = lv_v2i << lv_v2u;
    let bvv_i_shr: vec3<i32> = lv_v3i >> lv_v3u;

    // Expression::Binary — U32 vector OP U32 vector
    let bvv_u_add: vec4<u32>  = lv_v4u + comp_v4u;
    let bvv_u_sub: vec3<u32>  = lv_v3u - lv_v3u;
    let bvv_u_mul: vec4<u32>  = lv_v4u * comp_v4u;
    let bvv_u_div: vec3<u32>  = lv_v3u / lv_v3u;
    let bvv_u_mod: vec2<u32>  = lv_v2u % lv_v2u;
    let bvv_u_and: vec2<u32>  = lv_v2u & lv_v2u;
    let bvv_u_or:  vec3<u32>  = lv_v3u | lv_v3u;
    let bvv_u_xor: vec4<u32>  = lv_v4u ^ comp_v4u;
    let bvv_u_shl: vec3<u32>  = lv_v3u << lv_v3u;
    let bvv_u_shr: vec4<u32>  = lv_v4u >> lv_v4u;
    let bvv_u_eq:  vec2<bool> = lv_v2u == lv_v2u;
    let bvv_u_ne:  vec3<bool> = lv_v3u != lv_v3u;
    let bvv_u_lt:  vec3<bool> = lv_v3u < lv_v3u;
    let bvv_u_le:  vec4<bool> = lv_v4u <= lv_v4u;
    let bvv_u_gt:  vec2<bool> = lv_v2u > lv_v2u;
    let bvv_u_ge:  vec4<bool> = lv_v4u >= comp_v4u;

    // -----------------------------------------------------------------------
    // Expression::Unary
    // -----------------------------------------------------------------------
    let un_neg_f:  f32       = -lv_f32;
    let un_neg_i:  i32       = -lv_i32;
    let un_neg_v3: vec3<f32> = -lv_v3f;
    let un_neg_vi: vec4<i32> = -lv_v4i;
    let un_not_b:  bool      = !bi_i_eq;   // LogicalNot (bool → U32 0/1)
    let un_bnot_i: i32       = ~lv_i32;    // BitwiseNot i32
    let un_bnot_u: u32       = ~lv_u32;    // BitwiseNot u32

    // -----------------------------------------------------------------------
    // Expression::Select
    // -----------------------------------------------------------------------
    let sel_f: f32 = select(0.0,  1.0,  lv_f32 > 0.0);
    let sel_i: i32 = select(-1i,  1i,   lv_i32 > 0i);
    let sel_u: u32 = select(0u,   lv_u32, lv_u32 > 0u);

    // -----------------------------------------------------------------------
    // Expression::As — numeric conversion  (convert = Some(4) → 32-bit)
    // -----------------------------------------------------------------------
    let as_f_to_i:   i32        = i32(lv_f32);
    let as_f_to_u:   u32        = u32(lv_f32);
    let as_i_to_f:   f32        = f32(lv_i32);
    let as_u_to_f:   f32        = f32(lv_u32);
    let as_i_to_u:   u32        = u32(lv_i32);
    let as_u_to_i:   i32        = i32(lv_u32);

    // Vector element-wise conversions (Compose of scalar As expressions)
    let as_v2i_to_f: vec2<f32>  = vec2<f32>(f32(lv_v2i.x), f32(lv_v2i.y));
    let as_v3u_to_f: vec3<f32>  = vec3<f32>(f32(lv_v3u.x), f32(lv_v3u.y), f32(lv_v3u.z));
    let as_v4f_to_i: vec4<i32>  = vec4<i32>(i32(lv_v4f.x), i32(lv_v4f.y), i32(lv_v4f.z), i32(lv_v4f.w));
    let as_v2f_to_u: vec2<u32>  = vec2<u32>(u32(lv_v2f.x), u32(lv_v2f.y));

    // -----------------------------------------------------------------------
    // Expression::As — bitcast  (convert = None)
    // -----------------------------------------------------------------------
    let bc_f_to_u:   u32        = bitcast<u32>(lv_f32);
    let bc_u_to_f:   f32        = bitcast<f32>(lv_u32);
    let bc_i_to_f:   f32        = bitcast<f32>(lv_i32);
    let bc_v2f_to_i: vec2<i32>  = bitcast<vec2<i32>>(lv_v2f);
    let bc_v3u_to_f: vec3<f32>  = bitcast<vec3<f32>>(lv_v3u);
    let bc_v4f_to_u: vec4<u32>  = bitcast<vec4<u32>>(lv_v4f);
    let bc_v2i_to_f: vec2<f32>  = bitcast<vec2<f32>>(lv_v2i);

    // -----------------------------------------------------------------------
    // Expression::Math — comparison / range
    // -----------------------------------------------------------------------
    let m_abs_f:   f32       = abs(lv_f32);
    let m_abs_i:   i32       = abs(lv_i32);
    let m_abs_vf:  vec3<f32> = abs(lv_v3f);
    let m_abs_vi:  vec3<i32> = abs(lv_v3i);
    let m_min_ff:  f32       = min(lv_f32, 2.0);
    let m_max_ff:  f32       = max(lv_f32, 0.0);
    let m_min_ii:  i32       = min(lv_i32, 0i);
    let m_max_uu:  u32       = max(lv_u32, 1u);
    let m_clamp_f: f32       = clamp(lv_f32, 0.0, 5.0);
    let m_clamp_i: i32       = clamp(lv_i32, -10i, 10i);
    let m_clamp_u: u32       = clamp(lv_u32, 0u, 100u);
    let m_sat:     f32       = saturate(lv_f32);

    // Expression::Math — trigonometry
    let m_cos:   f32 = cos(lv_f32);
    let m_sin:   f32 = sin(lv_f32);
    let m_tan:   f32 = tan(lv_f32);
    let m_acos:  f32 = acos(0.5f);
    let m_asin:  f32 = asin(0.5f);
    let m_atan:  f32 = atan(1.0f);
    let m_atan2: f32 = atan2(lv_f32, 1.0);
    let m_cosh:  f32 = cosh(lv_f32);
    let m_sinh:  f32 = sinh(lv_f32);
    let m_tanh:  f32 = tanh(0.5f);
    let m_acosh: f32 = acosh(1.5f);
    let m_asinh: f32 = asinh(lv_f32);
    let m_atanh: f32 = atanh(0.5f);
    let m_rad:   f32 = radians(180.0f);
    let m_deg:   f32 = degrees(3.14159f);

    // Expression::Math — decomposition
    let m_ceil:  f32 = ceil(lv_f32);
    let m_floor: f32 = floor(lv_f32);
    let m_round: f32 = round(lv_f32);
    let m_fract: f32 = fract(lv_f32);
    let m_trunc: f32 = trunc(lv_f32);
    // NOTE: ldexp(f32, i32) — evaluator's math_binary_f32 expects F32 for both args,
    // so this exercises the Ldexp branch but returns Uninitialized (known limitation).
    let m_ldexp: f32 = ldexp(lv_f32, 2i);

    // Expression::Math — exponent / power
    let m_exp:   f32 = exp(lv_f32);
    let m_exp2:  f32 = exp2(lv_f32);
    let m_log:   f32 = log(abs(lv_f32) + 0.001);
    let m_log2:  f32 = log2(abs(lv_f32) + 0.001);
    let m_pow:   f32 = pow(abs(lv_f32), 2.0);
    let m_sqrt:  f32 = sqrt(abs(lv_f32));
    let m_isqrt: f32 = inverseSqrt(abs(lv_f32) + 0.001);
    let m_q16:   f32 = quantizeToF16(lv_f32);

    // Expression::Math — geometry (vec3 variants are the richest)
    let m_dot:   f32       = dot(lv_v3f, C_VEC3F);
    let m_dot2:  f32       = dot(lv_v2f, sw_xy);
    let m_dot4:  f32       = dot(lv_v4f, sw_wzyx);
    let m_cross: vec3<f32> = cross(lv_v3f, C_VEC3F);
    let m_len:   f32       = length(lv_v3f);
    let m_len2:  f32       = length(lv_v2f);
    let m_norm:  vec3<f32> = normalize(lv_v3f);
    let m_dist:  f32       = distance(lv_v3f, C_VEC3F);
    let m_ff:    vec3<f32> = faceForward(lv_v3f, C_VEC3F, lv_v3f);
    let m_refl:  vec3<f32> = reflect(lv_v3f, normalize(C_VEC3F));

    // Expression::Math — computational
    let m_sign_f:  f32       = sign(lv_f32);
    let m_sign_i:  i32       = sign(lv_i32);
    let m_fma:     f32       = fma(lv_f32, 2.0, 1.0);
    let m_mix_f:   f32       = mix(0.0f, 1.0f, 0.5f);
    let m_mix_vvf: vec3<f32> = mix(lv_v3f, splat_f3, 0.5f);   // vec, vec, scalar-t
    let m_mix_vvv: vec2<f32> = mix(lv_v2f, sw_xy,    lv_v2f);  // vec, vec, vec-t
    let m_step:    f32       = step(0.5f, lv_f32);
    let m_smooth:  f32       = smoothstep(0.0f, 1.0f, lv_f32);

    // Expression::Math — bit manipulation, u32
    let m_ctz:   u32 = countTrailingZeros(lv_u32);
    let m_clz:   u32 = countLeadingZeros(lv_u32);
    let m_pop:   u32 = countOneBits(lv_u32);
    let m_rev:   u32 = reverseBits(lv_u32);
    let m_ftb_u: u32 = firstTrailingBit(lv_u32);
    let m_flb_u: u32 = firstLeadingBit(lv_u32);
    let m_ext_u: u32 = extractBits(lv_u32, 0u, 8u);
    let m_ins_u: u32 = insertBits(lv_u32, 0xFFu, 0u, 8u);

    // Expression::Math — bit manipulation, i32
    let m_ctz_i: i32 = countTrailingZeros(lv_i32);
    let m_clz_i: i32 = countLeadingZeros(lv_i32);
    let m_pop_i: i32 = countOneBits(lv_i32);
    let m_rev_i: i32 = reverseBits(lv_i32);
    let m_ftb_i: i32 = firstTrailingBit(lv_i32);
    let m_flb_i: i32 = firstLeadingBit(lv_i32);
    let m_ext_i: i32 = extractBits(lv_i32, 1u, 4u);
    let m_ins_i: i32 = insertBits(lv_i32, 7i, 0u, 4u);

    // -----------------------------------------------------------------------
    // Expression::Relational — all, any
    // (IsNan / IsInf not reachable from standard WGSL surface syntax)
    // -----------------------------------------------------------------------
    let bv3:      vec3<bool> = lv_v3f > splat_f3;
    let rel_all:  bool       = all(bv3);
    let rel_any:  bool       = any(bv3);
    let bv2:      vec2<bool> = lv_v2i == lv_v2i;
    let rel_all2: bool       = all(bv2);
    let rel_any2: bool       = any(bv2);

    // -----------------------------------------------------------------------
    // Expression::Constant  (module-level const used in expressions)
    // -----------------------------------------------------------------------
    let use_c_f32: f32       = C_F32 * 2.0;
    let use_c_i32: i32       = C_I32 + 5i;
    let use_c_u32: u32       = C_U32 + 1u;
    let use_c_v3f: vec3<f32> = C_VEC3F + splat_f3;

    // -----------------------------------------------------------------------
    // Expression::Override  (pipeline override used in expression)
    // -----------------------------------------------------------------------
    let use_ovr: f32 = lv_f32 * SCALE;

    // -----------------------------------------------------------------------
    // Expression::GlobalVariable  (direct access to storage buffer element)
    // Expression::ArrayLength     (runtime length of storage buffer)
    // -----------------------------------------------------------------------
    let arr_len: u32 = arrayLength(&buf);
    let g_val:   f32 = buf[0u];

    // -----------------------------------------------------------------------
    // Expression::Access  (dynamic index into global storage buffer)
    // -----------------------------------------------------------------------
    let rt_idx: u32 = idx % arr_len;
    let acc_g:  f32 = buf[rt_idx];

    // -----------------------------------------------------------------------
    // Expression::CallResult  (result of a function call)
    // -----------------------------------------------------------------------
    let cr_f: f32 = mul_add_f(lv_f32, 2.0, use_c_f32);
    let cr_i: i32 = clamp_i(lv_i32, -100i, 100i);

    // -----------------------------------------------------------------------
    // Sink: write results to prevent dead-code elimination
    // -----------------------------------------------------------------------
    if idx < arr_len {
        buf[idx] = cr_f + acc_g + g_val
                 + m_dot + m_len + m_cos + m_sin + m_fma + m_mix_f
                 + m_smooth + m_sqrt + m_abs_f
                 + as_i_to_f + as_u_to_f
                 + sel_f + use_ovr + load_f
                 + acc_vf + sw_xy.x + aci_x
                 + bvs_f_add.x + bvv_f_add.x + splat_f3.x
                 + bc_u_to_f + m_sign_f + m_sat + m_norm.x
                 + zv_f32 + comp_v2f.x + use_c_f32;

        out_buf[idx] = arr_len
                     + u32(cr_i) + bi_u_add + bi_u_shl + m_ctz
                     + m_pop + m_rev + m_ext_u + m_ins_u + m_clz
                     + bi_u_and + bi_u_or + bi_u_xor + m_max_uu
                     + bc_f_to_u + u32(as_f_to_i) + use_c_u32
                     + bvs_u_add.x + bvv_u_add.x + acc_vu;
    }
}
