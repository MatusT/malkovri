# Plan: Graphics Pipeline Support (Vertex + Fragment Shaders)

## Context

The current debugger runs compute shaders. Vertex and fragment builtins already exist in `EntryPointInputs`, and naga already parses `@builtin` / `@location` bindings on function arguments and return types. However:
- Entry point return values are currently **discarded** (`evaluator.rs:431-449` only caches return values for non-entry-point function calls)
- `@location(N)` vertex shader inputs return `Value::Uninitialized` (`eval_expressions.rs:255`)
- `StepResult` carries no output data

The graphics pipeline requires: **vertex stage → rasterization → fragment stage**, where rasterization is a CPU-side software step that maps vertex outputs to fragment inputs via barycentric interpolation.

---

## Critical Files

- `malkovri_wgsl_debugger/src/evaluator.rs` — `apply_return()` needs to capture entry point return values
- `malkovri_wgsl_debugger/src/eval_expressions.rs` — `@location(N)` on vertex args needs vertex buffer data
- `malkovri_wgsl_debugger/src/entry_point_inputs.rs` — extend for vertex buffer inputs
- `malkovri_wgsl_debugger_dap/src/parse_input.rs` — parse vertex buffers and draw call from launch config
- `malkovri_wgsl_debugger_dap/src/debug_adapter.rs` — pipeline stage transitions

---

## Data Flow

```
VertexBuffer(s) + DrawCall
        ↓
  VertexDebugger          (one PerThread per vertex invocation)
        ↓  outputs: Vec<VertexOutput>
    Rasterizer             (software, CPU — not debuggable, automatic)
        ↓  outputs: Vec<FragmentInput>
  FragmentDebugger         (one PerThread per fragment)
        ↓  outputs: Vec<FragmentOutput>
```

---

## Stage 1 — Capture Entry Point Return Values (~1 day)

**Change to `Evaluator::apply_return()`:** When the returning frame is an entry point (i.e., `call_result_handle` is `None`), store the return value in a new field instead of discarding it.

```rust
// In Evaluator (or PerThread after the compute plan refactor):
pub entry_point_output: Option<Value>,
```

When the entry point returns, walk the return type's struct fields (via `module.types[result_ty].inner`) and their `naga::Binding` annotations to produce a labelled output:

```rust
pub struct ShaderOutput {
    pub builtins: HashMap<naga::BuiltIn, Value>,
    pub locations: HashMap<u32, Value>,
}
```

This is used by both vertex (to extract `position` + varyings) and fragment (to extract `frag_depth` + color outputs) stages.

---

## Stage 2 — Vertex Buffer Inputs (~1 day)

Currently `@location(N)` on vertex shader function arguments returns `Value::Uninitialized`. For vertex shaders, `@location(N)` means "read from vertex buffer attribute N for the current vertex."

**New input type:**
```rust
pub struct VertexBuffer {
    pub attributes: HashMap<u32, Vec<Value>>,  // location → per-vertex values
}
```

The user provides this in the launch config alongside the draw call:
```json
{
  "vertexBuffers": [
    { "location": 0, "values": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] }
  ],
  "drawCall": { "vertexCount": 3, "instanceCount": 1 }
}
```

In `eval_expressions.rs`, the `Binding::Location` arm for vertex function arguments reads `vertex_buffer.attributes[location][vertex_index]` instead of returning `Uninitialized`.

---

## Stage 3 — `VertexDebugger` (~1 day)

Essentially the same as `WorkgroupDebugger` from PLAN.md but for vertex invocations. Uses `Evaluator` with `HashMap<ThreadId, PerThread>`, one thread per vertex.

Per-thread inputs generated from `vertex_index` and `instance_index`:
```rust
EntryPointInputs {
    vertex_index: i as u32,
    instance_index: 0,
    // @location(N) comes from VertexBuffer
}
```

No workgroup barriers or subgroup ops needed for basic vertex shaders (vertex shaders are fully independent). The multi-thread `Evaluator` from PLAN.md still handles them correctly — threads simply never hit a barrier.

After all threads finish, collect `ShaderOutput` per thread into `Vec<VertexOutput>`:
```rust
pub struct VertexOutput {
    pub position: [f32; 4],                  // @builtin(position) — clip space
    pub locations: HashMap<u32, Value>,      // @location(N) varyings
}
```

---

## Stage 4 — Software Rasterizer (~3 days)

The rasterizer is **not debuggable** — it runs automatically between vertex and fragment stages. It takes `Vec<VertexOutput>` + viewport config and produces `Vec<FragmentInput>`.

### Spec conformance

All algorithms below are derived from the WebGPU and WGSL specifications:

| Algorithm | Spec status | Source |
|---|---|---|
| Viewport transform | Derivable — perspective divide + scale/offset | WebGPU §"Coordinate Systems" |
| `@interpolate(perspective)` | Fully specified | WGSL §"Interpolation": `(Σ vᵢ/wᵢ·λᵢ) / (Σ λᵢ/wᵢ)` |
| `@interpolate(linear)` | Fully specified | WGSL: `Σ vᵢ·λᵢ` (screen-space, no w-correction) |
| `@interpolate(flat)` / provoking vertex | Fully specified | WGSL: first vertex of each primitive |
| Depth interpolation | Fully specified | Linear over viewport-space z |
| Pixel coverage / fill rule | **Implementation-defined** | WebGPU says top-left tie-breaking only |

### Input configuration

```rust
pub struct RasterizerConfig {
    pub viewport_width: u32,
    pub viewport_height: u32,
    pub topology: PrimitiveTopology,      // TriangleList | TriangleStrip (initially TriangleList)
    pub coverage_mode: RasterizationMode, // see below
}

/// Pixel coverage rule. All major WebGPU-capable GPUs use the same top-left
/// convention and pixel-center sampling — differences are only in sub-pixel
/// precision and exact edge tie-breaking, which only matters at triangle boundaries.
pub enum RasterizationMode {
    /// Standard top-left fill convention. Pixel center at (x+0.5, y+0.5).
    /// Covers a pixel if its center lies strictly inside the triangle,
    /// or exactly on a top edge (horizontal, y decreases) or left edge
    /// (non-horizontal, going downward). Matches D3D12/Vulkan/WebGPU baseline.
    /// Source: WebGPU spec §"Rasterization", D3D12 spec §"Triangle Rasterization Rules".
    TopLeft,

    /// NVIDIA (Pascal/Turing/Ampere/Ada). Same top-left rule, 1/256 sub-pixel
    /// precision (8 fixed-point bits). Documented in NVIDIA OpenGL conformance
    /// and D3D12 HLK test suite results.
    Nvidia,

    /// AMD GCN/RDNA. Same top-left rule, 1/256 sub-pixel precision.
    /// Source: AMD "Rasterizer Order Views" and D3D12 conformance documentation.
    Amd,

    /// Apple GPU (M-series/A-series via Metal). Same top-left rule, 1/256
    /// sub-pixel precision. Source: Metal Feature Set Tables and Metal spec
    /// §"Rasterization".
    Apple,

    /// ARM Mali (Valhall/5th-gen). Follows Vulkan fill rules (top-left).
    /// Source: ARM Mali GPU Best Practices Guide §"Geometry".
    Mali,

    /// Qualcomm Adreno. Follows Vulkan fill rules.
    /// Source: Qualcomm Adreno GPU Developer Guide.
    Adreno,

    /// Permissive — covers any pixel whose center is inside or exactly on
    /// any edge. Useful when debugging shader logic and exact coverage
    /// at seams doesn't matter; maximises visible fragments.
    Permissive,
}
```

In practice, for pixels whose centers are clearly inside a triangle (the vast majority during shader debugging), all modes produce identical results. Differences only appear at triangle boundaries.

### Algorithm (per triangle, vertices A/B/C)

1. **Viewport transform** (WebGPU §"Coordinate Systems"):
   - Perspective divide: `ndc = clip.xyz / clip.w`
   - Screen space: `sx = (ndc.x + 1) × (width/2)`, `sy = (1 − ndc.y) × (height/2)` (y flipped: NDC bottom-up, screen top-down)
   - Depth: `sz = ndc.z × (maxDepth − minDepth) + minDepth`

2. **Bounding box** of triangle in screen space, clamped to viewport.

3. **For each pixel (px, py) in bounding box**, sample at `(px + 0.5, py + 0.5)`:
   - Compute edge functions (signed areas): `e0 = (B−A)×(P−A)`, etc.
   - Coverage test per `RasterizationMode` (top-left: inside if all `e ≥ 0` with tie-breaking on shared edges).
   - If covered: compute barycentric coordinates `λ = (e0, e1, e2) / (e0+e1+e2)`.
   - Interpolate varyings (see below).
   - Determine `front_facing`: positive signed area → front face.
   - Emit a `FragmentInput`.

### Interpolation formulas (from WGSL spec)

**`@interpolate(perspective)` (default)** — perspective-correct:
```
interp(v) = (λ₀·v₀/w₀ + λ₁·v₁/w₁ + λ₂·v₂/w₂) / (λ₀/w₀ + λ₁/w₁ + λ₂/w₂)
```
where `w₀/w₁/w₂` are the clip-space W values from vertex outputs.

**`@interpolate(linear)`** — screen-space linear (no perspective correction):
```
interp(v) = λ₀·v₀ + λ₁·v₁ + λ₂·v₂
```

**`@interpolate(flat)`** — no interpolation, provoking vertex (WebGPU spec: first vertex of primitive):
```
interp(v) = v₀
```

**Depth** (`@builtin(position).z`) — interpolated linearly in viewport space (same as `linear`).

**`@builtin(position).w`** in fragment shader — set to `1/clip.w` (perspective-correct reciprocal).

```rust
pub struct FragmentInput {
    pub position: [f32; 4],              // screen xy, viewport-space z, 1/clip.w
    pub front_facing: bool,
    pub locations: HashMap<u32, Value>,  // interpolated varyings
}
```

---

## Stage 5 — `FragmentDebugger` (~1 day)

Like `VertexDebugger` but each thread is a fragment. Per-thread `EntryPointInputs`:
```rust
EntryPointInputs {
    position: fragment_input.position,
    front_facing: fragment_input.front_facing,
    // @location(N) comes from fragment_input.locations
}
```

`@location(N)` on **fragment** shader arguments reads from `fragment_input.locations[N]` (interpolated varyings from rasterizer).

After all fragments finish, collect `ShaderOutput` per thread into `Vec<FragmentOutput>`:
```rust
pub struct FragmentOutput {
    pub screen_position: [u32; 2],         // which pixel
    pub locations: HashMap<u32, Value>,    // @location(N) color outputs
    pub frag_depth: Option<f32>,           // @builtin(frag_depth) if written
}
```

---

## Stage 6 — `GraphicsPipelineDebugger` (~2 days)

**New file:** `malkovri_wgsl_debugger/src/graphics_pipeline_debugger.rs`

Orchestrates the three stages with a state machine:

```rust
pub struct GraphicsPipelineDebugger {
    state: PipelineState,
    source: String,
    rasterizer_config: RasterizerConfig,
}

enum PipelineState {
    Vertex(VertexDebugger),
    Fragment {
        debugger: FragmentDebugger,
        vertex_outputs: Vec<VertexOutput>,   // kept for inspection
    },
    Finished {
        vertex_outputs: Vec<VertexOutput>,
        fragment_outputs: Vec<FragmentOutput>,
    },
}

impl GraphicsPipelineDebugger {
    pub fn new(source, vertex_ep_index, fragment_ep_index, config, vertex_buffers, draw_call, rasterizer_config, bindings) -> Result<Self>
    pub fn step(&mut self) -> Result<PipelineStepResult>
    pub fn stage(&self) -> PipelineStage          // Vertex | Fragment | Finished
    pub fn threads(&self) -> Vec<ThreadInfo>       // delegates to current stage
    pub fn current_location(&self, thread_id: u32) -> Option<SourceLocation>
    pub fn local_variables(&self, thread_id: u32) -> Vec<Variable>
    pub fn argument_variables(&self, thread_id: u32) -> Vec<Variable>
    pub fn global_variables(&self) -> Vec<Variable>
    // Output inspection:
    pub fn vertex_outputs(&self) -> &[VertexOutput]    // available after vertex stage
    pub fn fragment_outputs(&self) -> &[FragmentOutput] // available after Finished
    pub fn source(&self) -> &str
}

pub enum PipelineStepResult {
    Continue,
    StageTransition(PipelineStage),  // vertex → fragment, or fragment → finished
    Finished,
}
```

**`step()` transitions:**
- While in `Vertex`: delegate to `VertexDebugger::step()`. When it returns `WorkgroupStepResult::Finished`, run the rasterizer automatically, transition to `Fragment`.
- While in `Fragment`: delegate to `FragmentDebugger::step()`. When finished, transition to `Finished`.

---

## Stage 7 — DAP Adapter Changes (~2 days)

Extend `parse_input.rs` to parse vertex buffers and draw call from launch config. Add `GraphicsPipelineDebugger` to `DebuggerKind`:

```rust
enum DebuggerKind {
    Single(Debugger),
    Workgroup(WorkgroupDebugger),
    Graphics(GraphicsPipelineDebugger),
}
```

The DAP `threads` response labels threads differently per stage:
- Vertex: `"vertex {vertex_index}"`
- Fragment: `"fragment [{px},{py}]"`

A custom `output` event is sent at stage transitions so the user can see in VS Code when rasterization happens and how many fragments were generated.

---

## What Is NOT Included (future work)

- **Clipping** — vertices behind the near plane produce incorrect results; can be added later.
- **Depth testing / depth buffer** — fragments are not discarded based on depth; all are passed to the fragment shader.
- **MSAA** — single sample per pixel only.
- **Centroid / sample interpolation** — only `center` and `flat` modes are implemented.
- **Index buffers** — only non-indexed draws (`vertexCount` vertices in order). Index buffer support is a straightforward extension.
- **Multiple instances** — `instanceCount > 1` is not supported initially.
- **Geometry / mesh shaders** — not in WGSL scope.

---

## Edge Cases

- **No fragments generated** (triangle fully outside viewport): fragment stage is skipped, `PipelineStepResult::Finished` immediately after rasterization.
- **Degenerate triangles** (zero area): skipped by rasterizer (edge function sum is zero).
- **`@interpolate(flat)`** — provoking vertex semantics: use the values from vertex 0 of each triangle (WGSL default is "first" provoking vertex).
- **`position.w == 0`** — perspective division would divide by zero; skip the triangle or clamp.
- **Fragment discard (`Statement::Kill`)** — currently a no-op; a killed fragment's `FragmentOutput` entry should be omitted or flagged.

---

## Estimated Effort

| Stage | Days |
|---|---|
| 1. Capture entry point return values | 1 |
| 2. Vertex buffer inputs | 1 |
| 3. VertexDebugger | 1 |
| 4. Software rasterizer | 3 |
| 5. FragmentDebugger | 1 |
| 6. GraphicsPipelineDebugger | 2 |
| 7. DAP adapter | 2 |
| **Total** | **~11 days** |
