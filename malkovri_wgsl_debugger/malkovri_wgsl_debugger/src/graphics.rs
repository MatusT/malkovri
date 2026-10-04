//! CPU triangle interpolation shared by supplied and shader-generated vertex outputs.
use std::collections::BTreeMap;

use crate::{LocationInput, Primitive, ShaderProgram, Value};
use naga::{Interpolation, Sampling, ScalarKind};

pub const MAX_FRAGMENT_INVOCATIONS: usize = 65_536;

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Viewport {
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PixelRange {
    pub from: [u32; 2],
    pub to: [u32; 2],
}

#[derive(Clone, Debug)]
pub struct RasterConfig {
    pub viewport: Viewport,
    pub pixel_range: Option<PixelRange>,
}

#[derive(Clone, Debug)]
pub struct VertexOutput {
    pub position: [f32; 4],
    pub locations: BTreeMap<u32, Value>,
    pub vertex_index: u32,
    pub instance_index: u32,
}

#[derive(Clone, Debug)]
pub struct FragmentInput {
    pub position: [f32; 4],
    pub front_facing: bool,
    pub sample_mask: u32,
    pub locations: BTreeMap<u32, Value>,
}

#[derive(Clone, Debug)]
pub struct FragmentQuad {
    pub origin: [u32; 2],
    pub primitive_index: u32,
    pub instance_index: u32,
    /// Geometrically covered lanes within the viewport.
    pub coverage: u8,
    /// Covered lanes inside the requested range. Other lanes are helpers.
    pub selected: u8,
    pub inputs: [FragmentInput; 4],
}

impl RasterConfig {
    pub fn validate(&self) -> Result<PixelRange, String> {
        let Viewport { width, height } = self.viewport;
        if width == 0 || height == 0 || width > 1 << 23 || height > 1 << 23 {
            return Err(
                "viewport dimensions must be in [1, 8388608] for exact f32 pixel centers".into(),
            );
        }
        let range = self.pixel_range.clone().unwrap_or(PixelRange {
            from: [0, 0],
            to: [width, height],
        });
        for (axis, limit) in [width, height].into_iter().enumerate() {
            if range.from[axis] >= range.to[axis] || range.to[axis] > limit {
                return Err(
                    "pixelRange requires 0 <= from < to <= viewport on each axis (to is exclusive)"
                        .into(),
                );
            }
        }
        Ok(range)
    }
}

fn edge(a: [f64; 2], b: [f64; 2], p: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
}
fn top_left(a: [f64; 2], b: [f64; 2]) -> bool {
    b[1] < a[1] || (b[1] == a[1] && b[0] > a[0])
}
fn covered(points: &[[f64; 2]; 3], p: [f64; 2]) -> bool {
    (0..3).all(|i| {
        let a = points[i];
        let b = points[(i + 1) % 3];
        let e = edge(a, b, p);
        e > 0. || (e == 0. && top_left(a, b))
    })
}

impl ShaderProgram {
    /// Generate only quads needed by covered pixels in the requested range.
    pub fn interpolate_fragments(
        &self,
        fragment_entry: usize,
        vertices: &[VertexOutput],
        config: &RasterConfig,
    ) -> Result<Vec<FragmentQuad>, String> {
        if self
            .module()
            .entry_points
            .get(fragment_entry)
            .is_none_or(|e| e.stage != naga::ShaderStage::Fragment)
        {
            return Err("expected fragment entry point".into());
        }
        let range = config.validate()?;
        if vertices.is_empty() || !vertices.len().is_multiple_of(3) {
            return Err("vertex outputs must contain a nonempty multiple of three records".into());
        }
        let fields = self.input_locations(fragment_entry)?;
        for field in &fields {
            if !matches!(
                field.sampling,
                None | Some(Sampling::Center | Sampling::First | Sampling::Either)
            ) {
                return Err(format!(
                    "unsupported interpolation sampling at @location({})",
                    field.location
                ));
            }
        }
        for (index, vertex) in vertices.iter().enumerate() {
            let [_, _, z, w] = vertex.position;
            if vertex.position.iter().any(|v| !v.is_finite()) || w <= 0. || z < 0. || z > w {
                return Err(format!(
                    "vertexOutputs[{index}].position requires finite coordinates, positive W, and 0 <= Z <= W; clipping is not supported"
                ));
            }
            for f in &fields {
                f.ty.validate(vertex.locations.get(&f.location).ok_or_else(|| {
                    format!("vertexOutputs[{index}] missing @location({})", f.location)
                })?)
                .map_err(|e| format!("vertexOutputs[{index}] @location({}): {e}", f.location))?;
            }
        }
        let mut quads = Vec::new();
        let mut work = 0usize;
        let mut primitive_counts = BTreeMap::<u32, u32>::new();
        for triangle in vertices.chunks_exact(3) {
            let instance = triangle[0].instance_index;
            if triangle.iter().any(|v| v.instance_index != instance) {
                return Err("triangle crosses instance boundaries".into());
            }
            let primitive = primitive_counts.entry(instance).or_default();
            let primitive_index = *primitive;
            *primitive += 1;
            let mut v = [&triangle[0], &triangle[1], &triangle[2]];
            let mut points = v.map(|v| {
                let [x, y, _, w] = v.position.map(f64::from);
                [
                    (x / w + 1.) * f64::from(config.viewport.width) / 2.,
                    (1. - y / w) * f64::from(config.viewport.height) / 2.,
                ]
            });
            let area = edge(points[0], points[1], points[2]);
            if area == 0. {
                continue;
            }
            let front_facing = area < 0.; // Default CCW in NDC, Y is flipped in framebuffer space.
            if area < 0. {
                points.swap(1, 2);
                v.swap(1, 2);
            }
            let area = area.abs();
            let from: [u32; 2] = std::array::from_fn(|axis| {
                points
                    .iter()
                    .map(|p| p[axis])
                    .fold(f64::INFINITY, f64::min)
                    .floor()
                    .max(f64::from(range.from[axis]))
                    .min(f64::from(range.to[axis])) as u32
            });
            let to: [u32; 2] = std::array::from_fn(|axis| {
                points
                    .iter()
                    .map(|p| p[axis])
                    .fold(f64::NEG_INFINITY, f64::max)
                    .ceil()
                    .max(f64::from(range.from[axis]))
                    .min(f64::from(range.to[axis])) as u32
            });
            if (0..2).any(|i| from[i] >= to[i]) {
                continue;
            }
            let from = from.map(|x| x & !1);
            let to = to.map(|x| (x + 1) & !1);
            let candidates = u64::from((to[0] - from[0]) / 2) * u64::from((to[1] - from[1]) / 2);
            if candidates > 1_000_000 || work as u64 + candidates > 1_000_000 {
                return Err(
                    "pixel range exceeds rasterization work limit; select a smaller range".into(),
                );
            }
            work += candidates as usize;
            for y in (from[1]..to[1]).step_by(2) {
                for x in (from[0]..to[0]).step_by(2) {
                    let pixels: [[u32; 2]; 4] =
                        std::array::from_fn(|lane| [x + (lane as u32 % 2), y + (lane as u32 / 2)]);
                    let mut coverage = 0;
                    let mut selected = 0;
                    for (lane, p) in pixels.iter().enumerate() {
                        if p[0] < config.viewport.width
                            && p[1] < config.viewport.height
                            && covered(&points, [f64::from(p[0]) + 0.5, f64::from(p[1]) + 0.5])
                        {
                            coverage |= 1 << lane;
                            if (0..2).all(|i| p[i] >= range.from[i] && p[i] < range.to[i]) {
                                selected |= 1 << lane;
                            }
                        }
                    }
                    if selected == 0 {
                        continue;
                    }
                    if quads.len() >= MAX_FRAGMENT_INVOCATIONS / 4 {
                        return Err(
                            "fragment invocation limit exceeded; select a smaller pixelRange"
                                .into(),
                        );
                    }
                    let mut inputs = Vec::with_capacity(4);
                    for (lane, pixel) in pixels.iter().enumerate() {
                        let p = pixel.map(|p| f64::from(p) + 0.5);
                        let lambda = [
                            edge(points[1], points[2], p) / area,
                            edge(points[2], points[0], p) / area,
                            edge(points[0], points[1], p) / area,
                        ];
                        let reciprocal_w: f64 = (0..3)
                            .map(|i| lambda[i] / f64::from(v[i].position[3]))
                            .sum();
                        if reciprocal_w == 0. || !reciprocal_w.is_finite() {
                            return Err(
                                "invalid interpolation denominator (including helper lanes)".into(),
                            );
                        }
                        let depth: f64 = (0..3)
                            .map(|i| {
                                lambda[i] * f64::from(v[i].position[2])
                                    / f64::from(v[i].position[3])
                            })
                            .sum();
                        let locations = fields
                            .iter()
                            .map(|f| {
                                Ok((
                                    f.location,
                                    interpolate(f, &v, &triangle[0], lambda, reciprocal_w)?,
                                ))
                            })
                            .collect::<Result<_, String>>()?;
                        let position =
                            [p[0] as f32, p[1] as f32, depth as f32, reciprocal_w as f32];
                        if position.iter().any(|v| !v.is_finite()) {
                            return Err("non-finite interpolated position".into());
                        }
                        inputs.push(FragmentInput {
                            position,
                            front_facing,
                            sample_mask: u32::from(coverage & (1 << lane) != 0),
                            locations,
                        });
                    }
                    quads.push(FragmentQuad {
                        origin: [x, y],
                        primitive_index,
                        instance_index: instance,
                        coverage,
                        selected,
                        inputs: inputs.try_into().unwrap(),
                    });
                }
            }
        }
        Ok(quads)
    }
}

fn interpolate(
    field: &LocationInput,
    vertices: &[&VertexOutput; 3],
    provoking: &VertexOutput,
    lambda: [f64; 3],
    reciprocal_w: f64,
) -> Result<Value, String> {
    let mode = field.interpolation.unwrap_or(Interpolation::Perspective);
    if mode == Interpolation::Flat {
        return Ok(provoking.locations[&field.location].clone());
    }
    if field.ty.kind != ScalarKind::Float {
        return Err("integer varyings require flat interpolation".into());
    }
    let values = vertices.map(|v| {
        v.locations[&field.location]
            .as_primitive()
            .unwrap()
            .as_f32_slice()
            .unwrap()
    });
    let result = (0..field.ty.components as usize)
        .map(|component| {
            let value: f64 = (0..3)
                .map(|i| {
                    let term = lambda[i] * f64::from(values[i][component]);
                    if mode == Interpolation::Perspective {
                        term / f64::from(vertices[i].position[3])
                    } else {
                        term
                    }
                })
                .sum();
            (if mode == Interpolation::Perspective {
                value / reciprocal_w
            } else {
                value
            }) as f32
        })
        .collect::<Vec<_>>();
    if result.iter().any(|v| !v.is_finite()) {
        return Err("non-finite interpolated varying".into());
    }
    Ok(Primitive::from(&result).into())
}

/// Inspection identity and output eligibility of a fragment invocation.
#[derive(Clone, Debug)]
pub struct FragmentInfo {
    pub pixel: [u32; 2],
    pub quad_index: usize,
    pub lane: u32,
    pub primitive_index: u32,
    pub instance_index: u32,
    pub helper: bool,
    pub discarded: bool,
}
