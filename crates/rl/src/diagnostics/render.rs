//! Optional GUI-independent 2D scenes captured only during explicit evaluation.
//!
//! World coordinates use x right/y up and uniform letterbox scaling. Colors are
//! straight-alpha sRGB RGBA. Stroke widths and centered text sizes are logical
//! screen points. Paint ascending layers, preserving item order within a layer.
//! Polygons must be strictly convex, without a repeated closing vertex.
//!
//! Scene bytes, primitive/point counts and decoded UTF-8 strings are admitted
//! before typed allocations. JSON parser escape scratch is bounded by the 64 KiB
//! scene input; decoding retains at most one scene. Scene trees use at most 256
//! item slots, at most 4096 variable point capacity slots and 4096 text bytes (plus fixed fields).
//! Trajectory/evaluation byte limits count retained byte and offset capacities.
//! One bounded encoder buffer and one returned source scene are temporary extra
//! ownership. An arbitrary RenderSource can allocate internally before returning;
//! the framework cannot police that implementation or interior-mutability RNGs.
use super::{
    codec,
    schema::{Array, ArrayKey, Chunk, Record},
};
use crate::{Error, Result, core::ensure};
use serde::{
    Deserialize, Serialize,
    de::{self, DeserializeSeed, MapAccess, SeqAccess, Visitor},
};
use std::{
    collections::BTreeMap,
    fmt,
    sync::{Arc, Weak},
};

pub const RENDER_VERSION: u32 = 1;
pub type RenderPoint = [f64; 2];
pub type RenderColor = [u8; 4];
pub trait RenderSource {
    /// Read-only presentation: implementations must not consume simulator RNGs.
    fn render_scene(&self) -> Result<RenderScene>;
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RenderBounds {
    pub min: RenderPoint,
    pub max: RenderPoint,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RenderStroke {
    pub color: RenderColor,
    pub width: f32,
}
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RenderScene {
    pub version: u32,
    pub bounds: RenderBounds,
    pub background: RenderColor,
    pub items: Vec<RenderItem>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RenderItem {
    pub layer: i16,
    pub primitive: RenderPrimitive,
}
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RenderPrimitive {
    Line {
        from: RenderPoint,
        to: RenderPoint,
        stroke: RenderStroke,
    },
    Polyline {
        points: Vec<RenderPoint>,
        stroke: RenderStroke,
    },
    Polygon {
        points: Vec<RenderPoint>,
        fill: Option<RenderColor>,
        stroke: Option<RenderStroke>,
    },
    Circle {
        center: RenderPoint,
        radius: f64,
        fill: Option<RenderColor>,
        stroke: Option<RenderStroke>,
    },
    Rectangle {
        min: RenderPoint,
        max: RenderPoint,
        fill: Option<RenderColor>,
        stroke: Option<RenderStroke>,
    },
    Text {
        position: RenderPoint,
        text: String,
        color: RenderColor,
        size: f32,
    },
}
#[derive(Debug, Clone, Copy)]
pub struct RenderLimits {
    /// Maximum encoded bytes in one scene (ceiling 64 KiB).
    pub max_scene_bytes: usize,
    pub max_primitives: usize,
    pub max_points: usize,
    pub max_text_bytes: usize,
    pub max_frames: usize,
    /// Retained byte/offset Vec capacities during capture; wire lengths on read.
    pub max_trajectory_bytes: usize,
    /// Sum of retained render capacities for selected episodes; capture only.
    pub max_evaluation_bytes: usize,
}
impl Default for RenderLimits {
    fn default() -> Self {
        Self {
            max_scene_bytes: 64 * 1024,
            max_primitives: 256,
            max_points: 2048,
            max_text_bytes: 4096,
            max_frames: 4096,
            max_trajectory_bytes: 8 * 1024 * 1024,
            max_evaluation_bytes: 32 * 1024 * 1024,
        }
    }
}
impl RenderLimits {
    pub fn validate(&self) -> Result<()> {
        let hard = Self::default();
        for (value, ceiling) in [
            (self.max_scene_bytes, hard.max_scene_bytes),
            (self.max_primitives, hard.max_primitives),
            (self.max_points, hard.max_points),
            (self.max_text_bytes, hard.max_text_bytes),
            (self.max_frames, hard.max_frames),
            (self.max_trajectory_bytes, hard.max_trajectory_bytes),
            (self.max_evaluation_bytes, hard.max_evaluation_bytes),
        ] {
            ensure(
                value <= ceiling,
                "render limits may only tighten hard ceilings",
            )?;
        }
        Ok(())
    }
}
/// Render only this many of the numerical evaluation trajectories; zero disables
/// all render calls. Existing evaluation entrypoints do not invoke RenderSource.
#[derive(Debug, Clone)]
pub struct RenderCaptureConfig {
    pub trajectories: usize,
    pub limits: RenderLimits,
}
impl Default for RenderCaptureConfig {
    fn default() -> Self {
        Self {
            trajectories: 1,
            limits: RenderLimits::default(),
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RenderTrack {
    pub version: u32,
    pub offsets: ArrayKey,
    pub data: ArrayKey,
}

fn point(p: &RenderPoint) -> bool {
    p.iter().all(|n| n.is_finite())
}
fn bounds(min: &RenderPoint, max: &RenderPoint) -> bool {
    point(min)
        && point(max)
        && (0..2).all(|i| {
            let span = max[i] - min[i];
            span.is_finite() && span > 0.
        })
}
fn stroke(s: &RenderStroke) -> bool {
    s.width.is_finite() && s.width > 0. && s.width <= 1024.
}
fn closed(fill: &Option<RenderColor>, line: &Option<RenderStroke>) -> bool {
    (fill.is_some() || line.is_some()) && line.as_ref().is_none_or(stroke)
}
impl RenderScene {
    pub fn validate(&self, limits: &RenderLimits) -> Result<()> {
        limits.validate()?;
        ensure(
            self.version == RENDER_VERSION && bounds(&self.bounds.min, &self.bounds.max),
            "invalid render scene version or bounds",
        )?;
        ensure(
            self.items.len() <= limits.max_primitives,
            "render primitive budget exceeded",
        )?;
        let mut points = 0usize;
        let mut text_bytes = 0usize;
        for item in &self.items {
            let valid = match &item.primitive {
                RenderPrimitive::Line {
                    from,
                    to,
                    stroke: s,
                } => point(from) && point(to) && stroke(s),
                RenderPrimitive::Polyline {
                    points: p,
                    stroke: s,
                } => {
                    points = points.saturating_add(p.len());
                    p.len() >= 2 && p.iter().all(point) && stroke(s)
                }
                RenderPrimitive::Polygon {
                    points: p,
                    fill,
                    stroke: s,
                } => {
                    points = points.saturating_add(p.len());
                    // Check the count before the bounded quadratic convexity test.
                    ensure(points <= limits.max_points, "render point budget exceeded")?;
                    closed(fill, s) && convex(p)
                }
                RenderPrimitive::Circle {
                    center,
                    radius,
                    fill,
                    stroke: s,
                } => {
                    point(center)
                        && radius.is_finite()
                        && *radius > 0.
                        && closed(fill, s)
                        && center
                            .iter()
                            .all(|c| (c + radius).is_finite() && (c - radius).is_finite())
                }
                RenderPrimitive::Rectangle {
                    min,
                    max,
                    fill,
                    stroke: s,
                } => bounds(min, max) && closed(fill, s),
                RenderPrimitive::Text {
                    position,
                    text,
                    size,
                    ..
                } => {
                    text_bytes = text_bytes.saturating_add(text.len());
                    point(position)
                        && text.len() <= 1024
                        && size.is_finite()
                        && *size > 0.
                        && *size <= 1024.
                }
            };
            ensure(valid, "invalid render primitive geometry or style")?;
            ensure(
                points <= limits.max_points && text_bytes <= limits.max_text_bytes,
                "render point or text budget exceeded",
            )?;
        }
        Ok(())
    }
}
fn convex(p: &[RenderPoint]) -> bool {
    if p.len() < 3 || !p.iter().all(point) {
        return false;
    }
    // O(n log n) monotone hull: all input vertices must be strict hull vertices,
    // and their original order must be a cyclic traversal (either orientation).
    // This rejects stars/self-intersections unlike consecutive-turn tests alone.
    // Bounded scratch: n sorted indices and at most 2n hull indices.
    let cross = |a: usize, b: usize, c: usize| {
        (p[b][0] - p[a][0]) * (p[c][1] - p[a][1]) - (p[b][1] - p[a][1]) * (p[c][0] - p[a][0])
    };
    let mut sorted: Vec<usize> = (0..p.len()).collect();
    sorted.sort_unstable_by(|&a, &b| {
        p[a][0]
            .total_cmp(&p[b][0])
            .then(p[a][1].total_cmp(&p[b][1]))
    });
    if sorted.windows(2).any(|w| p[w[0]] == p[w[1]]) {
        return false;
    }
    let mut hull = Vec::with_capacity(2 * p.len());
    for &index in &sorted {
        while hull.len() >= 2 {
            let turn = cross(hull[hull.len() - 2], hull[hull.len() - 1], index);
            if !turn.is_finite() {
                return false;
            }
            if turn > 0. {
                break;
            }
            hull.pop();
        }
        hull.push(index);
    }
    let lower = hull.len();
    for &index in sorted.iter().rev().skip(1) {
        while hull.len() > lower {
            let turn = cross(hull[hull.len() - 2], hull[hull.len() - 1], index);
            if !turn.is_finite() {
                return false;
            }
            if turn > 0. {
                break;
            }
            hull.pop();
        }
        hull.push(index);
    }
    hull.pop();
    hull.len() == p.len()
        && ((0..hull.len()).all(|i| hull[(i + 1) % hull.len()] == (hull[i] + 1) % p.len())
            || (0..hull.len())
                .all(|i| hull[(i + 1) % hull.len()] == (hull[i] + p.len() - 1) % p.len()))
}

// Seeds admit each variable-size value before allocating it. Fixed fields use
// ordinary serde visitors; unknown/duplicate fields are rejected at every level.
struct Budget<'a> {
    limits: &'a RenderLimits,
    primitives: usize,
    points: usize,
    text: usize,
}
impl<'a> Budget<'a> {
    fn new(limits: &'a RenderLimits) -> Self {
        Self {
            limits,
            primitives: 0,
            points: 0,
            text: 0,
        }
    }
}
fn required<T, E: de::Error>(v: Option<T>, name: &'static str) -> std::result::Result<T, E> {
    v.ok_or_else(|| E::missing_field(name))
}
fn once<E: de::Error>(seen: &mut u32, bit: u32) -> std::result::Result<(), E> {
    if *seen & bit != 0 {
        return Err(E::custom("duplicate render field"));
    }
    *seen |= bit;
    Ok(())
}
fn push<T, E: de::Error>(v: &mut Vec<T>, value: T, ceiling: usize) -> std::result::Result<(), E> {
    if v.len() >= ceiling {
        return Err(E::custom("render sequence budget exceeded"));
    }
    if v.len() == v.capacity() {
        let target = v.capacity().saturating_mul(2).max(1).min(ceiling);
        v.try_reserve_exact(target - v.len()).map_err(E::custom)?;
    }
    v.push(value);
    Ok(())
}
#[derive(Deserialize)]
#[serde(field_identifier, rename_all = "snake_case")]
enum SceneField {
    Version,
    Bounds,
    Background,
    Items,
}
struct SceneSeed<'a>(&'a RenderLimits);
impl<'de> DeserializeSeed<'de> for SceneSeed<'_> {
    type Value = RenderScene;
    fn deserialize<D: de::Deserializer<'de>>(
        self,
        d: D,
    ) -> std::result::Result<Self::Value, D::Error> {
        self.0.validate().map_err(de::Error::custom)?;
        d.deserialize_map(self)
    }
}
impl<'de> Visitor<'de> for SceneSeed<'_> {
    type Value = RenderScene;
    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("bounded render scene")
    }
    fn visit_map<M: MapAccess<'de>>(
        self,
        mut map: M,
    ) -> std::result::Result<Self::Value, M::Error> {
        let (mut version, mut bounds, mut background, mut items) = (None, None, None, None);
        let mut seen = 0;
        let mut budget = Budget::new(self.0);
        while let Some(key) = map.next_key::<SceneField>()? {
            match key {
                SceneField::Version => {
                    once(&mut seen, 1)?;
                    version = Some(map.next_value()?);
                }
                SceneField::Bounds => {
                    once(&mut seen, 2)?;
                    bounds = Some(map.next_value()?);
                }
                SceneField::Background => {
                    once(&mut seen, 4)?;
                    background = Some(map.next_value()?);
                }
                SceneField::Items => {
                    once(&mut seen, 8)?;
                    items = Some(map.next_value_seed(ItemsSeed(&mut budget))?);
                }
            }
        }
        let scene = RenderScene {
            version: required(version, "version")?,
            bounds: required(bounds, "bounds")?,
            background: required(background, "background")?,
            items: required(items, "items")?,
        };
        scene.validate(self.0).map_err(de::Error::custom)?;
        Ok(scene)
    }
}
impl<'de> Deserialize<'de> for RenderScene {
    fn deserialize<D: de::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        SceneSeed(&RenderLimits::default()).deserialize(d)
    }
}
struct ItemsSeed<'a, 'b>(&'a mut Budget<'b>);
impl<'de> DeserializeSeed<'de> for ItemsSeed<'_, '_> {
    type Value = Vec<RenderItem>;
    fn deserialize<D: de::Deserializer<'de>>(
        self,
        d: D,
    ) -> std::result::Result<Self::Value, D::Error> {
        d.deserialize_seq(self)
    }
}
impl<'de> Visitor<'de> for ItemsSeed<'_, '_> {
    type Value = Vec<RenderItem>;
    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("bounded render items")
    }
    fn visit_seq<A: SeqAccess<'de>>(
        self,
        mut seq: A,
    ) -> std::result::Result<Self::Value, A::Error> {
        let mut out = Vec::new();
        while let Some(item) = seq.next_element_seed(ItemSeed(self.0))? {
            push(&mut out, item, self.0.limits.max_primitives)?;
        }
        Ok(out)
    }
}
#[derive(Deserialize)]
#[serde(field_identifier, rename_all = "snake_case")]
enum ItemField {
    Layer,
    Primitive,
}
struct ItemSeed<'a, 'b>(&'a mut Budget<'b>);
impl<'de> DeserializeSeed<'de> for ItemSeed<'_, '_> {
    type Value = RenderItem;
    fn deserialize<D: de::Deserializer<'de>>(
        self,
        d: D,
    ) -> std::result::Result<Self::Value, D::Error> {
        if self.0.primitives >= self.0.limits.max_primitives {
            return Err(de::Error::custom("render primitive budget exceeded"));
        }
        self.0.primitives += 1;
        d.deserialize_map(self)
    }
}
impl<'de> Visitor<'de> for ItemSeed<'_, '_> {
    type Value = RenderItem;
    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("render item")
    }
    fn visit_map<M: MapAccess<'de>>(
        self,
        mut map: M,
    ) -> std::result::Result<Self::Value, M::Error> {
        let (mut layer, mut primitive) = (None, None);
        let mut seen = 0;
        while let Some(key) = map.next_key::<ItemField>()? {
            match key {
                ItemField::Layer => {
                    once(&mut seen, 1)?;
                    layer = Some(map.next_value()?);
                }
                ItemField::Primitive => {
                    once(&mut seen, 2)?;
                    primitive = Some(map.next_value_seed(PrimitiveSeed(self.0))?);
                }
            }
        }
        Ok(RenderItem {
            layer: required(layer, "layer")?,
            primitive: required(primitive, "primitive")?,
        })
    }
}
struct PointsSeed<'a, 'b>(&'a mut Budget<'b>);
impl<'de> DeserializeSeed<'de> for PointsSeed<'_, '_> {
    type Value = Vec<RenderPoint>;
    fn deserialize<D: de::Deserializer<'de>>(
        self,
        d: D,
    ) -> std::result::Result<Self::Value, D::Error> {
        d.deserialize_seq(self)
    }
}
impl<'de> Visitor<'de> for PointsSeed<'_, '_> {
    type Value = Vec<RenderPoint>;
    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("bounded render points")
    }
    fn visit_seq<A: SeqAccess<'de>>(
        self,
        mut seq: A,
    ) -> std::result::Result<Self::Value, A::Error> {
        let mut out = Vec::new();
        while let Some(p) = seq.next_element_seed(PointSeed(self.0))? {
            push(&mut out, p, self.0.limits.max_points)?;
        }
        Ok(out)
    }
}
struct PointSeed<'a, 'b>(&'a mut Budget<'b>);
impl<'de> DeserializeSeed<'de> for PointSeed<'_, '_> {
    type Value = RenderPoint;
    fn deserialize<D: de::Deserializer<'de>>(
        self,
        d: D,
    ) -> std::result::Result<Self::Value, D::Error> {
        if self.0.points >= self.0.limits.max_points {
            return Err(de::Error::custom("render point budget exceeded"));
        }
        self.0.points += 1;
        RenderPoint::deserialize(d)
    }
}
struct TextSeed<'a, 'b>(&'a mut Budget<'b>);
impl<'de> DeserializeSeed<'de> for TextSeed<'_, '_> {
    type Value = String;
    fn deserialize<D: de::Deserializer<'de>>(
        self,
        d: D,
    ) -> std::result::Result<Self::Value, D::Error> {
        d.deserialize_str(self)
    }
}
impl<'de> Visitor<'de> for TextSeed<'_, '_> {
    type Value = String;
    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("bounded render text")
    }
    fn visit_str<E: de::Error>(self, v: &str) -> std::result::Result<String, E> {
        if v.len() > 1024 || v.len() > self.0.limits.max_text_bytes.saturating_sub(self.0.text) {
            return Err(E::custom("render text budget exceeded"));
        }
        self.0.text += v.len();
        Ok(v.to_owned())
    }
}
#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum Kind {
    Line,
    Polyline,
    Polygon,
    Circle,
    Rectangle,
    Text,
}
#[derive(Deserialize, Clone, Copy)]
#[serde(field_identifier, rename_all = "snake_case")]
enum PrimitiveField {
    Kind,
    From,
    To,
    Stroke,
    Points,
    Fill,
    Center,
    Radius,
    Min,
    Max,
    Position,
    Text,
    Color,
    Size,
}
struct PrimitiveSeed<'a, 'b>(&'a mut Budget<'b>);
impl<'de> DeserializeSeed<'de> for PrimitiveSeed<'_, '_> {
    type Value = RenderPrimitive;
    fn deserialize<D: de::Deserializer<'de>>(
        self,
        d: D,
    ) -> std::result::Result<Self::Value, D::Error> {
        d.deserialize_map(self)
    }
}
impl<'de> Visitor<'de> for PrimitiveSeed<'_, '_> {
    type Value = RenderPrimitive;
    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("render primitive")
    }
    fn visit_map<M: MapAccess<'de>>(
        self,
        mut map: M,
    ) -> std::result::Result<Self::Value, M::Error> {
        use PrimitiveField as F;
        let (
            mut kind,
            mut from,
            mut to,
            mut stroke,
            mut points,
            mut fill,
            mut center,
            mut radius,
            mut min,
            mut max,
            mut position,
            mut text,
            mut color,
            mut size,
        ) = (
            None, None, None, None, None, None, None, None, None, None, None, None, None, None,
        );
        let mut seen = 0;
        while let Some(key) = map.next_key::<F>()? {
            once(&mut seen, 1 << key as u32)?;
            match key {
                F::Kind => kind = Some(map.next_value::<Kind>()?),
                F::From => from = Some(map.next_value()?),
                F::To => to = Some(map.next_value()?),
                F::Stroke => stroke = map.next_value::<Option<RenderStroke>>()?,
                F::Points => points = Some(map.next_value_seed(PointsSeed(self.0))?),
                F::Fill => fill = map.next_value::<Option<RenderColor>>()?,
                F::Center => center = Some(map.next_value()?),
                F::Radius => radius = Some(map.next_value()?),
                F::Min => min = Some(map.next_value()?),
                F::Max => max = Some(map.next_value()?),
                F::Position => position = Some(map.next_value()?),
                F::Text => text = Some(map.next_value_seed(TextSeed(self.0))?),
                F::Color => color = Some(map.next_value()?),
                F::Size => size = Some(map.next_value()?),
            }
        }
        let (value, fields): (_, &[F]) = match required(kind, "kind")? {
            Kind::Line => (
                RenderPrimitive::Line {
                    from: required(from, "from")?,
                    to: required(to, "to")?,
                    stroke: required(stroke, "stroke")?,
                },
                &[F::From, F::To, F::Stroke],
            ),
            Kind::Polyline => (
                RenderPrimitive::Polyline {
                    points: required(points, "points")?,
                    stroke: required(stroke, "stroke")?,
                },
                &[F::Points, F::Stroke],
            ),
            Kind::Polygon => (
                RenderPrimitive::Polygon {
                    points: required(points, "points")?,
                    fill,
                    stroke,
                },
                &[F::Points, F::Fill, F::Stroke],
            ),
            Kind::Circle => (
                RenderPrimitive::Circle {
                    center: required(center, "center")?,
                    radius: required(radius, "radius")?,
                    fill,
                    stroke,
                },
                &[F::Center, F::Radius, F::Fill, F::Stroke],
            ),
            Kind::Rectangle => (
                RenderPrimitive::Rectangle {
                    min: required(min, "min")?,
                    max: required(max, "max")?,
                    fill,
                    stroke,
                },
                &[F::Min, F::Max, F::Fill, F::Stroke],
            ),
            Kind::Text => (
                RenderPrimitive::Text {
                    position: required(position, "position")?,
                    text: required(text, "text")?,
                    color: required(color, "color")?,
                    size: required(size, "size")?,
                },
                &[F::Position, F::Text, F::Color, F::Size],
            ),
        };
        let allowed = fields.iter().fold(1u32, |a, f| a | (1 << *f as u32));
        if seen & !allowed != 0 {
            return Err(de::Error::custom(
                "unexpected field for render primitive kind",
            ));
        }
        Ok(value)
    }
}
impl<'de> Deserialize<'de> for RenderPrimitive {
    fn deserialize<D: de::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let limits = RenderLimits::default();
        PrimitiveSeed(&mut Budget::new(&limits)).deserialize(d)
    }
}
fn decode_scene(bytes: &[u8], limits: &RenderLimits) -> Result<RenderScene> {
    ensure(
        bytes.len() <= limits.max_scene_bytes,
        "render scene byte budget exceeded",
    )?;
    let mut decoder = serde_json::Deserializer::from_slice(bytes);
    let scene = SceneSeed(limits).deserialize(&mut decoder)?;
    decoder.end()?;
    Ok(scene)
}

/// Encoded captured scenes. Private fields prevent unvalidated tracks being
/// constructed through evaluation; recording readers independently validate.
#[derive(Debug, Clone)]
pub struct RenderFrames {
    offsets: Vec<u64>,
    bytes: Vec<u8>,
}
impl RenderFrames {
    pub fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn retained_bytes(&self) -> usize {
        self.bytes
            .capacity()
            .saturating_add(self.offsets.capacity().saturating_mul(8))
    }
    pub(super) fn new() -> Self {
        Self {
            offsets: Vec::new(),
            bytes: Vec::new(),
        }
    }
    pub(super) fn capture(
        &mut self,
        scene: &RenderScene,
        limits: &RenderLimits,
        total: &mut usize,
    ) -> Result<()> {
        ensure(
            self.len() < limits.max_frames,
            "render frame budget exceeded",
        )?;
        scene.validate(limits)?;
        let encoded = codec::bounded_json(scene, limits.max_scene_bytes)?;
        let bytes = self
            .bytes
            .len()
            .checked_add(encoded.len())
            .ok_or_else(|| Error::Shape("render byte overflow".into()))?;
        let offsets = self
            .offsets
            .len()
            .max(1)
            .checked_add(1)
            .ok_or_else(|| Error::Shape("render offset overflow".into()))?;
        let charge = offsets
            .checked_mul(8)
            .and_then(|n| n.checked_add(bytes))
            .ok_or_else(|| Error::Shape("render ownership overflow".into()))?;
        let previous = if self.is_empty() {
            0
        } else {
            self.retained_bytes()
        };
        let next = total
            .checked_sub(previous)
            .and_then(|n| n.checked_add(charge))
            .ok_or_else(|| Error::Shape("render evaluation ownership overflow".into()))?;
        ensure(
            charge <= limits.max_trajectory_bytes && next <= limits.max_evaluation_bytes,
            "render trajectory/evaluation byte budget exceeded",
        )?;
        self.bytes
            .try_reserve_exact(bytes - self.bytes.len())
            .map_err(|e| Error::State(e.to_string()))?;
        self.offsets
            .try_reserve_exact(offsets - self.offsets.len())
            .map_err(|e| Error::State(e.to_string()))?;
        // Allocators may provide additional capacity; account the actual result.
        let actual = self.retained_bytes();
        let next = total
            .checked_sub(previous)
            .and_then(|n| n.checked_add(actual))
            .ok_or_else(|| Error::Shape("render evaluation ownership overflow".into()))?;
        ensure(
            actual <= limits.max_trajectory_bytes && next <= limits.max_evaluation_bytes,
            "render retained capacity budget exceeded",
        )?;
        self.bytes.extend_from_slice(&encoded);
        if self.offsets.is_empty() {
            self.offsets.push(0);
        }
        self.offsets.push(bytes as u64);
        *total = next;
        Ok(())
    }
    pub(super) fn arrays(&self, arrays: &mut BTreeMap<String, Array>) -> Result<RenderTrack> {
        arrays.insert(
            "render/data".into(),
            Array::u8(vec![self.bytes.len()], self.bytes.clone())?,
        );
        arrays.insert(
            "render/offsets".into(),
            Array::u64(vec![self.offsets.len()], self.offsets.clone())?,
        );
        Ok(RenderTrack {
            version: RENDER_VERSION,
            data: "render/data".into(),
            offsets: "render/offsets".into(),
        })
    }
}
#[cfg(test)]
thread_local! {
    static TRACK_PAIR_VISITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn track_parts<'a>(
    chunk: &'a Chunk,
    limits: &RenderLimits,
) -> Result<Option<(&'a [u64], &'a [u8])>> {
    limits.validate()?;
    let Record::Trajectory(record) = &chunk.record else {
        return Err(Error::State("render scene requires a trajectory".into()));
    };
    let Some(track) = &record.render else {
        return Ok(None);
    };
    ensure(
        track.version == RENDER_VERSION,
        "unsupported render track version",
    )?;
    let get = |key: &str| {
        chunk
            .arrays
            .get(key)
            .ok_or_else(|| Error::Shape("missing render array".into()))
    };
    let observations = get(&record.observations)?;
    ensure(
        observations.shape().len() == 2,
        "invalid render observation shape",
    )?;
    let frames = observations.shape()[0];
    let rewards = get(&record.rewards)?;
    ensure(
        rewards.shape().first().and_then(|n| n.checked_add(1)) == Some(frames),
        "render scenes must align with T+1 observations",
    )?;
    let offsets = get(&track.offsets)?;
    let data = get(&track.data)?;
    ensure(
        frames > 0
            && frames <= limits.max_frames
            && frames.checked_add(1) == offsets.shape().first().copied()
            && offsets.shape().len() == 1
            && data.shape().len() == 1,
        "invalid render track shape or frame budget",
    )?;
    let offsets = offsets.as_u64()?;
    let data = data.as_u8()?;
    ensure(
        offsets.len() == frames + 1,
        "render offset count differs from frame count",
    )?;
    let charge = data
        .len()
        .checked_add(
            offsets
                .len()
                .checked_mul(8)
                .ok_or_else(|| Error::Shape("render track overflow".into()))?,
        )
        .ok_or_else(|| Error::Shape("render track overflow".into()))?;
    ensure(
        charge <= limits.max_trajectory_bytes,
        "render trajectory byte budget exceeded",
    )?;
    ensure(
        offsets.first() == Some(&0) && offsets.last().copied() == u64::try_from(data.len()).ok(),
        "invalid render offsets endpoints",
    )?;
    for pair in offsets.windows(2) {
        #[cfg(test)]
        TRACK_PAIR_VISITS.with(|count| count.set(count.get() + 1));
        let a =
            usize::try_from(pair[0]).map_err(|_| Error::Shape("render offset overflow".into()))?;
        let b =
            usize::try_from(pair[1]).map_err(|_| Error::Shape("render offset overflow".into()))?;
        ensure(
            a < b && b <= data.len() && b - a <= limits.max_scene_bytes,
            "invalid render offset or scene byte budget",
        )?;
    }
    Ok(Some((offsets, data)))
}
/// Validating wire adapter for arbitrary chunks. None means no render track; invalid records,
/// steps, versions or budgets produce errors. Decodes only the selected scene.
pub fn scene_at(chunk: &Chunk, step: usize, limits: &RenderLimits) -> Result<Option<RenderScene>> {
    let Some((offsets, data)) = track_parts(chunk, limits)? else {
        return Ok(None);
    };
    selected_scene(offsets, data, step, limits).map(Some)
}

fn selected_scene(
    offsets: &[u64],
    data: &[u8],
    step: usize,
    limits: &RenderLimits,
) -> Result<RenderScene> {
    ensure(
        step < offsets.len().saturating_sub(1),
        "render step outside trajectory",
    )?;
    let a = usize::try_from(offsets[step])
        .map_err(|_| Error::Shape("render offset overflow".into()))?;
    let b = usize::try_from(offsets[step + 1])
        .map_err(|_| Error::Shape("render offset overflow".into()))?;
    let bytes = data
        .get(a..b)
        .ok_or_else(|| Error::Shape("invalid render offset or scene byte budget".into()))?;
    decode_scene(bytes, limits)
}

/// A once-validated track bound to one immutable `Arc<Chunk>` allocation and
/// the complete limits supplied at construction. Retains no raw chunk payload.
/// Selected scenes are still decoded and validated on every lookup.
pub struct ValidatedRenderTrack {
    owner: Weak<Chunk>,
    limits: RenderLimits,
}
impl ValidatedRenderTrack {
    /// Validate the entire track's structure and encoded scene bounds once.
    /// This does not replace full `Chunk::validate` when loading a recording.
    pub fn new(chunk: &Arc<Chunk>, limits: RenderLimits) -> Result<Self> {
        track_parts(chunk, &limits)?;
        Ok(Self {
            owner: Arc::downgrade(chunk),
            limits,
        })
    }
    /// The Weak witness prevents allocation-address reuse and mutable access;
    /// `Arc::make_mut` instead creates an owner requiring a new proof.
    pub fn matches(&self, chunk: &Arc<Chunk>) -> bool {
        self.owner.as_ptr() == Arc::as_ptr(chunk)
    }
    /// Decode the selected scene under the construction limits. An unrelated
    /// owner is rejected; a matching trajectory without render data returns None.
    pub fn scene_at(&self, chunk: &Arc<Chunk>, step: usize) -> Result<Option<RenderScene>> {
        ensure(
            self.matches(chunk),
            "render track proof belongs to a different raw owner",
        )?;
        // The matching allocation is immutable for this proof's lifetime. Use
        // checked array access without repeating global offset validation.
        let Record::Trajectory(record) = &chunk.record else {
            return Err(Error::State("render scene requires a trajectory".into()));
        };
        let Some(track) = &record.render else {
            return Ok(None);
        };
        let get = |key: &str| {
            chunk
                .arrays
                .get(key)
                .ok_or_else(|| Error::Shape("missing render array".into()))
        };
        let offsets = get(&track.offsets)?.as_u64()?;
        let data = get(&track.data)?.as_u8()?;
        selected_scene(offsets, data, step, &self.limits).map(Some)
    }
}
pub(super) fn validate_track(chunk: &Chunk) -> Result<()> {
    let limits = RenderLimits::default();
    if let Some((offsets, data)) = track_parts(chunk, &limits)? {
        for pair in offsets.windows(2) {
            decode_scene(&data[pair[0] as usize..pair[1] as usize], &limits)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn proof_validates_offset_pairs_once_for_complete_episode_playback() {
        use crate::diagnostics::schema::{
            ActionCondition, Conditioning, EvaluationMetadata, RecordContext, TrajectoryRecord,
        };
        const FRAMES: usize = 16;
        let bytes = serde_json::to_vec(&empty()).unwrap();
        let chunk = Arc::new(Chunk {
            context: RecordContext::new("offset-work", 0),
            record: Record::Trajectory(TrajectoryRecord {
                id: "episode".into(),
                environment: "test".into(),
                observations: "s".into(),
                actions: "a".into(),
                rewards: "r".into(),
                terminated: "term".into(),
                truncated: "trunc".into(),
                simulation_times: None,
                action_masks: None,
                frames: BTreeMap::new(),
                render: Some(RenderTrack {
                    version: RENDER_VERSION,
                    offsets: "o".into(),
                    data: "d".into(),
                }),
                evaluation: EvaluationMetadata {
                    conditioning: Conditioning::InitialScenario,
                    initial_action: ActionCondition::CurrentPolicy,
                    continuation_policy: "test".into(),
                    stochastic_policy: false,
                    discount: 0.9,
                    independent_future_streams: true,
                    seed_methodology: "test".into(),
                    truncation_handling: "termination".into(),
                    complete_returns: true,
                },
            }),
            arrays: BTreeMap::from([
                (
                    "s".into(),
                    Array::f32(vec![FRAMES, 1], vec![0.; FRAMES]).unwrap(),
                ),
                (
                    "r".into(),
                    Array::f32(vec![FRAMES - 1], vec![0.; FRAMES - 1]).unwrap(),
                ),
                (
                    "o".into(),
                    Array::u64(
                        vec![FRAMES + 1],
                        (0..=FRAMES).map(|i| (i * bytes.len()) as u64).collect(),
                    )
                    .unwrap(),
                ),
                (
                    "d".into(),
                    Array::u8(vec![FRAMES * bytes.len()], bytes.repeat(FRAMES)).unwrap(),
                ),
            ]),
        });
        let limits = RenderLimits::default();
        TRACK_PAIR_VISITS.with(|count| count.set(0));
        for step in 0..FRAMES {
            scene_at(&chunk, step, &limits).unwrap();
        }
        assert_eq!(TRACK_PAIR_VISITS.with(|count| count.get()), FRAMES * FRAMES);
        TRACK_PAIR_VISITS.with(|count| count.set(0));
        let proof = ValidatedRenderTrack::new(&chunk, limits).unwrap();
        for step in 0..FRAMES {
            assert_eq!(proof.scene_at(&chunk, step).unwrap(), Some(empty()));
        }
        assert_eq!(TRACK_PAIR_VISITS.with(|count| count.get()), FRAMES);
    }
    fn empty() -> RenderScene {
        RenderScene {
            version: RENDER_VERSION,
            bounds: RenderBounds {
                min: [0., 0.],
                max: [1., 1.],
            },
            background: [0; 4],
            items: vec![],
        }
    }
    #[test]
    fn capture_frame_ceiling_and_retained_byte_boundaries() {
        let limits = RenderLimits::default();
        let scene = empty();
        let mut frames = RenderFrames::new();
        let mut total = 0;
        for _ in 0..limits.max_frames {
            frames.capture(&scene, &limits, &mut total).unwrap();
        }
        assert_eq!(frames.len(), 4096);
        assert_eq!(total, frames.retained_bytes());
        assert!(frames.capture(&scene, &limits, &mut total).is_err());
        let mut first = RenderFrames::new();
        let mut first_total = 0;
        first.capture(&scene, &limits, &mut first_total).unwrap();
        for delta in [0, 1] {
            let mut track = RenderFrames::new();
            let mut charge = 0;
            let bounded = RenderLimits {
                max_trajectory_bytes: first_total - delta,
                max_evaluation_bytes: first_total - delta,
                ..limits
            };
            assert_eq!(
                track.capture(&scene, &bounded, &mut charge).is_ok(),
                delta == 0
            );
        }
        let mut second = RenderFrames::new();
        assert!(
            second
                .capture(
                    &scene,
                    &RenderLimits {
                        max_evaluation_bytes: first_total,
                        ..limits
                    },
                    &mut first_total
                )
                .is_err()
        );
    }
    #[test]
    fn escaped_text_and_unknown_duplicate_fields_are_bounded() {
        let good = r#"{"version":1,"bounds":{"min":[0,0],"max":[1,1]},"background":[0,0,0,0],"items":[{"layer":0,"primitive":{"kind":"text","position":[0,0],"text":"\u00e9","color":[0,0,0,0],"size":12}}]}"#;
        assert!(
            decode_scene(
                good.as_bytes(),
                &RenderLimits {
                    max_text_bytes: 2,
                    ..Default::default()
                }
            )
            .is_ok()
        );
        assert!(
            decode_scene(
                good.as_bytes(),
                &RenderLimits {
                    max_text_bytes: 1,
                    ..Default::default()
                }
            )
            .is_err()
        );
        for bad in [
            good.replace("\"layer\":0", "\"layer\":0,\"layer\":0"),
            good.replace("\"size\":12", "\"size\":12,\"radius\":1"),
            good.replace("\"kind\":\"text\"", "\"kind\":\"text\",\"kind\":\"text\""),
            good.replace("\"min\":[0,0]", "\"min\":[0,0],\"unexpected\":0"),
        ] {
            assert!(decode_scene(bad.as_bytes(), &RenderLimits::default()).is_err());
        }
    }
}
