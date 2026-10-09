use core::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use self_cell::self_cell;
use skrifa::{FontRef, GlyphId, MetadataProvider as _, instance::Size, outline::DrawSettings};

use super::Blob;

pub use skrifa::{
    instance::NormalizedCoord,
    outline::{DrawError as OutlineDrawError, OutlinePen},
};

/// Opaque identity of an installed face at resolved normalized variation coordinates.
///
/// Stable while its font instance is alive. A replacement installation gets a new
/// identity, even when its name and bytes match. Size, DPI, color, bitmap hinting,
/// subpixel bins, and render transforms do not affect this identity.
/// This is an in-process cache key, not a persistent or serialized identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FontInstanceId(usize);

impl FontInstanceId {
    fn new() -> Self {
        static NEXT_ID: AtomicUsize = AtomicUsize::new(1);
        Self(
            NEXT_ID
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
                .expect("font instance identity space exhausted"),
        )
    }
}

self_cell! {
    struct PaintFontCell {
        owner: Blob,

        #[covariant]
        dependent: FontRef,
    }
}

/// Shared font bytes and immutable parsed views, independent of bitmap hinting state.
pub(super) struct PaintFontData {
    font: PaintFontCell,
    face_index: u32,
}

impl PaintFontData {
    pub(super) fn new(bytes: Blob, face_index: u32) -> Result<Self, Box<dyn core::error::Error>> {
        let font = PaintFontCell::try_new(bytes, |bytes| {
            skrifa::FontRef::from_index(AsRef::<[u8]>::as_ref(bytes.as_ref()), face_index)
        })?;
        Ok(Self { font, face_index })
    }
}

/// Owned unhinted outline provider for an actual resolved font instance.
///
/// Font faces intern these by their complete resolved location, so labels at
/// different sizes share an `Arc`. The provider keeps shared bytes and parsed
/// views alive after a font store or cached layout is discarded. Consumers must
/// use the glyph ID supplied by shaping, never map the text's characters again.
/// Color/bitmap/custom-rasterizer selection remains the layout engine's job;
/// the presence of an outline alone does not establish the paint source.
pub struct FontPaintInstance {
    id: FontInstanceId,
    data: Arc<PaintFontData>,
    location: skrifa::instance::Location,
}

impl core::fmt::Debug for FontPaintInstance {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("FontPaintInstance")
            .field("id", &self.id)
            .field("face_index", &self.face_index())
            .field("location", &self.location)
            .finish_non_exhaustive()
    }
}

impl PartialEq for FontPaintInstance {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for FontPaintInstance {}

impl FontPaintInstance {
    pub(super) fn new(data: Arc<PaintFontData>, location: skrifa::instance::Location) -> Self {
        Self {
            id: FontInstanceId::new(),
            data,
            location,
        }
    }

    /// Identity for outline cache lookup, independent of display size and paint.
    pub fn id(&self) -> FontInstanceId {
        self.id
    }

    /// Shared bytes, including all faces when this is a font collection.
    pub fn font_data(&self) -> &Blob {
        self.data.font.borrow_owner()
    }

    /// Collection face index within [`Self::font_data`].
    pub fn face_index(&self) -> u32 {
        self.data.face_index
    }

    /// Resolved normalized coordinates in the font's axis order, including `avar` mapping.
    pub fn normalized_coords(&self) -> &[NormalizedCoord] {
        self.location.coords()
    }

    /// Number of design units per em used by [`Self::draw_unhinted`].
    pub fn units_per_em(&self) -> u16 {
        self.data
            .font
            .borrow_dependent()
            .metrics(Size::unscaled(), &self.location)
            .units_per_em
    }

    pub(crate) fn outline_bounds(&self, glyph_id: u32) -> (super::GlyphPaintSource, emath::Rect) {
        use super::GlyphPaintSource;
        let mut pen = BoundsPen::default();
        match self.draw_unhinted(glyph_id, &mut pen) {
            Ok(true) if pen.has_segments => (GlyphPaintSource::Outline, pen.bounds),
            Ok(true) => (GlyphPaintSource::Empty, emath::Rect::NOTHING),
            Ok(false) | Err(_) => (GlyphPaintSource::Unsupported, emath::Rect::NOTHING),
        }
    }

    /// Emit an unhinted path for a resolved glyph in design units, with Y pointing up.
    ///
    /// There is no size, subpixel translation, synthetic italic shear, or font
    /// tweak baked into the path. Apply these in placement data. Winding and
    /// contour closure follow Skrifa. Cubics are emitted intact for the consumer
    /// to convert using its own encoding policy.
    ///
    /// Returns `Ok(false)` if the face has no outline for this ID. `Ok(true)`
    /// can emit no commands for an empty outline such as a space. An error may
    /// follow partial pen output: discard that output and use fallback.
    pub fn draw_unhinted(
        &self,
        glyph_id: u32,
        pen: &mut impl OutlinePen,
    ) -> Result<bool, OutlineDrawError> {
        let font = self.data.font.borrow_dependent();
        let Some(outline) = font.outline_glyphs().get(GlyphId::new(glyph_id)) else {
            return Ok(false);
        };
        outline.draw(
            DrawSettings::unhinted(Size::unscaled(), &self.location),
            pen,
        )?;
        Ok(true)
    }
}

struct BoundsPen {
    bounds: emath::Rect,
    has_segments: bool,
}

impl Default for BoundsPen {
    fn default() -> Self {
        Self {
            bounds: emath::Rect::NOTHING,
            has_segments: false,
        }
    }
}

impl BoundsPen {
    fn include(&mut self, x: f32, y: f32) {
        self.bounds.extend_with(emath::pos2(x, y));
    }
}

impl OutlinePen for BoundsPen {
    fn move_to(&mut self, x: f32, y: f32) {
        self.include(x, y);
    }
    fn line_to(&mut self, x: f32, y: f32) {
        self.has_segments = true;
        self.include(x, y);
    }
    fn quad_to(&mut self, cx: f32, cy: f32, x: f32, y: f32) {
        self.has_segments = true;
        self.include(cx, cy);
        self.include(x, y);
    }
    fn curve_to(&mut self, cx0: f32, cy0: f32, cx1: f32, cy1: f32, x: f32, y: f32) {
        self.has_segments = true;
        self.include(cx0, cy0);
        self.include(cx1, cy1);
        self.include(x, y);
    }
    fn close(&mut self) {}
}

#[cfg(all(test, feature = "default_fonts"))]
mod tests {
    use super::*;
    use crate::text::{FontData, TextOptions, font_face::FontFace};

    /// Keep every verb and coordinate so a placement or hinting change is observable.
    #[derive(Default, Debug, PartialEq)]
    struct Path(Vec<(u8, Vec<f32>)>);

    impl OutlinePen for Path {
        fn move_to(&mut self, x: f32, y: f32) {
            self.0.push((0, vec![x, y]));
        }
        fn line_to(&mut self, x: f32, y: f32) {
            self.0.push((1, vec![x, y]));
        }
        fn quad_to(&mut self, cx: f32, cy: f32, x: f32, y: f32) {
            self.0.push((2, vec![cx, cy, x, y]));
        }
        fn curve_to(&mut self, cx: f32, cy: f32, cx1: f32, cy1: f32, x: f32, y: f32) {
            self.0.push((3, vec![cx, cy, cx1, cy1, x, y]));
        }
        fn close(&mut self) {
            self.0.push((4, vec![]));
        }
    }

    fn face(data: &FontData) -> FontFace {
        FontFace::new(
            TextOptions::default(),
            "fixture".into(),
            Arc::clone(&data.font),
            data.index,
            data.tweak.clone(),
        )
        .unwrap()
    }

    fn glyphs(face: &FontFace, text: &str) -> Vec<u32> {
        let mut buffer = harfrust::UnicodeBuffer::new();
        buffer.push_str(text);
        buffer.guess_segment_properties();
        face.shaper_data()
            .shaper(face.skrifa_font_ref())
            .build()
            .shape(buffer, harfrust::ShapeOptions::new())
            .glyph_infos()
            .iter()
            .map(|info| info.glyph_id)
            .collect()
    }

    #[test]
    fn font_paint_instance_is_shared_across_size_dpi_and_bitmap_options() {
        let data = FontData::from_static(epaint_default_fonts::UBUNTU_LIGHT);
        let mut face = face(&data);
        let small = face.styled_metrics(1.0, 12.0, &Default::default());
        let large = face.styled_metrics(2.0, 80.0, &Default::default());
        let a = face.paint_instance(&small);
        let b = face.paint_instance(&large);
        assert!(Arc::ptr_eq(&a, &b));
        assert!(Arc::ptr_eq(a.font_data(), &data.font));
        assert_eq!(a.face_index(), data.index);
        assert_eq!(a.units_per_em(), 1000);
        assert!(a.normalized_coords().is_empty());

        let glyph = glyphs(&face, "A")[0];
        let mut before = Path::default();
        assert!(a.draw_unhinted(glyph, &mut before).unwrap());
        assert!(!before.0.is_empty());
        // Unscaled geometry has no pixel-grid, size, or bitmap hinting dependency.
        assert!(
            before
                .0
                .iter()
                .flat_map(|(_, p)| p)
                .any(|p| p.abs() > 100.0)
        );
        face.set_options(TextOptions {
            font_hinting: false,
            subpixel_binning: false,
            ..Default::default()
        });
        let c = face.paint_instance(&small);
        assert!(Arc::ptr_eq(&a, &c));
        let mut after = Path::default();
        assert!(c.draw_unhinted(glyph, &mut after).unwrap());
        assert_eq!(before, after);
    }

    #[test]
    fn font_paint_instance_survives_replacement_and_store_drop() {
        let data = FontData::from_static(epaint_default_fonts::UBUNTU_LIGHT);
        let mut old_face = face(&data);
        let metrics = old_face.styled_metrics(1.0, 14.0, &Default::default());
        let old = old_face.paint_instance(&metrics);
        let glyph = glyphs(&old_face, "A")[0];
        let mut expected = Path::default();
        old.draw_unhinted(glyph, &mut expected).unwrap();
        drop(old_face);

        // A reinstall under the same name must not alias a live old instance.
        let mut replacement = face(&data);
        let new = replacement.paint_instance(&metrics);
        assert_ne!(old.id(), new.id());
        assert!(Arc::ptr_eq(old.font_data(), new.font_data()));
        drop(replacement);
        drop(data);
        let mut actual = Path::default();
        old.draw_unhinted(glyph, &mut actual).unwrap();
        assert_eq!(expected, actual);
    }

    #[test]
    fn font_paint_instances_compare_full_locations() {
        let data = FontData::from_static(epaint_default_fonts::UBUNTU_LIGHT);
        let mut face = face(&data);
        let mut metrics = face.styled_metrics(1.0, 14.0, &Default::default());
        // Exercise interning with explicit normalized locations, independently
        // of axis discovery; actual variable-font resolution needs its own fixture.
        metrics.location = skrifa::instance::Location::new(2);
        metrics.location.coords_mut()[0] = NormalizedCoord::from_f32(0.5);
        let a = face.paint_instance(&metrics);
        metrics.location.coords_mut()[1] = NormalizedCoord::from_f32(-0.5);
        let b = face.paint_instance(&metrics);
        assert_ne!(a.id(), b.id());
        assert_eq!(b.normalized_coords()[0], NormalizedCoord::from_f32(0.5));
        assert_eq!(b.normalized_coords()[1], NormalizedCoord::from_f32(-0.5));
        metrics.location.coords_mut()[1] = NormalizedCoord::from_f32(0.0);
        assert!(Arc::ptr_eq(&a, &face.paint_instance(&metrics)));
    }

    #[test]
    fn font_paint_provider_draws_shaped_ligature_and_empty_outline() {
        let data = FontData::from_static(epaint_default_fonts::UBUNTU_LIGHT);
        let mut face = face(&data);
        let metrics = face.styled_metrics(1.0, 14.0, &Default::default());
        let instance = face.paint_instance(&metrics);
        let ligature = glyphs(&face, "fi");
        assert_eq!(ligature.len(), 1, "fixture must actually shape a ligature");
        assert_ne!(ligature[0], glyphs(&face, "f")[0]);
        let mut path = Path::default();
        assert!(instance.draw_unhinted(ligature[0], &mut path).unwrap());
        assert!(!path.0.is_empty());
        assert!(path.0.iter().any(|(verb, _)| *verb == 4));
        path.0.clear();
        assert!(
            instance
                .draw_unhinted(glyphs(&face, " ")[0], &mut path)
                .unwrap()
        );
        assert!(path.0.is_empty());
        assert!(!instance.draw_unhinted(u32::MAX, &mut path).unwrap());
        assert!(path.0.is_empty());
    }

    #[test]
    fn font_paint_allocation_retains_discovered_face() {
        use crate::text::{
            FallbackRequest, FontDefinitions, FontFamily, FontInsert, FontPriority,
            InsertFontFamily, font_face::ShapedGlyph, fonts::FontsImpl,
        };
        let data = FontData::from_static(epaint_default_fonts::UBUNTU_LIGHT);
        let provided = data.clone();
        let mut fonts = FontsImpl::new(TextOptions::default(), FontDefinitions::empty());
        fonts.set_font_providers(vec![Arc::new(move |request: &FallbackRequest<'_>| {
            Some(FontInsert::new(
                "discovered",
                provided.clone(),
                vec![InsertFontFamily {
                    family: request.family.clone(),
                    priority: FontPriority::Lowest,
                }],
            ))
        })]);
        let family = fonts.family_key(&FontFamily::Proportional);
        let key = fonts.resolve_cluster_face(family, "fi");
        let face = fonts.face(key).unwrap();
        assert_eq!(face.name(), "discovered");
        let metrics = face.styled_metrics(1.0, 14.0, &Default::default());
        let shaped = glyphs(face, "fi");
        assert_eq!(shaped.len(), 1);
        let allocation = fonts.allocate_glyph(
            key,
            &metrics,
            &ShapedGlyph {
                glyph_id: GlyphId::new(shaped[0]),
                h_pos: 0.375,
                is_cjk: false,
            },
        );
        assert!(!allocation.allocation.uv_rect.is_nothing());
        let instance = allocation.paint_font.unwrap();
        assert!(Arc::ptr_eq(instance.font_data(), &data.font));
        assert_eq!(fonts.discovered_fonts().len(), 1);
        drop(fonts);
        let mut path = Path::default();
        assert!(instance.draw_unhinted(shaped[0], &mut path).unwrap());
        assert!(!path.0.is_empty());
    }
}
