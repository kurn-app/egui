use emath::{NumExt as _, Vec2, vec2};
use nohash_hasher::IntMap;
use skrifa::GlyphId;

use crate::{
    FontColorTransferFunction, ImageDelta, TextOptions, TextureAtlas,
    text::{
        FontFamily, FontPriority, GlyphBitmap, GlyphRasterizer, GlyphRasterizerRequest,
        MAX_GLYPH_SIZE, RasterizedGlyph,
        face_store::FontFaceKey,
        font_face::{FontFace, ShapedGlyph},
        styled_metrics::StyledMetrics,
    },
};

/// Where a glyph ended up in the font atlas, in UV coordinates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct UvRect {
    /// X/Y offset for nice rendering (unit: points).
    pub offset: Vec2,

    /// Screen size (in points) of this glyph.
    /// Note that the height is different from the font height.
    pub size: Vec2,

    /// Top left corner UV in texture.
    pub min: [u16; 2],

    /// Bottom right corner (exclusive).
    pub max: [u16; 2],
}

impl UvRect {
    pub fn is_nothing(&self) -> bool {
        self.min == self.max
    }
}

/// Where a glyph is in the font atlas.
///
/// The default value is the invisible, zero-size glyph.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GlyphAllocation {
    /// UV rectangle for drawing.
    pub uv_rect: UvRect,

    /// A color glyph (e.g. emoji), stored with its own colors in the atlas.
    /// Do not tint it with the text color.
    pub is_color: bool,
}

/// An outline glyph in the atlas, positioned for one [`ShapedGlyph`].
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(not(feature = "glyph_paint"), derive(Copy, Default))]
pub(crate) struct OutlineGlyph {
    /// Resolved source, retained independently of bitmap atlas allocation.
    #[cfg(feature = "glyph_paint")]
    pub paint_font: Option<std::sync::Arc<crate::text::FontPaintInstance>>,

    #[cfg(feature = "glyph_paint")]
    pub paint_source: crate::text::GlyphPaintSource,

    /// Unhinted conservative control bounds in design units, Y up.
    #[cfg(feature = "glyph_paint")]
    pub outline_bounds: emath::Rect,

    pub allocation: GlyphAllocation,

    /// Laid out without its bitmap: the allocation holds the bitmap's offset and size, with an
    /// empty UV range. Rasterized at a lookup that wants its bitmap.
    #[cfg(feature = "glyph_paint")]
    pub skipped: bool,

    /// Left edge of the glyph's pixel grid, in physical pixels.
    ///
    /// This is [`ShapedGlyph::h_pos`] rounded to the pixel (or sub-pixel bin)
    /// the bitmap was rendered for. Draw the bitmap here, not at the
    /// un-rounded `h_pos`, or it will be blurry.
    pub x_px: i32,
}

impl OutlineGlyph {
    /// A glyph with no resolved font to paint from, such as a custom rasterizer's bitmap
    /// or a character without a glyph ID.
    pub fn unpainted(allocation: GlyphAllocation, x_px: i32) -> Self {
        Self {
            allocation,
            x_px,
            #[cfg(feature = "glyph_paint")]
            paint_font: None,
            #[cfg(feature = "glyph_paint")]
            paint_source: crate::text::GlyphPaintSource::Unsupported,
            #[cfg(feature = "glyph_paint")]
            outline_bounds: emath::Rect::NOTHING,
            #[cfg(feature = "glyph_paint")]
            skipped: false,
        }
    }
}

#[cfg(feature = "glyph_paint")]
impl Default for OutlineGlyph {
    fn default() -> Self {
        Self::unpainted(GlyphAllocation::default(), 0)
    }
}

/// A glyph from a [`GlyphRasterizer`], allocated in the atlas.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RasterGlyphAllocation {
    pub allocation: GlyphAllocation,
    pub advance_px: f32,
}

// ----------------------------------------------------------------------------

// Subpixel binning, taken from cosmic-text:
// https://github.com/pop-os/cosmic-text/blob/974ddaed96b334f560b606ebe5d2ca2d2f9f23ef/src/glyph_cache.rs

/// Bin for subpixel positioning of glyphs.
///
/// For accurate glyph positioning, we want to render each glyph at a subpixel coordinate. However, we also want to
/// cache each glyph's bitmap. As a compromise, we bin each subpixel offset into one of four fractional values. This
/// means one glyph can have up to four subpixel-positioned bitmaps in the cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub(crate) enum SubpixelBin {
    #[default]
    Zero,
    One,
    Two,
    Three,
}

impl SubpixelBin {
    /// Bin the given position and return the new integral coordinate.
    fn new(pos: f32) -> (i32, Self) {
        let trunc = pos as i32;
        let fract = pos - trunc as f32;

        if pos.is_sign_negative() {
            if fract > -0.125 {
                (trunc, Self::Zero)
            } else if fract > -0.375 {
                (trunc - 1, Self::Three)
            } else if fract > -0.625 {
                (trunc - 1, Self::Two)
            } else if fract > -0.875 {
                (trunc - 1, Self::One)
            } else {
                (trunc - 1, Self::Zero)
            }
        } else {
            if fract < 0.125 {
                (trunc, Self::Zero)
            } else if fract < 0.375 {
                (trunc, Self::One)
            } else if fract < 0.625 {
                (trunc, Self::Two)
            } else if fract < 0.875 {
                (trunc, Self::Three)
            } else {
                (trunc + 1, Self::Zero)
            }
        }
    }

    pub fn as_float(&self) -> f32 {
        match self {
            Self::Zero => 0.0,
            Self::One => 0.25,
            Self::Two => 0.5,
            Self::Three => 0.75,
        }
    }
}

// ----------------------------------------------------------------------------

/// Hash of everything that decides what an outline glyph looks like in the atlas.
#[derive(Hash, PartialEq, Eq)]
struct OutlineGlyphKey(u64);

impl nohash_hasher::IsEnabled for OutlineGlyphKey {}

impl OutlineGlyphKey {
    #[inline]
    fn new(
        face_key: FontFaceKey,
        glyph_id: GlyphId,
        metrics: &StyledMetrics,
        bin: SubpixelBin,
    ) -> Self {
        let StyledMetrics {
            pixels_per_point,
            px_scale_factor,
            location_hash,
            ..
        } = *metrics;
        debug_assert!(
            0.0 < pixels_per_point && pixels_per_point.is_finite(),
            "Bad pixels_per_point {pixels_per_point}"
        );
        debug_assert!(
            0.0 < px_scale_factor && px_scale_factor.is_finite(),
            "Bad px_scale_factor: {px_scale_factor}"
        );
        Self(crate::util::hash((
            face_key,
            glyph_id,
            pixels_per_point.to_bits(),
            px_scale_factor.to_bits(),
            bin,
            location_hash,
        )))
    }
}

/// Hash of `(priority, cluster, family, pixels_per_point, font_size)`,
/// so that cache lookups do not allocate.
#[derive(Hash, PartialEq, Eq)]
struct RasterGlyphKey(u64);

impl nohash_hasher::IsEnabled for RasterGlyphKey {}

impl RasterGlyphKey {
    fn new(
        priority: FontPriority,
        cluster: &str,
        family: &FontFamily,
        pixels_per_point: f32,
        font_size: f32,
    ) -> Self {
        Self(crate::util::hash((
            priority,
            cluster,
            family,
            pixels_per_point.to_bits(),
            font_size.to_bits(),
        )))
    }
}

// ----------------------------------------------------------------------------

/// Which glyphs get bitmaps, and the bitmap work done for outline glyphs.
///
/// Kept when the atlas starts over, for the life of the [`crate::text::Fonts`].
#[cfg(feature = "glyph_paint")]
#[derive(Default)]
struct GlyphPaintState {
    mode: crate::text::GlyphPaintMode,

    /// Glyphs that get their bitmaps in every mode, by font instance and glyph ID.
    requested: ahash::HashSet<(crate::text::FontInstanceId, u32)>,

    counters: crate::text::GlyphPaintCounters,
}

/// The font atlas, plus every cache that points into it.
///
/// Cleared as a unit when the atlas is getting full.
/// Knows nothing about families or fallback chains.
pub(crate) struct GlyphAtlas {
    atlas: TextureAtlas,

    /// Glyphs rendered from font outlines.
    outline_glyphs: IntMap<OutlineGlyphKey, OutlineGlyph>,

    /// Glyphs from the [`GlyphRasterizer`]s.
    ///
    /// `None` means no rasterizer of that priority could handle the cluster.
    raster_glyphs: IntMap<RasterGlyphKey, Option<RasterGlyphAllocation>>,

    #[cfg(feature = "glyph_paint")]
    paint: GlyphPaintState,
}

impl GlyphAtlas {
    pub fn new(options: TextOptions) -> Self {
        let texture_width = options.max_texture_side.at_most(16 * 1024);
        let initial_height = 32; // Keep initial font atlas small, so it is fast to upload to GPU. This will expand as needed anyways.
        Self {
            atlas: TextureAtlas::new([texture_width, initial_height], options),
            outline_glyphs: Default::default(),
            raster_glyphs: Default::default(),
            #[cfg(feature = "glyph_paint")]
            paint: Default::default(),
        }
    }

    #[inline]
    pub fn options(&self) -> &TextOptions {
        self.atlas.options()
    }

    #[inline]
    pub fn atlas(&self) -> &TextureAtlas {
        &self.atlas
    }

    #[inline]
    pub fn take_delta(&mut self) -> Option<ImageDelta> {
        self.atlas.take_delta()
    }

    /// How full is the atlas?
    #[inline]
    pub fn fill_ratio(&self) -> f32 {
        self.atlas.fill_ratio()
    }

    /// Forget every glyph and start over with an empty atlas.
    ///
    /// Anything holding a [`UvRect`] into the old atlas (e.g. a cached galley) is now stale.
    pub fn clear(&mut self) {
        self.reset(*self.atlas.options());
    }

    /// Forget every glyph and start over with an empty atlas for `options`.
    ///
    /// Which glyphs get bitmaps, and the counters, are kept.
    pub fn reset(&mut self, options: TextOptions) {
        #[cfg(feature = "glyph_paint")]
        let paint = core::mem::take(&mut self.paint);
        *self = Self::new(options);
        #[cfg(feature = "glyph_paint")]
        {
            self.paint = paint;
        }
    }

    #[cfg(feature = "glyph_paint")]
    pub fn glyph_paint_mode(&self) -> crate::text::GlyphPaintMode {
        self.paint.mode
    }

    /// Lay glyphs out in `mode` from now on. Returns whether it changed.
    ///
    /// Bitmaps already prepared stay; a glyph laid out without its bitmap gets it at its next
    /// lookup in [`crate::text::GlyphPaintMode::Bitmaps`].
    #[cfg(feature = "glyph_paint")]
    pub fn set_glyph_paint_mode(&mut self, mode: crate::text::GlyphPaintMode) -> bool {
        core::mem::replace(&mut self.paint.mode, mode) != mode
    }

    /// Give these glyphs their bitmaps in every mode, from their next lookup on. Returns how many
    /// were not requested before.
    #[cfg(feature = "glyph_paint")]
    pub fn request_glyph_bitmaps(
        &mut self,
        glyphs: impl IntoIterator<Item = (crate::text::FontInstanceId, u32)>,
    ) -> usize {
        glyphs
            .into_iter()
            .filter(|glyph| self.paint.requested.insert(*glyph))
            .count()
    }

    #[cfg(feature = "glyph_paint")]
    pub fn glyph_paint_counters(&self) -> crate::text::GlyphPaintCounters {
        self.paint.counters
    }

    /// How many glyphs' bitmaps were requested.
    #[cfg(all(test, feature = "glyph_paint"))]
    pub fn requested_glyph_bitmaps(&self) -> usize {
        self.paint.requested.len()
    }

    /// Forget what the [`GlyphRasterizer`]s produced, e.g. because they were replaced.
    pub fn clear_raster_glyphs(&mut self) {
        self.raster_glyphs.clear();
    }

    /// Get or render the glyph of `face` for `shaped`.
    ///
    /// [`GlyphId::NOTDEF`] renders the face's `.notdef` glyph ("tofu"),
    /// which is what we show for characters no font has.
    ///
    /// The bitmap is rendered at the sub-pixel bin of [`ShapedGlyph::h_pos`]
    /// (if the face has sub-pixel binning on, and the glyph is not CJK),
    /// so the same glyph can occupy up to four slots in the atlas.
    ///
    /// Never fails: see [`Self::allocate_bitmap`] for what happens when the atlas is full.
    pub fn allocate_outline(
        &mut self,
        face_key: FontFaceKey,
        face: &mut FontFace,
        metrics: &StyledMetrics,
        shaped: &ShapedGlyph,
    ) -> OutlineGlyph {
        let ShapedGlyph {
            glyph_id,
            h_pos,
            is_cjk,
        } = *shaped;

        let subpixel_binning =
            face.subpixel_binning() && !is_cjk && !face.is_color_glyph(metrics, glyph_id);
        let (x_px, bin) = if subpixel_binning {
            SubpixelBin::new(h_pos)
        } else {
            // CJK scripts contain a lot of characters and could hog the glyph atlas
            // if we stored 4 subpixel offsets per glyph.
            // Color glyphs (emoji) are big, and gain nothing from subpixel positioning.
            (h_pos.round() as i32, SubpixelBin::Zero)
        };

        let key = OutlineGlyphKey::new(face_key, glyph_id, metrics, bin);

        #[cfg(not(feature = "glyph_paint"))]
        let cached = {
            let Self {
                atlas,
                outline_glyphs,
                ..
            } = self;
            *outline_glyphs.entry(key).or_insert_with(|| {
                let (allocation, _) = Self::rasterize_outline(atlas, face, metrics, glyph_id, bin);
                OutlineGlyph::unpainted(allocation, 0)
            })
        };

        #[cfg(feature = "glyph_paint")]
        let cached = {
            let Self {
                atlas,
                outline_glyphs,
                paint,
                ..
            } = self;
            let cached = outline_glyphs.entry(key).or_insert_with(|| {
                Self::prepare_outline(atlas, paint, face, metrics, glyph_id, bin)
            });
            if cached.skipped
                && let Some(paint_font) = &cached.paint_font
                && (paint.mode == crate::text::GlyphPaintMode::Bitmaps
                    || paint
                        .requested
                        .contains(&(paint_font.id(), glyph_id.to_u32())))
            {
                let paint_font = std::sync::Arc::clone(paint_font);
                *cached = Self::rasterized_outline(
                    atlas, paint, face, metrics, glyph_id, bin, paint_font,
                );
            }
            cached.clone()
        };

        OutlineGlyph { x_px, ..cached }
    }

    /// Rasterize a glyph and copy it into the atlas: its allocation and, when it rendered a
    /// bitmap, whether that is a color one.
    fn rasterize_outline(
        atlas: &mut TextureAtlas,
        face: &mut FontFace,
        metrics: &StyledMetrics,
        glyph_id: GlyphId,
        bin: SubpixelBin,
    ) -> (GlyphAllocation, Option<bool>) {
        let bitmap = face.rasterize_glyph(metrics, glyph_id, bin);
        let is_color = bitmap.as_ref().map(|bitmap| bitmap.is_color);
        let allocation = bitmap
            .and_then(|bitmap| {
                let transfer = Self::transfer_function(atlas, bitmap.is_color);
                let mut uv_rect =
                    Self::allocate_bitmap(atlas, &bitmap, metrics.pixels_per_point, transfer)?;
                uv_rect.offset.y += metrics.y_offset_in_points;
                Some(GlyphAllocation {
                    uv_rect,
                    is_color: bitmap.is_color,
                })
            })
            .unwrap_or_default();
        (allocation, is_color)
    }

    /// A new atlas entry for a glyph of `face`.
    ///
    /// In [`crate::text::GlyphPaintMode::Outlines`] the paint source is decided before
    /// rasterizing: an outline or empty glyph whose bitmap was not requested is measured
    /// without being filled, and skipped. Every other glyph is rasterized.
    #[cfg(feature = "glyph_paint")]
    fn prepare_outline(
        atlas: &mut TextureAtlas,
        paint: &mut GlyphPaintState,
        face: &mut FontFace,
        metrics: &StyledMetrics,
        glyph_id: GlyphId,
        bin: SubpixelBin,
    ) -> OutlineGlyph {
        use crate::text::{GlyphPaintMode, GlyphPaintSource};

        let paint_font = face.paint_instance(metrics);
        if paint.mode == GlyphPaintMode::Outlines && !face.is_color_glyph(metrics, glyph_id) {
            let (paint_source, outline_bounds) = paint_font.outline_bounds(glyph_id.to_u32());
            let requested = paint
                .requested
                .contains(&(paint_font.id(), glyph_id.to_u32()));
            if matches!(
                paint_source,
                GlyphPaintSource::Outline | GlyphPaintSource::Empty
            ) && !requested
            {
                let uv_rect = face
                    .glyph_pixel_box(metrics, glyph_id, bin)
                    .and_then(|(offset_px, size)| {
                        Self::bitmap_rect(offset_px, size, metrics.pixels_per_point)
                    })
                    .map(|mut uv_rect| {
                        uv_rect.offset.y += metrics.y_offset_in_points;
                        uv_rect
                    });
                return OutlineGlyph {
                    allocation: GlyphAllocation {
                        uv_rect: uv_rect.unwrap_or_default(),
                        is_color: false,
                    },
                    x_px: 0,
                    paint_font: Some(paint_font),
                    paint_source,
                    outline_bounds,
                    skipped: uv_rect.is_some(),
                };
            }
        }
        Self::rasterized_outline(atlas, paint, face, metrics, glyph_id, bin, paint_font)
    }

    /// An atlas entry for a glyph of `face`, rasterized, counting the bitmap of an outline glyph.
    #[cfg(feature = "glyph_paint")]
    fn rasterized_outline(
        atlas: &mut TextureAtlas,
        paint: &mut GlyphPaintState,
        face: &mut FontFace,
        metrics: &StyledMetrics,
        glyph_id: GlyphId,
        bin: SubpixelBin,
        paint_font: std::sync::Arc<crate::text::FontPaintInstance>,
    ) -> OutlineGlyph {
        use crate::text::GlyphPaintSource;

        let (allocation, rendered) = Self::rasterize_outline(atlas, face, metrics, glyph_id, bin);
        let (paint_source, outline_bounds) = if rendered == Some(true) {
            (GlyphPaintSource::FontColor, emath::Rect::NOTHING)
        } else {
            paint_font.outline_bounds(glyph_id.to_u32())
        };
        if paint_source == GlyphPaintSource::Outline && rendered.is_some() {
            let UvRect { min, max, .. } = allocation.uv_rect;
            let texels = u64::from(max[0] - min[0]) * u64::from(max[1] - min[1]);
            paint.counters.outline_bitmaps += 1;
            paint.counters.outline_bitmap_texels += texels;
        }
        OutlineGlyph {
            allocation,
            x_px: 0,
            paint_font: Some(paint_font),
            paint_source,
            outline_bounds,
            skipped: false,
        }
    }

    /// Get or rasterize `cluster` using the [`GlyphRasterizer`]s of the given `priority`.
    ///
    /// The rasterizers are asked in order, and the first to return `Some` wins.
    ///
    /// Failures are cached too, so the (potentially slow) rasterizers
    /// are asked at most once per cluster and size.
    ///
    /// See [`Self::allocate_bitmap`] for what happens when the atlas is full.
    pub fn allocate_raster<'r>(
        &mut self,
        rasterizers: impl IntoIterator<Item = &'r GlyphRasterizer>,
        priority: FontPriority,
        cluster: &str,
        family: &FontFamily,
        pixels_per_point: f32,
        font_size: f32,
    ) -> Option<RasterGlyphAllocation> {
        let key = RasterGlyphKey::new(priority, cluster, family, pixels_per_point, font_size);
        if let Some(allocation) = self.raster_glyphs.get(&key) {
            return *allocation;
        }
        let request = GlyphRasterizerRequest {
            cluster,
            family,
            font_size_px: font_size * pixels_per_point,
            subpixel_offset_px: 0.0,
        };
        let allocation = rasterizers
            .into_iter()
            .filter(|rasterizer| rasterizer.priority == priority)
            .find_map(|rasterizer| (rasterizer.rasterize)(&request))
            .and_then(|RasterizedGlyph { bitmap, advance_px }| {
                let is_color = bitmap.is_color;
                let transfer = Self::transfer_function(&self.atlas, is_color);
                let uv_rect =
                    Self::allocate_bitmap(&mut self.atlas, &bitmap, pixels_per_point, transfer)?;
                Some(RasterGlyphAllocation {
                    allocation: GlyphAllocation { uv_rect, is_color },
                    advance_px,
                })
            });
        self.raster_glyphs.insert(key, allocation);
        allocation
    }

    /// The transfer function assumes white coverage glyphs and discards color,
    /// so color glyphs (e.g. emoji) must skip it.
    fn transfer_function(atlas: &TextureAtlas, is_color: bool) -> FontColorTransferFunction {
        if is_color {
            FontColorTransferFunction::Off
        } else {
            atlas.options().color_transfer_function
        }
    }

    /// Copy a bitmap into the atlas.
    ///
    /// Returns `None` for empty and oversized bitmaps.
    ///
    /// Never fails for lack of space: the [`TextureAtlas`] grows in height as needed,
    /// and once it hits its maximum it logs a warning and starts overwriting old glyphs
    /// (see [`TextureAtlas::allocate`]). We avoid getting there because `Fonts::begin_pass`
    /// calls [`Self::clear`] as soon as [`Self::fill_ratio`] passes 80%.
    fn allocate_bitmap(
        atlas: &mut TextureAtlas,
        bitmap: &GlyphBitmap,
        pixels_per_point: f32,
        transfer: FontColorTransferFunction,
    ) -> Option<UvRect> {
        let [width, height] = bitmap.image.size;
        let uv_rect = Self::bitmap_rect(bitmap.offset_px, bitmap.image.size, pixels_per_point)?;
        let (glyph_pos, image) = atlas.allocate((width, height));
        for y in 0..height {
            for x in 0..width {
                image[(glyph_pos.0 + x, glyph_pos.1 + y)] =
                    transfer.to_atlas_color(bitmap.image[(x, y)]);
            }
        }
        Some(UvRect {
            min: [glyph_pos.0 as u16, glyph_pos.1 as u16],
            max: [(glyph_pos.0 + width) as u16, (glyph_pos.1 + height) as u16],
            ..uv_rect
        })
    }

    /// The offset and size in points of a bitmap of `size` pixels at `offset_px`, with an empty
    /// UV range, or `None` for an empty or oversized bitmap, which the atlas does not hold.
    fn bitmap_rect(offset_px: Vec2, size: [usize; 2], pixels_per_point: f32) -> Option<UvRect> {
        let [width, height] = size;
        if width == 0 || height == 0 || MAX_GLYPH_SIZE < width || MAX_GLYPH_SIZE < height {
            return None;
        }
        Some(UvRect {
            offset: offset_px / pixels_per_point,
            size: vec2(width as f32, height as f32) / pixels_per_point,
            min: [0, 0],
            max: [0, 0],
        })
    }
}
