use core::ops::Range;
use std::sync::Arc;

use emath::{Pos2, Rect, Vec2};

use super::FontPaintInstance;
use crate::Color32;

/// The source selected by layout, independently of bitmap atlas UV availability.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GlyphPaintSource {
    /// An unhinted monochrome outline in the resolved font.
    Outline,
    /// A font's embedded bitmap or color paint graph, rendered by the stock path.
    FontColor,
    /// A custom rasterizer (including synthetic fontless tofu), rendered by the stock path.
    CustomRasterizer,
    /// A successfully resolved empty outline, such as a space.
    Empty,
    /// No usable outline, or an outline draw error. Use the stock path.
    #[default]
    Unsupported,
}

/// The contiguous run of a row mesh's index list that draws its glyph quads.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GlyphIndexRange {
    pub start: u32,
    pub end: u32,
}

impl GlyphIndexRange {
    /// The run as a range of positions in the mesh's index list.
    pub fn range(self) -> Range<usize> {
        self.start as usize..self.end as usize
    }
}

/// How a row's glyph paint relates to its stock mesh. Layout checks it once per
/// row, so a painter replacing glyph quads need not re-walk the row every frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GlyphPaintCheck {
    /// Never checked, as for a deserialized row. Paint it with the stock mesh.
    #[default]
    Unchecked,
    /// Paint occurrences are in strictly increasing glyph order with finite or
    /// [`Rect::NOTHING`] bounds, and each glyph with a stock quad has one. Those
    /// quads, four vertices each in glyph order, fill
    /// [`super::RowVisuals::glyph_vertex_range`], and their indices, six per
    /// quad addressing only that quad, fill `glyphs`. Every other index
    /// addresses a mesh vertex outside the glyph quads, and the indices before
    /// `glyphs` are whole triangles.
    ///
    /// Triangles before `glyphs` (backgrounds, a selection highlight) paint
    /// under the glyphs, the rest (underline, strikethrough) over them. A
    /// painter may draw any occurrence itself and keep the others' stock quads.
    /// With no stock quad, `glyphs` is empty at the end of the index list.
    Separable { glyphs: GlyphIndexRange },
    /// The paint occurrences or the mesh layout do not match the glyphs.
    Malformed,
}

impl super::Row {
    /// Check the row's glyph paint against its glyphs and stock mesh. Layout
    /// stores the result in [`super::Row::glyph_paint_check`]; code that edits
    /// a row's glyphs, paint or mesh structure afterwards recomputes it.
    pub fn check_glyph_paint(&self) -> GlyphPaintCheck {
        let mesh = &self.visuals.mesh;
        let quads = self.visuals.glyph_vertex_range.clone();
        if quads.start > quads.end || quads.end > mesh.vertices.len() {
            return GlyphPaintCheck::Malformed;
        }
        let mut previous = None;
        for paint in self.glyph_paint.iter() {
            if paint.glyph_index >= self.glyphs.len()
                || previous.is_some_and(|p| p >= paint.glyph_index)
                || (paint.bounds != Rect::NOTHING && !paint.bounds.is_finite())
            {
                return GlyphPaintCheck::Malformed;
            }
            previous = Some(paint.glyph_index);
        }
        // The glyph run starts at the first index into the first quad, or at
        // the end of the list when there is no quad.
        let first_quad = quads.start..(quads.start + 4).min(quads.end);
        let start = mesh
            .indices
            .iter()
            .position(|&i| first_quad.contains(&(i as usize)))
            .unwrap_or(mesh.indices.len());
        let mut paints = self.glyph_paint.iter().peekable();
        let (mut vertex, mut index) = (quads.start, start);
        for (glyph_index, glyph) in self.glyphs.iter().enumerate() {
            if glyph.uv_rect.is_nothing() {
                continue;
            }
            while paints.next_if(|p| p.glyph_index < glyph_index).is_some() {}
            let owned = paints.peek().is_some_and(|p| p.glyph_index == glyph_index);
            let quad = vertex..vertex + 4;
            let indices = mesh.indices.get(index..index + 6);
            if !owned
                || glyph.first_vertex as usize != vertex
                || !indices.is_some_and(|i| i.iter().all(|&i| quad.contains(&(i as usize))))
            {
                return GlyphPaintCheck::Malformed;
            }
            vertex += 4;
            index += 6;
        }
        let end = index;
        if vertex != quads.end || start % 3 != 0 {
            return GlyphPaintCheck::Malformed;
        }
        let outside = mesh.indices[..start].iter().chain(&mesh.indices[end..]);
        for &i in outside {
            let i = i as usize;
            if i >= mesh.vertices.len() || quads.contains(&i) {
                return GlyphPaintCheck::Malformed;
            }
        }
        let (Ok(start), Ok(end)) = (u32::try_from(start), u32::try_from(end)) else {
            return GlyphPaintCheck::Malformed;
        };
        GlyphPaintCheck::Separable {
            glyphs: GlyphIndexRange { start, end },
        }
    }
}

/// One resolved paint occurrence, separate from character/cursor records.
///
/// Coordinates are relative to the original, untransformed row. The row's current
/// position and the text shape's render scale/rotation must be applied separately.
/// Continuation and editing-only invisible characters have no record. Empty
/// shaped glyphs do have a record, preserving their actual font/glyph identity.
/// Font handles keep data alive through layout-cache reuse and font replacement.
#[derive(Clone, Debug, PartialEq)]
pub struct GlyphPaint {
    /// Index of the logical glyph owning the stock bitmap quad, if there is one.
    pub glyph_index: usize,
    /// Actual resolved font. Custom rasterizers have no font handle.
    pub font: Option<Arc<FontPaintInstance>>,
    /// Actual shaped ID (or resolved fallback/overflow ID), never reconstructed from a character.
    pub glyph_id: Option<u32>,
    pub source: GlyphPaintSource,
    /// Unrounded horizontal shaping origin and final rounded baseline, including font Y tweaks.
    pub origin: Pos2,
    /// Shaper offsets in UI points, with Y pointing down; not baked into the origin.
    pub offset: Vec2,
    /// Design-unit to UI-point scale, including font tweaks and Y-up to Y-down conversion.
    pub outline_scale: Vec2,
    /// Synthetic italic shear: `x += shear * (shear_origin_y - y)`.
    pub shear: f32,
    /// Row-local pivot for synthetic italic shear.
    pub shear_origin_y: f32,
    /// Resolved section color, possibly PLACEHOLDER. Apply shape color overrides and opacity at paint time.
    pub color: Color32,
    /// Whether stock painting preserves source RGB and applies only text alpha.
    pub is_color: bool,
    /// Conservative row-local visual bounds, independent of outline bitmap allocation.
    /// Empty/unsupported glyphs without a bitmap have NOTHING bounds.
    pub bounds: Rect,
}

impl GlyphPaint {
    /// Convert an outline point in design units (Y up) to original row coordinates.
    pub fn local_point(&self, design_point: Vec2) -> Pos2 {
        let mut point = self.origin + self.offset + self.outline_scale * design_point;
        point.x += self.shear * (self.shear_origin_y - point.y);
        point
    }
}

#[cfg(all(test, feature = "default_fonts"))]
mod tests {
    use super::*;
    use crate::text::{
        FontData, FontDefinitions, FontFamily, FontId, FontPriority, FontTweak, GlyphBitmap,
        GlyphRasterizer, GlyphRasterizerRequest, LayoutJob, RasterizedGlyph, TextOptions,
        VariationCoords,
        fonts::FontsImpl,
        galley_cache::GalleyCache,
        text_layout::{ITALIC_SHEAR, layout},
    };
    use crate::{Mesh, Shape, Tessellator, TextShape};
    use emath::{Rot2, TSTransform, pos2, vec2};

    fn ubuntu(tweak: FontTweak) -> FontsImpl {
        single_font(epaint_default_fonts::UBUNTU_LIGHT, tweak)
    }

    fn single_font(bytes: &'static [u8], tweak: FontTweak) -> FontsImpl {
        let mut definitions = FontDefinitions::empty();
        let mut data = FontData::from_static(bytes);
        data.tweak = tweak;
        definitions
            .font_data
            .insert("Ubuntu".into(), Arc::new(data));
        definitions
            .families
            .insert(FontFamily::Proportional, vec!["Ubuntu".into()]);
        FontsImpl::new(TextOptions::default(), definitions)
    }

    fn job(text: &str) -> LayoutJob {
        LayoutJob::simple(
            text.into(),
            FontId::proportional(17.0),
            Color32::BLUE,
            f32::INFINITY,
        )
    }

    fn close(a: Pos2, b: Pos2) {
        assert!((a - b).length() < 1e-4, "{a:?} != {b:?}");
    }

    fn separable(row: &super::super::Row) -> GlyphIndexRange {
        match row.check_glyph_paint() {
            GlyphPaintCheck::Separable { glyphs } => glyphs,
            check => panic!("expected a separable row: {check:?}"),
        }
    }

    #[test]
    fn layout_checks_row_glyph_paint_against_glyphs_and_mesh() {
        let mut fonts = ubuntu(FontTweak::default());
        let galley = layout(&mut fonts, 1.0, Arc::new(job("Ab c")));
        let row = &galley.rows[0].row;
        // A plain row's glyph run is its whole index list.
        let whole = GlyphIndexRange {
            start: 0,
            end: row.visuals.mesh.indices.len() as u32,
        };
        assert_eq!(whole.end, 18, "three quads of six indices");
        assert_eq!(
            row.glyph_paint_check,
            GlyphPaintCheck::Separable { glyphs: whole }
        );
        assert_eq!(separable(row), whole);

        // A background, an underline and a strikethrough: the run starts at the
        // first glyph index, six indices per stock quad.
        let mut rich = job("Ab");
        rich.sections[0].format.background = Color32::GREEN;
        rich.sections[0].format.underline = crate::Stroke::new(1.0, Color32::RED);
        rich.sections[0].format.strikethrough = crate::Stroke::new(1.0, Color32::RED);
        let rich = layout(&mut fonts, 1.0, Arc::new(rich));
        let decorated = &rich.rows[0].row;
        let quads = decorated
            .glyphs
            .iter()
            .filter(|glyph| !glyph.uv_rect.is_nothing())
            .count();
        assert_eq!(quads, 2);
        let start = decorated.visuals.glyph_index_start;
        assert!(start > 0, "the background draws first");
        let glyphs = GlyphIndexRange {
            start: start as u32,
            end: (start + 6 * quads) as u32,
        };
        assert!(
            glyphs.range().end < decorated.visuals.mesh.indices.len(),
            "the decorations draw last"
        );
        assert_eq!(
            decorated.glyph_paint_check,
            GlyphPaintCheck::Separable { glyphs }
        );
        assert_eq!(separable(decorated), glyphs);

        let edited = |row: &super::super::Row, edit: &dyn Fn(&mut super::super::Row)| {
            let mut row = row.clone();
            edit(&mut row);
            row.check_glyph_paint()
        };
        let with_paints = |row: &mut super::super::Row, edit: &dyn Fn(&mut Vec<GlyphPaint>)| {
            let mut paints = row.glyph_paint.to_vec();
            edit(&mut paints);
            row.glyph_paint = paints.into();
        };
        // A background index moved into the second glyph's quad.
        let second_quad = decorated.glyphs[1].first_vertex;
        assert_eq!(
            edited(decorated, &|row| row.visuals.mesh.indices[0] = second_quad),
            GlyphPaintCheck::Malformed
        );
        // A decoration index at the vertex count.
        assert_eq!(
            edited(decorated, &|row| {
                let past = row.visuals.mesh.vertices.len() as u32;
                *row.visuals.mesh.indices.last_mut().unwrap() = past;
            }),
            GlyphPaintCheck::Malformed
        );
        // Paint occurrences out of glyph order.
        assert_eq!(
            edited(row, &|row| with_paints(row, &|p| p.swap(0, 1))),
            GlyphPaintCheck::Malformed
        );
        // A stock quad without its paint occurrence.
        assert_eq!(
            edited(row, &|row| with_paints(row, &|p| {
                p.remove(0);
            })),
            GlyphPaintCheck::Malformed
        );
        // Non-finite paint bounds.
        assert_eq!(
            edited(row, &|row| with_paints(row, &|p| {
                p[0].bounds = Rect::from_min_max(pos2(0.0, 0.0), pos2(f32::INFINITY, 1.0));
            })),
            GlyphPaintCheck::Malformed
        );
        // A triangle reaching into the next glyph's quad.
        assert_eq!(
            edited(row, &|row| row.visuals.mesh.indices[0] = 4),
            GlyphPaintCheck::Malformed
        );
        // Unserialized rows are never checked.
        assert_eq!(GlyphPaintCheck::default(), GlyphPaintCheck::Unchecked);
    }

    #[test]
    fn row_paint_retains_shaped_ids_origins_offsets_and_font_tweaks() {
        let tweak = FontTweak {
            scale: 1.25,
            y_offset: 2.0,
            ..Default::default()
        };
        let mut fonts = single_font(epaint_default_fonts::HACK_REGULAR, tweak);
        let family = fonts.family_key(&FontFamily::Proportional);
        let key = fonts.resolve_face(family, 'x');
        let face = fonts.face(key).unwrap();
        let metrics = face.styled_metrics(1.3, 17.0, &VariationCoords::default());
        let mut buffer = harfrust::UnicodeBuffer::new();
        buffer.push_str("Ax\u{301}");
        buffer.guess_segment_properties();
        let expected = face
            .shaper_data()
            .shaper(face.skrifa_font_ref())
            .build()
            .shape(buffer, harfrust::ShapeOptions::new());
        assert!(
            expected
                .glyph_positions()
                .iter()
                .any(|p| p.x_offset != 0 || p.y_offset != 0)
        );
        let galley = layout(&mut fonts, 1.3, Arc::new(job("Ax\u{301}")));
        let row = &galley.rows[0].row;
        assert_eq!(row.glyph_paint.len(), expected.len());
        let scale = metrics.px_scale_factor / 1.3;
        let mut advance = 0.0;
        for ((paint, info), position) in row
            .glyph_paint
            .iter()
            .zip(expected.glyph_infos())
            .zip(expected.glyph_positions())
        {
            assert_eq!(paint.source, GlyphPaintSource::Outline);
            assert_eq!(paint.glyph_id, Some(info.glyph_id));
            assert!(paint.bounds.is_positive());
            close(
                paint.origin,
                pos2(
                    advance,
                    row.glyphs[paint.glyph_index].pos.y + metrics.y_offset_in_points,
                ),
            );
            close(
                paint.offset.to_pos2(),
                pos2(
                    position.x_offset as f32 * scale,
                    -position.y_offset as f32 * scale,
                ),
            );
            assert_eq!(paint.outline_scale, vec2(scale, -scale));
            assert_eq!(paint.color, Color32::BLUE);
            advance += position.x_advance as f32 * scale;
        }
        let first_font = row.glyph_paint[0].font.as_ref().unwrap();
        assert!(
            row.glyph_paint
                .iter()
                .all(|paint| Arc::ptr_eq(first_font, paint.font.as_ref().unwrap()))
        );
        assert!(
            row.glyphs
                .iter()
                .all(|glyph| glyph.paint_index == u32::MAX && glyph.section_index == u32::MAX)
        );
    }

    #[test]
    fn row_paint_ligature_continuations_and_invisible_characters() {
        let mut fonts = ubuntu(FontTweak::default());
        let plain_f = layout(&mut fonts, 1.0, Arc::new(job("f")));
        let ligature = layout(&mut fonts, 1.0, Arc::new(job("fi")));
        let row = &ligature.rows[0].row;
        assert_eq!(row.text(), "fi");
        assert_eq!(row.glyphs.len(), 2);
        assert_eq!(
            row.glyph_paint.len(),
            1,
            "fixture must shape a real ligature"
        );
        let paint = &row.glyph_paint[0];
        assert_eq!(paint.source, GlyphPaintSource::Outline);
        assert_ne!(paint.glyph_id, plain_f.rows[0].glyph_paint[0].glyph_id);
        assert_eq!(paint.glyph_index, 0);
        assert_eq!(row.glyphs[1].advance_width, 0.0);
        assert!(row.glyphs[1].uv_rect.is_nothing());
        assert_eq!(row.char_count_excluding_newline().0, 2);
        assert_eq!(row.x_offset(crate::text::CharIndex(1)), row.glyphs[1].pos.x);

        let invisible = layout(&mut fonts, 1.0, Arc::new(job("A\u{200b} \tB")));
        let row = &invisible.rows[0].row;
        assert_eq!(row.glyphs.len(), 5);
        assert!(!row.glyph_paint.iter().any(|paint| paint.glyph_index == 1));
        assert_eq!(
            row.glyph_paint
                .iter()
                .map(|paint| paint.source)
                .collect::<Vec<_>>(),
            vec![
                GlyphPaintSource::Outline,
                GlyphPaintSource::Empty,
                GlyphPaintSource::Empty,
                GlyphPaintSource::Outline
            ]
        );
    }

    #[test]
    fn row_paint_is_shared_through_composed_transforms_and_paragraph_cache() {
        let mut fonts = ubuntu(FontTweak::default());
        let mut cache = GalleyCache::default();
        let original = cache.layout(&mut fonts, 1.0, job("A\nfi"), true);
        let changed = cache.layout(&mut fonts, 1.0, job("B\nfi"), true);
        assert!(Arc::ptr_eq(&original.rows[1].row, &changed.rows[1].row));
        let records = Arc::clone(&original.rows[1].glyph_paint);
        let cursor_glyphs = original.rows[1].glyphs.clone();
        let mut shape =
            TextShape::new(pos2(7.25, 9.5), Arc::clone(&original), Color32::WHITE).with_angle(0.7);
        let before = shape.clone();
        let transforms = [
            TSTransform::new(vec2(3.3, -2.7), 1.75),
            TSTransform::new(vec2(-4.4, 8.2), 0.65),
        ];
        for transform in transforms {
            shape.transform(transform);
        }
        assert!(Arc::ptr_eq(&records, &shape.galley.rows[1].glyph_paint));
        assert!(Arc::ptr_eq(&original.job, &shape.galley.job));
        assert_eq!(shape.galley.rows[1].glyphs, cursor_glyphs);
        assert_eq!(original.rows[1].glyphs, cursor_glyphs);
        for point in [
            records[0].origin,
            records[0].bounds.min,
            records[0].bounds.max,
        ] {
            let initial = before.pos
                + Rot2::from_angle(before.angle)
                    * (original.rows[1].pos.to_vec2() + point.to_vec2());
            close(
                shape.glyph_paint_pos(&shape.galley.rows[1], point),
                transforms[1] * (transforms[0] * initial),
            );
        }
        assert_eq!(shape.glyph_paint_scale, 1.75 * 0.65);
    }

    #[test]
    fn row_paint_outline_source_survives_empty_bitmap_allocation() {
        let mut fonts = ubuntu(FontTweak::default());
        let mut input = job("A");
        input.sections[0].format.font_id.size = 0.01;
        let galley = layout(&mut fonts, 1.0, Arc::new(input));
        assert!(galley.rows[0].glyphs[0].uv_rect.is_nothing());
        let paint = &galley.rows[0].glyph_paint[0];
        assert_eq!(paint.source, GlyphPaintSource::Outline);
        assert!(paint.font.is_some() && paint.glyph_id.is_some());
        assert!(paint.outline_scale.x > 0.0 && paint.outline_scale.y < 0.0);
    }

    #[test]
    fn row_paint_overflow_retains_resolved_id_after_wrapping() {
        let mut fonts = ubuntu(FontTweak::default());
        let ellipsis = layout(&mut fonts, 1.0, Arc::new(job("…")));
        let mut input = job("AV fi test test test");
        input.wrap.max_width = 70.0;
        input.wrap.max_rows = 1;
        input.wrap.overflow_character = Some('…');
        let galley = layout(&mut fonts, 1.0, Arc::new(input));
        assert!(galley.elided);
        let row = &galley.rows[0].row;
        assert_eq!(row.glyphs.last().unwrap().chr, '…');
        let paint = row.glyph_paint.last().unwrap();
        assert_eq!(paint.glyph_id, ellipsis.rows[0].glyph_paint[0].glyph_id);
        assert_eq!(paint.glyph_index, row.glyphs.len() - 1);
        assert_eq!(paint.source, GlyphPaintSource::Outline);
        assert!(paint.bounds.is_positive());
    }

    fn raster(is_color: bool) -> GlyphRasterizer {
        GlyphRasterizer::new(
            "paint-source-test",
            move |request: &GlyphRasterizerRequest<'_>| {
                (request.cluster == "A").then(|| RasterizedGlyph {
                    bitmap: GlyphBitmap {
                        image: crate::ColorImage::new([2, 2], vec![Color32::WHITE; 4]),
                        offset_px: vec2(1.0, -3.0),
                        is_color,
                    },
                    advance_px: 8.0,
                })
            },
        )
        .with_priority(FontPriority::Highest)
    }

    #[test]
    fn row_paint_preserves_rasterizer_priority_and_stock_color_semantics() {
        for is_color in [false, true] {
            let mut fonts = ubuntu(FontTweak::default()).with_glyph_rasterizer(raster(is_color));
            for (color, override_color, opacity) in [
                (Color32::BLUE, None, 1.0),
                (Color32::PLACEHOLDER, None, 0.5),
                (Color32::BLUE, Some(Color32::from_black_alpha(128)), 0.75),
            ] {
                let mut input = job("AB");
                input.sections[0].format.color = color;
                let galley = Arc::new(layout(&mut fonts, 1.0, Arc::new(input)));
                let paint = &galley.rows[0].glyph_paint[0];
                assert_eq!(paint.source, GlyphPaintSource::CustomRasterizer);
                assert!(paint.font.is_none() && paint.glyph_id.is_none());
                assert_eq!(paint.is_color, is_color);
                assert!(paint.bounds.is_positive());
                assert_eq!(
                    galley.rows[0].glyph_paint[1].source,
                    GlyphPaintSource::Outline
                );
                let mut shape = TextShape::new(Pos2::ZERO, Arc::clone(&galley), Color32::GREEN)
                    .with_opacity_factor(opacity);
                shape.override_text_color = override_color;
                let mut mesh = Mesh::default();
                Tessellator::new(1.0, Default::default(), [1024, 1024], vec![])
                    .tessellate_text(&shape, &mut mesh);
                for paint in galley.rows[0].glyph_paint.iter() {
                    let first_vertex =
                        galley.rows[0].glyphs[paint.glyph_index].first_vertex as usize;
                    assert_eq!(
                        shape.glyph_paint_color(paint),
                        mesh.vertices[first_vertex].color
                    );
                }
            }
        }
    }

    #[test]
    fn row_paint_italic_bitmap_quad_matches_stock_mesh() {
        let mut fonts = ubuntu(FontTweak::default()).with_glyph_rasterizer(raster(false));
        let mut input = job("AB");
        input.sections[0].format.italics = true;
        let galley = layout(&mut fonts, 1.0, Arc::new(input));
        let row = &galley.rows[0];
        let sources = row
            .glyph_paint
            .iter()
            .map(|paint| paint.source)
            .collect::<Vec<_>>();
        assert_eq!(
            sources,
            vec![
                GlyphPaintSource::CustomRasterizer,
                GlyphPaintSource::Outline
            ]
        );
        let mut checked = 0;
        for paint in row.glyph_paint.iter() {
            let glyph = &row.glyphs[paint.glyph_index];
            if glyph.uv_rect.is_nothing() {
                continue;
            }
            let first = glyph.first_vertex as usize;
            let quad = &row.visuals.mesh.vertices[first..first + 4];
            let (top_left, bottom_left) = (quad[0].pos, quad[2].pos);
            assert_eq!(paint.shear, ITALIC_SHEAR);
            assert!((paint.shear_origin_y - bottom_left.y).abs() < 1e-4);
            assert!(
                (top_left.x - bottom_left.x - paint.shear * (bottom_left.y - top_left.y)).abs()
                    < 1e-4
            );
            if paint.source == GlyphPaintSource::CustomRasterizer {
                let mesh_bounds =
                    Rect::from_points(&quad.iter().map(|v| v.pos).collect::<Vec<_>>());
                close(paint.bounds.min, mesh_bounds.min);
                close(paint.bounds.max, mesh_bounds.max);
            }
            checked += 1;
        }
        assert_eq!(checked, 2, "both glyphs must have a stock bitmap quad");
    }

    #[test]
    fn outline_bounds_drawn_once_per_instance_glyph() {
        let text = "Hello wall";
        let sized = |size: f32| {
            let mut input = job(text);
            input.sections[0].format.font_id.size = size;
            Arc::new(input)
        };
        let mut fonts = ubuntu(FontTweak::default());
        let galleys = [12.0, 17.0, 23.0].map(|size| layout(&mut fonts, 1.0, sized(size)));
        let paints = &galleys[1].rows[0].glyph_paint;
        let instance = paints[0].font.as_ref().unwrap();
        for galley in &galleys {
            assert!(
                galley.rows[0]
                    .glyph_paint
                    .iter()
                    .all(|paint| { Arc::ptr_eq(instance, paint.font.as_ref().unwrap()) })
            );
        }
        let mut glyph_ids = paints
            .iter()
            .map(|paint| paint.glyph_id.unwrap())
            .collect::<Vec<_>>();
        glyph_ids.sort_unstable();
        glyph_ids.dedup();
        assert!(
            glyph_ids.len() < paints.len(),
            "fixture must repeat a glyph"
        );
        assert_eq!(instance.outline_bounds_draws(), glyph_ids.len());

        let mut alone = ubuntu(FontTweak::default());
        let expected = layout(&mut alone, 1.0, sized(17.0));
        let expected = &expected.rows[0].glyph_paint;
        assert_eq!(expected.len(), paints.len());
        for (paint, expected) in paints.iter().zip(expected.iter()) {
            assert_eq!(paint.glyph_id, expected.glyph_id);
            assert_eq!(paint.source, expected.source);
            assert_eq!(paint.bounds, expected.bounds);
        }
    }

    #[test]
    fn row_paint_color_adjustment_uses_copy_on_write() {
        let mut fonts = ubuntu(FontTweak::default());
        let galley = Arc::new(layout(&mut fonts, 1.0, Arc::new(job("A"))));
        let records = Arc::clone(&galley.rows[0].glyph_paint);
        let mut shape = Shape::Text(TextShape::new(
            Pos2::ZERO,
            Arc::clone(&galley),
            Color32::WHITE,
        ));
        crate::shape_transform::adjust_colors(&mut shape, |color| {
            *color = color.gamma_multiply(0.5);
        });
        let Shape::Text(shape) = shape else {
            unreachable!()
        };
        assert_eq!(records[0].color, Color32::BLUE);
        assert!(!Arc::ptr_eq(&records, &shape.galley.rows[0].glyph_paint));
        let vertex = &shape.galley.rows[0].visuals.mesh.vertices[0];
        assert_eq!(shape.galley.rows[0].glyph_paint[0].color, vertex.color);
        assert_eq!(galley.rows[0].glyph_paint[0].color, Color32::BLUE);
    }

    #[test]
    fn row_paint_fallback_font_and_missing_outline_remain_explicit() {
        // The icon is a private-use glyph only the bundled icon face has, so it resolves
        // through fallback to a second instance under every default font set.
        let mut fonts = FontsImpl::new(TextOptions::default(), FontDefinitions::default());
        let text = format!("A{}", epaint_default_fonts::special_emojis::GITHUB);
        let galley = layout(&mut fonts, 1.0, Arc::new(job(&text)));
        let paint = &galley.rows[0].glyph_paint;
        assert_eq!(paint.len(), 2);
        assert_eq!(paint[1].source, GlyphPaintSource::Outline);
        assert_ne!(
            paint[0].font.as_ref().unwrap().id(),
            paint[1].font.as_ref().unwrap().id()
        );
        let provider = paint[0].font.as_ref().unwrap();
        assert_eq!(
            provider.outline_bounds(u32::MAX),
            (GlyphPaintSource::Unsupported, Rect::NOTHING)
        );
    }
    #[test]
    fn row_paint_keeps_discovered_font_alive() {
        use crate::text::{FallbackRequest, FontInsert, InsertFontFamily};
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
        let galley = layout(&mut fonts, 1.0, Arc::new(job("fi")));
        assert_eq!(fonts.discovered_fonts().len(), 1);
        assert_eq!(galley.rows[0].glyph_paint.len(), 1);
        let paint = &galley.rows[0].glyph_paint[0];
        let provider = paint.font.as_ref().unwrap();
        assert!(Arc::ptr_eq(provider.font_data(), &data.font));
        drop(fonts);
        assert_eq!(
            provider.outline_bounds(paint.glyph_id.unwrap()).0,
            GlyphPaintSource::Outline
        );

        let mut fontless = FontsImpl::new(TextOptions::default(), FontDefinitions::empty());
        let tofu = layout(&mut fontless, 1.0, Arc::new(job("A")));
        assert_eq!(
            tofu.rows[0].glyph_paint[0].source,
            GlyphPaintSource::CustomRasterizer
        );
        assert!(tofu.rows[0].glyph_paint[0].font.is_none());
    }
}
