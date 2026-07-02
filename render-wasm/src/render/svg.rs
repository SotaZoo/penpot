use std::collections::HashSet;

use skia_safe::{self as skia, Paint};

use crate::error::Result;
use crate::shapes::{radius_to_sigma, BlurType, Fill, Shape, Stroke, StrokeKind, Type};
use crate::state::ShapesPoolRef;
use crate::uuid::Uuid;

use super::shape_renderer::ShapeRenderer;
use super::vector::{
    children_paint_order, draw_shape_geometry, render_leaf_content, ExportState, VectorRenderer,
    VectorTarget,
};
use super::RenderState;

/// Collects the registered font aliases used by every text span in the subtree
/// rooted at `id`, so the exporter can embed exactly those fonts.
fn collect_font_aliases(tree: ShapesPoolRef, id: &Uuid, out: &mut HashSet<String>) {
    let Some(shape) = tree.get(id) else {
        return;
    };

    if let Type::Text(_) = &shape.shape_type {
        for paragraph in shape.get_text_content().paragraphs() {
            for span in paragraph.children() {
                out.insert(format!("{}", span.font_family));
            }
        }
    }

    for child_id in shape.children_ids_iter_forward(true) {
        collect_font_aliases(tree, child_id, out);
    }
}

/// Renders a shape tree to an SVG document and returns the raw SVG bytes.
///
/// Dedicated vector-SVG render path. Leaf content (paths, text, fills, images)
/// is emitted as real SVG markup via short-lived Skia SVG canvases, while
/// composite effects — container/leaf opacity, blend mode, layer blur and masks
/// — are composed as native SVG `<g>` wrappers (`opacity`, `mix-blend-mode`,
/// `filter="feGaussianBlur"`, `clip-path`). This keeps everything vectorial and
/// faithful to the GPU/PDF output, sidestepping `SkSVGDevice`'s inability to
/// keep `save_layer` content.
///
/// Text is emitted as real `<text>` elements (no `CONVERT_TEXT_TO_PATHS`), so
/// the output stays selectable/editable. Skia's SVG backend does not embed
/// fonts, so we inject `@font-face` rules with the used fonts base64-embedded to
/// keep the document self-contained without relying on the viewer's fonts.
pub fn render_to_svg(
    shared: &mut RenderState,
    id: &Uuid,
    tree: ShapesPoolRef,
    scale: f32,
) -> Result<Vec<u8>> {
    let mut ctx = ExportState {
        fonts: &shared.fonts,
        images: Some(&mut shared.images as &mut dyn super::ImageProvider),
        sampling_options: shared.sampling_options,
    };
    render_tree_to_svg(&mut ctx, id, tree, scale)
}

/// Core SVG export, decoupled from `RenderState` via [`ExportState`] so it can
/// run on a plain CPU Skia canvas (and in headless native tests, which cannot
/// build a GPU-backed `RenderState`).
pub(crate) fn render_tree_to_svg(
    shared: &mut ExportState,
    id: &Uuid,
    tree: ShapesPoolRef,
    scale: f32,
) -> Result<Vec<u8>> {
    // Text shaping helpers read fonts from the global GPU `RenderState`; point
    // them at this export's `FontStore` for the duration of the render so text
    // (glyph shaping, strokes) works on `ExportState` alone — including headless
    // tests, which have no global state.
    let _fonts_guard =
        crate::globals::set_export_fonts(shared.fonts, crate::utils::Browser::Chrome as u8);

    let shape = tree
        .get(id)
        .ok_or_else(|| crate::error::Error::CriticalError("Shape not found for SVG".to_string()))?;
    let bounds = shape.extrect(tree, scale);

    let page_w = bounds.width() * scale;
    let page_h = bounds.height() * scale;
    let rect = skia::Rect::from_xywh(0., 0., page_w, page_h);

    let (defs, body) =
        render_svg_body(shared, id, tree, scale, rect, -bounds.left(), -bounds.top())?;

    // Embed the fonts used by the subtree as `@font-face` rules so the SVG
    // renders faithfully without relying on the viewer having them installed.
    let mut aliases = HashSet::new();
    collect_font_aliases(tree, id, &mut aliases);
    let font_css = shared.fonts.font_face_css_for_aliases(&aliases);

    let mut out = String::with_capacity(body.len() + defs.len() + font_css.len() + 256);
    out.push_str("<?xml version=\"1.0\" encoding=\"utf-8\" ?>\n");
    out.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" \
         width=\"{page_w}\" height=\"{page_h}\">"
    ));

    if !font_css.is_empty() || !defs.is_empty() {
        out.push_str("<defs>");
        if !font_css.is_empty() {
            out.push_str(&format!(
                "<style type=\"text/css\"><![CDATA[{font_css}]]></style>"
            ));
        }
        out.push_str(&defs);
        out.push_str("</defs>");
    }

    out.push_str(&body);
    out.push_str("</svg>");

    Ok(out.into_bytes())
}

// ===========================================================================
// SVG compositor
// ===========================================================================
//
// Skia's SVG backend (`SkSVGDevice`) silently drops everything drawn inside a
// `save_layer`, so composite effects (container/leaf opacity, blend mode, layer
// blur, masks) rendered the PDF way (one canvas + `save_layer`) vanish in SVG.
//
// Instead of one canvas, the SVG path composes the document itself: leaf
// content is drawn into short-lived `skia::svg::Canvas` *fragments* (real
// `<path>`/`<text>`/… vector markup), and composite effects become native SVG
// `<g>` wrappers (`opacity`, `mix-blend-mode`, `filter="feGaussianBlur"`,
// `clip-path`). This keeps the output fully vectorial and matches the GPU/PDF
// result without rasterizing.

/// Accumulates the SVG document body while drawing.
///
/// Leaf draws land in the current `pending` fragment; opening/closing a group
/// flushes it so its markup nests correctly inside the `<g>`.
struct SvgLayerCanvas {
    scale: f32,
    page_rect: skia::Rect,
    tx: f32,
    ty: f32,
    /// Composed body (nested `<g>` + fragment markup), in document order.
    out: String,
    /// Global `<defs>` (clip paths, blur filters).
    defs: String,
    /// Fragment collecting consecutive leaf draws, if any.
    pending: Option<skia::svg::Canvas>,
    /// Counter for unique def ids (`blur0`, `clip1`, …).
    next_id: usize,
    /// Counter for per-fragment id prefixes (avoids id clashes across frags).
    frag_no: usize,
    /// Interned reusable defs (`body` → id), deduping identical filters shared
    /// by several shapes (e.g. the same drop shadow).
    def_cache: std::collections::HashMap<String, String>,
}

impl SvgLayerCanvas {
    fn new(scale: f32, page_rect: skia::Rect, tx: f32, ty: f32) -> Self {
        Self {
            scale,
            page_rect,
            tx,
            ty,
            out: String::new(),
            defs: String::new(),
            pending: None,
            next_id: 0,
            frag_no: 0,
            def_cache: std::collections::HashMap::new(),
        }
    }

    fn unique(&mut self, prefix: &str) -> String {
        let id = format!("{prefix}{}", self.next_id);
        self.next_id += 1;
        id
    }

    /// Creates a fragment canvas configured with the page transform
    /// (scale + translate to the export bounds), matching the GPU/PDF page.
    fn new_fragment(&self) -> skia::svg::Canvas {
        let canvas = skia::svg::Canvas::new(self.page_rect, None);
        {
            let cv: &skia::Canvas = &*canvas;
            cv.scale((self.scale, self.scale));
            cv.translate((self.tx, self.ty));
        }
        canvas
    }

    /// Returns the current leaf-drawing canvas, creating a fragment if needed.
    fn canvas(&mut self) -> &skia::Canvas {
        if self.pending.is_none() {
            self.pending = Some(self.new_fragment());
        }
        self.pending.as_deref().unwrap()
    }

    /// Finalizes the pending fragment and appends its markup to `out`.
    fn flush(&mut self) {
        let Some(canvas) = self.pending.take() else {
            return;
        };
        let data = canvas.end();
        let doc = String::from_utf8_lossy(data.as_bytes());
        let inner = extract_inner_svg(&doc);
        if inner.trim().is_empty() {
            return;
        }
        let prefix = format!("f{}_", self.frag_no);
        self.frag_no += 1;
        self.out.push_str(&remap_ids(inner, &prefix));
    }

    fn open_group(&mut self, attrs: &str) {
        self.flush();
        self.out.push_str("<g ");
        self.out.push_str(attrs);
        self.out.push('>');
    }

    fn close_group(&mut self) {
        self.flush();
        self.out.push_str("</g>");
    }

    fn push_def(&mut self, def: &str) {
        self.defs.push_str(def);
    }

    /// Interns a reusable `<def>` keyed by its (id-independent) body. If an
    /// identical def was already emitted, its id is returned; otherwise a fresh
    /// id is minted, the def emitted via `render(id)` and cached. Lets several
    /// shapes share one filter instead of duplicating identical `<filter>`s.
    fn intern_def(&mut self, prefix: &str, key: &str, render: impl Fn(&str) -> String) -> String {
        if let Some(id) = self.def_cache.get(key) {
            return id.clone();
        }
        let id = self.unique(prefix);
        let def = render(&id);
        self.push_def(&def);
        self.def_cache.insert(key.to_string(), id.clone());
        id
    }

    /// Emits a `<clipPath>` from a mask shape's geometry (in device/page space)
    /// and returns nothing; the caller wraps content in
    /// `<g clip-path="url(#id)">`.
    ///
    /// A mask can be a group (Penpot masked groups often use a group of shapes
    /// as the mask). Since a group has no geometry of its own, we recurse into
    /// its descendants and accumulate their geometry — mirroring the GPU/PDF
    /// path, which re-renders the whole mask subtree with `DstIn`.
    fn push_clip_path(&mut self, id: &str, shape: &Shape, tree: ShapesPoolRef) {
        let canvas = self.new_fragment();
        {
            let cv: &skia::Canvas = &*canvas;
            let mut paint = Paint::default();
            paint.set_anti_alias(true);
            paint.set_color(skia::Color::BLACK);
            draw_clip_geometry(cv, shape, tree, &paint);
        }
        self.finish_clip_path_fragment(id, canvas);
    }

    /// Renders the mask subtree rooted at `mask_id` into an isolated fragment
    /// and registers it as an alpha `<mask>` def, returning the def id for the
    /// caller to reference via `mask="url(#id)"`.
    ///
    /// Unlike a geometric `<clipPath>`, this captures the mask's rendered
    /// *alpha* (fills included, so images/gradients/soft masks compose
    /// faithfully), matching the GPU/PDF `DstIn` mask. The subtree is rendered
    /// with a fresh [`SvgLayerCanvas`] and its ids are namespaced so they can't
    /// collide with the surrounding document.
    fn push_alpha_mask(
        &mut self,
        shared: &mut ExportState,
        mask_id: &Uuid,
        tree: ShapesPoolRef,
        scale: f32,
    ) -> Result<String> {
        let mut sub = SvgLayerCanvas::new(self.scale, self.page_rect, self.tx, self.ty);
        svg_render_tree(&mut sub, shared, mask_id, tree, scale, true)?;
        sub.flush();

        let prefix = format!("{}_", self.unique("m"));
        let mask_ref = self.unique("mask");
        let sub_defs = remap_ids(&sub.defs, &prefix);
        let sub_body = remap_ids(&sub.out, &prefix);

        if !sub_defs.is_empty() {
            self.defs.push_str(&sub_defs);
        }
        self.defs.push_str(&format!(
            "<mask id=\"{mask_ref}\" maskUnits=\"userSpaceOnUse\" \
             mask-type=\"alpha\">{sub_body}</mask>"
        ));
        Ok(mask_ref)
    }

    /// Builds an alpha `<mask>` from a stroke's opaque silhouette and returns its
    /// id, for confining an image-filled stroke to the stroke region. The
    /// silhouette is drawn opaque (alpha 1 in the stroke area), so `mask-type:
    /// alpha` shows the image exactly where the stroke paints.
    fn push_stroke_alpha_mask(
        &mut self,
        shared: &mut ExportState,
        element: &Shape,
        stroke: &Stroke,
        scale: f32,
    ) -> Result<String> {
        let canvas = self.new_fragment();
        {
            let cv: &skia::Canvas = &*canvas;
            cv.save();
            cv.concat(&element.centered_transform());
            let mut renderer = VectorRenderer::new(cv, shared, scale, VectorTarget::Svg);
            renderer.draw_stroke_silhouette(element, stroke)?;
            cv.restore();
        }

        let id = self.unique("simask");
        self.finish_alpha_mask_fragment(&id, canvas);
        Ok(id)
    }

    /// Finalizes a fragment canvas as an alpha `<mask>` def: its emitted markup's
    /// *alpha* becomes the mask (opaque shows, transparent hides), independent of
    /// color — unlike a luminance mask, so an opaque black silhouette works.
    fn finish_alpha_mask_fragment(&mut self, id: &str, canvas: skia::svg::Canvas) {
        let data = canvas.end();
        let doc = String::from_utf8_lossy(data.as_bytes());
        let inner = extract_inner_svg(&doc);
        let prefix = format!("f{}_", self.frag_no);
        self.frag_no += 1;
        let geometry = remap_ids(inner, &prefix);
        self.defs.push_str(&format!(
            "<mask id=\"{id}\" maskUnits=\"userSpaceOnUse\" \
             mask-type=\"alpha\">{geometry}</mask>"
        ));
    }

    /// Finalizes a fragment canvas as a `<clipPath>` def: its emitted markup
    /// (shape geometry or text glyph silhouette) becomes the clip geometry.
    fn finish_clip_path_fragment(&mut self, id: &str, canvas: skia::svg::Canvas) {
        let data = canvas.end();
        let doc = String::from_utf8_lossy(data.as_bytes());
        let inner = extract_inner_svg(&doc);
        let prefix = format!("f{}_", self.frag_no);
        self.frag_no += 1;
        let geometry = remap_ids(inner, &prefix);
        self.defs.push_str(&format!(
            "<clipPath id=\"{id}\" clipPathUnits=\"userSpaceOnUse\">{geometry}</clipPath>"
        ));
    }

    /// Finalizes a fragment canvas as a luminance `<mask>` def: its emitted
    /// markup becomes the mask (white shows, black hides).
    fn finish_mask_fragment(&mut self, id: &str, canvas: skia::svg::Canvas) {
        let data = canvas.end();
        let doc = String::from_utf8_lossy(data.as_bytes());
        let inner = extract_inner_svg(&doc);
        let prefix = format!("f{}_", self.frag_no);
        self.frag_no += 1;
        let geometry = remap_ids(inner, &prefix);
        self.defs.push_str(&format!(
            "<mask id=\"{id}\" maskUnits=\"userSpaceOnUse\">{geometry}</mask>"
        ));
    }
}

/// Draws a mask's clip geometry into `cv` (already set up with the page
/// transform). Leaf shapes contribute their own geometry under their
/// `centered_transform`; groups contribute nothing themselves but recurse into
/// their children (a group carries no geometry, and its children hold absolute
/// coordinates with their own transforms — the group transform is not
/// propagated, matching the GPU/PDF group rendering).
fn draw_clip_geometry(cv: &skia::Canvas, shape: &Shape, tree: ShapesPoolRef, paint: &Paint) {
    if let Type::Group(_) = &shape.shape_type {
        for child_id in shape.children_ids_iter_forward(true) {
            if let Some(child) = tree.get(child_id) {
                draw_clip_geometry(cv, child, tree, paint);
            }
        }
        return;
    }

    cv.save();
    cv.concat(&shape.centered_transform());
    draw_shape_geometry(cv, shape, paint);
    cv.restore();
}

/// Renders `id`'s subtree to an SVG body, returning `(defs, body)`.
fn render_svg_body(
    shared: &mut ExportState,
    id: &Uuid,
    tree: ShapesPoolRef,
    scale: f32,
    page_rect: skia::Rect,
    tx: f32,
    ty: f32,
) -> Result<(String, String)> {
    let mut builder = SvgLayerCanvas::new(scale, page_rect, tx, ty);
    svg_render_tree(&mut builder, shared, id, tree, scale, true)?;
    builder.flush();
    Ok((builder.defs, builder.out))
}

/// `render_shadows` is `false` while building a container's drop-shadow
/// silhouette: descendants must contribute their *shape* alpha but not their own
/// drop shadows (matching the GPU, which clears nested shadows when rendering a
/// container shadow). It is `true` for the normal document render.
fn svg_render_tree(
    builder: &mut SvgLayerCanvas,
    shared: &mut ExportState,
    id: &Uuid,
    tree: ShapesPoolRef,
    scale: f32,
    render_shadows: bool,
) -> Result<()> {
    let Some(element) = tree.get(id) else {
        return Ok(());
    };
    if element.hidden {
        return Ok(());
    }

    match &element.shape_type {
        Type::Group(group) => {
            svg_render_group(builder, shared, element, group.masked, tree, scale, render_shadows)
        }
        Type::Frame(_) => svg_render_frame(builder, shared, element, tree, scale, render_shadows),
        Type::Rect(_)
        | Type::Circle
        | Type::Path(_)
        | Type::Bool(_)
        | Type::Text(_)
        | Type::SVGRaw(_) => svg_render_leaf(builder, shared, element, scale, render_shadows),
    }
}

fn svg_render_group(
    builder: &mut SvgLayerCanvas,
    shared: &mut ExportState,
    element: &Shape,
    masked: bool,
    tree: ShapesPoolRef,
    scale: f32,
    render_shadows: bool,
) -> Result<()> {
    let effects = svg_effect_attrs(builder, element, scale);
    if let Some(attrs) = &effects {
        builder.open_group(attrs);
    }

    // Group drop shadow: the silhouette is the group's subtree (rendered
    // without its descendants' own shadows), drawn behind the content.
    if render_shadows {
        svg_render_container_drop_shadow(builder, shared, element, tree, scale)?;
    }

    // A Penpot mask is an *alpha* mask: content is clipped to the mask shape's
    // rendered alpha (geometry AND fill alpha), not merely its outline. Mirror
    // the GPU/PDF `DstIn` compose with an SVG `<mask mask-type="alpha">` that
    // holds the fully-rendered mask subtree — so image/gradient/soft masks work
    // (a geometric `<clipPath>` can only capture the outline).
    let mask_id = masked
        .then(|| element.mask_id())
        .flatten()
        .filter(|mid| tree.get(mid).is_some())
        .copied();
    if let Some(mid) = mask_id {
        let mask_ref = builder.push_alpha_mask(shared, &mid, tree, scale)?;
        builder.open_group(&format!("mask=\"url(#{mask_ref})\""));
    }

    for child_id in &children_paint_order(tree, element) {
        svg_render_tree(builder, shared, child_id, tree, scale, render_shadows)?;
    }

    if mask_id.is_some() {
        builder.close_group();
    }
    if effects.is_some() {
        builder.close_group();
    }
    Ok(())
}

fn svg_render_frame(
    builder: &mut SvgLayerCanvas,
    shared: &mut ExportState,
    element: &Shape,
    tree: ShapesPoolRef,
    scale: f32,
    render_shadows: bool,
) -> Result<()> {
    let matrix = element.centered_transform();

    let effects = svg_effect_attrs(builder, element, scale);
    if let Some(attrs) = &effects {
        builder.open_group(attrs);
    }

    // Frame drop shadow: like the GPU/PDF path, the silhouette is the frame's
    // rendered content (background + strokes + children) — but the descendants'
    // *own* drop shadows must not bleed into it (that would trace a second,
    // doubled silhouette). It is emitted as a separate shadow-only pass drawn
    // behind the content, sitting outside the content clip so the offset shadow
    // is not clipped to the frame bounds.
    if render_shadows {
        svg_render_container_drop_shadow(builder, shared, element, tree, scale)?;
    }

    let clipped = element.clip_content;
    if clipped {
        let clip_id = builder.unique("clip");
        builder.push_clip_path(&clip_id, element, tree);
        builder.open_group(&format!("clip-path=\"url(#{clip_id})\""));
    }

    // Frame background + inner shadows (frame space).
    if !element.fills.is_empty() {
        let canvas = builder.canvas();
        canvas.save();
        canvas.concat(&matrix);
        let mut renderer = VectorRenderer::new(canvas, shared, scale, VectorTarget::Svg);
        renderer.draw_fills(element, &element.fills)?;
        canvas.restore();
    }

    // Frame-background inner shadows (native `<g filter>`; the shared renderer's
    // `save_layer` version is dropped by `SkSVGDevice`). Over the background,
    // under the children — matching the GPU order.
    svg_render_inner_shadows(builder, shared, element, scale)?;

    // Children (absolute coords).
    for child_id in &children_paint_order(tree, element) {
        svg_render_tree(builder, shared, child_id, tree, scale, render_shadows)?;
    }

    // Strokes over children (frame space).
    let visible_strokes: Vec<&Stroke> = element.visible_strokes().collect();
    if !visible_strokes.is_empty() {
        let canvas = builder.canvas();
        canvas.save();
        canvas.concat(&matrix);
        let mut renderer = VectorRenderer::new(canvas, shared, scale, VectorTarget::Svg);
        renderer.draw_strokes(element, &visible_strokes)?;
        canvas.restore();

        // Image-filled frame strokes are deferred by the shared renderer; re-emit
        // the texture masked to the stroke region.
        svg_render_image_strokes(builder, shared, element, scale)?;
    }

    if clipped {
        builder.close_group();
    }
    if effects.is_some() {
        builder.close_group();
    }
    Ok(())
}

/// Emits a container's (frame/group) drop shadows as a shadow-only `<g filter>`
/// group drawn *behind* the content. The group holds the container's silhouette
/// — its own fills/strokes plus its descendants rendered with `render_shadows =
/// false` — so the filter's `SourceAlpha` is the shape silhouette *without* the
/// descendants' own drop shadows (which would otherwise cast a second, offset
/// green copy: the "doubled shadow"). The filter merges only the tinted/offset
/// shadow (no `SourceGraphic`), since the real content is drawn separately on
/// top.
///
/// For a frame, the descendants' silhouette is clipped to the frame geometry —
/// mirroring the GPU (`get_nested_shadow_clip_bounds` clips each child's shadow
/// to the frame selrect). The frame's *own* fills/strokes are left unclipped
/// (the GPU renders them with `clip_content = false`). The clip is applied
/// pre-offset; the filter's `feOffset` then shifts the clipped silhouette,
/// matching the GPU's selrect-shifted-by-offset clip.
fn svg_render_container_drop_shadow(
    builder: &mut SvgLayerCanvas,
    shared: &mut ExportState,
    element: &Shape,
    tree: ShapesPoolRef,
    scale: f32,
) -> Result<()> {
    let Some(attrs) = svg_drop_shadow_attr(builder, element, scale, false) else {
        return Ok(());
    };

    let matrix = element.centered_transform();
    builder.open_group(&attrs);

    // Frame background silhouette (frame space, unclipped).
    if !element.fills.is_empty() {
        let canvas = builder.canvas();
        canvas.save();
        canvas.concat(&matrix);
        let mut renderer = VectorRenderer::new(canvas, shared, scale, VectorTarget::Svg);
        renderer.draw_fills(element, &element.fills)?;
        canvas.restore();
    }

    // Descendants (absolute coords), with their own shadows suppressed. Clipped
    // to the frame geometry to match the GPU nested-shadow clip.
    let children = children_paint_order(tree, element);
    let clip_children = !children.is_empty() && matches!(element.shape_type, Type::Frame(_));
    if clip_children {
        let clip_id = builder.unique("clip");
        builder.push_clip_path(&clip_id, element, tree);
        builder.open_group(&format!("clip-path=\"url(#{clip_id})\""));
    }
    for child_id in &children {
        svg_render_tree(builder, shared, child_id, tree, scale, false)?;
    }
    if clip_children {
        builder.close_group();
    }

    // Frame strokes silhouette (frame space, unclipped).
    let visible_strokes: Vec<&Stroke> = element.visible_strokes().collect();
    if !visible_strokes.is_empty() {
        let canvas = builder.canvas();
        canvas.save();
        canvas.concat(&matrix);
        let mut renderer = VectorRenderer::new(canvas, shared, scale, VectorTarget::Svg);
        renderer.draw_strokes(element, &visible_strokes)?;
        canvas.restore();
    }

    builder.close_group();
    Ok(())
}

fn svg_render_leaf(
    builder: &mut SvgLayerCanvas,
    shared: &mut ExportState,
    element: &Shape,
    scale: f32,
    render_shadows: bool,
) -> Result<()> {
    let effects = svg_effect_attrs(builder, element, scale);
    if let Some(attrs) = &effects {
        builder.open_group(attrs);
    }

    // Drop shadows sit behind (and below the opacity/blend layer of) the shape.
    // `SkSVGDevice` drops the GPU/PDF `save_layer` shadow, so emit a native SVG
    // filter and let it produce the shadow + the shape on top. Suppressed while
    // rendering a parent container's shadow silhouette (`render_shadows` false).
    let shadow = if render_shadows {
        svg_drop_shadow_attr(builder, element, scale, true)
    } else {
        None
    };
    if let Some(attrs) = &shadow {
        builder.open_group(attrs);
    }

    {
        let matrix = element.centered_transform();
        let canvas = builder.canvas();
        canvas.save();
        canvas.concat(&matrix);
        let mut renderer = VectorRenderer::new(canvas, shared, scale, VectorTarget::Svg);
        render_leaf_content(&mut renderer, element)?;
        canvas.restore();
    }

    // Inner shadows compose inside a `save_layer` the SVG backend drops; re-emit
    // them natively over the fill (matches the GPU fill/inner-shadow/stroke order).
    svg_render_inner_shadows(builder, shared, element, scale)?;

    // Outer strokes on closed paths/bools are deferred by the shared renderer on
    // SVG (their `save_layer` + `Clear` composition is dropped by `SkSVGDevice`);
    // re-emit them as a nested `<g>` clipped to the shape's exterior.
    svg_render_path_outer_strokes(builder, element, scale)?;

    // Dotted inner/outer strokes on rect/circle are likewise deferred (their
    // boundary-ring clip lives in a dropped `save_layer`); re-emit them here
    // clipped/masked to the shape interior/exterior.
    svg_render_rect_circle_dotted_strokes(builder, element)?;

    // Image-filled strokes are deferred too (their `SrcIn` composition lives in a
    // dropped `save_layer`); re-emit the texture masked to the stroke region.
    svg_render_image_strokes(builder, shared, element, scale)?;

    // Semi-transparent text strokes are skipped by the shared renderer on SVG
    // (their opacity layer is a dropped `save_layer`); re-emit them here inside
    // a native `<g opacity>` wrapping the fully-opaque stroke.
    if let Type::Text(_) = &element.shape_type {
        svg_render_text_alpha_strokes(builder, shared, element, scale)?;
        svg_render_text_inner_strokes(builder, shared, element, scale)?;
        svg_render_text_outer_strokes(builder, shared, element, scale)?;
    }

    if shadow.is_some() {
        builder.close_group();
    }
    if effects.is_some() {
        builder.close_group();
    }
    Ok(())
}

/// Re-emits the *outer* solid/gradient strokes of a closed path/bool as a
/// nested `<g>` whose content (the shape stroked at double width) is clipped to
/// the shape's *exterior* via a luminance `<mask>` (white canvas minus the
/// shape silhouette). This mirrors the GPU/PDF `save_layer` + `Clear` (DstOut of
/// the shape), which `SkSVGDevice` drops — so the shared renderer skips these on
/// SVG and lets the compositor nest them here.
///
/// Image-fill outer strokes are out of scope (they need a GPU-backed store).
fn svg_render_path_outer_strokes(
    builder: &mut SvgLayerCanvas,
    element: &Shape,
    scale: f32,
) -> Result<()> {
    if !matches!(element.shape_type, Type::Path(_) | Type::Bool(_)) || element.is_open() {
        return Ok(());
    }

    let matrix = element.centered_transform();
    for stroke in element.visible_strokes() {
        if stroke.render_kind(false) != StrokeKind::Outer {
            continue;
        }

        // Inverse-of-shape luminance mask: white everywhere, black over the
        // shape. Only the stroke's outer half (outside the shape) survives.
        let mask_id = builder.unique("smask");
        {
            let canvas = builder.new_fragment();
            {
                let cv: &skia::Canvas = &*canvas;
                let mut white = Paint::default();
                white.set_color(skia::Color::WHITE);
                cv.draw_rect(
                    skia::Rect::from_ltrb(-100_000.0, -100_000.0, 100_000.0, 100_000.0),
                    &white,
                );
                cv.save();
                cv.concat(&matrix);
                let mut black = Paint::default();
                black.set_anti_alias(true);
                black.set_color(skia::Color::BLACK);
                draw_shape_geometry(cv, element, &black);
                cv.restore();
            }
            builder.finish_mask_fragment(&mask_id, canvas);
        }

        builder.open_group(&format!("mask=\"url(#{mask_id})\""));
        {
            let canvas = builder.canvas();
            canvas.save();
            canvas.concat(&matrix);
            let svg_attrs = element.svg_attrs.as_ref();
            let paint = stroke.to_stroked_paint(false, &element.selrect, svg_attrs, true);
            draw_shape_geometry(canvas, element, &paint);
            canvas.restore();
        }
        builder.close_group();
    }
    Ok(())
}

/// Re-emits *image-filled* strokes. On the GPU/PDF path the texture is confined
/// to the stroke with a `save_layer` + `SrcIn` over the stroke silhouette, which
/// `SkSVGDevice` drops — so the shared renderer skips them on SVG and the
/// compositor draws the image under a `<mask>` built from that silhouette.
///
/// - Solid/center strokes: the silhouette is the stroke geometry, used directly
///   as the mask.
/// - Dotted inner/outer strokes: the silhouette is the boundary-centered dot
///   ring; the image is drawn under that ring mask *and* inside a second group
///   that restricts it to the shape interior (inner) or exterior (outer), since
///   the ring straddles the boundary.
///
/// Without an image store (headless export/tests) there is nothing to draw.
fn svg_render_image_strokes(
    builder: &mut SvgLayerCanvas,
    shared: &mut ExportState,
    element: &Shape,
    scale: f32,
) -> Result<()> {
    if shared.images.is_none() {
        return Ok(());
    }

    let matrix = element.centered_transform();
    for stroke in element.visible_strokes() {
        if !matches!(stroke.fill, Fill::Image(_)) {
            continue;
        }

        match stroke.clip_op() {
            // Solid / dotted-center: mask straight to the stroke silhouette.
            None => {
                let mask_id = builder.push_stroke_alpha_mask(shared, element, stroke, scale)?;
                builder.open_group(&format!("mask=\"url(#{mask_id})\""));
                svg_draw_stroke_image(builder, shared, element, stroke, scale, &matrix)?;
                builder.close_group();
            }
            // Dotted inner/outer: restrict to interior/exterior, then to the ring.
            Some(clip_op) => {
                if clip_op == skia::ClipOp::Intersect {
                    let clip_id = push_leaf_clip_path(builder, element, &matrix);
                    builder.open_group(&format!("clip-path=\"url(#{clip_id})\""));
                } else {
                    let mask_id = push_inverse_shape_mask(builder, element, &matrix);
                    builder.open_group(&format!("mask=\"url(#{mask_id})\""));
                }
                let ring_id = push_dotted_ring_alpha_mask(builder, element, stroke, &matrix);
                builder.open_group(&format!("mask=\"url(#{ring_id})\""));
                svg_draw_stroke_image(builder, shared, element, stroke, scale, &matrix)?;
                builder.close_group(); // ring
                builder.close_group(); // interior/exterior restriction
            }
        }
    }
    Ok(())
}

/// Draws an image-filled stroke's texture over its destination rect into the
/// current group (the caller supplies the mask/clip that confines it).
fn svg_draw_stroke_image(
    builder: &mut SvgLayerCanvas,
    shared: &mut ExportState,
    element: &Shape,
    stroke: &Stroke,
    scale: f32,
    matrix: &skia::Matrix,
) -> Result<()> {
    let canvas = builder.canvas();
    canvas.save();
    canvas.concat(matrix);
    let mut renderer = VectorRenderer::new(canvas, shared, scale, VectorTarget::Svg);
    renderer.draw_stroke_image(element, stroke)?;
    canvas.restore();
    Ok(())
}

/// Registers a `<clipPath>` from a leaf shape's geometry and returns its id.
fn push_leaf_clip_path(
    builder: &mut SvgLayerCanvas,
    element: &Shape,
    matrix: &skia::Matrix,
) -> String {
    let clip_id = builder.unique("dclip");
    let canvas = builder.new_fragment();
    {
        let cv: &skia::Canvas = &*canvas;
        cv.concat(matrix);
        let mut black = Paint::default();
        black.set_anti_alias(true);
        black.set_color(skia::Color::BLACK);
        draw_shape_geometry(cv, element, &black);
    }
    builder.finish_clip_path_fragment(&clip_id, canvas);
    clip_id
}

/// Registers an inverse-of-shape luminance `<mask>` (white everywhere, black over
/// the shape) and returns its id, keeping only content *outside* the shape.
fn push_inverse_shape_mask(
    builder: &mut SvgLayerCanvas,
    element: &Shape,
    matrix: &skia::Matrix,
) -> String {
    let mask_id = builder.unique("dmask");
    let canvas = builder.new_fragment();
    {
        let cv: &skia::Canvas = &*canvas;
        let mut white = Paint::default();
        white.set_color(skia::Color::WHITE);
        cv.draw_rect(
            skia::Rect::from_ltrb(-100_000.0, -100_000.0, 100_000.0, 100_000.0),
            &white,
        );
        cv.save();
        cv.concat(matrix);
        let mut black = Paint::default();
        black.set_anti_alias(true);
        black.set_color(skia::Color::BLACK);
        draw_shape_geometry(cv, element, &black);
        cv.restore();
    }
    builder.finish_mask_fragment(&mask_id, canvas);
    mask_id
}

/// Registers an alpha `<mask>` of a stroke's boundary-centered dot ring (the
/// shape geometry stroked with the dotted paint, opaque) and returns its id.
fn push_dotted_ring_alpha_mask(
    builder: &mut SvgLayerCanvas,
    element: &Shape,
    stroke: &Stroke,
    matrix: &skia::Matrix,
) -> String {
    let svg_attrs = element.svg_attrs.as_ref();
    let mask_id = builder.unique("dimask");
    let canvas = builder.new_fragment();
    {
        let cv: &skia::Canvas = &*canvas;
        cv.concat(matrix);
        // Keep the dotted paint's dash effect and width, but paint it opaque so
        // the alpha mask captures the exact dot footprint (fill is irrelevant).
        let mut paint = stroke.to_paint(&element.selrect, svg_attrs, true);
        paint.set_shader(None);
        paint.set_color(skia::Color::BLACK);
        draw_shape_geometry(cv, element, &paint);
    }
    builder.finish_alpha_mask_fragment(&mask_id, canvas);
    mask_id
}

/// Re-emits the *dotted inner/outer* strokes of a rect/circle. On the GPU/PDF
/// path these stamp a ring of dots centered on the shape boundary and clip it
/// to the shape interior (inner) or exterior (outer) inside a `save_layer` that
/// `SkSVGDevice` drops — so the shared renderer skips them on SVG and the
/// compositor re-emits the same ring of dots here, clipped natively: a
/// `<g clip-path>` (inner) or a `<g mask>` with the inverse-of-shape luminance
/// mask (outer, since SVG `<clipPath>` cannot subtract).
fn svg_render_rect_circle_dotted_strokes(builder: &mut SvgLayerCanvas, element: &Shape) -> Result<()> {
    if !matches!(element.shape_type, Type::Rect(_) | Type::Circle) {
        return Ok(());
    }

    let matrix = element.centered_transform();
    let svg_attrs = element.svg_attrs.as_ref();

    for stroke in element.visible_strokes() {
        let Some(clip_op) = stroke.clip_op() else {
            continue;
        };
        // Image-filled dotted strokes are re-emitted by `svg_render_image_strokes`
        // (the dots must be filled with the texture, not a solid paint).
        if matches!(stroke.fill, Fill::Image(_)) {
            continue;
        }

        // The ring of dots: the shape geometry stroked with the dotted paint
        // (`outer_rect` returns the boundary for dotted inner/outer, so the
        // dots center on it exactly as on the GPU/PDF backends).
        let paint = stroke.to_paint(&element.selrect, svg_attrs, true);

        if clip_op == skia::ClipOp::Intersect {
            // Inner: keep only the dot halves inside the shape.
            let clip_id = builder.unique("dclip");
            {
                let canvas = builder.new_fragment();
                {
                    let cv: &skia::Canvas = &*canvas;
                    cv.concat(&matrix);
                    let mut black = Paint::default();
                    black.set_anti_alias(true);
                    black.set_color(skia::Color::BLACK);
                    draw_shape_geometry(cv, element, &black);
                }
                builder.finish_clip_path_fragment(&clip_id, canvas);
            }
            builder.open_group(&format!("clip-path=\"url(#{clip_id})\""));
        } else {
            // Outer: keep only the dot halves outside the shape via an
            // inverse-of-shape luminance mask (white minus the silhouette).
            let mask_id = builder.unique("dmask");
            {
                let canvas = builder.new_fragment();
                {
                    let cv: &skia::Canvas = &*canvas;
                    let mut white = Paint::default();
                    white.set_color(skia::Color::WHITE);
                    cv.draw_rect(
                        skia::Rect::from_ltrb(-100_000.0, -100_000.0, 100_000.0, 100_000.0),
                        &white,
                    );
                    cv.save();
                    cv.concat(&matrix);
                    let mut black = Paint::default();
                    black.set_anti_alias(true);
                    black.set_color(skia::Color::BLACK);
                    draw_shape_geometry(cv, element, &black);
                    cv.restore();
                }
                builder.finish_mask_fragment(&mask_id, canvas);
            }
            builder.open_group(&format!("mask=\"url(#{mask_id})\""));
        }

        {
            let canvas = builder.canvas();
            canvas.save();
            canvas.concat(&matrix);
            draw_shape_geometry(canvas, element, &paint);
            canvas.restore();
        }
        builder.close_group();
    }
    Ok(())
}

/// Emits each semi-transparent *center* text stroke as a `<g opacity>` wrapper
/// around the fully-opaque stroke geometry, matching the GPU/PDF opacity-layer
/// result without a `save_layer` (which `SkSVGDevice` would drop). Inner strokes
/// are handled by `svg_render_text_inner_strokes` and outer strokes (any
/// opacity) by `svg_render_text_outer_strokes`.
fn svg_render_text_alpha_strokes(
    builder: &mut SvgLayerCanvas,
    shared: &mut ExportState,
    element: &Shape,
    scale: f32,
) -> Result<()> {
    let matrix = element.centered_transform();
    for stroke in element.visible_strokes() {
        if stroke.render_kind(false) != StrokeKind::Center {
            continue;
        }
        let opacity = stroke.fill.opacity();
        if opacity >= 1.0 {
            // Opaque strokes were already drawn by `render_leaf_content`.
            continue;
        }

        builder.open_group(&format!("opacity=\"{opacity}\""));
        {
            let canvas = builder.canvas();
            canvas.save();
            canvas.concat(&matrix);
            let mut renderer = VectorRenderer::new(canvas, shared, scale, VectorTarget::Svg);
            renderer.draw_text_stroke_opaque(element, stroke)?;
            canvas.restore();
        }
        builder.close_group();
    }
    Ok(())
}

/// Emits each inner text stroke as a `<g clip-path>` (glyph-silhouette clip)
/// wrapping the fully-opaque double-width stroke, plus a `<g opacity>` when the
/// stroke is semi-transparent. Reproduces the GPU/PDF mask + `SrcIn` + `DstOver`
/// inner-stroke composition, which `SkSVGDevice` drops (it lives inside
/// `save_layer`s). Inner strokes vanish from SVG regardless of opacity, so all
/// of them are handled here.
fn svg_render_text_inner_strokes(
    builder: &mut SvgLayerCanvas,
    shared: &mut ExportState,
    element: &Shape,
    scale: f32,
) -> Result<()> {
    let matrix = element.centered_transform();
    for stroke in element.visible_strokes() {
        if stroke.render_kind(false) != StrokeKind::Inner {
            continue;
        }

        // Clip path from the opaque glyph silhouette: clipping the double-width
        // stroke to the glyph interior keeps only its inner half.
        let clip_id = builder.unique("tclip");
        {
            let canvas = builder.new_fragment();
            {
                let cv: &skia::Canvas = &*canvas;
                cv.concat(&matrix);
                let mut renderer = VectorRenderer::new(cv, shared, scale, VectorTarget::Svg);
                renderer.draw_text_glyph_silhouette(element)?;
            }
            builder.finish_clip_path_fragment(&clip_id, canvas);
        }

        let opacity = stroke.fill.opacity();
        let mut attrs = format!("clip-path=\"url(#{clip_id})\"");
        if opacity < 1.0 {
            attrs.push_str(&format!(" opacity=\"{opacity}\""));
        }
        builder.open_group(&attrs);
        {
            let canvas = builder.canvas();
            canvas.save();
            canvas.concat(&matrix);
            let mut renderer = VectorRenderer::new(canvas, shared, scale, VectorTarget::Svg);
            renderer.draw_text_stroke_opaque(element, stroke)?;
            canvas.restore();
        }
        builder.close_group();
    }
    Ok(())
}

/// Emits each *outer* text stroke as a `<g mask>` (glyph-exterior mask) wrapping
/// the fully-opaque double-width stroke, plus a `<g opacity>` when the stroke is
/// semi-transparent. The shared renderer draws the outer stroke at double width
/// centered on the glyph outline and relies on a `save_layer` + `Clear` (keep
/// only the outer half) that `SkSVGDevice` drops — so on SVG it is skipped there
/// and re-emitted here masked to the glyph exterior (an inverse-of-glyph
/// luminance mask: white canvas minus the glyphs, since `<clipPath>` cannot
/// subtract), keeping only the outer half and matching the GPU/PDF width.
fn svg_render_text_outer_strokes(
    builder: &mut SvgLayerCanvas,
    shared: &mut ExportState,
    element: &Shape,
    scale: f32,
) -> Result<()> {
    let matrix = element.centered_transform();
    for stroke in element.visible_strokes() {
        if stroke.render_kind(false) != StrokeKind::Outer {
            continue;
        }

        // Inverse-of-glyph luminance mask: white everywhere, black over the
        // glyphs. Only the stroke's outer half (outside the glyphs) survives.
        let mask_id = builder.unique("tmask");
        {
            let canvas = builder.new_fragment();
            {
                let cv: &skia::Canvas = &*canvas;
                let mut white = Paint::default();
                white.set_color(skia::Color::WHITE);
                cv.draw_rect(
                    skia::Rect::from_ltrb(-100_000.0, -100_000.0, 100_000.0, 100_000.0),
                    &white,
                );
                cv.save();
                cv.concat(&matrix);
                let mut renderer = VectorRenderer::new(cv, shared, scale, VectorTarget::Svg);
                renderer.draw_text_glyph_silhouette(element)?;
                cv.restore();
            }
            builder.finish_mask_fragment(&mask_id, canvas);
        }

        let opacity = stroke.fill.opacity();
        let mut attrs = format!("mask=\"url(#{mask_id})\"");
        if opacity < 1.0 {
            attrs.push_str(&format!(" opacity=\"{opacity}\""));
        }
        builder.open_group(&attrs);
        {
            let canvas = builder.canvas();
            canvas.save();
            canvas.concat(&matrix);
            let mut renderer = VectorRenderer::new(canvas, shared, scale, VectorTarget::Svg);
            renderer.draw_text_stroke_opaque(element, stroke)?;
            canvas.restore();
        }
        builder.close_group();
    }
    Ok(())
}

/// Registers an SVG `<filter>` reproducing the shape's visible drop shadows
/// (Penpot paints them behind the shape) and returns the `<g filter=…>`
/// attribute referencing it, or `None` when there are none.
///
/// `SkSVGDevice` drops the `save_layer`-based shadow the GPU/PDF path uses, so
/// the SVG backend emits a native filter instead. Each shadow is built from
/// `SourceAlpha` (optionally dilated for spread), blurred, offset and tinted,
/// then merged. Blur maps to `stdDeviation` via the same `radius_to_sigma` used
/// by the GPU filter, and filtering happens in sRGB to match the rasterized
/// output.
///
/// When `include_source` is true the shape (`SourceGraphic`) is merged on top of
/// its shadows — the normal leaf case. When false the filter emits only the
/// shadow (used for a container's separate silhouette pass, where the real
/// content is drawn on top afterwards). The filter is interned so shapes sharing
/// the same shadow parameters reuse a single `<filter>` def.
fn svg_drop_shadow_attr(
    builder: &mut SvgLayerCanvas,
    element: &Shape,
    scale: f32,
    include_source: bool,
) -> Option<String> {
    let shadows: Vec<&crate::shapes::Shadow> = element.drop_shadows_visible().collect();
    if shadows.is_empty() {
        return None;
    }

    // The filter region must contain every shadow's full spread, otherwise the
    // blurred result is clipped and looks like a denser, hard-edged block. Track
    // the largest reach (blur ~3σ + |offset| + spread) so the region can be sized
    // to it as a fraction of the shape's bounds (`objectBoundingBox`).
    let mut margin_x = 0.0_f32;
    let mut margin_y = 0.0_f32;

    let mut prims = String::new();
    let mut merges = String::new();
    for (i, shadow) in shadows.iter().enumerate() {
        let sigma = radius_to_sigma(shadow.blur * scale);
        let dx = shadow.offset.0 * scale;
        let dy = shadow.offset.1 * scale;
        // 3σ captures ~99.7% of the Gaussian; add the offset (either direction)
        // and the spread dilation so the region always covers the shadow.
        let reach = 3.0 * sigma + shadow.spread * scale;
        margin_x = margin_x.max(reach + dx.abs());
        margin_y = margin_y.max(reach + dy.abs());
        let color = shadow.color;
        let hex = format!("#{:02X}{:02X}{:02X}", color.r(), color.g(), color.b());
        let opacity = color.a() as f32 / 255.0;

        let mut input = "SourceAlpha".to_string();
        if shadow.spread > 0.0 {
            let dilated = format!("shsp{i}");
            prims.push_str(&format!(
                "<feMorphology in=\"{input}\" operator=\"dilate\" \
                 radius=\"{}\" result=\"{dilated}\"/>",
                shadow.spread * scale
            ));
            input = dilated;
        }
        let blurred = format!("shbl{i}");
        prims.push_str(&format!(
            "<feGaussianBlur in=\"{input}\" stdDeviation=\"{sigma}\" result=\"{blurred}\"/>"
        ));
        let offset = format!("shof{i}");
        prims.push_str(&format!(
            "<feOffset in=\"{blurred}\" dx=\"{dx}\" dy=\"{dy}\" result=\"{offset}\"/>"
        ));
        let flood = format!("shfl{i}");
        prims.push_str(&format!(
            "<feFlood flood-color=\"{hex}\" flood-opacity=\"{opacity}\" result=\"{flood}\"/>"
        ));
        let tinted = format!("shad{i}");
        prims.push_str(&format!(
            "<feComposite in=\"{flood}\" in2=\"{offset}\" \
             operator=\"in\" result=\"{tinted}\"/>"
        ));
        merges.push_str(&format!("<feMergeNode in=\"{tinted}\"/>"));
    }
    if include_source {
        // The shape itself sits on top of all its shadows.
        merges.push_str("<feMergeNode in=\"SourceGraphic\"/>");
    }

    // Convert the pixel margins to `objectBoundingBox` fractions of the shape's
    // (scaled) bounds. A tiny shape with a large blur therefore gets a region
    // hundreds of percent wide, which is exactly what a soft shadow needs.
    let bbox = element.selrect();
    let w = (bbox.width() * scale).abs().max(1.0);
    let h = (bbox.height() * scale).abs().max(1.0);
    let fx = -margin_x / w;
    let fy = -margin_y / h;
    let fw = 1.0 + 2.0 * margin_x / w;
    let fh = 1.0 + 2.0 * margin_y / h;

    let body = format!(
        "x=\"{:.4}%\" y=\"{:.4}%\" width=\"{:.4}%\" height=\"{:.4}%\" \
         primitiveUnits=\"userSpaceOnUse\" color-interpolation-filters=\"sRGB\">\
         {prims}<feMerge>{merges}</feMerge>",
        fx * 100.0,
        fy * 100.0,
        fw * 100.0,
        fh * 100.0,
    );
    let id = builder.intern_def("shadow", &body, |id| format!("<filter id=\"{id}\" {body}</filter>"));
    Some(format!("filter=\"url(#{id})\""))
}

/// Builds an `filter="url(#…)"` attribute for a shape's *inner* shadows, or
/// `None` when it has none. The GPU/PDF path composes inner shadows inside a
/// `save_layer` (+ image filter) that `SkSVGDevice` drops, so — mirroring the
/// drop-shadow compositor — we emit a native SVG `<filter>` instead.
///
/// The filter output is the tinted inner-shadow band(s) *only* (no
/// `SourceGraphic`); the caller applies it to an opaque silhouette drawn right
/// over the shape's fill, so the shadow darkens the interior. For each shadow:
/// blur+offset the silhouette alpha to build an occluder, take the part of the
/// shape *not* covered by it (`operator="out"`), tint it, optionally grow it by
/// the spread, and finally clip it back inside the shape (`operator="in"`).
fn svg_inner_shadow_attr(
    builder: &mut SvgLayerCanvas,
    element: &Shape,
    scale: f32,
) -> Option<String> {
    let shadows: Vec<&crate::shapes::Shadow> = element.inner_shadows_visible().collect();
    if shadows.is_empty() {
        return None;
    }

    // Intermediates (offset occluder) extend beyond the shape by blur+offset, so
    // the region must contain them even though the final output is clipped back
    // to the shape. Same adaptive sizing as the drop-shadow filter.
    let mut margin_x = 0.0_f32;
    let mut margin_y = 0.0_f32;

    let mut prims = String::new();
    let mut merges = String::new();
    for (i, shadow) in shadows.iter().enumerate() {
        let sigma = radius_to_sigma(shadow.blur * scale);
        let dx = shadow.offset.0 * scale;
        let dy = shadow.offset.1 * scale;
        let reach = 3.0 * sigma + shadow.spread * scale;
        margin_x = margin_x.max(reach + dx.abs());
        margin_y = margin_y.max(reach + dy.abs());
        let color = shadow.color;
        let hex = format!("#{:02X}{:02X}{:02X}", color.r(), color.g(), color.b());
        let opacity = color.a() as f32 / 255.0;

        // Occluder: the silhouette alpha, blurred then offset.
        let blurred = format!("isbl{i}");
        prims.push_str(&format!(
            "<feGaussianBlur in=\"SourceAlpha\" stdDeviation=\"{sigma}\" result=\"{blurred}\"/>"
        ));
        let occluder = format!("isof{i}");
        prims.push_str(&format!(
            "<feOffset in=\"{blurred}\" dx=\"{dx}\" dy=\"{dy}\" result=\"{occluder}\"/>"
        ));
        // Inner band: the shape minus the occluder (the edge away from the offset).
        let band = format!("isbd{i}");
        prims.push_str(&format!(
            "<feComposite in=\"SourceAlpha\" in2=\"{occluder}\" \
             operator=\"out\" result=\"{band}\"/>"
        ));
        let flood = format!("isfl{i}");
        prims.push_str(&format!(
            "<feFlood flood-color=\"{hex}\" flood-opacity=\"{opacity}\" result=\"{flood}\"/>"
        ));
        let tinted = format!("istn{i}");
        prims.push_str(&format!(
            "<feComposite in=\"{flood}\" in2=\"{band}\" \
             operator=\"in\" result=\"{tinted}\"/>"
        ));
        // Spread grows the shadow (dilate), then it is clipped back inside the shape.
        let mut clip_input = tinted.clone();
        if shadow.spread > 0.0 {
            let dilated = format!("issp{i}");
            prims.push_str(&format!(
                "<feMorphology in=\"{clip_input}\" operator=\"dilate\" \
                 radius=\"{}\" result=\"{dilated}\"/>",
                shadow.spread * scale
            ));
            clip_input = dilated;
        }
        let shade = format!("issh{i}");
        prims.push_str(&format!(
            "<feComposite in=\"{clip_input}\" in2=\"SourceAlpha\" \
             operator=\"in\" result=\"{shade}\"/>"
        ));
        merges.push_str(&format!("<feMergeNode in=\"{shade}\"/>"));
    }

    let bbox = element.selrect();
    let w = (bbox.width() * scale).abs().max(1.0);
    let h = (bbox.height() * scale).abs().max(1.0);
    let fx = -margin_x / w;
    let fy = -margin_y / h;
    let fw = 1.0 + 2.0 * margin_x / w;
    let fh = 1.0 + 2.0 * margin_y / h;

    let body = format!(
        "x=\"{:.4}%\" y=\"{:.4}%\" width=\"{:.4}%\" height=\"{:.4}%\" \
         primitiveUnits=\"userSpaceOnUse\" color-interpolation-filters=\"sRGB\">\
         {prims}<feMerge>{merges}</feMerge>",
        fx * 100.0,
        fy * 100.0,
        fw * 100.0,
        fh * 100.0,
    );
    let id = builder.intern_def("inshadow", &body, |id| {
        format!("<filter id=\"{id}\" {body}</filter>")
    });
    Some(format!("filter=\"url(#{id})\""))
}

/// Emits a shape's inner shadows as a `<g filter>` wrapping an opaque silhouette
/// (the shape geometry, or the glyph silhouette for text) drawn over the fill.
/// The filter turns that silhouette into the interior shadow band(s).
///
/// Matches the GPU/PDF order: inner shadows sit over the fill and (for non-text
/// leaves) require fills to be present.
fn svg_render_inner_shadows(
    builder: &mut SvgLayerCanvas,
    shared: &mut ExportState,
    element: &Shape,
    scale: f32,
) -> Result<()> {
    let is_text = matches!(element.shape_type, Type::Text(_));
    if !is_text && !element.has_fills() {
        return Ok(());
    }
    let Some(attrs) = svg_inner_shadow_attr(builder, element, scale) else {
        return Ok(());
    };

    builder.open_group(&attrs);
    {
        let matrix = element.centered_transform();
        let canvas = builder.canvas();
        canvas.save();
        canvas.concat(&matrix);
        if is_text {
            let mut renderer = VectorRenderer::new(canvas, shared, scale, VectorTarget::Svg);
            renderer.draw_text_glyph_silhouette(element)?;
        } else {
            let mut paint = Paint::default();
            paint.set_anti_alias(true);
            paint.set_color(skia::Color::BLACK);
            draw_shape_geometry(canvas, element, &paint);
        }
        canvas.restore();
    }
    builder.close_group();
    Ok(())
}

/// Builds the `<g>` attribute string for a shape's composite effects (opacity,
/// blend mode, layer blur), registering any needed `<defs>`. Returns `None`
/// when the shape needs no wrapper.
fn svg_effect_attrs(
    builder: &mut SvgLayerCanvas,
    element: &Shape,
    scale: f32,
) -> Option<String> {
    let mut parts: Vec<String> = Vec::new();

    let opacity = element.opacity();
    if opacity < 1.0 {
        parts.push(format!("opacity=\"{opacity}\""));
    }

    if let Some(css) = blend_css(element.blend_mode().0) {
        parts.push(format!("style=\"mix-blend-mode:{css}\""));
    }

    if let Some(value) = layer_blur_value(element) {
        let sigma = radius_to_sigma(value * scale);
        let id = builder.unique("blur");
        builder.push_def(&format!(
            "<filter id=\"{id}\" x=\"-50%\" y=\"-50%\" width=\"200%\" height=\"200%\">\
             <feGaussianBlur stdDeviation=\"{sigma}\"/></filter>"
        ));
        parts.push(format!("filter=\"url(#{id})\""));
    }

    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" "))
    }
}

/// Layer-blur radius of a shape, if it has a visible layer blur.
fn layer_blur_value(element: &Shape) -> Option<f32> {
    element
        .blur
        .and_then(|b| (!b.hidden && b.blur_type == BlurType::LayerBlur && b.value > 0.0).then_some(b.value))
}

/// Maps a Skia blend mode to its CSS `mix-blend-mode` keyword. Returns `None`
/// for `SrcOver` (normal) and modes without a CSS equivalent.
fn blend_css(mode: skia::BlendMode) -> Option<&'static str> {
    use skia::BlendMode::*;
    Some(match mode {
        Multiply => "multiply",
        Screen => "screen",
        Overlay => "overlay",
        Darken => "darken",
        Lighten => "lighten",
        ColorDodge => "color-dodge",
        ColorBurn => "color-burn",
        HardLight => "hard-light",
        SoftLight => "soft-light",
        Difference => "difference",
        Exclusion => "exclusion",
        Hue => "hue",
        Saturation => "saturation",
        Color => "color",
        Luminosity => "luminosity",
        _ => return None,
    })
}

/// Returns the inner body of a Skia SVG document (everything between the
/// opening `<svg …>` tag and the closing `</svg>`).
fn extract_inner_svg(doc: &str) -> &str {
    let start = doc
        .find("<svg")
        .and_then(|s| doc[s..].find('>').map(|e| s + e + 1));
    let end = doc.rfind("</svg>");
    match (start, end) {
        (Some(s), Some(e)) if s <= e => &doc[s..e],
        _ => "",
    }
}

/// Prefixes every id defined in a fragment (and its `url(#…)` / `#…`
/// references) so ids stay unique once fragments are merged into one document.
fn remap_ids(body: &str, prefix: &str) -> String {
    // Collect id definitions.
    let needle = "id=\"";
    let mut ids: Vec<&str> = Vec::new();
    let mut offset = 0;
    while let Some(pos) = body[offset..].find(needle) {
        let start = offset + pos + needle.len();
        let Some(end_rel) = body[start..].find('"') else {
            break;
        };
        let id = &body[start..start + end_rel];
        if !id.is_empty() {
            ids.push(id);
        }
        offset = start + end_rel + 1;
    }

    ids.sort_unstable();
    ids.dedup();
    // Longest-first so a shorter id can't collide inside a longer one; the
    // quote/paren delimiters below already prevent partial matches.
    ids.sort_by(|a, b| b.len().cmp(&a.len()));

    let mut out = body.to_string();
    for id in ids {
        let new_id = format!("{prefix}{id}");
        out = out.replace(&format!("id=\"{id}\""), &format!("id=\"{new_id}\""));
        out = out.replace(&format!("url(#{id})"), &format!("url(#{new_id})"));
        out = out.replace(&format!("=\"#{id}\""), &format!("=\"#{new_id}\""));
    }
    out
}

// ===========================================================================
// Tests
// ===========================================================================
//
// These are fast, headless native tests (`cargo test --bin render_wasm`) for
// the SVG exporter. They bypass the GPU/browser stack entirely: shapes are
// built directly into a `ShapesPool` and rendered through
// [`render_tree_to_svg`] with a GPU-free [`ExportState`] (a standalone
// `FontStore`, no `ImageStore`). Output is checked with `insta` snapshots.
//
// To (re)generate snapshots after a deliberate change:
//   cargo insta test --accept --bin render_wasm
// (or run the tests and `cargo insta accept`).
#[cfg(test)]
mod tests {
    use super::{render_tree_to_svg, ExportState};
    use crate::render::{FontStore, ImageProvider};
    use crate::shapes::{
        BlendMode, Blur, BlurType, Fill, FontFamily, FontStyle, Frame, GrowType, Group, ImageFill,
        Paragraph, Path, Rect, Segment, Shadow, ShadowStyle, SolidColor, Stroke, StrokeKind,
        StrokeStyle, TextAlign, TextContent, TextDirection, TextSpan, Type,
    };
    use crate::state::ShapesPool;
    use crate::utils::uuid_from_u32_quartet;
    use crate::uuid::Uuid;
    use skia_safe as skia;

    /// A GPU-free export context: real fonts (embedded, no GPU), no image store
    /// (image fills need a GPU-backed store and are out of scope for these fast
    /// tests).
    fn export_state(fonts: &FontStore) -> ExportState<'_> {
        ExportState {
            fonts,
            images: None,
            sampling_options: skia::SamplingOptions::new(
                skia::FilterMode::Linear,
                skia::MipmapMode::Nearest,
            ),
        }
    }

    /// Deterministic UUID from a small integer, keeping snapshots stable.
    fn uid(n: u32) -> Uuid {
        uuid_from_u32_quartet(0, 0, 0, n)
    }

    /// Adds a solid-filled rectangle to the pool and returns nothing; callers
    /// tweak the returned shape via `pool.get_mut` when they need effects.
    fn add_solid_rect(
        pool: &mut ShapesPool,
        id: Uuid,
        parent: Uuid,
        (l, t, r, b): (f32, f32, f32, f32),
        color: skia::Color,
    ) {
        let shape = pool.add_shape(id);
        shape.set_parent(parent);
        shape.set_shape_type(Type::Rect(Rect::default()));
        shape.set_selrect(l, t, r, b);
        shape.set_fills(vec![Fill::Solid(SolidColor(color))]);
    }

    fn render(pool: &ShapesPool, root: Uuid) -> String {
        let fonts = FontStore::try_new().expect("font store");
        let mut ctx = export_state(&fonts);
        let bytes = render_tree_to_svg(&mut ctx, &root, pool, 1.0).expect("svg export");
        String::from_utf8(bytes).expect("utf8 svg")
    }

    #[test]
    fn exports_a_solid_rect() {
        let mut pool = ShapesPool::new();
        let id = uid(1);
        add_solid_rect(
            &mut pool,
            id,
            Uuid::nil(),
            (0.0, 0.0, 100.0, 80.0),
            skia::Color::from_rgb(255, 0, 0),
        );

        insta::assert_snapshot!(render(&pool, id));
    }

    #[test]
    fn exports_leaf_opacity_and_blend_mode_as_group_wrappers() {
        let mut pool = ShapesPool::new();
        let id = uid(1);
        add_solid_rect(
            &mut pool,
            id,
            Uuid::nil(),
            (0.0, 0.0, 100.0, 100.0),
            skia::Color::from_rgb(0, 128, 255),
        );
        {
            let shape = pool.get_mut(&id).unwrap();
            shape.set_opacity(0.5);
            shape.set_blend_mode(BlendMode(skia::BlendMode::Multiply));
        }

        let svg = render(&pool, id);
        // Composite effects must become native SVG group wrappers, not dropped
        // `save_layer`s.
        assert!(svg.contains("opacity=\"0.5\""), "missing opacity wrapper: {svg}");
        assert!(
            svg.contains("mix-blend-mode:multiply"),
            "missing blend-mode wrapper: {svg}"
        );
        insta::assert_snapshot!(svg);
    }

    #[test]
    fn exports_a_group_with_two_rects_and_group_opacity() {
        let mut pool = ShapesPool::new();
        let group_id = uid(1);
        let a = uid(2);
        let b = uid(3);

        {
            let group = pool.add_shape(group_id);
            group.set_parent(Uuid::nil());
            group.set_shape_type(Type::Group(Group { masked: false }));
            group.set_selrect(0.0, 0.0, 200.0, 100.0);
            group.set_opacity(0.7);
            group.add_child(a);
            group.add_child(b);
        }

        add_solid_rect(
            &mut pool,
            a,
            group_id,
            (0.0, 0.0, 90.0, 100.0),
            skia::Color::from_rgb(0, 0, 255),
        );
        add_solid_rect(
            &mut pool,
            b,
            group_id,
            (110.0, 0.0, 200.0, 100.0),
            skia::Color::from_rgb(0, 200, 0),
        );

        insta::assert_snapshot!(render(&pool, group_id));
    }

    /// A minimal masked group: the first child is a solid rectangle acting as
    /// the mask, clipping a single solid-filled rectangle to its rendered
    /// alpha. A Penpot mask is an alpha mask, so the mask needs a fill; an
    /// opaque solid fill clips exactly to the mask's geometry.
    #[test]
    fn exports_a_masked_group() {
        let mut pool = ShapesPool::new();
        let group_id = uid(1);
        let mask = uid(2); // first child — the mask
        let content = uid(3);

        {
            let group = pool.add_shape(group_id);
            group.set_parent(Uuid::nil());
            group.set_shape_type(Type::Group(Group { masked: true }));
            group.set_selrect(0.0, 0.0, 100.0, 100.0);
            // First child is the mask; the rest is the masked content.
            group.add_child(mask);
            group.add_child(content);
        }

        // Mask: a solid rectangle covering the left half of the group. Its
        // opaque alpha defines the clip region.
        add_solid_rect(
            &mut pool,
            mask,
            group_id,
            (0.0, 0.0, 50.0, 100.0),
            skia::Color::from_rgb(0, 0, 0),
        );

        // Masked content: a solid rectangle spanning the whole group.
        add_solid_rect(
            &mut pool,
            content,
            group_id,
            (0.0, 0.0, 100.0, 100.0),
            skia::Color::from_rgb(255, 0, 0),
        );

        insta::assert_snapshot!(render(&pool, group_id));
    }

    /// A masked group whose *mask is itself a group* of shapes (a very common
    /// Penpot pattern). The mask region is the union of the mask group's
    /// descendants — here a horizontal + vertical band forming a plus sign —
    /// and the masked content is an ellipse.
    ///
    /// Regression test: a group carries no geometry of its own, so the SVG
    /// exporter must render the whole mask subtree into the `<mask>`. Otherwise
    /// the mask comes out empty and the content renders unmasked (or fully
    /// hidden).
    ///
    /// Mirrors the transit shape (with small, round coordinates):
    ///   group (masked-group=true) with children
    ///     [ group (the mask) with [Rectangle #B1B2B5, Rectangle #B1B2B5],
    ///       Ellipse #1a4de5 (content) ]
    #[test]
    fn exports_a_masked_group_whose_mask_is_a_group() {
        let mut pool = ShapesPool::new();
        let group_id = uid(1);
        let mask_group = uid(2); // first child — the mask (a group)
        let ellipse = uid(3); // masked content
        let band_h = uid(4); // horizontal band (mask geometry)
        let band_v = uid(5); // vertical band (mask geometry)

        {
            let group = pool.add_shape(group_id);
            group.set_parent(Uuid::nil());
            group.set_shape_type(Type::Group(Group { masked: true }));
            group.set_selrect(0.0, 0.0, 100.0, 100.0);
            // First child is the mask group; the rest is the masked content.
            group.add_child(mask_group);
            group.add_child(ellipse);
        }

        // Mask: a group of two rectangles forming a plus sign. No geometry of
        // its own — its mask region comes from its children.
        {
            let group = pool.add_shape(mask_group);
            group.set_parent(group_id);
            group.set_shape_type(Type::Group(Group { masked: false }));
            group.set_selrect(0.0, 0.0, 100.0, 100.0);
            group.add_child(band_h);
            group.add_child(band_v);
        }
        add_solid_rect(
            &mut pool,
            band_h,
            mask_group,
            (0.0, 40.0, 100.0, 60.0),
            skia::Color::from_rgb(0xB1, 0xB2, 0xB5),
        );
        add_solid_rect(
            &mut pool,
            band_v,
            mask_group,
            (40.0, 0.0, 60.0, 100.0),
            skia::Color::from_rgb(0xB1, 0xB2, 0xB5),
        );

        // Masked content: an ellipse (oval from its selrect).
        {
            let shape = pool.add_shape(ellipse);
            shape.set_parent(group_id);
            shape.set_shape_type(Type::Circle);
            shape.set_selrect(0.0, 0.0, 100.0, 100.0);
            shape.set_fills(vec![Fill::Solid(SolidColor(skia::Color::from_rgb(
                0x1A, 0x4D, 0xE5,
            )))]);
        }

        let svg = render(&pool, group_id);
        // The mask is a group, so the `<mask>` must contain the descendants'
        // rendered content: the content group references a real mask and the
        // mask def is not empty.
        assert!(
            svg.contains("mask=\"url(#"),
            "content must reference a mask: {svg}"
        );
        assert!(
            !svg.contains("mask-type=\"alpha\"></mask>"),
            "mask from a group must not be empty: {svg}"
        );
        insta::assert_snapshot!(svg);
    }

    /// A simple leaf shape with a drop shadow: a solid-filled ellipse with one
    /// visible drop shadow (offset, blur, tint). The GPU/PDF path draws the
    /// shadow via a `save_layer` image filter, which `SkSVGDevice` drops — so
    /// the SVG backend must emit a native `<filter>` (blur + offset + flood)
    /// producing the shadow behind the shape.
    ///
    /// Mirrors the transit shape (with small, round coordinates):
    ///   Ellipse #ffbbbb with drop-shadow
    ///     { color #380000, opacity 1, offset (10, 4), blur 4, spread 0 }
    #[test]
    fn exports_a_shape_with_a_drop_shadow() {
        let mut pool = ShapesPool::new();
        let id = uid(1);

        {
            let shape = pool.add_shape(id);
            shape.set_parent(Uuid::nil());
            shape.set_shape_type(Type::Circle);
            shape.set_selrect(0.0, 0.0, 100.0, 80.0);
            shape.set_fills(vec![Fill::Solid(SolidColor(skia::Color::from_rgb(
                0xFF, 0xBB, 0xBB,
            )))]);
            shape.add_shadow(Shadow::new(
                skia::Color::from_argb(0xFF, 0x38, 0x00, 0x00),
                4.0,          // blur
                0.0,          // spread
                (10.0, 4.0),  // offset (x, y)
                ShadowStyle::Drop,
                false, // hidden
            ));
        }

        let svg = render(&pool, id);
        // The drop shadow must survive as a native SVG filter (the GPU/PDF
        // `save_layer` shadow is dropped by `SkSVGDevice`).
        assert!(
            svg.contains("<filter") && svg.contains("feOffset") && svg.contains("feGaussianBlur"),
            "drop shadow must be emitted as an SVG filter: {svg}"
        );
        assert!(
            svg.contains("flood-color=\"#380000\""),
            "shadow tint must be preserved: {svg}"
        );
        insta::assert_snapshot!(svg);
    }

    /// A simple leaf shape with a layer blur: a solid rectangle with a visible
    /// `LayerBlur`. The GPU/PDF path blurs via a `save_layer` image filter,
    /// which `SkSVGDevice` drops — so the SVG backend emits a native
    /// `<filter><feGaussianBlur>` wrapper (`stdDeviation` from `radius_to_sigma`).
    #[test]
    fn exports_a_shape_with_a_layer_blur() {
        let mut pool = ShapesPool::new();
        let id = uid(1);
        add_solid_rect(
            &mut pool,
            id,
            Uuid::nil(),
            (0.0, 0.0, 100.0, 100.0),
            skia::Color::from_rgb(0, 128, 255),
        );
        {
            let shape = pool.get_mut(&id).unwrap();
            shape.set_blur(Some(Blur::new(BlurType::LayerBlur, false, 8.0)));
        }

        let svg = render(&pool, id);
        // The layer blur must survive as a native SVG filter (the GPU/PDF
        // `save_layer` blur is dropped by `SkSVGDevice`).
        assert!(
            svg.contains("<filter") && svg.contains("feGaussianBlur"),
            "layer blur must be emitted as an SVG feGaussianBlur filter: {svg}"
        );
        assert!(
            svg.contains("filter=\"url(#blur"),
            "shape must reference the blur filter: {svg}"
        );
        insta::assert_snapshot!(svg);
    }

    /// A frame ("Board") with no fill, an inner solid stroke and a drop shadow.
    /// The GPU/PDF path casts the frame shadow from its rendered silhouette
    /// (including the stroke) via a `save_layer` image filter that `SkSVGDevice`
    /// drops — so the SVG backend must emit a native `<filter>` wrapping the
    /// frame content, sitting outside the content clip so the offset shadow is
    /// not clipped to the frame bounds.
    ///
    /// Mirrors the transit shape: a 354x204 board, inner 10px black stroke, drop
    /// shadow { color #000000 opacity 0.2, offset (40, 40), blur 4, spread 0 }.
    #[test]
    fn exports_a_frame_with_a_drop_shadow() {
        let mut pool = ShapesPool::new();
        let id = uid(1);
        {
            let shape = pool.add_shape(id);
            shape.set_parent(Uuid::nil());
            shape.set_shape_type(Type::Frame(Frame {
                corners: None,
                layout: None,
            }));
            shape.set_selrect(0.0, 0.0, 354.0, 204.0);
            let mut stroke =
                Stroke::new_inner_stroke(10.0, StrokeStyle::Solid, None, None, None, None);
            stroke.fill = Fill::Solid(SolidColor(skia::Color::from_rgb(0, 0, 0)));
            shape.add_stroke(stroke);
            shape.add_shadow(Shadow::new(
                skia::Color::from_argb(0x33, 0x00, 0x00, 0x00), // #000000 @ ~0.2
                4.0,          // blur
                0.0,          // spread
                (40.0, 40.0), // offset (x, y)
                ShadowStyle::Drop,
                false, // hidden
            ));
        }

        let svg = render(&pool, id);
        // The frame drop shadow must survive as a native SVG filter (the GPU/PDF
        // `save_layer` shadow is dropped by `SkSVGDevice`).
        assert!(
            svg.contains("<filter") && svg.contains("feOffset") && svg.contains("feGaussianBlur"),
            "frame drop shadow must be emitted as an SVG filter: {svg}"
        );
        assert!(
            svg.contains("filter=\"url(#shadow"),
            "frame content must reference the shadow filter: {svg}"
        );
        // The stroke (shadow silhouette source) must still be present.
        assert!(
            svg.contains("stroke-width=\"10\""),
            "frame stroke must be present: {svg}"
        );
        insta::assert_snapshot!(svg);
    }

    #[test]
    fn frame_drop_shadow_does_not_double_child_shadows() {
        // A frame with a drop shadow, containing a child that has its *own* drop
        // shadow. The frame shadow silhouette must trace the child's shape only,
        // never the child's shadow — otherwise the frame shadow is painted twice
        // (a second copy offset by the child shadow). The silhouette pass renders
        // descendants with their shadows suppressed, so only one frame shadow
        // filter and one child shadow filter exist, each used once.
        let mut pool = ShapesPool::new();
        let frame_id = uid(1);
        let child_id = uid(2);
        {
            let frame = pool.add_shape(frame_id);
            frame.set_parent(Uuid::nil());
            frame.set_shape_type(Type::Frame(Frame {
                corners: None,
                layout: None,
            }));
            frame.set_selrect(0.0, 0.0, 300.0, 200.0);
            frame.add_shadow(Shadow::new(
                skia::Color::from_rgb(0x58, 0xEA, 0x66), // green frame shadow
                0.0,
                0.0,
                (50.0, 50.0),
                ShadowStyle::Drop,
                false,
            ));
            frame.add_child(child_id);
        }
        {
            let child = pool.add_shape(child_id);
            child.set_parent(frame_id);
            child.set_shape_type(Type::Rect(Rect::default()));
            child.set_selrect(40.0, 40.0, 160.0, 120.0);
            child.add_fill(Fill::Solid(SolidColor(skia::Color::from_rgb(
                0xE1, 0x7F, 0xDA,
            ))));
            child.add_shadow(Shadow::new(
                skia::Color::from_rgb(0x19, 0x00, 0xFF), // blue child shadow
                0.0,
                0.0,
                (20.0, 20.0),
                ShadowStyle::Drop,
                false,
            ));
        }

        let svg = render(&pool, frame_id);

        // Both shadows are emitted as distinct filters, each defined exactly once
        // (no redundant duplicated `<filter>`s, no doubled frame silhouette).
        assert_eq!(
            svg.matches("flood-color=\"#58EA66\"").count(),
            1,
            "frame (green) shadow filter must be defined once: {svg}"
        );
        assert_eq!(
            svg.matches("flood-color=\"#1900FF\"").count(),
            1,
            "child (blue) shadow filter must be defined once: {svg}"
        );
        // The child's own shadow must not leak into the frame silhouette: it is
        // referenced by exactly one `<g filter>` (the real content), while the
        // silhouette pass renders the child with shadows suppressed.
        assert_eq!(
            svg.matches("filter=\"url(#shadow").count(),
            2,
            "exactly the frame-silhouette + child-content shadow refs: {svg}"
        );
        insta::assert_snapshot!(svg);
    }

    /// Adds a solid-filled rectangle with a single solid stroke of the given
    /// kind/width/color.
    fn add_stroked_rect(
        pool: &mut ShapesPool,
        id: Uuid,
        rect: (f32, f32, f32, f32),
        fill: skia::Color,
        kind: StrokeKind,
        width: f32,
        stroke_color: skia::Color,
    ) {
        add_styled_stroked_rect(
            pool,
            id,
            rect,
            fill,
            kind,
            StrokeStyle::Solid,
            width,
            stroke_color,
        );
    }

    /// Adds a solid-filled rectangle with a single stroke of the given
    /// kind/style/width/color.
    fn add_styled_stroked_rect(
        pool: &mut ShapesPool,
        id: Uuid,
        (l, t, r, b): (f32, f32, f32, f32),
        fill: skia::Color,
        kind: StrokeKind,
        style: StrokeStyle,
        width: f32,
        stroke_color: skia::Color,
    ) {
        let shape = pool.add_shape(id);
        shape.set_parent(Uuid::nil());
        shape.set_shape_type(Type::Rect(Rect::default()));
        shape.set_selrect(l, t, r, b);
        shape.set_fills(vec![Fill::Solid(SolidColor(fill))]);

        let mut stroke = match kind {
            StrokeKind::Inner => Stroke::new_inner_stroke(width, style, None, None, None, None),
            StrokeKind::Center => Stroke::new_center_stroke(width, style, None, None, None, None),
            StrokeKind::Outer => Stroke::new_outer_stroke(width, style, None, None, None, None),
        };
        stroke.fill = Fill::Solid(SolidColor(stroke_color));
        shape.add_stroke(stroke);
    }

    /// A simple rectangle with a solid *inner* stroke: the stroke sits fully
    /// inside the shape's geometry.
    #[test]
    fn exports_a_shape_with_a_solid_inner_stroke() {
        let mut pool = ShapesPool::new();
        let id = uid(1);
        add_stroked_rect(
            &mut pool,
            id,
            (0.0, 0.0, 100.0, 80.0),
            skia::Color::from_rgb(0xCC, 0xCC, 0xCC),
            StrokeKind::Inner,
            10.0,
            skia::Color::from_rgb(0xFF, 0x00, 0x00),
        );

        let svg = render(&pool, id);
        // Skia's SVG backend serializes colors as CSS names / short hex, so
        // assert on the emitted stroke rather than an exact hex string.
        assert!(
            svg.contains("stroke-width=\"10\"") && svg.contains("stroke=\"red\""),
            "inner stroke must be present: {svg}"
        );
        insta::assert_snapshot!(svg);
    }

    /// A simple rectangle with a solid *center* stroke: the stroke straddles the
    /// geometry (half inside, half outside).
    #[test]
    fn exports_a_shape_with_a_solid_center_stroke() {
        let mut pool = ShapesPool::new();
        let id = uid(1);
        add_stroked_rect(
            &mut pool,
            id,
            (0.0, 0.0, 100.0, 80.0),
            skia::Color::from_rgb(0xCC, 0xCC, 0xCC),
            StrokeKind::Center,
            10.0,
            skia::Color::from_rgb(0x00, 0x88, 0x00),
        );

        let svg = render(&pool, id);
        assert!(
            svg.contains("stroke-width=\"10\"") && svg.contains("stroke=\"#080\""),
            "center stroke must be present: {svg}"
        );
        insta::assert_snapshot!(svg);
    }

    /// A simple rectangle with a solid *outer* stroke: the stroke sits fully
    /// outside the shape's geometry.
    #[test]
    fn exports_a_shape_with_a_solid_outer_stroke() {
        let mut pool = ShapesPool::new();
        let id = uid(1);
        add_stroked_rect(
            &mut pool,
            id,
            (0.0, 0.0, 100.0, 80.0),
            skia::Color::from_rgb(0xCC, 0xCC, 0xCC),
            StrokeKind::Outer,
            10.0,
            skia::Color::from_rgb(0x00, 0x00, 0xFF),
        );

        let svg = render(&pool, id);
        assert!(
            svg.contains("stroke-width=\"10\"") && svg.contains("stroke=\"blue\""),
            "outer stroke must be present: {svg}"
        );
        insta::assert_snapshot!(svg);
    }

    // -- image-filled strokes ------------------------------------------------
    //
    // Image-filled strokes are the trickiest SVG case: the GPU/PDF path paints
    // the texture inside a `save_layer` + `SrcIn` composition that `SkSVGDevice`
    // drops. The SVG compositor instead emits the raw `<image>` confined by an
    // alpha `<mask>` of the stroke silhouette (solid/dotted-center) or, for
    // dotted inner/outer, an interior/exterior restriction plus a dotted-ring
    // alpha mask. These tests exercise both branches with a GPU-free
    // [`ImageProvider`] so they stay fast and headless.

    /// A CPU-only [`ImageProvider`] returning one tiny raster image for a fixed
    /// id, so image-stroke tests need no GPU-backed `ImageStore`.
    struct FakeImages {
        id: Uuid,
        image: skia::Image,
    }

    impl ImageProvider for FakeImages {
        fn get_cpu_image(&mut self, id: &Uuid) -> Option<skia::Image> {
            (*id == self.id).then(|| self.image.clone())
        }
    }

    /// A 2×2 non-uniform raster image (kept tiny so its base64 stays small and
    /// deterministic in snapshots).
    fn tiny_image() -> skia::Image {
        let info = skia::ImageInfo::new_n32_premul((2, 2), None);
        let mut surface = skia::surfaces::raster(&info, None, None).expect("raster surface");
        let canvas = surface.canvas();
        canvas.clear(skia::Color::from_rgb(0x00, 0x99, 0xFF));
        let mut paint = skia::Paint::default();
        paint.set_color(skia::Color::from_rgb(0xFF, 0x33, 0x00));
        canvas.draw_rect(skia::Rect::from_xywh(0.0, 0.0, 1.0, 1.0), &paint);
        canvas.draw_rect(skia::Rect::from_xywh(1.0, 1.0, 1.0, 1.0), &paint);
        surface.image_snapshot()
    }

    /// Renders `root` with an image provider available (image draws enabled).
    fn render_with_images(
        pool: &ShapesPool,
        root: Uuid,
        images: &mut dyn ImageProvider,
    ) -> String {
        let fonts = FontStore::try_new().expect("font store");
        let mut ctx = ExportState {
            fonts: &fonts,
            images: Some(images),
            sampling_options: skia::SamplingOptions::new(
                skia::FilterMode::Linear,
                skia::MipmapMode::Nearest,
            ),
        };
        let bytes = render_tree_to_svg(&mut ctx, &root, pool, 1.0).expect("svg export");
        String::from_utf8(bytes).expect("utf8 svg")
    }

    /// Adds a solid-filled rectangle whose single stroke is *image-filled* (the
    /// stroke references `image_id` via an [`ImageFill`]).
    fn add_image_stroked_rect(
        pool: &mut ShapesPool,
        id: Uuid,
        image_id: Uuid,
        (l, t, r, b): (f32, f32, f32, f32),
        fill: skia::Color,
        kind: StrokeKind,
        style: StrokeStyle,
        width: f32,
    ) {
        let shape = pool.add_shape(id);
        shape.set_parent(Uuid::nil());
        shape.set_shape_type(Type::Rect(Rect::default()));
        shape.set_selrect(l, t, r, b);
        shape.set_fills(vec![Fill::Solid(SolidColor(fill))]);

        let mut stroke = match kind {
            StrokeKind::Inner => Stroke::new_inner_stroke(width, style, None, None, None, None),
            StrokeKind::Center => Stroke::new_center_stroke(width, style, None, None, None, None),
            StrokeKind::Outer => Stroke::new_outer_stroke(width, style, None, None, None, None),
        };
        stroke.fill = Fill::Image(ImageFill::new(image_id, 255, 2, 2, false));
        shape.add_stroke(stroke);
    }

    /// A rectangle with a solid, image-filled *center* stroke: the texture is
    /// re-emitted as an `<image>` confined by an alpha `<mask>` of the stroke
    /// silhouette (no dropped `save_layer`).
    #[test]
    fn exports_a_shape_with_a_solid_center_image_stroke() {
        let mut pool = ShapesPool::new();
        let id = uid(1);
        let image_id = uid(9);
        add_image_stroked_rect(
            &mut pool,
            id,
            image_id,
            (0.0, 0.0, 100.0, 80.0),
            skia::Color::from_rgb(0xCC, 0xCC, 0xCC),
            StrokeKind::Center,
            StrokeStyle::Solid,
            16.0,
        );

        let mut images = FakeImages {
            id: image_id,
            image: tiny_image(),
        };
        let svg = render_with_images(&pool, id, &mut images);
        assert!(
            svg.contains("<image"),
            "stroke image must be emitted: {svg}"
        );
        assert!(
            svg.contains("mask-type=\"alpha\""),
            "stroke silhouette alpha mask must be present: {svg}"
        );
        insta::assert_snapshot!(svg);
    }

    /// A rectangle with a *dotted inner* image-filled stroke: the texture is
    /// confined to the dot ring (alpha mask) and to the shape interior
    /// (`clip-path`).
    #[test]
    fn exports_a_shape_with_a_dotted_inner_image_stroke() {
        let mut pool = ShapesPool::new();
        let id = uid(1);
        let image_id = uid(9);
        add_image_stroked_rect(
            &mut pool,
            id,
            image_id,
            (0.0, 0.0, 100.0, 80.0),
            skia::Color::from_rgb(0xCC, 0xCC, 0xCC),
            StrokeKind::Inner,
            StrokeStyle::Dotted,
            16.0,
        );

        let mut images = FakeImages {
            id: image_id,
            image: tiny_image(),
        };
        let svg = render_with_images(&pool, id, &mut images);
        assert!(svg.contains("<image"), "stroke image must be emitted: {svg}");
        // Dotted inner: interior restriction is a clip-path, the dot ring is an
        // alpha mask.
        assert!(
            svg.contains("clip-path=\"url(#") && svg.contains("mask-type=\"alpha\""),
            "dotted-inner image stroke needs a clip-path + alpha ring mask: {svg}"
        );
        insta::assert_snapshot!(svg);
    }

    /// A rectangle with a *dotted outer* image-filled stroke: the texture is
    /// confined to the dot ring (alpha mask) and to the shape exterior
    /// (inverse-of-shape luminance mask, since SVG `<clipPath>` cannot subtract).
    #[test]
    fn exports_a_shape_with_a_dotted_outer_image_stroke() {
        let mut pool = ShapesPool::new();
        let id = uid(1);
        let image_id = uid(9);
        add_image_stroked_rect(
            &mut pool,
            id,
            image_id,
            (0.0, 0.0, 100.0, 80.0),
            skia::Color::from_rgb(0xCC, 0xCC, 0xCC),
            StrokeKind::Outer,
            StrokeStyle::Dotted,
            16.0,
        );

        let mut images = FakeImages {
            id: image_id,
            image: tiny_image(),
        };
        let svg = render_with_images(&pool, id, &mut images);
        assert!(svg.contains("<image"), "stroke image must be emitted: {svg}");
        // Dotted outer: exterior restriction + dot-ring both via <mask>.
        assert!(
            svg.contains("mask=\"url(#") && svg.contains("mask-type=\"alpha\""),
            "dotted-outer image stroke needs restriction + alpha ring masks: {svg}"
        );
        insta::assert_snapshot!(svg);
    }

    /// A rectangle with a solid, image-filled *inner* stroke: like the center
    /// case, the texture is confined by an alpha `<mask>` of the stroke
    /// silhouette, but the silhouette sits fully inside the shape geometry.
    #[test]
    fn exports_a_shape_with_a_solid_inner_image_stroke() {
        let mut pool = ShapesPool::new();
        let id = uid(1);
        let image_id = uid(9);
        add_image_stroked_rect(
            &mut pool,
            id,
            image_id,
            (0.0, 0.0, 100.0, 80.0),
            skia::Color::from_rgb(0xCC, 0xCC, 0xCC),
            StrokeKind::Inner,
            StrokeStyle::Solid,
            16.0,
        );

        let mut images = FakeImages {
            id: image_id,
            image: tiny_image(),
        };
        let svg = render_with_images(&pool, id, &mut images);
        assert!(svg.contains("<image"), "stroke image must be emitted: {svg}");
        assert!(
            svg.contains("mask-type=\"alpha\""),
            "inner stroke silhouette alpha mask must be present: {svg}"
        );
        insta::assert_snapshot!(svg);
    }

    /// A rectangle with a solid, image-filled *outer* stroke: the silhouette
    /// sits fully outside the shape geometry, again confined by an alpha
    /// `<mask>`.
    #[test]
    fn exports_a_shape_with_a_solid_outer_image_stroke() {
        let mut pool = ShapesPool::new();
        let id = uid(1);
        let image_id = uid(9);
        add_image_stroked_rect(
            &mut pool,
            id,
            image_id,
            (0.0, 0.0, 100.0, 80.0),
            skia::Color::from_rgb(0xCC, 0xCC, 0xCC),
            StrokeKind::Outer,
            StrokeStyle::Solid,
            16.0,
        );

        let mut images = FakeImages {
            id: image_id,
            image: tiny_image(),
        };
        let svg = render_with_images(&pool, id, &mut images);
        assert!(svg.contains("<image"), "stroke image must be emitted: {svg}");
        assert!(
            svg.contains("mask-type=\"alpha\""),
            "outer stroke silhouette alpha mask must be present: {svg}"
        );
        insta::assert_snapshot!(svg);
    }

    /// Adds a rectangle whose *fill* is an image (references `image_id` via an
    /// [`ImageFill`]).
    fn add_image_filled_rect(
        pool: &mut ShapesPool,
        id: Uuid,
        image_id: Uuid,
        (l, t, r, b): (f32, f32, f32, f32),
    ) {
        let shape = pool.add_shape(id);
        shape.set_parent(Uuid::nil());
        shape.set_shape_type(Type::Rect(Rect::default()));
        shape.set_selrect(l, t, r, b);
        shape.set_fills(vec![Fill::Image(ImageFill::new(image_id, 255, 2, 2, false))]);
    }

    /// A rectangle filled with an image: the texture is emitted as an `<image>`
    /// clipped to the shape geometry (no dropped `save_layer`).
    #[test]
    fn exports_a_shape_with_an_image_fill() {
        let mut pool = ShapesPool::new();
        let id = uid(1);
        let image_id = uid(9);
        add_image_filled_rect(&mut pool, id, image_id, (0.0, 0.0, 100.0, 80.0));

        let mut images = FakeImages {
            id: image_id,
            image: tiny_image(),
        };
        let svg = render_with_images(&pool, id, &mut images);
        assert!(svg.contains("<image"), "image fill must be emitted: {svg}");
        insta::assert_snapshot!(svg);
    }

    /// Adds a solid-filled closed triangular path with a single solid stroke of
    /// the given kind/width/color. Closed paths honor the stroke kind (open
    /// paths always render centered).
    fn add_stroked_path(
        pool: &mut ShapesPool,
        id: Uuid,
        fill: skia::Color,
        kind: StrokeKind,
        width: f32,
        stroke_color: skia::Color,
    ) {
        let shape = pool.add_shape(id);
        shape.set_parent(Uuid::nil());
        shape.set_shape_type(Type::Path(Path::new(vec![
            Segment::MoveTo((50.0, 0.0)),
            Segment::LineTo((100.0, 80.0)),
            Segment::LineTo((0.0, 80.0)),
            Segment::Close,
        ])));
        shape.set_selrect(0.0, 0.0, 100.0, 80.0);
        shape.set_fills(vec![Fill::Solid(SolidColor(fill))]);

        let mut stroke = match kind {
            StrokeKind::Inner => {
                Stroke::new_inner_stroke(width, StrokeStyle::Solid, None, None, None, None)
            }
            StrokeKind::Center => {
                Stroke::new_center_stroke(width, StrokeStyle::Solid, None, None, None, None)
            }
            StrokeKind::Outer => {
                Stroke::new_outer_stroke(width, StrokeStyle::Solid, None, None, None, None)
            }
        };
        stroke.fill = Fill::Solid(SolidColor(stroke_color));
        shape.add_stroke(stroke);
    }

    /// A simple closed path (triangle) with a solid *inner* stroke.
    #[test]
    fn exports_a_path_with_a_solid_inner_stroke() {
        let mut pool = ShapesPool::new();
        let id = uid(1);
        add_stroked_path(
            &mut pool,
            id,
            skia::Color::from_rgb(0xCC, 0xCC, 0xCC),
            StrokeKind::Inner,
            10.0,
            skia::Color::from_rgb(0xFF, 0x00, 0x00),
        );

        insta::assert_snapshot!(render(&pool, id));
    }

    /// A simple closed path (triangle) with a solid *center* stroke.
    #[test]
    fn exports_a_path_with_a_solid_center_stroke() {
        let mut pool = ShapesPool::new();
        let id = uid(1);
        add_stroked_path(
            &mut pool,
            id,
            skia::Color::from_rgb(0xCC, 0xCC, 0xCC),
            StrokeKind::Center,
            10.0,
            skia::Color::from_rgb(0x00, 0x88, 0x00),
        );

        insta::assert_snapshot!(render(&pool, id));
    }

    /// A simple closed path (triangle) with a solid *outer* stroke.
    #[test]
    fn exports_a_path_with_a_solid_outer_stroke() {
        let mut pool = ShapesPool::new();
        let id = uid(1);
        add_stroked_path(
            &mut pool,
            id,
            skia::Color::from_rgb(0xCC, 0xCC, 0xCC),
            StrokeKind::Outer,
            10.0,
            skia::Color::from_rgb(0x00, 0x00, 0xFF),
        );

        let svg = render(&pool, id);
        // The outer stroke is composed as a nested `<g>` masked to the shape's
        // exterior (double-width stroke = 2 * 10). It must not vanish.
        assert!(
            svg.contains("<mask") && svg.contains("mask=\"url(#smask"),
            "outer path stroke must be composed via an exterior mask: {svg}"
        );
        assert!(
            svg.contains("stroke-width=\"20\""),
            "outer path stroke must emit the double-width stroke: {svg}"
        );
        insta::assert_snapshot!(svg);
    }

    /// A simple rectangle with a *dotted inner* stroke. The GPU/PDF path clips a
    /// boundary ring of dots to the shape interior inside a `save_layer` that
    /// `SkSVGDevice` drops; the compositor re-emits it as a `<g clip-path>`.
    #[test]
    fn exports_a_shape_with_a_dotted_inner_stroke() {
        let mut pool = ShapesPool::new();
        let id = uid(1);
        add_styled_stroked_rect(
            &mut pool,
            id,
            (0.0, 0.0, 100.0, 80.0),
            skia::Color::from_rgb(0xCC, 0xCC, 0xCC),
            StrokeKind::Inner,
            StrokeStyle::Dotted,
            10.0,
            skia::Color::from_rgb(0xFF, 0x00, 0x00),
        );

        let svg = render(&pool, id);
        // The dotted inner stroke must be composed as a nested `<g>` clipped to
        // the shape interior (it must not vanish with the dropped `save_layer`).
        assert!(
            svg.contains("<clipPath") && svg.contains("clip-path=\"url(#dclip"),
            "dotted inner stroke must be composed via an interior clip: {svg}"
        );
        // Dots are emitted as filled paths (not a dashed stroke).
        assert!(
            svg.contains("fill=\"red\""),
            "dotted inner stroke color must be present: {svg}"
        );
        insta::assert_snapshot!(svg);
    }

    /// A simple rectangle with a *dotted center* stroke.
    #[test]
    fn exports_a_shape_with_a_dotted_center_stroke() {
        let mut pool = ShapesPool::new();
        let id = uid(1);
        add_styled_stroked_rect(
            &mut pool,
            id,
            (0.0, 0.0, 100.0, 80.0),
            skia::Color::from_rgb(0xCC, 0xCC, 0xCC),
            StrokeKind::Center,
            StrokeStyle::Dotted,
            10.0,
            skia::Color::from_rgb(0x00, 0x88, 0x00),
        );

        insta::assert_snapshot!(render(&pool, id));
    }

    /// A simple rectangle with a *dotted outer* stroke. The GPU/PDF path clips a
    /// boundary ring of dots to the shape exterior inside a `save_layer` that
    /// `SkSVGDevice` drops; the compositor re-emits it masked to the exterior
    /// (an inverse-of-shape luminance mask, since `<clipPath>` cannot subtract).
    #[test]
    fn exports_a_shape_with_a_dotted_outer_stroke() {
        let mut pool = ShapesPool::new();
        let id = uid(1);
        add_styled_stroked_rect(
            &mut pool,
            id,
            (0.0, 0.0, 100.0, 80.0),
            skia::Color::from_rgb(0xCC, 0xCC, 0xCC),
            StrokeKind::Outer,
            StrokeStyle::Dotted,
            10.0,
            skia::Color::from_rgb(0x00, 0x00, 0xFF),
        );

        let svg = render(&pool, id);
        // The dotted outer stroke must be composed as a nested `<g>` masked to
        // the shape exterior (it must not vanish with the dropped `save_layer`).
        assert!(
            svg.contains("<mask") && svg.contains("mask=\"url(#dmask"),
            "dotted outer stroke must be composed via an exterior mask: {svg}"
        );
        // Dots are emitted as filled paths (not a dashed stroke).
        assert!(
            svg.contains("fill=\"blue\""),
            "dotted outer stroke color must be present: {svg}"
        );
        insta::assert_snapshot!(svg);
    }

    /// Adds a single-line text shape ("HOLA"-style) using the embedded default
    /// font (registered under the nil-UUID family, weight 400), optionally with
    /// a single solid stroke of the given kind. Font shaping resolves against the
    /// export `FontStore` (installed for the render), so no global state or GPU
    /// is needed.
    fn add_text(
        pool: &mut ShapesPool,
        id: Uuid,
        (l, t, r, b): (f32, f32, f32, f32),
        text: &str,
        font_size: f32,
        fill: skia::Color,
        stroke: Option<(StrokeKind, f32, skia::Color)>,
    ) {
        let bounds = skia::Rect::from_ltrb(l, t, r, b);
        let mut content = TextContent::new(bounds, GrowType::Fixed);
        // `line_height` is a multiplier of the font size (Skia `set_height` with
        // height override), NOT an absolute pixel value.
        let line_height = 1.2;
        let span = TextSpan::new(
            text.to_string(),
            FontFamily::new(Uuid::nil(), 400, FontStyle::Normal),
            font_size,
            line_height,
            0.0,
            None,
            None,
            TextDirection::LTR,
            400,
            Uuid::nil(),
            vec![Fill::Solid(SolidColor(fill))],
        );
        content.add_paragraph(Paragraph::new(
            TextAlign::Left,
            TextDirection::LTR,
            None,
            None,
            line_height,
            0.0,
            vec![span],
        ));

        let shape = pool.add_shape(id);
        shape.set_parent(Uuid::nil());
        // Set the selrect *before* the text type: `set_selrect` on a text shape
        // eagerly relayouts (needing the font collection), which isn't available
        // until the export installs it. The render recomputes text layout from
        // the selrect anyway.
        shape.set_selrect(l, t, r, b);
        shape.set_shape_type(Type::Text(content));
        if let Some((kind, width, color)) = stroke {
            let mut stroke = match kind {
                StrokeKind::Inner => {
                    Stroke::new_inner_stroke(width, StrokeStyle::Solid, None, None, None, None)
                }
                StrokeKind::Center => {
                    Stroke::new_center_stroke(width, StrokeStyle::Solid, None, None, None, None)
                }
                StrokeKind::Outer => {
                    Stroke::new_outer_stroke(width, StrokeStyle::Solid, None, None, None, None)
                }
            };
            stroke.fill = Fill::Solid(SolidColor(color));
            shape.add_stroke(stroke);
        }
    }

    /// Simple text with a solid *inner* stroke. `SkSVGDevice` drops the
    /// glyph-fill mask the shared renderer uses, so the compositor re-emits the
    /// inner stroke clipped to the glyph silhouette (`<clipPath>`).
    #[test]
    fn exports_text_with_a_solid_inner_stroke() {
        let mut pool = ShapesPool::new();
        let id = uid(1);
        add_text(
            &mut pool,
            id,
            (0.0, 0.0, 560.0, 240.0),
            "HOLA",
            200.0, // large font + thin stroke so the fill shows inside the glyph stems
            skia::Color::from_rgb(0xE1, 0x7F, 0xDA), // pink fill
            Some((StrokeKind::Inner, 2.0, skia::Color::from_rgb(0x00, 0x00, 0xFF))), // blue stroke
        );

        let svg = render(&pool, id);
        assert!(svg.contains("<text"), "text glyphs must be present: {svg}");
        assert!(
            svg.contains("<clipPath") && svg.contains("clip-path=\"url(#tclip"),
            "inner text stroke must be composed via a glyph-silhouette clip: {svg}"
        );
        insta::assert_snapshot!(svg);
    }

    /// Simple text with a solid *center* stroke: drawn inline by the shared
    /// renderer (a stroked `<text>` straddling the glyph outline).
    #[test]
    fn exports_text_with_a_solid_center_stroke() {
        let mut pool = ShapesPool::new();
        let id = uid(1);
        add_text(
            &mut pool,
            id,
            (0.0, 0.0, 560.0, 240.0),
            "HOLA",
            200.0, // large font + thin stroke so the fill shows inside the glyph stems
            skia::Color::from_rgb(0xE1, 0x7F, 0xDA), // pink fill
            Some((StrokeKind::Center, 2.0, skia::Color::from_rgb(0x00, 0x00, 0xFF))), // blue stroke
        );

        let svg = render(&pool, id);
        assert!(svg.contains("<text"), "text glyphs must be present: {svg}");
        assert!(
            svg.contains("stroke-width=\"2\"") && svg.contains("stroke=\"blue\""),
            "center text stroke must be present: {svg}"
        );
        insta::assert_snapshot!(svg);
    }

    /// Simple text with a solid *outer* stroke.
    #[test]
    fn exports_text_with_a_solid_outer_stroke() {
        let mut pool = ShapesPool::new();
        let id = uid(1);
        add_text(
            &mut pool,
            id,
            (0.0, 0.0, 560.0, 240.0),
            "HOLA",
            200.0,
            skia::Color::from_rgb(0xE1, 0x7F, 0xDA), // pink fill
            Some((StrokeKind::Outer, 6.0, skia::Color::from_rgb(0x00, 0x00, 0xFF))), // blue stroke
        );

        let svg = render(&pool, id);
        assert!(svg.contains("<text"), "text glyphs must be present: {svg}");
        // The outer stroke must be masked to the glyph exterior (keeping only its
        // outer half), not painted at full double width over the fill.
        assert!(
            svg.contains("<mask") && svg.contains("mask=\"url(#tmask"),
            "outer text stroke must be composed via an exterior mask: {svg}"
        );
        assert!(
            svg.contains("stroke=\"blue\""),
            "outer text stroke color must be present: {svg}"
        );
        insta::assert_snapshot!(svg);
    }

    /// A filled shape with an *inner* shadow. The GPU/PDF path composes it inside
    /// a `save_layer` (+ image filter) that `SkSVGDevice` drops, so the SVG
    /// compositor must re-emit it as a native `<filter>` that darkens the shape
    /// interior (blur+offset occluder, `operator="out"`, then clipped back in).
    #[test]
    fn exports_a_shape_with_an_inner_shadow() {
        let mut pool = ShapesPool::new();
        let id = uid(1);
        add_solid_rect(
            &mut pool,
            id,
            Uuid::nil(),
            (0.0, 0.0, 120.0, 90.0),
            skia::Color::from_rgb(0xB1, 0xB2, 0xB5),
        );
        {
            let shape = pool.get_mut(&id).unwrap();
            shape.add_shadow(Shadow::new(
                skia::Color::from_argb(0x66, 0x00, 0x00, 0x00), // #000000 @ ~0.4
                12.0,         // blur
                0.0,          // spread
                (8.0, 6.0),   // offset (x, y)
                ShadowStyle::Inner,
                false, // hidden
            ));
        }

        let svg = render(&pool, id);
        // The inner shadow must survive as a native SVG filter, not a dropped
        // `save_layer`, and must be clipped to the shape interior.
        assert!(
            svg.contains("filter=\"url(#inshadow"),
            "inner shadow must be emitted as an SVG filter: {svg}"
        );
        assert!(
            svg.contains("operator=\"out\"") && svg.contains("flood-color=\"#000000\""),
            "inner shadow must build a tinted interior band: {svg}"
        );
        insta::assert_snapshot!(svg);
    }

    /// Text with an *inner* shadow: the shared renderer paints it via an
    /// image-filter overlay dropped by `SkSVGDevice`, so the compositor re-emits
    /// it over the glyph silhouette.
    #[test]
    fn exports_text_with_an_inner_shadow() {
        let mut pool = ShapesPool::new();
        let id = uid(1);
        add_text(
            &mut pool,
            id,
            (0.0, 0.0, 560.0, 240.0),
            "HOLA",
            200.0,
            skia::Color::from_rgb(0xEE, 0x19, 0x19), // red fill
            None,
        );
        {
            let shape = pool.get_mut(&id).unwrap();
            shape.add_shadow(Shadow::new(
                skia::Color::from_argb(0x99, 0x00, 0x00, 0x00),
                6.0,
                0.0,
                (4.0, 4.0),
                ShadowStyle::Inner,
                false,
            ));
        }

        let svg = render(&pool, id);
        assert!(svg.contains("<text"), "text glyphs must be present: {svg}");
        assert!(
            svg.contains("filter=\"url(#inshadow"),
            "text inner shadow must be emitted as an SVG filter: {svg}"
        );
        insta::assert_snapshot!(svg);
    }
}
