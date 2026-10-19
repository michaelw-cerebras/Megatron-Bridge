#!/usr/bin/env python3
"""Architecture diagrams for the model-surgery variants.

One diagram per variant, all drawn from the same vocabulary so they can be put
side by side in a deck:

* every sublayer is drawn explicitly -- norm, attention, and MLP/MoE each get
  their own labelled box, so the reader never has to assume where a norm sits;
* the residual stream is always the grey line down the middle;
* the governing equation is always typeset in the same place at the bottom.

Text can never overflow its box: ``fit_text`` measures the rendered extent and
shrinks the font until it fits. Hand-tuning font sizes per label is what made
the previous version drift.

Each diagram is a zero-argument function that returns a Figure, so in a
notebook a cell is just the call::

    from draw_surgery_arch import *
    v1_1()

Figures are built off pyplot deliberately: a pyplot figure would be shown once
by the inline backend and again by the returned value, giving a duplicate.

To write files::

    v1_1().savefig("v1_1.png", dpi=300)
    save_all()                      # every diagram -> ./exports
"""

from __future__ import annotations

import functools
import inspect
from pathlib import Path

import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from matplotlib.transforms import Bbox


# One hue per role, held fixed across every diagram so a reader who learns the
# vocabulary on one slide keeps it on the next.
COLORS = {
    "ink": "#24364A",
    "muted": "#7B8CA1",
    "border": "#A2A9B1",
    # io nodes sit on a near-white panel, so they need a darker edge than the
    # panel outlines do -- "border" at lw 1.1 all but disappeared.
    "io_edge": "#8FA3B8",
    "panel": "#FBFCFE",
    "norm": "#EEF2F7",
    "norm_edge": "#8C9DAE",
    "attention": "#E1EDFF",
    "attention_edge": "#5E9BEE",
    "mlp": "#F0E6FF",
    "mlp_edge": "#A47BE6",
    "moe": "#FFF0E3",
    "moe_edge": "#E08B3C",
    "dense": "#EDECFF",
    "dense_edge": "#7E76C8",
    "residual": "#E3F6EC",
    "residual_edge": "#4FB184",
    "correction": "#FFF1D5",
    "correction_edge": "#CA9039",
    "gate": "#FFE6EE",
    "gate_edge": "#DD6E96",
    "emphasis": "#FFFFFF",
    "emphasis_edge": "#24364A",
    "unselected": "#FFFFFF",
    "unselected_edge": "#99A0A8",
    "decoder": "#EDF1F6",
    "decoder_edge": "#6B7F96",
    "frozen": "#F2F4F7",
    "frozen_edge": "#939EAB",
}

FIG_W, FIG_H = 16.0, 9.0
TITLE_Y, SUBTITLE_Y = 8.48, 8.06
FORMULA_Y, NOTE_Y = 0.74, 0.22

# The drawing area nearly fills the canvas, so point sizes have to grow with it
# or the type would shrink relative to the boxes. Both scale together, which
# leaves fit_text's measurements -- and therefore every layout -- unchanged.
_FONT_SCALE = [1.23]

# Titles, legends and equations are off: these render as bare diagrams for a
# deck that supplies its own captions. Flip to True to get them back.
SHOW_CAPTIONS = False


def setup_style() -> None:
    """Apply shared rcParams: editable text in SVG/PDF, no axis furniture."""
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "stix",
            "axes.linewidth": 0,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def canvas() -> tuple[Figure, "mpl.axes.Axes"]:
    """Create the shared 16x9 drawing surface in data units.

    Built without pyplot so the figure is not added to pyplot's registry; in a
    notebook that is what keeps the inline backend from drawing it twice.
    """
    fig = Figure(figsize=(FIG_W, FIG_H))
    FigureCanvasAgg(fig)  # fit_text needs a renderer to measure against
    # add_subplot would inset the drawing area by matplotlib's default margins
    # (12.5% left, 10% right, 12% top, 11% bottom), shrinking every diagram to
    # roughly 60% of the canvas area. These diagrams carry their own margins in
    # data units, so the axes takes essentially the whole figure.
    ax = fig.add_axes([0.02, 0.025, 0.96, 0.95])
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")
    return fig, ax


DIAGRAMS = {}  # name -> zero-argument render function

FIT_MARGIN = 0.30


def _content_bbox(ax):
    """Union of every drawn artist, in data coordinates."""
    renderer = _renderer(ax)
    skip = {ax.patch, ax.xaxis, ax.yaxis, *ax.spines.values()}
    boxes = []
    for child in ax.get_children():
        if child in skip:
            continue
        try:
            box = child.get_window_extent(renderer)
        except TypeError:
            box = child.get_window_extent()
        except Exception:  # pragma: no cover - artist without an extent
            continue
        if box.width > 0 and box.height > 0:
            boxes.append(box)
    if not boxes:
        return None
    return Bbox(ax.transData.inverted().transform(Bbox.union(boxes).get_points()))


def _fit_axes(ax):
    """Frame the drawn content, padded and widened to the canvas aspect.

    Returns:
        How much the view zoomed in relative to the nominal 16-unit width, so
        the caller can scale type by the same factor.
    """
    box = _content_bbox(ax)
    if box is None:
        return 1.0
    width, height = box.width + 2 * FIT_MARGIN, box.height + 2 * FIT_MARGIN
    aspect = FIG_W / FIG_H
    if width / height < aspect:
        width = height * aspect
    else:
        height = width / aspect
    mid_x, mid_y = box.x0 + box.width / 2, box.y0 + box.height / 2
    ax.set_xlim(mid_x - width / 2, mid_x + width / 2)
    ax.set_ylim(mid_y - height / 2, mid_y + height / 2)
    return FIG_W / width


def diagram(draw):
    """Turn a ``draw(ax)`` body into a zero-argument entry point.

    ``v1_1()`` builds its own figure and returns it, which is all a
    notebook cell needs. Passing ``ax`` draws into an existing axes instead,
    which is how several diagrams can share one figure.
    """

    @functools.wraps(draw)
    def render(ax=None):
        if ax is not None:
            draw(ax)
            return ax.figure
        fig, ax = canvas()
        draw(ax)
        zoom = _fit_axes(ax)
        if zoom > 1.02:
            # Redraw with type scaled by the same factor, so the diagram grows
            # as a whole instead of boxes outgrowing their labels.
            previous = _FONT_SCALE[0]
            _FONT_SCALE[0] = previous * zoom
            try:
                fig, ax = canvas()
                draw(ax)
                _fit_axes(ax)
            finally:
                _FONT_SCALE[0] = previous
        return fig

    # functools.wraps sets __wrapped__, which inspect.signature follows -- without
    # this a notebook tooltip would advertise the inner draw(ax) and imply ax is
    # required. Advertise what render actually accepts.
    render.__signature__ = inspect.Signature(
        [inspect.Parameter("ax", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=None)]
    )
    DIAGRAMS[draw.__name__] = render
    return render


# --------------------------------------------------------------------------
# text fitting
# --------------------------------------------------------------------------


def _renderer(ax):
    fig = ax.figure
    try:
        return fig.canvas.get_renderer()
    except AttributeError:  # pragma: no cover - non-Agg backends
        fig.canvas.draw()
        return fig.canvas.get_renderer()


def text_size(ax, artist) -> tuple[float, float]:
    """Return a text artist's rendered size in data units.

    Args:
        ax: The axes the text lives on.
        artist: The text artist to measure.

    Returns:
        Width and height in data coordinates.
    """
    extent = artist.get_window_extent(renderer=_renderer(ax))
    box = extent.transformed(ax.transData.inverted())
    return box.width, box.height


def fit_text(ax, cx, cy, s, max_w, max_h, *, fontsize=13.0, min_fontsize=5.5, halo=None, **kwargs):
    """Draw centered text shrunk until it fits inside ``max_w`` x ``max_h``.

    Measuring beats guessing: label lengths change whenever an equation is
    edited, and a font size that fit yesterday silently overflows today.

    Args:
        ax: Target axes.
        cx: Center x in data units.
        cy: Center y in data units.
        s: The string, which may contain mathtext and newlines.
        max_w: Maximum width in data units.
        max_h: Maximum height in data units.
        fontsize: Starting font size.
        min_fontsize: Floor; below this the text is left slightly tight rather
            than rendered unreadably small.
        halo: Optional background colour painted behind the glyphs, so a label
            placed on a connector masks the line instead of being struck
            through by it. Pass the colour of whatever it sits on.
        **kwargs: Forwarded to ``Axes.text``.

    Returns:
        The text artist.
    """
    fontsize *= _FONT_SCALE[0]
    min_fontsize *= _FONT_SCALE[0]
    kwargs.setdefault("ha", "center")
    kwargs.setdefault("va", "center")
    if halo is not None:
        kwargs.setdefault("bbox", dict(boxstyle="round,pad=0.22", fc=halo, ec="none"))
    kwargs.setdefault("color", COLORS["ink"])
    kwargs.setdefault("linespacing", 1.35)
    artist = ax.text(cx, cy, s, fontsize=fontsize, **kwargs)
    size = fontsize
    for _ in range(60):
        width, height = text_size(ax, artist)
        if width <= max_w and height <= max_h:
            break
        size *= 0.94
        if size < min_fontsize:
            artist.set_fontsize(min_fontsize)
            break
        artist.set_fontsize(size)
    return artist


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------


def frame(ax, cx, cy, w, h, *, fc="white", ec=None, lw=1.3, radius=0.14, zorder=2):
    """Draw a rounded rectangle centered on ``(cx, cy)`` with exact size w x h."""
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec or COLORS["border"],
        linewidth=lw,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def chip(ax, cx, cy, w, h, label, kind, *, fontsize=12.0, weight="bold", pad=0.12):
    """Draw one sublayer box with its label fitted inside.

    Args:
        ax: Target axes.
        cx: Center x.
        cy: Center y.
        w: Box width.
        h: Box height.
        label: Text, possibly multi-line mathtext.
        kind: Colour role key, e.g. ``"attention"`` or ``"moe"``.
        fontsize: Starting font size before fitting.
        weight: Font weight.
        pad: Inner margin kept clear of the border.
    """
    frame(ax, cx, cy, w, h, fc=COLORS[kind], ec=COLORS[f"{kind}_edge"], lw=1.4)
    fit_text(ax, cx, cy, label, w - 2 * pad, h - 2 * pad, fontsize=fontsize, weight=weight, zorder=4)


def io_node(ax, cx, cy, label, *, w=0.66, h=0.34, fontsize=13.0):
    """Draw a small white node for an input, output or intermediate state."""
    frame(ax, cx, cy, w, h, fc="white", ec=COLORS["io_edge"], lw=1.4, radius=0.1)
    fit_text(ax, cx, cy, label, w - 0.14, h - 0.10, fontsize=fontsize, zorder=4)


def state_node(ax, cx, cy, label, *, w=0.66, h=0.34, fontsize=13.0):
    """Draw a residual-state node, tinted to distinguish it from a sublayer."""
    frame(ax, cx, cy, w, h, fc=COLORS["residual"], ec=COLORS["residual_edge"], lw=1.2, radius=0.1)
    fit_text(ax, cx, cy, label, w - 0.14, h - 0.10, fontsize=fontsize, zorder=4)


def plus(ax, cx, cy, r=0.17):
    """Draw a residual addition node."""
    ax.add_patch(
        Circle((cx, cy), r, facecolor=COLORS["residual"], edgecolor=COLORS["residual_edge"], lw=1.3, zorder=3)
    )
    ax.text(cx, cy - 0.012, "+", ha="center", va="center", fontsize=15, color=COLORS["ink"], zorder=4)
    return r


def arrow(ax, p1, p2, *, color=None, lw=1.6, scale=11, zorder=1, style="arc3"):
    """Draw a straight arrow between two points."""
    patch = FancyArrowPatch(
        p1,
        p2,
        arrowstyle="-|>",
        mutation_scale=scale,
        linewidth=lw,
        color=color or COLORS["ink"],
        connectionstyle=style,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def path(ax, points, *, color=None, lw=1.5, zorder=1, head=True, scale=11, linestyle="-"):
    """Draw an orthogonal polyline, optionally with an arrowhead at the end.

    Args:
        ax: Target axes.
        points: Sequence of ``(x, y)`` vertices.
        color: Line colour.
        lw: Line width.
        zorder: Draw order.
        head: Whether to place an arrowhead on the final segment.
        scale: Arrowhead size.
        linestyle: Matplotlib line style; dashed marks a line that crosses
            another element without connecting to it.
    """
    colour = color or COLORS["ink"]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.plot(xs, ys, color=colour, lw=lw, zorder=zorder, linestyle=linestyle,
            solid_capstyle="round", solid_joinstyle="round")
    if head and len(points) >= 2:
        # Draw the head over a short stub, not the whole final segment: on a
        # two-point path that segment is the entire line, and the solid arrow
        # patch would paint over a dashed linestyle completely.
        tail, tip = points[-2], points[-1]
        dx, dy = tip[0] - tail[0], tip[1] - tail[1]
        length = (dx * dx + dy * dy) ** 0.5
        if length > 0.3:
            tail = (tip[0] - dx * 0.3 / length, tip[1] - dy * 0.3 / length)
        arrow(ax, tail, tip, color=colour, lw=lw, scale=scale, zorder=zorder)


def title(ax, text, subtitle=None, *, fontsize=23.0):
    """Draw the diagram title, unless captions are switched off."""
    if not SHOW_CAPTIONS:
        return
    fit_text(ax, FIG_W / 2, TITLE_Y, text, FIG_W - 1.0, 0.56, fontsize=fontsize, weight="bold", color="black")
    if subtitle:
        fit_text(ax, FIG_W / 2, SUBTITLE_Y, subtitle, FIG_W - 1.6, 0.34, fontsize=13.0, color=COLORS["muted"])


def formula(ax, text, *, note=None, fontsize=15.0):
    """Typeset the governing equation in the fixed bottom slot."""
    if not SHOW_CAPTIONS:
        return
    fit_text(ax, FIG_W / 2, FORMULA_Y, text, FIG_W - 1.4, 0.62, fontsize=fontsize, family="serif")
    if note:
        fit_text(ax, FIG_W / 2, NOTE_Y, note, FIG_W - 1.6, 0.34, fontsize=11.5, color=COLORS["muted"], family="serif")


def panel(ax, cx, cy, w, h, heading, subheading=None, *, heading_fs=17.0):
    """Draw a titled background panel and return its inner drawing bounds.

    Args:
        ax: Target axes.
        cx: Panel center x.
        cy: Panel center y.
        w: Panel width.
        h: Panel height.
        heading: Panel title.
        subheading: Optional second line.
        heading_fs: Starting font size for the heading.

    Returns:
        ``(top, bottom)`` y bounds available for content.
    """
    frame(ax, cx, cy, w, h, fc=COLORS["panel"], ec=COLORS["border"], lw=1.5, radius=0.2, zorder=0)
    top = cy + h / 2
    fit_text(ax, cx, top - 0.34, heading, w - 0.5, 0.36, fontsize=heading_fs, family="serif")
    inner_top = top - 0.72
    if subheading:
        fit_text(ax, cx, top - 0.66, subheading, w - 0.6, 0.28, fontsize=11.0, color=COLORS["muted"], family="serif")
        inner_top = top - 1.00
    return inner_top, cy - h / 2 + 0.22


def legend(ax, entries, *, cy=7.62):
    """Draw the colour key, unless captions are switched off."""
    if not SHOW_CAPTIONS:
        return
    gap = 2.05
    start = FIG_W / 2 - gap * (len(entries) - 1) / 2
    for index, (kind, label) in enumerate(entries):
        cx = start + index * gap
        frame(ax, cx - 0.66, cy, 0.30, 0.20, fc=COLORS[kind], ec=COLORS[f"{kind}_edge"], lw=1.2, radius=0.06)
        fit_text(ax, cx + 0.22, cy, label, 1.02, 0.26, fontsize=10.5, color=COLORS["muted"], ha="center")


# --------------------------------------------------------------------------
# composite building blocks
# --------------------------------------------------------------------------

CHIP_W, CHIP_H = 1.72, 0.52
GAP = 0.42


def sublayer(ax, cx, cy, norm_label, op_label, op_kind, *, w=CHIP_W, h=CHIP_H,
             norm_fs=11.5, op_fs=11.5, frozen=False):
    """Draw a norm box feeding an operator box, the unit every variant reuses.

    Args:
        ax: Target axes.
        cx: Shared center x.
        cy: Center y of the pair.
        norm_label: Label for the norm chip.
        op_label: Label for the attention / MLP / MoE chip.
        op_kind: Colour role of the operator chip.
        w: Width of both chips.
        h: Height of the operator chip.
        norm_fs: Starting font size for the norm label.
        op_fs: Starting font size for the operator label.
        frozen: Draw the operator greyed, for frozen-weight variants.

    Returns:
        ``(top_y, bottom_y)`` of the drawn pair.
    """
    # 0.44 with pad 0.08 leaves exactly the usable height of an operator
    # chip (0.52 with pad 0.12), so one font size fits both.
    norm_h = 0.44 * (h / CHIP_H)
    top = cy + (norm_h + h) / 2 + 0.10
    chip(ax, cx, top - norm_h / 2, w, norm_h, norm_label, "norm", fontsize=norm_fs, weight="normal", pad=0.08)
    op_cy = top - norm_h - 0.14 - h / 2
    chip(ax, cx, op_cy, w, h, op_label, "frozen" if frozen else op_kind, fontsize=op_fs)
    arrow(ax, (cx, top - norm_h), (cx, op_cy + h / 2), scale=9, lw=1.4)
    return top, op_cy - h / 2


def sequential_decoder(ax, cx, top, *, attn_kind="attention", mlp_kind="moe", mlp_label="MoE",
                       frozen=False, w=CHIP_W, norm_fs=11.5, op_fs=11.5, state_fs=11.5, tag="",
                       mirror=False):
    """Draw the original sequential block: norm->attn->add->norm->mlp->add.

    Returns:
        The y coordinate of the block's output.
    """
    # mirror puts the skip rail on the right, so a pair of decoders can be
    # drawn as each other's reflection
    side = 1.0 if mirror else -1.0
    rail = cx + side * (w / 2 + 0.56)
    norm_a = f"RMSNorm$^a_{tag}$" if tag else "RMSNorm$_1$"
    norm_m = f"RMSNorm$^m_{tag}$" if tag else "RMSNorm$_2$"
    attn_name = f"Attention  $a_{tag}$" if tag else "Attention  $a$"
    mlp_name = f"{mlp_label}  $m_{tag}$" if tag else f"{mlp_label}  $m$"
    state_text = f"$z_{tag}=x+a_{tag}(x)$" if tag else "$z=x+a(x)$"
    norm_top, attn_bottom = sublayer(ax, cx, top - 1.00, norm_a, attn_name, attn_kind,
                                     w=w, norm_fs=norm_fs, op_fs=op_fs, frozen=frozen)
    arrow(ax, (cx, top - 0.17), (cx, norm_top), scale=9, lw=1.4)
    add1 = attn_bottom - 0.42
    plus(ax, cx, add1)
    arrow(ax, (cx, attn_bottom), (cx, add1 + 0.17), scale=9, lw=1.4)
    path(ax, [(cx, top - 0.17), (rail, top - 0.17), (rail, add1), (cx + side * 0.17, add1)],
         color=COLORS["muted"], lw=1.4, scale=9)

    state_y = add1 - 0.46
    # height scales with the block too, or chip padding leaves no room for the
    # label and fit_text shrinks it to the floor
    state_node(ax, cx, state_y, state_text, w=w * 0.91, h=0.32 * (w / CHIP_W), fontsize=state_fs)
    arrow(ax, (cx, add1 - 0.17), (cx, state_y + 0.16), scale=9, lw=1.4)

    _, mlp_bottom = sublayer(ax, cx, state_y - 1.02, norm_m, mlp_name, mlp_kind,
                             w=w, norm_fs=norm_fs, op_fs=op_fs, frozen=frozen)
    arrow(ax, (cx, state_y - 0.16), (cx, state_y - 0.44), scale=9, lw=1.4)
    add2 = mlp_bottom - 0.42
    plus(ax, cx, add2)
    arrow(ax, (cx, mlp_bottom), (cx, add2 + 0.17), scale=9, lw=1.4)
    path(ax, [(cx + side * w * 0.455, state_y), (rail, state_y), (rail, add2), (cx + side * 0.17, add2)],
         color=COLORS["muted"], lw=1.4, scale=9)
    return add2 - 0.17


def parallel_branches(ax, cx, top, branches, *, spread=2.62, add_y=None,
                      residual_from=None, stem=0.52, w=CHIP_W, norm_fs=11.5, op_fs=11.5):
    """Draw N branches reading the same input and summing into one node.

    Args:
        ax: Target axes.
        cx: Center x of the fan.
        top: y of the shared input node.
        branches: ``(norm_label, op_label, op_kind, frozen)`` per branch.
        spread: Horizontal distance between branch centers.
        add_y: y of the summation node; derived if omitted.
        residual_from: ``"center"`` runs the skip straight down the middle,
            which is only unambiguous with an even number of branches; a float
            places it at that x; the default is just left of the leftmost
            branch.
        stem: Drop between the input node and the fan-out rail, so the branches
            do not appear glued to the input.
        w: Width of every branch's boxes. Size this to the longest label:
            fit_text shrinks per box, so a box that is too narrow for one label
            makes that branch's type smaller than its neighbours'.
        norm_fs: Starting font size for the norm labels.
        op_fs: Starting font size for the operator labels.

    Returns:
        ``(add_y, branch_xs)``.

    Raises:
        ValueError: If a centre residual is asked for with a branch on the
            centre line, where the two would be drawn on top of each other.
    """
    count = len(branches)
    start = cx - spread * (count - 1) / 2
    xs = [start + index * spread for index in range(count)]
    if residual_from == "center" and any(abs(bx - cx) < 1e-6 for bx in xs):
        raise ValueError("A centre residual needs the centre line free; use an even branch count")

    rail = top - 0.17 - stem
    bottoms = []
    for bx, (norm_label, op_label, op_kind, frozen) in zip(xs, branches):
        branch_top, bottom = sublayer(ax, bx, rail - 0.82, norm_label, op_label, op_kind,
                                      w=w, norm_fs=norm_fs, op_fs=op_fs, frozen=frozen)
        path(ax, [(cx, rail), (bx, rail), (bx, branch_top)], lw=1.4, scale=9)
        bottoms.append(bottom)

    join = add_y if add_y is not None else min(bottoms) - 0.82
    plus(ax, cx, join)
    for bx, bottom in zip(xs, bottoms):
        if abs(bx - cx) < 1e-6:
            arrow(ax, (bx, bottom), (bx, join + 0.17), scale=9, lw=1.4)
        else:
            side = -0.17 if bx < cx else 0.17
            path(ax, [(bx, bottom), (bx, join), (cx + side, join)], lw=1.4, scale=9)

    if residual_from == "center":
        # x runs straight down the middle; the branches tap off it and it
        # carries on into the sum, which is both symmetric and literally true.
        path(ax, [(cx, top - 0.17), (cx, join + 0.17)], color=COLORS["muted"], lw=1.4, scale=9)
    else:
        skip_x = residual_from if residual_from is not None else min(xs) - 1.22
        path(ax, [(cx, top - 0.17), (cx, rail), (skip_x, rail), (skip_x, join), (cx - 0.17, join)],
             color=COLORS["muted"], lw=1.4, scale=9)
    return join, xs


# --------------------------------------------------------------------------
# diagrams
# --------------------------------------------------------------------------


@diagram
def v0(ax):
    """v0: the unmodified decoder block.

    No legend and no prose: the boxes are labelled in full, so the only thing
    left to say is the equation.
    """
    title(ax, "v0 — baseline decoder")

    cx = 8.0
    top, _ = panel(ax, cx, 4.88, 7.40, 6.05, "One decoder block")
    io_node(ax, cx, top, "$x$", w=0.86, h=0.40, fontsize=15)
    out = sequential_decoder(ax, cx, top, w=2.90, norm_fs=15.0, op_fs=15.0, state_fs=13.0)
    io_node(ax, cx, out - 0.40, "$y$", w=0.86, h=0.40, fontsize=15)
    arrow(ax, (cx, out), (cx, out - 0.20), scale=9, lw=1.4)

    formula(ax, r"$z = x + a(x),\qquad y = z + m(z) = x + a(x) + m(x + a(x))$")


@diagram
def v1(ax):
    """v1: attention and MLP both read the block input."""
    cx = 8.0
    top, _ = panel(ax, cx, 4.70, 9.80, 4.90, "Parallel attention / MLP")
    io_node(ax, cx, top, "$x$", w=0.86, h=0.40, fontsize=15)
    join, _ = parallel_branches(
        ax,
        cx,
        top,
        [
            ("RMSNorm$_1$", "Attention  $a$", "attention", False),
            ("RMSNorm$_2$", "MoE  $m$", "moe", False),
        ],
        spread=3.60,
        residual_from="center",
    )
    io_node(ax, cx, join - 0.62, "$y$", w=0.86, h=0.40, fontsize=15)
    arrow(ax, (cx, join - 0.17), (cx, join - 0.42), scale=9, lw=1.4)


@diagram
def v2(ax):
    """v2: layer i's MLP runs beside layer i+1's attention."""
    PANEL_W = 13.6
    LEFT = 8.0 - PANEL_W / 2          # headings hang off the panel's left edge

    def heading(y, text, aside=None):
        fit_text(ax, LEFT + 0.45, y, text, 5.2, 0.32, fontsize=13.5, weight="bold",
                 ha="left", family="serif")
        if aside:
            fit_text(ax, 8.0 + PANEL_W / 2 - 0.45, y, aside, 5.6, 0.28, fontsize=10.5,
                     ha="right", color=COLORS["muted"], family="serif")

    def row(y, widths, gap):
        """Centre a row of boxes in the panel; returns their centres."""
        span = sum(widths) + gap * (len(widths) - 1)
        cursor = 8.0 - span / 2
        centres = []
        for width in widths:
            centres.append(cursor + width / 2)
            cursor += width + gap
        return centres

    # --- panel 1: the original chain -----------------------------------
    frame(ax, 8.0, 6.62, PANEL_W, 1.40, fc=COLORS["panel"], ec=COLORS["border"], lw=1.4, radius=0.18, zorder=0)
    heading(7.08, "Original sequential order")
    labels = ["$x$", "$A_1$", "$M_1$", "$A_2$", "$M_2$", "$A_3$", "$M_3$", "$A_4$", "$M_4$", "$y$"]
    widths = [0.60 if label in ("$x$", "$y$") else 1.04 for label in labels]
    centres = row(6.45, widths, 0.38)
    for index, (label, bx, width) in enumerate(zip(labels, centres, widths)):
        if label.startswith("$A"):
            chip(ax, bx, 6.45, width, 0.46, label, "attention", fontsize=12.5, weight="normal")
        elif label.startswith("$M"):
            chip(ax, bx, 6.45, width, 0.46, label, "moe", fontsize=12.5, weight="normal")
        else:
            io_node(ax, bx, 6.45, label, w=width, h=0.36, fontsize=12)
        if index:
            arrow(ax, (centres[index - 1] + widths[index - 1] / 2, 6.45), (bx - width / 2, 6.45),
                  scale=9, lw=1.3)

    # --- panel 2: the same operators, regrouped -------------------------
    frame(ax, 8.0, 4.46, PANEL_W, 2.28, fc=COLORS["panel"], ec=COLORS["border"], lw=1.4, radius=0.18, zorder=0)
    heading(5.30, "Rearranged order", "Boxes stacked together act on the same residual state")

    widths = [0.60, 1.16, 1.62, 1.62, 1.62, 1.16, 0.60]
    centres = row(4.26, widths, 0.62)
    io_node(ax, centres[0], 4.26, "$x$", w=0.60, h=0.36, fontsize=12)
    chip(ax, centres[1], 4.26, 1.16, 0.46, "$A_1$", "attention", fontsize=12.5, weight="normal")
    for slot, index in zip((2, 3, 4), (1, 2, 3)):
        bx = centres[slot]
        # Same problem the io nodes had: white fill on a near-white panel with
        # the palette's lightest edge. Use the io edge so the group reads.
        frame(ax, bx, 4.26, 1.62, 1.46, fc="white", ec=COLORS["io_edge"], lw=1.4, radius=0.12, zorder=0)
        fit_text(ax, bx, 4.82, "parallel", 1.3, 0.24, fontsize=10, color=COLORS["muted"], family="serif")
        chip(ax, bx, 4.42, 1.30, 0.44, f"$M_{index}$", "moe", fontsize=12, weight="normal")
        chip(ax, bx, 3.90, 1.30, 0.44, f"$A_{index + 1}$", "attention", fontsize=12, weight="normal")
    chip(ax, centres[5], 4.26, 1.16, 0.46, "$M_4$", "moe", fontsize=12.5, weight="normal")
    io_node(ax, centres[6], 4.26, "$y$", w=0.60, h=0.36, fontsize=12)
    for index in range(1, len(centres)):
        arrow(ax, (centres[index - 1] + widths[index - 1] / 2, 4.26),
              (centres[index] - widths[index] / 2, 4.26), scale=9, lw=1.3)

    # --- panel 3: one pair expanded, norms included ---------------------
    frame(ax, 8.0, 2.10, PANEL_W, 2.12, fc=COLORS["panel"], ec=COLORS["border"], lw=1.4, radius=0.18, zorder=0)
    heading(2.92, "Inside the first parallel pair",
            r"$M_2$ sees $A_2$'s update, so only the MLP one step back is missing")

    # This panel zooms inside the chain above, so its endpoints are named for
    # where they sit in that chain rather than reusing the block-level x and y.
    io_node(ax, 2.62, 2.02, "output of $A_1$", w=2.44, h=0.48, fontsize=12)
    rows = ((2.42, "$N^m_1$", "MLP  $m_1$", "moe"), (1.62, "$N^a_2$", "Attention  $a_2$", "attention"))
    for row_y, norm_label, op_label, kind in rows:
        chip(ax, 5.60, row_y, 1.40, 0.44, norm_label, "norm", fontsize=11.5, weight="normal", pad=0.08)
        chip(ax, 8.00, row_y, 2.20, 0.46, op_label, kind, fontsize=12)
        arrow(ax, (6.30, row_y), (6.90, row_y), scale=9, lw=1.3)
        path(ax, [(3.84, 2.02), (4.24, 2.02), (4.24, row_y), (4.90, row_y)], lw=1.3, scale=9)
        path(ax, [(9.10, row_y), (9.72, row_y), (9.72, 2.02), (9.95, 2.02)], lw=1.3, scale=9)
    plus(ax, 10.12, 2.02)
    path(ax, [(3.84, 2.02), (4.02, 2.02), (4.02, 1.22), (10.12, 1.22), (10.12, 1.85)],
         color=COLORS["muted"], lw=1.3, scale=9)
    io_node(ax, 12.26, 2.02, "input to the next pair", w=3.00, h=0.48, fontsize=12)
    arrow(ax, (10.29, 2.02), (10.76, 2.02), scale=9, lw=1.3)


@diagram
def v3(ax):
    """v3: two whole decoders applied to the same input."""
    cx, BX = 8.0, (4.30, 11.70)
    top, _ = panel(ax, cx, 4.40, 14.4, 7.70, "Parallel decoders")
    io_node(ax, cx, top, "$x$", w=0.86, h=0.40, fontsize=15)

    rail = top - 0.72
    outs = []
    for bx, tag, mirror in zip(BX, ("1", "2"), (False, True)):
        path(ax, [(cx, top - 0.20), (cx, rail), (bx, rail)], lw=1.4, head=False)
        out = sequential_decoder(ax, bx, rail + 0.17, w=2.50, tag=tag, mirror=mirror,
                                 norm_fs=12.0, op_fs=13.0, state_fs=12.0)
        outs.append(out)
        state_node(ax, bx, out - 0.44, f"$B_{tag}(x) = x + a_{tag} + m_{tag}$", w=3.50, h=0.46, fontsize=12.5)
        arrow(ax, (bx, out), (bx, out - 0.21), scale=9, lw=1.4)

    join = min(outs) - 1.22
    plus(ax, cx, join)
    for bx in BX:
        side = -0.17 if bx < cx else 0.17
        path(ax, [(bx, min(outs) - 0.67), (bx, join), (cx + side, join)], lw=1.4, scale=9)

    # The correction gets a box of its own: both decoders returned a copy of x,
    # and nothing else on the page says one of them has to come back out.
    mid = (top + join) / 2 + 0.25
    path(ax, [(cx, top - 0.20), (cx, mid + 0.30)], color=COLORS["muted"], lw=1.4, head=False)
    frame(ax, cx, mid, 1.70, 0.60, fc=COLORS["emphasis"], ec=COLORS["emphasis_edge"], lw=2.0, radius=0.12)
    fit_text(ax, cx, mid, "$-\\,x$", 1.40, 0.38, fontsize=17, weight="bold", family="serif")
    fit_text(ax, cx, mid - 0.56, "one copy of $x$ is returned twice", 3.20, 0.30,
             fontsize=11.5, color=COLORS["muted"], family="serif")
    arrow(ax, (cx, mid - 0.30), (cx, join + 0.17), color=COLORS["muted"], lw=1.4, scale=9)

    io_node(ax, cx, join - 0.66, "$y$", w=0.86, h=0.40, fontsize=15)
    arrow(ax, (cx, join - 0.17), (cx, join - 0.46), scale=9, lw=1.4)


@diagram
def v3_1(ax):
    """v3_1: the pair's two expert pools merge into one global top-2k selection."""
    cx, BX = 8.0, (4.60, 11.40)

    io_node(ax, cx, 8.62, "$x$", w=0.86, h=0.40, fontsize=15)
    rail = 7.94
    for bx, tag in zip(BX, ("1", "2")):
        top, bottom = sublayer(ax, bx, 7.20, f"RMSNorm$^a_{tag}$", f"Attention  $a_{tag}$",
                               "attention", w=2.70, norm_fs=12.5, op_fs=13.5)
        path(ax, [(cx, 8.42), (cx, rail), (bx, rail), (bx, top)], lw=1.4, scale=9)
        path(ax, [(bx, bottom), (bx, 6.28), (cx + (-0.17 if bx < cx else 0.17), 6.28)], lw=1.4, scale=9)
    plus(ax, cx, 6.28)
    state_node(ax, cx, 5.78, "$z = x + a_1(x) + a_2(x)$", w=3.80, h=0.46, fontsize=13)
    arrow(ax, (cx, 6.11), (cx, 6.01), scale=9, lw=1.4)

    # Each router reads its own norm's view of z -- see the note in the chat:
    # router 1 never sees RMSNorm_2's output. What is shared is the selection.
    rail2 = 5.30
    for bx, tag in zip(BX, ("1", "2")):
        path(ax, [(cx, 5.55), (cx, rail2), (bx, rail2), (bx, 5.16)], lw=1.4, scale=9)
        chip(ax, bx, 4.94, 2.70, 0.44, f"RMSNorm$^m_{tag}$", "norm", fontsize=12.5,
             weight="normal", pad=0.08)
        chip(ax, bx, 4.34, 2.70, 0.48, f"Router {tag}", "gate", fontsize=13)
        arrow(ax, (bx, 4.72), (bx, 4.58), scale=9, lw=1.4)

    for bx, tag in zip(BX, ("1", "2")):
        arrow(ax, (bx, 4.10), (bx, 3.34), scale=9, lw=1.4)
        label = "Experts $1..E$" if tag == "1" else "Experts $E{+}1..2E$"
        chip(ax, bx, 3.10, 2.70, 0.48, label, "moe", fontsize=13)

    combine = 2.42
    state_node(ax, cx, combine, "sum over $2k$ selected expert outputs",
               w=5.20, h=0.50, fontsize=13)
    for bx in BX:
        side = -0.17 if bx < cx else 0.17
        path(ax, [(bx, 2.86), (bx, combine), (cx + side * 15.3, combine)], lw=1.4, scale=9)

    join = 1.70
    plus(ax, cx, join)
    # The combine output steps aside so the residual can keep the centre line.
    path(ax, [(cx - 1.80, combine - 0.25), (cx - 1.80, join), (cx - 0.17, join)], lw=1.4, scale=9)
    # z crosses the combine box without joining it: dashed, and drawn on top so
    # the crossing reads as a crossing rather than a connection.
    path(ax, [(cx, 5.55), (cx, join + 0.17)], color=COLORS["muted"], lw=1.4, scale=9,
         linestyle=(0, (5, 4)), zorder=6)
    fit_text(ax, cx + 0.42, 3.90, "$z$", 0.5, 0.30, fontsize=12.5, color=COLORS["muted"],
             family="serif", halo="white")
    io_node(ax, cx, 1.16, "$y$", w=0.86, h=0.40, fontsize=15)
    arrow(ax, (cx, 1.53), (cx, 1.36), scale=9, lw=1.4)

    # --- separate panel: what one joint top-2k actually does --------------
    frame(ax, cx, -0.06, 12.4, 1.80, fc=COLORS["panel"], ec=COLORS["border"], lw=1.4, radius=0.18, zorder=0)
    fit_text(ax, 1.90, 0.64, "Each token sees the experts behind both routers and selects the top $k_1 + k_2$ of them",
             8.2, 0.30, fontsize=12.5, weight="bold", ha="left", family="serif")

    E, HALF_W, GAP = 8, 3.30, 0.46
    cell_w = HALF_W / E - 0.06
    half_left = {1: cx - GAP / 2 - HALF_W, 2: cx + GAP / 2}
    for index in (1, 2):
        fit_text(ax, half_left[index] + HALF_W / 2, 0.28, f"pool {index}", 2.4, 0.24,
                 fontsize=11, color=COLORS["muted"], family="serif")
    fit_text(ax, 12.10, 0.28, "$k_1 + k_2$", 1.5, 0.24, fontsize=11,
             color=COLORS["muted"], ha="left", family="serif")

    for name, row_y, masks, split in (
        ("token A", -0.06, {1: [1, 1, 0, 1, 0, 1, 1, 0], 2: [0, 1, 0, 0, 1, 1, 0, 0]}, "5 + 3"),
        ("token B", -0.42, {1: [0, 0, 1, 0, 0, 0, 0, 0], 2: [1, 1, 0, 1, 1, 1, 1, 1]}, "1 + 7"),
    ):
        fit_text(ax, 3.90, row_y, name, 1.80, 0.26, fontsize=11.5, color=COLORS["ink"],
                 ha="right", family="serif")
        for index in (1, 2):
            for slot, taken in enumerate(masks[index]):
                frame(ax, half_left[index] + (slot + 0.5) * (HALF_W / E), row_y, cell_w, 0.26,
                      fc=COLORS["moe"] if taken else COLORS["unselected"],
                      ec=COLORS["moe_edge"] if taken else COLORS["unselected_edge"],
                      lw=1.3 if taken else 1.2, radius=0.05)
        fit_text(ax, 12.10, row_y, split, 1.50, 0.26, fontsize=12.5, weight="bold",
                 color=COLORS["ink"], ha="left", family="serif")


@diagram
def v1_1(ax):
    """v1_1: a third, trainable correction branch beside the frozen originals."""
    cx = 8.0
    top, _ = panel(ax, cx, 4.70, 13.20, 4.70, "Parallel block + compensatory MLP")
    io_node(ax, cx, top, "$x$", w=0.86, h=0.40, fontsize=15)
    join, xs = parallel_branches(
        ax,
        cx,
        top,
        [
            ("RMSNorm$_1$", "Attention  FROZEN", "attention", True),
            ("RMSNorm$_2$", "MoE  FROZEN", "moe", True),
            ("compensatory norm", "SiLU MLP  TRAINABLE", "correction", False),
        ],
        spread=3.90,
        w=3.40,
        norm_fs=13.0,
        op_fs=13.0,
    )
    for bx, label in zip(xs, ("$a(x)$", "$m(x)$", "$c_\\theta(x)$")):
        fit_text(ax, bx, join + 0.56, label, 1.5, 0.32, fontsize=13,
                 color=COLORS["muted"], halo=COLORS["panel"])
    io_node(ax, cx, join - 0.62, "$y$", w=0.86, h=0.40, fontsize=15)
    arrow(ax, (cx, join - 0.17), (cx, join - 0.42), scale=9, lw=1.4)


def _cross_pair(ax, gate_label, gate_fs=11.0, gate_w=2.30):
    """Shared body for the two cross-injection diagrams; only the gate box differs.

    ``gate_label`` is called as ``gate_label(index, other)`` so each gate is
    labelled with its own subscripts -- g_1 reads branch 2 and vice versa.
    """
    CX, BX = 8.0, {1: 4.40, 2: 11.60}
    FEED = 6.95          # fan-out rail, set below x rather than flush with it
    io_node(ax, CX, 7.40, "$x$", w=0.80, h=0.38)

    # --- attention, one per branch -------------------------------------
    state_y, tap_y = {}, {}
    for index, bx in BX.items():
        top, bottom = sublayer(ax, bx, 6.15, f"RMSNorm$^a_{index}$",
                               f"Attention  $a_{index}$", "attention")
        path(ax, [(CX, 7.21), (CX, FEED), (bx, FEED), (bx, top)], lw=1.4, scale=9)
        state_y[index] = bottom - 0.46
        # The gate reads attentions[1 - index] -- the attention output itself,
        # before the residual add -- so the cross lane taps this segment, not
        # the h node below it.
        tap_y[index] = bottom - 0.14
        state_node(ax, bx, state_y[index], f"$h_{index} = x + a_{index}$",
                   w=2.10, h=0.36, fontsize=11)
        arrow(ax, (bx, bottom), (bx, state_y[index] + 0.18), scale=9, lw=1.4)

    # --- cross lanes ----------------------------------------------------
    # Each lane gets its own height so neither one crosses the far gate box.
    GATE_Y, GATE_H = 4.00, 0.54
    GATE_X = {1: CX - 1.75, 2: CX + 1.75}
    GATE_HALF = gate_w / 2
    LANE_Y = {1: 4.72, 2: 4.48}
    LABEL_X = {1: 9.30, 2: 6.60}
    for index, other in ((1, 2), (2, 1)):
        gx, lane = GATE_X[index], LANE_Y[index]
        src = BX[other]
        chip(ax, gx, GATE_Y, gate_w, GATE_H, f"$g_{index}$:  {gate_label(index, other)}",
             "gate", fontsize=gate_fs)
        # Step inboard first: dropping straight down from the tap would run
        # through the other branch's own h node.
        jog = src + (1.25 if src < CX else -1.25)
        path(ax, [(src, tap_y[other]), (jog, tap_y[other]), (jog, lane), (gx, lane),
                  (gx, GATE_Y + GATE_H / 2)],
             color=COLORS["gate_edge"], lw=1.4, scale=9)
        fit_text(ax, LABEL_X[index], lane + 0.15, f"$a_{other}$", 0.6, 0.24,
                 fontsize=10.5, color=COLORS["gate_edge"], ha="center", family="serif",
                 halo="white")

    # --- injection, MLP, per branch -------------------------------------
    for index, bx in BX.items():
        gx = GATE_X[index]
        plus(ax, bx, GATE_Y)
        path(ax, [(bx, state_y[index] - 0.18), (bx, GATE_Y + 0.17)], lw=1.4, scale=9)
        arrow(ax, (gx + (-GATE_HALF if gx > bx else GATE_HALF), GATE_Y),
              (bx + (0.17 if gx > bx else -0.17), GATE_Y),
              color=COLORS["gate_edge"], scale=9, lw=1.4)
        chip(ax, bx, 3.35, 2.10, 0.44, f"RMSNorm$^m_{index}$", "norm",
             fontsize=11.5, weight="normal", pad=0.08)
        arrow(ax, (bx, GATE_Y - 0.17), (bx, 3.52), scale=9, lw=1.4)
        # 0.52 (CHIP_H) so the usable height matches the norm and attention
        # boxes; at 0.46 this label alone was being shrunk.
        chip(ax, bx, 2.75, 2.10, 0.52, f"MoE  $m_{index}$", "moe", fontsize=11.5)
        arrow(ax, (bx, 3.18), (bx, 2.98), scale=9, lw=1.4)

    # --- output sum ------------------------------------------------------
    join = 2.15
    plus(ax, CX, join)
    for index, bx in BX.items():
        side = -0.17 if bx < CX else 0.17
        out = bx + (-1.30 if bx < CX else 1.30)
        path(ax, [(bx, 2.49), (bx, join), (CX + side, join)], lw=1.4, scale=9)
        # h_i also reaches the output directly -- the pair of residuals the
        # trailing "- x" has to cancel
        # Branch off the main line a little below the node, not flush with its
        # bottom edge -- tangent to the box, the turn reads as part of the box.
        branch_y = state_y[index] - 0.46
        path(ax, [(bx, branch_y), (out, branch_y), (out, join), (CX + side, join)],
             color=COLORS["muted"], lw=1.3, scale=9, head=False)
    path(ax, [(CX, 7.21), (CX, FEED), (1.55, FEED), (1.55, join), (CX - 0.17, join)],
         color=COLORS["muted"], lw=1.4, scale=9)
    fit_text(ax, 1.55, (FEED + join) / 2, "$-x$", 0.8, 0.30, fontsize=12,
             color=COLORS["muted"], family="serif", halo="white")
    io_node(ax, CX, 1.62, "$y$", w=0.72, h=0.36)
    arrow(ax, (CX, join - 0.17), (CX, 1.80), scale=9, lw=1.4)


@diagram
def v3_2(ax):
    """v3_2: scalar gated cross-injection between the two branches."""
    title(ax, "v3_2 — parallel decoders with scalar cross-injection", "Each MLP also sees a learnable scalar share of the other branch's attention")
    legend(ax, [("attention", "Attention"), ("gate", "Cross gate"), ("moe", "MoE"), ("residual", "Residual")])
    _cross_pair(ax, lambda i, o: rf"$\alpha_{i}\,a_{o}$")
    formula(
        ax,
        r"$o = h_1 + m_1(h_1 + \alpha_1 a_2) + h_2 + m_2(h_2 + \alpha_2 a_1) - x$",
        note=r"Both $\alpha$ start at 0, so training begins from exactly v3. Norms are omitted above; the injection enters $before$ each MLP norm.",
    )


@diagram
def v3_3(ax):
    """v3_3: the same wiring with an input-modulated nonlinear gate."""
    title(ax, "v3_3 — parallel decoders with nonlinear cross-injection", "How much of the other branch leaks in is modulated per token by the block input")
    legend(ax, [("attention", "Attention"), ("gate", "Cross gate"), ("moe", "MoE"), ("residual", "Residual")])
    _cross_pair(ax, lambda i, o: rf"$W_{i}[\,V_{i}a_{o}\odot\mathrm{{SiLU}}(U_{i}x)\,]$",
                gate_fs=12.0, gate_w=2.90)
    formula(
        ax,
        r"$g(x,h) = W[\,V(h-x)\odot\mathrm{SiLU}(Ux)\,],"
        r"\qquad o = h_1 + m_1(h_1 + g_1) + h_2 + m_2(h_2 + g_2) - x$",
        note=r"$h-x=a$ is the other branch's attention output. $W$ starts at zero so the gate is inert; inner width defaults to $d=\mathrm{hidden}/4$ (111M on Mellum2).",
    )


def save_all(out_dir="exports", formats=("png",), dpi=300, names=None) -> list[Path]:
    """Render diagrams to ``out_dir``; returns the paths written.

    ``names`` selects a subset; the default renders all of them.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = {name: DIAGRAMS[name] for name in (names or DIAGRAMS)}
    written = []
    for name, render in selected.items():
        fig = render()
        for extension in formats:
            target = out_dir / f"{name}.{extension}"
            fig.savefig(target, **({"dpi": dpi} if extension == "png" else {}))
            written.append(target)
    return written


__all__ = [*DIAGRAMS, "save_all", "canvas", "DIAGRAMS"]

setup_style()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", choices=[*DIAGRAMS, []], default=[],
                        help="diagrams to render (default: all)")
    parser.add_argument("--out-dir", default="exports")
    parser.add_argument("--formats", default="png", help="comma-separated: png,svg,pdf")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    for written in save_all(
        out_dir=args.out_dir,
        formats=tuple(f.strip() for f in args.formats.split(",") if f.strip()),
        dpi=args.dpi,
        names=args.names or None,
    ):
        print(written)
