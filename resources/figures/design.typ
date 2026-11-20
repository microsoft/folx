// Figure 1: (a) unmodified JAX code is traced into (b) a jaxpr; (c) every jaxpr equation
// maps the triplets (value, Jacobian, Laplacian) of its inputs to a new triplet via an
// annotated (blue) or custom (orange) forward Laplacian rule.
#import "style.typ": *
#show: setup

// ---------------------------------------------------------------------------------------
// Geometry. Panels (b) and (c) share one row per jaxpr line.
#let cw = 0.602 * 7pt          // advance of one DejaVu Sans Mono character at 7pt
#let rh = 20pt                 // row height
#let top0 = 15pt               // room above the frames for their titles
#let pad = 6pt                 // vertical inset of the jaxpr frame
#let jw = 22 * cw + 12pt       // width of the jaxpr frame
#let rule-len = 75pt           // length of the rule arrows between (b) and (c)
#let tx = jw + rule-len        // left edge of the triplets
#let tw = 146pt                // triplet width
#let th = 15.5pt                 // triplet height
#let re = tx + tw              // right edge of the triplets
#let lane = (re + 4.5pt, re + 9pt, re + 13.5pt) // x positions of the dataflow lanes
#let stub = 3.5pt              // dataflow arrows leave below / enter above the row centre
#let rr = 2.5pt                // corner radius of the dataflow connectors
#let nrows = 7
#let height = top0 + 2 * pad + nrows * rh
#let yc(i) = top0 + pad + (i + 0.5) * rh

#let code-fill = rgb("#f5f5f5")
#let frame-stroke = 0.7pt + rgb("#c8c8c8")
#let flow = rgb("#8c8c8c")
#let flow-stroke = 0.7pt + flow

#let tok(s, color) = text(fill: color, raw(s))
#let ind = h(2 * cw)
#let title(tag, name) = box(text(size: 8pt)[*(#tag)* #name])

// Gray frame with its title placed just above the top border.
#let frame(tag, name, ix: 6pt, it: 6pt, ib: 6pt, body) = box(
  fill: code-fill,
  stroke: frame-stroke,
  radius: 4pt,
  inset: (x: ix, top: it, bottom: ib),
  {
    body
    place(top + left, dx: -ix + 1pt, dy: -it - 13pt, title(tag, name))
  },
)

// ---------------------------------------------------------------------------------------
// (a) The program.
#let program = frame("a", "Program", ix: 7pt, it: 10pt, ib: 8pt, stack(
  spacing: 6pt,
  raw("def f(r):"),
  [#ind#raw("h = jnp.")#tok("tanh", blue)#raw("(r ")#tok("@", blue)#raw(" W)")],
  [#ind#raw("s = h.")#tok("sum", blue)#raw("()")],
  [#ind#raw("d = ")#tok("slogdet", orange)#raw("(h)[1]")],
  [#ind#raw("return d ")#tok("+", blue)#raw(" s")],
))

// ---------------------------------------------------------------------------------------
// (b) The jaxpr, one line per row.
#let jrow(body) = box(height: rh, align(left + horizon, body))
#let eqn(lhs, prim, rhs, color) = jrow[#ind#raw(lhs + " = ")#tok(prim, color)#raw(" " + rhs)]
#let jaxpr = frame("b", "Jaxpr", it: pad, ib: pad, box(width: jw - 12pt, stack(
  jrow(text(fill: gray, raw("{ lambda W; r. let"))),
  eqn("a", "dot_general", "r W", blue),
  eqn("b", "tanh", "a", blue),
  eqn("c", "reduce_sum", "b", blue),
  eqn("_, d", "slogdet", "b", orange),
  eqn("e", "add", "d c", blue),
  jrow(text(fill: gray, raw("in (e,) }"))),
)))

// ---------------------------------------------------------------------------------------
// (c) Rules and triplets.
#let triplet(body) = box(
  width: tw,
  height: th,
  fill: light-blue,
  stroke: 0.9pt + blue,
  radius: 3pt,
  align(center + horizon, text(size: 7.5pt, body)),
)
#let pill(name, color, fill) = box(
  fill: fill,
  stroke: 0.7pt + color,
  radius: 6pt,
  inset: (x: 5pt, y: 2.5pt),
  align(center, text(fill: color, raw(name))),
)
// Arrowhead with its tip at (x, y), pointing right (dir: 1) or left (dir: -1).
#let head(x, y, color, dir: 1, l: 4pt, w: 3.4pt) = place(
  top + left,
  polygon(fill: color, (x, y), (x - dir * l, y - w / 2), (x - dir * l, y + w / 2)),
)
// Arrow from a jaxpr line to its triplet, interrupted by the rule pill.
#let rule-arrow(i, color, p: none) = {
  let y = yc(i)
  let x0 = jw + 2pt
  place(top + left, line(start: (x0, y), end: (tx - 3pt, y), stroke: 0.8pt + color))
  head(tx - 0.6pt, y, color)
  if p != none {
    place(top + left, dx: x0, dy: y - 10pt,
      box(width: tx - 3pt - x0, height: 20pt, align(center + horizon, p)))
  }
}
// Dataflow connector from the triplet in row i to the one in row j along lane k.
#let connector(i, j, k, halo: false) = {
  let y0 = yc(i) + stub
  let y1 = yc(j) - stub
  let x = lane.at(k)
  let path(s) = place(top + left, curve(
    stroke: s,
    curve.move((re + 0.5pt, y0)),
    curve.line((x - rr, y0)),
    curve.quad((x, y0), (x, y0 + rr)),
    curve.line((x, y1 - rr)),
    curve.quad((x, y1), (x - rr, y1)),
    curve.line((re + 3.5pt, y1)),
  ))
  if halo { path(2.6pt + white) }
  path(flow-stroke)
  head(re + 0.5pt, y1, flow, dir: -1, l: 3.5pt, w: 3pt)
}

#let rhs = box(width: lane.at(2) + 1pt, height: height, {
  place(top + left, dy: top0, jaxpr)
  place(top + left, dx: jw + 8pt, dy: top0 - 13pt, title("c", "Forward Laplacian"))

  // Triplets.
  // Each triplet written out via its rule in terms of the triplets it consumes.
  let trips = (
    $(bold(r), I, 0)$,
    $(bold(r) W, W, 0)$,
    $(tanh bold(a), J_b nabla bold(a), H_b abs(nabla bold(a))^2)$,
    $(sum bold(b), sum nabla bold(b), sum Delta bold(b))$,
    $(log|det bold(b)|, J_d nabla bold(b), J_d Delta bold(b) + tr(nabla bold(b)^top H_d nabla bold(b)))$,
    $(d + c, nabla d + nabla c, Delta d + Delta c)$,
  )
  for (i, t) in trips.enumerate() {
    place(top + left, dx: tx, dy: yc(i) - th / 2, triplet(t))
  }

  // Rule arrows from each jaxpr line to its triplet.
  rule-arrow(0, flow)
  rule-arrow(1, blue, p: pill("LINEAR", blue, light-blue))
  rule-arrow(2, blue, p: pill("in_axes=()", blue, light-blue))
  rule-arrow(3, blue, p: pill("SUMMATION", blue, light-blue))
  rule-arrow(4, orange, p: pill("custom rule", orange, light-orange))
  rule-arrow(5, blue, p: pill("LINEAR", blue, light-blue))

  // Dataflow between triplets: x->a, a->b, b->c, b->d, c->e, d->e.
  connector(2, 4, 1)
  connector(3, 5, 2, halo: true)
  connector(0, 1, 0)
  connector(1, 2, 0)
  connector(2, 3, 0)
  connector(4, 5, 0)
})

#grid(
  columns: 3,
  column-gutter: 3.5pt,
  align: horizon,
  program,
  harrow(length: 16pt, label: [trace]),
  rhs,
)
