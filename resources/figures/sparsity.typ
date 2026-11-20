// Figure 2: compile-time sparsity for four electrons with three coordinates each.
// (a) storage of a sparse Jacobian, (b) elementwise, (c) concatenation, (d) reduction.
#import "style.typ": *
#show: setup

#let cell-w = 9.5pt
#let cell-h = 9pt
#let PAD = -1

// Index array with one row per output entry, colored by electron.
#let mask(rows) = table(
  columns: rows.first().len(),
  inset: 0pt,
  stroke: grid-stroke,
  ..rows.flatten().map(idx => table.cell(
    fill: if idx == PAD { rgb("#f2f2f2") } else { electron-colors.at(calc.div-euclid(idx, 3)) },
    box(width: cell-w, height: cell-h, align(center + horizon, text(
      font: "DejaVu Sans Mono",
      size: 6pt,
      fill: if idx == PAD { gray } else { ink },
      if idx == PAD { "-1" } else { str(idx) },
    ))),
  )),
)

// Matrix whose cell in column j is colored if j is a dependency of that row.
#let pattern(rows, n-cols) = table(
  columns: n-cols,
  inset: 0pt,
  stroke: grid-stroke,
  ..rows.map(deps => range(n-cols).map(j => table.cell(
    fill: if n-cols == deps.len() { electron-colors.at(calc.div-euclid(deps.at(j), 3)) }
    else if j in deps { electron-colors.at(calc.div-euclid(j, 3)) }
    else { white },
    box(width: cell-w, height: cell-h),
  ))).flatten(),
)

#let labeled(label, body) = stack(spacing: 5pt, align(center, label), body)
#let side(label, body) = grid(columns: 2, column-gutter: 3pt, align: horizon, label, body)

#let one = range(4).map(e => range(3).map(k => 3 * e + k))
#let pair = (range(6, 12),)
#let concat = (one.at(0) + (PAD,) * 3, one.at(1) + (PAD,) * 3, pair.at(0))

#let panel-a = panel("a", grid(
  columns: 4,
  column-gutter: 4pt,
  align: horizon,
  labeled($nabla bold(x)$, pattern(one, 12)),
  pad(top: 12pt, harrow(length: 16pt)),
  labeled($D_bold(x)$, pattern(one, 3)),
  labeled($cal(I)_bold(x)$, mask(one)),
))

#let panel-b = panel("b", grid(
  columns: 3,
  column-gutter: 4pt,
  align: horizon,
  labeled($cal(I)_bold(x)$, mask(one)),
  pad(top: 12pt, harrow(length: 20pt, label: raw("tanh"))),
  labeled($cal(I)_bold(y)$, mask(one)),
))

#let panel-d = panel("d", grid(
  columns: 3,
  column-gutter: 4pt,
  align: horizon,
  labeled($cal(I)_bold(x)$, mask(one.slice(0, 2))),
  pad(top: 12pt, harrow(length: 18pt, label: raw("sum"))),
  labeled($cal(I)_bold(y)$, mask((range(6),))),
))

#let panel-c = panel("c", grid(
  columns: 3,
  column-gutter: 4pt,
  align: horizon,
  stack(
    spacing: 4pt,
    side($cal(I)_bold(a)$, mask(one.slice(0, 2))),
    side($cal(I)_bold(b)$, mask(pair)),
  ),
  harrow(length: 40pt, label: raw("concatenate")),
  labeled($cal(I)_bold(y)$, mask(concat)),
))

#grid(
  columns: 2,
  column-gutter: 12pt,
  row-gutter: 12pt,
  panel-a, panel-b,
  panel-c, panel-d,
)
