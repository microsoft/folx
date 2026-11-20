// Shared style for the paper figures.
// Compile with: typst compile --format png --ppi 300 resources/figures/<name>.typ

#let ink = rgb("#2b2b2b")
#let gray = rgb("#7a7a7a")
#let blue = rgb("#2a6fb0")
#let light-blue = rgb("#eaf1f9")
#let orange = rgb("#d9822b")
#let light-orange = rgb("#fdf1e6")
#let grid-stroke = 0.5pt + rgb("#9a9a9a")
// One color per electron, i.e., per block of three input coordinates.
#let electron-colors = (rgb("#cfe0f1"), rgb("#fbdcc0"), rgb("#d3ead6"), rgb("#e4d7ee"))

// Match the paper's LaTeX math: \mathbf is upright bold, \mathcal is roundhand script.
#let bold(x) = math.upright(math.bold(x))

#let setup(body) = {
  set page(width: auto, height: auto, margin: 3pt, fill: white)
  set text(font: "New Computer Modern", size: 9pt, fill: ink)
  show math.equation: set text(stylistic-set: 1)
  show raw: set text(font: "DejaVu Sans Mono", size: 7pt)
  body
}

// Horizontal arrow with an optional label above.
#let harrow(length: 24pt, color: gray, label: none) = stack(
  spacing: 2pt,
  if label != none { align(center, text(size: 7pt, fill: color, label)) },
  box(height: 6pt, stack(
    dir: ltr,
    box(height: 6pt, align(horizon, line(length: length - 5pt, stroke: 1pt + color))),
    polygon(fill: color, (0pt, 0pt), (5pt, 3pt), (0pt, 6pt)),
  )),
)

// Vertical arrow pointing down.
#let varrow(length: 14pt, color: gray) = box(width: 6pt, stack(
  align(center, line(angle: 90deg, length: length - 5pt, stroke: 1.1pt + color)),
  polygon(fill: color, (0pt, 0pt), (6pt, 0pt), (3pt, 5pt)),
))

#let panel(tag, body) = stack(spacing: 5pt, text(weight: "bold")[(#tag)], body)
