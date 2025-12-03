#import "@preview/touying:0.6.1": *
#import "@preview/cetz:0.4.2"
#import "@preview/cetz-plot:0.1.3": chart, plot
#import "@preview/fletcher:0.5.8" as fletcher: node, edge, diagram
#import themes.university: *

#show: university-theme.with(
  aspect-ratio: "16-9",
  config-common(handout: false),
  config-info(
    title: [Context Parallelism for Scalable Million-Token Inference],
    author: [William Arnold],
    date: datetime.today(),
    institution: [DLAlgo Inference Reading Group],
  ),
)
#let cetz-canvas = touying-reducer.with(reduce: cetz.canvas, cover: cetz.draw.hide.with(bounds: true))
#let fletcher-diagram = touying-reducer.with(reduce: fletcher.diagram, cover: fletcher.hide)

#let smat(..args) = { math.mat(delim: "[", ..args) }
#let bvec(a) = { math.accent(math.bold(a), math.arrow)}

#title-slide()

== Background: Attention (single query)

For a single query vector $bvec(q)$:
$
bvec(a) &= bvec(q) K^T = smat(
bvec(q) dot bvec(k)_1, bvec(q) dot bvec(k)_2, dots.h, bvec(q) dot bvec(k)_S
) = smat(
a_1, a_2, dots.h, a_S
) \
"Softmax"(bvec(a)) &= smat((exp(a_1 - m)) / Z, (exp(a_2 - m))/Z, dots.h , (exp(a_S - m)) / Z) \
"where" m &= max_j a_j, #h(1em) Z = sum_(j=1)^S exp(a_j - m)
$ 

== Background: Flash Attention

$forall i in {1..S}$
$
x_i &= bvec(q) dot bvec(k_i) \
m_i &= max(m_(i-1), x_i) \
Z_i &= Z_(i-1)e^(m_(i-1) - m_i) + e^(a_i - m_i) \
bvec(o)_i &= bvec(o)_(i-1)
            e^(m_i - m_(i-1))
            Z_(i-1) / Z_i
            + e^(x_i - m_i) / Z_i bvec(v_i)
$

At $i=S$, $bvec(o)'_S$ is the correct output for query $bvec(q)$.
#footnote[#link("https://github.com/Aphoh/flash-attention-703/blob/main/main.pdf")[#underline[Flash Attention Explained]]]

Define $"FlashAttnBlock"(bvec(q), K, V, bvec(o)_(i-1), m_(i-1), Z_(i-1)) -> (bvec(o)', m, Z)$:

== Ring Attention
Compute attention on pieces of the sequence!

$N$ ranks, ${Q_1, ..., Q_N}, {K_1, ..., K_N}, {V_1, ..., V_N}$

On rank $i$, keep $Q_i$ and compute $forall i in {1..N}$
$
bvec(o)_i, m_i, Z_i <- "FlashAttnBlock"(Q_i, bold(K_i), bold(V_i), bvec(o)_i, m_(i-1), Z_(i-1))^
(\[#footnote[$o_i$ is initialized to zero]\])
$ 

$bvec(o)$ will have the correct output for $Q_i$ 

== Ring Attention: Communication

#slide(repeat: 5, self => [
  #let (uncover, only) = utils.methods(self)
  
  #align(center)[
    #text(size: 18pt)[
      #only(1)[Step 1: Each rank computes local attention]
      #only(2)[Step 2: Send KV to next rank]
      #only(3)[Step 3: Send KV to next rank]
      #only(4)[Step 4: Send KV to next rank]
      #only(5)[Complete! Each Q has seen all KVs]
    ]
  ]
  
  #v(0.5em)
  
  #align(center)[
  #cetz.canvas({
    import cetz.draw: *
    
    let self = utils.merge-dicts(
      self,
      config-methods(cover: utils.method-wrapper(hide.with(bounds: true))),
    )
    let (uncover, only) = utils.methods(self)
    
    // Colors
    let q-color = rgb("#2563eb")  // blue
    let kv-color = rgb("#dc2626") // red
    let node-fill = rgb("#f3f4f6")
    let arrow-color = rgb("#6b7280")
    
    // Layout parameters
    let radius = 3.5
    let box-size = 0.9
    let kv-box-w = 0.9
    let kv-box-h = 0.35
    
    // 4 positions: top, right, bottom, left
    let positions = (
      (0, radius),      // Rank 0 - top
      (radius, 0),      // Rank 1 - right  
      (0, -radius),     // Rank 2 - bottom
      (-radius, 0),     // Rank 3 - left
    )
    
    // KV offsets for each rank (outside the box)
    let kv-offsets = (
      (0, -box-size - 0.5),   // Rank 0: below box
      (-box-size - 0.7, 0),   // Rank 1: left of box
      (0, box-size + 0.5),    // Rank 2: above box
      (box-size + 0.7, 0),    // Rank 3: right of box
    )
    
    // Draw straight arrows between nodes
    for i in range(4) {
      let curr = positions.at(i)
      let next = positions.at(calc.rem(i + 1, 4))
      
      // Shorten arrows to not overlap boxes
      let dx = next.at(0) - curr.at(0)
      let dy = next.at(1) - curr.at(1)
      let len = calc.sqrt(dx * dx + dy * dy)
      let shrink = (box-size + 0.3) / len
      
      let start-x = curr.at(0) + dx * shrink
      let start-y = curr.at(1) + dy * shrink
      let end-x = next.at(0) - dx * shrink
      let end-y = next.at(1) - dy * shrink
      
      line(
        (start-x, start-y),
        (end-x, end-y),
        stroke: (paint: arrow-color, thickness: 1.5pt, dash: "dashed"),
        mark: (end: "stealth", fill: arrow-color),
      )
    }
    
    // Draw nodes and labels
    for i in range(4) {
      let pos = positions.at(i)
      
      // Square box
      rect(
        (pos.at(0) - box-size, pos.at(1) - box-size),
        (pos.at(0) + box-size, pos.at(1) + box-size),
        fill: node-fill,
        stroke: 1.5pt
      )
      
      // Rank label - on top except rank 2 on bottom
      let label-y = if i == 2 { pos.at(1) - box-size - 0.4 } else { pos.at(1) + box-size + 0.4 }
      content(
        (pos.at(0), label-y),
        text(size: 12pt, weight: "bold")[Rank #i]
      )
      
      // Q in the middle (code font, no subscript)
      content(
        pos,
        text(fill: q-color, weight: "bold", size: 14pt, font: "Menlo")[Q#i]
      )
    }
    
    // Animate KV positions based on subslide
    for kv-idx in range(4) {
      let get-rank(step) = calc.rem(kv-idx + step - 1, 4)
      
      for step in range(1, 6) {
        only(step, {
          let rank = get-rank(step)
          let pos = positions.at(rank)
          let offset = kv-offsets.at(rank)
          let kv-x = pos.at(0) + offset.at(0)
          let kv-y = pos.at(1) + offset.at(1)
          
          // Draw rectangle around KV
          rect(
            (kv-x - kv-box-w, kv-y - kv-box-h),
            (kv-x + kv-box-w, kv-y + kv-box-h),
            fill: rgb("#fef2f2"),
            stroke: (paint: kv-color, thickness: 1pt),
            radius: 0.1
          )
          content(
            (kv-x, kv-y),
            text(fill: kv-color, weight: "bold", size: 11pt, font: "Menlo")[K#kv-idx V#kv-idx]
          )
        })
      }
    }
    
  })
  ]
])

== Ring Attention: Complexity (Pass-KV)

For $N$ ranks, sequence length $T$, model dim $D_q$, $N_H$ query heads, $N_(K V)$ KV heads:, $e$ bytes/element

#table(columns: 2, inset: 15pt,
  [*FLOPS*],  [ $4 T^2 D$],
  [*Q bytes*], [$T D e$],
  [*KV bytes*], [$2 T D (N_(K V)) / (N_H) e$]
)

*Computation* (per rank): $display((4 T^2 D) / N)$ FLOPs

*Communication* (per rank): $display(2 T D (N_(K V)) / (N_H) e)$ bytes
#let BW = $"BW"$

Overlap condition: $ (4 T^2 D / N) / C >= (2 T D (N_(K V)) / (N_H) e) / BW $

$
  2 T / (C N) &>= N_(K V) / (N_H e BW) \
  T &>= (N_(K V) / N_H)  (C / (2 e BW))
$

#let sci(a, x) = a * calc.pow(10, x)
#let blackwell_mem_bw = sci(8, 12)
#let blackwell_c = sci(9, 15)

$C/BW$ for Blackwell FP4 is $approx$#(blackwell_c / blackwell_mem_bw)

== When Can We Hide Communication?

// T_min = (N_KV / N_H) * (C / (2 * e * BW))
// C scales with precision: FP4 = 9e15, FP8 = 4.5e15
#let blackwell_c_fp4 = sci(9, 15)
#let blackwell_c_fp8 = sci(4.5, 15)

#let t_min(n_kv, n_h, e, c) = (n_kv / n_h) * c / (2 * e * blackwell_mem_bw)

// Model configs
#let llama_nkv = 8
#let llama_nh = 128
#let gptoss_nkv = 8
#let gptoss_nh = 64

// Compute T_min for each config
#let llama_fp4 = t_min(llama_nkv, llama_nh, 0.5, blackwell_c_fp4)
#let llama_fp8 = t_min(llama_nkv, llama_nh, 1, blackwell_c_fp8)
#let gptoss_fp4 = t_min(gptoss_nkv, gptoss_nh, 0.5, blackwell_c_fp4)

Minimum $T$ for communication overlap (Blackwell):

#align(center)[
#cetz.canvas({
  import cetz.draw: *
  
  let data = (
    ([GPT-OSS FP4], gptoss_fp4),
    ([Llama 405B FP4], llama_fp4),
    ([Llama 405B FP8], llama_fp8),
  )
  
  set-style(barchart: (bar-width: 0.6))
  chart.barchart(
    size: (8, 5),
    label-key: 0,
    value-key: 1,
    data,
    x-label: text(size: 18pt)[$T_min$ (tokens)],
    x-tick-step: 100,
    x-format: x => text(size: 18pt, font: "Menlo")[#(plot.formats.sci(x))],
    bar-style: (idx) => {
      let colors = (rgb("#3b82f6"), rgb("#60a5fa"), rgb("#ef4444"))
      (fill: colors.at(idx), stroke: none)
    }
  )
})
]

#table(columns: 5, inset: 10pt, align: center,
  [*Model*], [$N_H$], [$N_(K V)$], [*Precision*], [$T_min$],
  [Llama 405B], [128], [8], [FP4], [#calc.round(llama_fp4)],
  [Llama 405B], [128], [8], [FP8], [#calc.round(llama_fp8)],
  [GPT-OSS], [64], [8], [FP4], [#calc.round(gptoss_fp4)],
)

More KV heads relative to Q heads → need longer sequences to hide comm

