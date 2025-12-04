#import "@preview/touying:0.6.1": *
#import "@preview/cetz:0.4.2"
#import "@preview/cetz-plot:0.1.3": chart, plot
#import "@preview/fletcher:0.5.8" as fletcher: diagram, edge, node
#import "@preview/numty:0.0.5" as nt
#import themes.university: *

#show: university-theme.with(
  aspect-ratio: "16-9",
  config-common(handout: false),
  config-info(
    title: [Context Parallelism for Scalable Million-Token Inference],
    author: [William Arnold],
    date: "2025-12-04",
    institution: [DLAlgo Inference Reading Group],
  ),
)
#let cetz-canvas = touying-reducer.with(reduce: cetz.canvas, cover: cetz.draw.hide.with(bounds: true))
#let fletcher-diagram = touying-reducer.with(reduce: fletcher.diagram, cover: fletcher.hide)

#let smat(..args) = { math.mat(delim: "[", ..args) }
#let bvec(a) = { math.accent(math.bold(a), math.arrow) }

// Color palette
#let colors = (
  blue: rgb("#3b82f6"),
  blue-light: rgb("#60a5fa"),
  red: rgb("#ef4444"),
  orange: rgb("#f97316"),
  purple: rgb("#8b5cf6"),
  purple-light: rgb("#a78bfa"),
  green: rgb("#22c55e"),
  green-light: rgb("#4ade80"),
  gray: rgb("#6b7280"),
  gray-light: rgb("#f3f4f6"),
)

// =============================================================================
// Hardware Constants
// =============================================================================
#let sci(a, x) = a * calc.pow(10, x)

// Blackwell B200
#let blackwell_mem_bw = sci(8, 12)       // 8 TB/s HBM
#let blackwell_ifb_bw = sci(200, 9)      // 200 GB/s InfiniBand
#let blackwell_nvlink_bw = sci(0.9, 12)  // 900 GB/s NVLink
#let blackwell_c_fp4 = sci(9, 15)        // 9 PFLOPS FP4
#let blackwell_c_fp8 = sci(4.5, 15)      // 4.5 PFLOPS FP8

// Hopper H100
#let h100_fp8_flops = sci(2, 15)         // 2 PFLOPS FP8
#let h100_ifb_bw = sci(200, 9)           // 200 GB/s InfiniBand

// =============================================================================
// Model Configs
// =============================================================================
#let llama_nkv = 8
#let llama_nh = 128
#let gptoss_nkv = 8
#let gptoss_nh = 64

// DeepSeek V3 MLA (Multi-Head Latent Attention)
// Single compressed KV cache shared across all query heads
#let dsv3_cv = 512      // compressed KV dimension
#let dsv3_c = 576       // key dim (512 + 64 RoPE)
#let dsv3_hq = 128      // query heads

// =============================================================================
// Plotting Helpers
// =============================================================================
// Human-readable tick marks for log scale axes
#let x_ticks = (
  (100, "100"),
  (300, "300"),
  (1000, "1K"),
  (5000, "5K"),
  (16000, "16K"),
  (128000, "128K"),
  (1000000, "1M"),
)

#let y_ticks = (
  (100, "100"),
  (300, "300"),
  (1000, "1K"),
  (2000, "2K"),
  (5000, "5K"),
  (10000, "10K"),
  (20000, "20K"),
  (40000, "40K"),
  (100000, "100K"),
  (300000, "300K"),
  (500000, "500K"),
  (1000000, "1M"),
)

// Create a log-log plot for T_max vs P with automatic axes
// series: array of (data, color, label) tuples
// data is array of (x, y) points
#let tmax-plot(
  series,
  x-label: [$P$ (prefix tokens)],
  y-label: [$T_max$ (new tokens)],
  size: (12, 7),
  x-min: 100,
  x-max: 1000000,
  y-min: auto,
  y-max: auto,
) = {
  // Calculate y bounds from data if auto
  let all_y = series.map(s => s.at(0).map(p => p.at(1))).flatten()
  let data_y_min = calc.min(..all_y)
  let data_y_max = calc.max(..all_y)

  let actual_y_min = if y-min == auto { data_y_min * 0.9 } else { y-min }
  let actual_y_max = if y-max == auto { data_y_max * 1.1 } else { y-max }

  cetz.canvas({
    import cetz.draw: *

    plot.plot(
      size: size,
      x-label: x-label,
      y-label: y-label,
      x-mode: "log",
      y-mode: "log",
      x-tick-step: none,
      y-tick-step: none,
      x-ticks: x_ticks,
      y-ticks: y_ticks,
      x-min: x-min,
      x-max: x-max,
      y-min: actual_y_min,
      y-max: actual_y_max,
      legend: "east",
      {
        for (data, color, label) in series {
          plot.add(
            data,
            style: (stroke: (paint: color, thickness: 2pt)),
            label: label,
          )
        }
      },
    )
  })
}

#title-slide()

== Background: Attention (single query)

For a single query vector $bvec(q)$:
$
             bvec(a) & = bvec(q) K^T = smat(bvec(q) dot bvec(k)_1, bvec(q) dot bvec(k)_2, dots.h, bvec(q) dot bvec(k)_S) = smat(
                         a_1, a_2, dots.h, a_S
                       ) \
  "Softmax"(bvec(a)) & = smat((exp(a_1 - m)) / Z, (exp(a_2 - m))/Z, dots.h, (exp(a_S - m)) / Z) \
           "where" m & = max_j a_j, #h(1em) Z = sum_(j=1)^S exp(a_j - m)
$

== Background: Flash Attention

$forall i in {1..S}$
$
        x_i & = bvec(q) dot bvec(k_i) \
        m_i & = max(m_(i-1), x_i) \
        Z_i & = Z_(i-1)e^(m_(i-1) - m_i) + e^(a_i - m_i) \
  bvec(o)_i & = bvec(o)_(i-1)
              e^(m_i - m_(i-1))
              Z_(i-1) / Z_i
              + e^(x_i - m_i) / Z_i bvec(v_i)
$

At $i=S$, $bvec(o)'_S$ is the correct output for query $bvec(q)$.
#footnote[#link(
  "https://github.com/Aphoh/flash-attention-703/blob/main/main.pdf",
)[#underline[Flash Attention Explained]]]

Define $"AttnBlock"(bvec(q), K, V, bvec(o)_(i-1), m_(i-1), Z_(i-1)) -> (bvec(o)', m, Z)$:

== Ring Attention
Compute attention on pieces of the sequence!

$N$ ranks, give each rank 1/N of the sequence

${Q_1, ..., Q_N}, {K_1, ..., K_N}, {V_1, ..., V_N}$

On rank $i$, keep $Q_i$ and compute $forall i in {1..N}$
$
  bvec(o)_i, m_i, Z_i <- "AttnBlock"(Q_i, bold(K_i), bold(V_i), bvec(o)_i, m_(i-1), Z_(i-1))^(\[#footnote[$o_i$ is initialized to zero]\])
$

$bvec(o)$ will have the correct output for $Q_i$

How to get $bold(K_i)$ and $bold(V_i)$?

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

      // Colors (using global palette)
      let q-color = colors.blue
      let kv-color = colors.red
      let node-fill = colors.gray-light
      let arrow-color = colors.gray

      // Layout parameters
      let radius = 3.5
      let box-size = 0.9
      let kv-box-w = 0.9
      let kv-box-h = 0.35

      // 4 positions: top, right, bottom, left
      let positions = (
        (0, radius), // Rank 0 - top
        (radius, 0), // Rank 1 - right
        (0, -radius), // Rank 2 - bottom
        (-radius, 0), // Rank 3 - left
      )

      // KV offsets for each rank (outside the box)
      let kv-offsets = (
        (0, -box-size - 0.5), // Rank 0: below box
        (-box-size - 0.7, 0), // Rank 1: left of box
        (0, box-size + 0.5), // Rank 2: above box
        (box-size + 0.7, 0), // Rank 3: right of box
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
          stroke: 1.5pt,
        )

        // Rank label - on top except rank 2 on bottom
        let label-y = if i == 2 { pos.at(1) - box-size - 0.4 } else { pos.at(1) + box-size + 0.4 }
        content(
          (pos.at(0), label-y),
          text(size: 12pt, weight: "bold")[Rank #i],
        )

        // Q in the middle (code font, no subscript)
        content(
          pos,
          text(fill: q-color, weight: "bold", size: 14pt, font: "Menlo")[Q#i],
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
              radius: 0.1,
            )
            content(
              (kv-x, kv-y),
              text(fill: kv-color, weight: "bold", size: 14pt, font: "Menlo")[K#kv-idx V#kv-idx],
            )
          })
        }
      }
    })
  ]
])

== Ring Attention: Complexity (Pass-KV)

For $N$ ranks, sequence length $T$, model dim $D_q$, $N_H$ query heads, $N_(K V)$ KV heads:, $e$ bytes/element

#table(
  columns: 2,
  inset: 15pt,
  [*FLOPS*], [ $2 T^2 D$],
  [*Q bytes*], [$T D e$],
  [*KV bytes*], [$2 T D (N_(K V)) / (N_H) e$],
)

*Computation*: $(2 T^2 D) / N$ FLOPs

*Communication* (unidirectional): $2 T D (N_(K V)) / (N_H) e$ bytes
#let BW = $"BW"$

== Ring Attention: Overlap Condition

$
  (2 T^2 D / N) / C >= (2 T D (N_(K V) / N_H) e) / BW \
                                            T / (C N) & >= (N_(K V) e) / (N_H BW) \
                                                    T & >= N (N_(K V) / N_H) ((C e) / (BW))
$

$(C e)/BW$ for Blackwell IFB is $approx$#(blackwell_c_fp4 * 0.5 / blackwell_ifb_bw)

$(C e)/BW$ for Blackwell NVLink is $approx$#(blackwell_c_fp4 * 0.5 / blackwell_nvlink_bw)

$(C e)/BW$ for Blackwell HMB is $approx$#(blackwell_c_fp4 * 0.5 / blackwell_mem_bw)

== When Can We Hide Communication?

#let t_min(n, n_kv, n_h, e, c, bw) = n * (n_kv / n_h) * c * e / (2 * bw)

// MLA T_min: T >= N × C_v × e × C_flops / (2 × H_q × (C + C_v) × BW)
// Note: MLA has single shared KV cache (no factor of 2), and different FLOPs formula
// Reordered to avoid numeric overflow: (c_v / (2 * h_q * (c_k + c_v))) is small
#let t_min_mla(n, c_v, h_q, c_k, e, c, bw) = n * (c_v / (2 * h_q * (c_k + c_v))) * c * e / bw

// Compute T_min for each config
#let llama_tmin_fp4_nvl_b200 = t_min(8, llama_nkv, llama_nh, 0.5, blackwell_c_fp4, blackwell_nvlink_bw)
#let llama_tmin_fp8_nvl_b200 = t_min(8, llama_nkv, llama_nh, 1, blackwell_c_fp8, blackwell_nvlink_bw)
#let gptoss_tmin_fp4_nvl_b200 = t_min(8, gptoss_nkv, gptoss_nh, 0.5, blackwell_c_fp4, blackwell_nvlink_bw)
#let llama_tmin_fp4_ib_b200 = t_min(8, llama_nkv, llama_nh, 0.5, blackwell_c_fp4, blackwell_ifb_bw)
#let llama_tmin_fp8_ib_b200 = t_min(8, llama_nkv, llama_nh, 1, blackwell_c_fp8, blackwell_ifb_bw)
#let gptoss_tmin_fp4_ib_b200 = t_min(8, gptoss_nkv, gptoss_nh, 0.5, blackwell_c_fp4, blackwell_ifb_bw)
#let llama_tmin_fp8_ib_h100 = t_min(4, llama_nkv, llama_nh, 1, h100_fp8_flops, h100_ifb_bw)
#let gptoss_tmin_fp4_ib_h100 = t_min(4, gptoss_nkv, gptoss_nh, 0.5, h100_fp8_flops, h100_ifb_bw)

// DeepSeek V3 MLA T_min values
#let dsv3_tmin_fp4_nvl_b200 = t_min_mla(8, dsv3_cv, dsv3_hq, dsv3_c, 0.5, blackwell_c_fp4, blackwell_nvlink_bw)
#let dsv3_tmin_fp8_nvl_b200 = t_min_mla(8, dsv3_cv, dsv3_hq, dsv3_c, 1, blackwell_c_fp8, blackwell_nvlink_bw)
#let dsv3_tmin_fp4_ib_b200 = t_min_mla(8, dsv3_cv, dsv3_hq, dsv3_c, 0.5, blackwell_c_fp4, blackwell_ifb_bw)
#let dsv3_tmin_fp8_ib_b200 = t_min_mla(8, dsv3_cv, dsv3_hq, dsv3_c, 1, blackwell_c_fp8, blackwell_ifb_bw)

// Horizontal line x-range for T_min plots
#let hline_x = (100, 1000000)

Minimum $T$ for communication overlap (8 $times$ B200) #footnote[Infiniband unidirectional @ 200GB/s, 4.5e12 FP8 FLOPS, 9e12 FP4 FLOPS]:

#align(center)[
  #set text(font: "Menlo", size: 14pt)
  #cetz.canvas({
    import cetz.draw: *

    let data = (
      ([GPT-OSS   FP4 NVL], gptoss_tmin_fp4_nvl_b200),
      ([Llama405B FP4 NVL], llama_tmin_fp4_nvl_b200),
      ([Llama405B FP8 NVL], llama_tmin_fp8_nvl_b200),
      ([GPT-OSS   FP4 IFB], gptoss_tmin_fp4_ib_b200),
      ([Llama405B FP4 IFB], llama_tmin_fp4_ib_b200),
      ([Llama405B FP8 IFB], llama_tmin_fp8_ib_b200),
    )

    set-style(barchart: (bar-width: 0.6))
    chart.barchart(
      size: (8, 6),
      label-key: 0,
      value-key: 1,
      data,
      x-label: [$T_min$ (tokens)],
      x-tick-step: 4000,
      x-format: x => text(size: 12pt)[#(plot.formats.sci(x))],
      bar-style: idx => {
        // NVLink (first 4): greens for MLA, others blues
        // IB (last 4): green for MLA, others oranges/purples
        let bar-colors = (
          colors.red, // GPT-OSS FP4 NVLink
          colors.blue, // Llama FP4 NVLink
          colors.blue-light, // Llama FP8 NVLink
          colors.orange, // GPT-OSS FP4 IB
          colors.purple, // Llama FP4 IB
          colors.purple-light, // Llama FP8 IB
        )
        (fill: bar-colors.at(idx), stroke: none)
      },
    )
  })
]

Independent of datatype since $C e$ is _theoretically_ constant.
NVL72 

== Ring Attention with Prefixes
With $P$ cached tokens,

#table(
  columns: 2,
  inset: 15pt,
  [*FLOPS*], [ $2 T(P + T) D$],
  [*Q bytes*], [$T D e$],
  [*KV bytes*], [$2 (P + T) D (N_(K V)) / (N_H) e$],
)

*Computation*: $display((2 (P + T)T D) / N)$ FLOPs

*Communication* (unidirectional): $display(2 (P + T) D (N_(K V)) / (N_H) e)$ bytes

== Ring Attention with Prefixes: Overlap Condition
$
  (2 T(P + T) D) /(C N) & >= (2 (P + T) D (N_(K V)) / (N_H) e) / BW \
              T / (C N) & >= N_(K V)/N_H e/BW \
                      T & >= N N_(K V) / N_H (C e) / BW
$

#emoji.excl it's the same!

But the new $T$ is _only new tokens_!

Must be 4k+ to hide communication!

== This Paper: Context Parallelism over Queries

From Meta using 128xH100 with IFB and TCP.

For small $T$, queries are much smaller than KV!

What if we ring-pass queries?

== Ring Attention: Pass-Q

#slide(repeat: 6, self => [
  #let (uncover, only) = utils.methods(self)

  #align(center)[
    #text(size: 18pt)[
      #only(1)[Step 1: Each rank computes local attention, stores $(o, m, Z)$]
      #only(2)[Step 2: Send Q to next rank, compute & store]
      #only(3)[Step 3: Send Q to next rank, compute & store]
      #only(4)[Step 4: Send Q to next rank, compute & store]
      #only(5)[All2All: Exchange partial outputs...]
      #only(6)[Done! Each rank has all partials for its own Q]
    ]
  ]

  #v(0.5em)

  #align(center + horizon)[
    #cetz.canvas({
      import cetz.draw: *

      let self = utils.merge-dicts(
        self,
        config-methods(cover: utils.method-wrapper(hide.with(bounds: true))),
      )
      let (uncover, only) = utils.methods(self)

      // Colors (using global palette)
      let q-color = colors.blue
      let kv-color = colors.red
      let state-color = colors.green
      let node-fill = colors.gray-light
      let arrow-color = colors.gray

      // Layout parameters
      let radius = 3.0
      let box-size = 0.75
      let q-box-w = 0.5
      let q-box-h = 0.3
      let state-box-w = 0.75
      let state-box-h = 0.2

      // 4 positions: top, right, bottom, left
      let positions = (
        (0, radius), // Rank 0 - top
        (radius, 0), // Rank 1 - right
        (0, -radius), // Rank 2 - bottom
        (-radius, 0), // Rank 3 - left
      )

      // Q offsets (outside the box)
      let q-offsets = (
        (0, -box-size - 0.4), // Rank 0: below box
        (-box-size - 0.55, 0), // Rank 1: left of box
        (0, box-size + 0.4), // Rank 2: above box
        (box-size + 0.55, 0), // Rank 3: right of box
      )

      // State offsets (on the other side, more spaced out)
      let state-offsets = (
        (0, box-size + 1.3), // Rank 0: above box
        (box-size + 1.5, 0), // Rank 1: right of box
        (0, -box-size - 1.3), // Rank 2: below box
        (-box-size - 1.5, 0), // Rank 3: left of box
      )

      // Draw arrows - ring for steps 1-4, all-to-all for steps 5-6
      // Steps 1-4: Ring arrows
      for step in range(1, 5) {
        only(step, {
          for i in range(4) {
            let curr = positions.at(i)
            let next = positions.at(calc.rem(i + 1, 4))

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
        })
      }

      // Step 5: All-to-All arrows and label
      let a2a-color = colors.orange
      only(5, {
        // Draw bidirectional arrows between all pairs
        for i in range(4) {
          for j in range(i + 1, 4) {
            let p1 = positions.at(i)
            let p2 = positions.at(j)

            let dx = p2.at(0) - p1.at(0)
            let dy = p2.at(1) - p1.at(1)
            let len = calc.sqrt(dx * dx + dy * dy)
            let shrink = (box-size + 0.25) / len

            let start-x = p1.at(0) + dx * shrink
            let start-y = p1.at(1) + dy * shrink
            let end-x = p2.at(0) - dx * shrink
            let end-y = p2.at(1) - dy * shrink

            line(
              (start-x, start-y),
              (end-x, end-y),
              stroke: (paint: a2a-color, thickness: 2pt),
              mark: (start: "stealth", end: "stealth", fill: a2a-color),
            )
          }
        }

        // Central "All2All" label
        rect(
          (-0.8, -0.35),
          (0.8, 0.35),
          fill: rgb("#fff7ed"),
          stroke: (paint: a2a-color, thickness: 1.5pt),
          radius: 0.15,
        )
        content(
          (0, 0),
          text(fill: a2a-color, weight: "bold", size: 12pt)[All2All],
        )
      })

      // Draw nodes with KV (fixed) and labels inside
      for i in range(4) {
        let pos = positions.at(i)

        // Square box
        rect(
          (pos.at(0) - box-size, pos.at(1) - box-size),
          (pos.at(0) + box-size, pos.at(1) + box-size),
          fill: node-fill,
          stroke: 1.5pt,
        )

        // Rank label inside the box at the top
        content(
          (pos.at(0), pos.at(1) + box-size - 0.22),
          text(size: 10pt, weight: "bold")[Rank #i],
        )

        // KV stays fixed in the middle-bottom
        content(
          (pos.at(0), pos.at(1) - 0.15),
          text(fill: kv-color, weight: "bold", size: 12pt, font: "Menlo")[K#i V#i],
        )
      }

      // Animate Q positions based on subslide
      for q-idx in range(4) {
        let get-rank(step) = calc.rem(q-idx + step - 1, 4)

        // Steps 1-4: Q's rotate around the ring
        for step in range(1, 5) {
          only(step, {
            let rank = get-rank(step)
            let pos = positions.at(rank)
            let offset = q-offsets.at(rank)
            let q-x = pos.at(0) + offset.at(0)
            let q-y = pos.at(1) + offset.at(1)

            rect(
              (q-x - q-box-w, q-y - q-box-h),
              (q-x + q-box-w, q-y + q-box-h),
              fill: rgb("#eff6ff"),
              stroke: (paint: q-color, thickness: 1pt),
              radius: 0.08,
            )
            content(
              (q-x, q-y),
              text(fill: q-color, weight: "bold", size: 12pt, font: "Menlo")[Q#q-idx],
            )
          })
        }

        // Steps 5-6: Q's return to their original rank
        for step in (5, 6) {
          only(step, {
            let rank = q-idx // Q returns to its home rank
            let pos = positions.at(rank)
            let offset = q-offsets.at(rank)
            let q-x = pos.at(0) + offset.at(0)
            let q-y = pos.at(1) + offset.at(1)

            rect(
              (q-x - q-box-w, q-y - q-box-h),
              (q-x + q-box-w, q-y + q-box-h),
              fill: rgb("#eff6ff"),
              stroke: (paint: q-color, thickness: 1pt),
              radius: 0.08,
            )
            content(
              (q-x, q-y),
              text(fill: q-color, weight: "bold", size: 12pt, font: "Menlo")[Q#q-idx],
            )
          })
        }
      }

      // Show computed attention results accumulating at each rank
      // For rank r, track which Q's have visited and computed Attn(Qi, KrVr)
      // Q rotation: Q_i is at rank (i + step - 1) mod 4 at step
      // So rank r has Q_(r - step + 1 mod 4) at step

      // Precompute which Q visits each rank at each step
      let get-q-at-rank(rank, step) = calc.rem(rank - step + 1 + 4, 4)
      let state-text-color = rgb("#166534") // darker green for readability
      let final-color = colors.purple // color for final result

      for rank in range(4) {
        let pos = positions.at(rank)
        let offset = state-offsets.at(rank)
        let state-x = pos.at(0) + offset.at(0)
        let state-y = pos.at(1) + offset.at(1)

        // Steps 1-4: Show accumulating partials (different Q's, same KV)
        for step in range(1, 5) {
          only(step, {
            let computed = ()
            for s in range(1, step + 1) {
              let q-id = get-q-at-rank(rank, s)
              computed.push("A(Q" + str(q-id) + ",K" + str(rank) + "V" + str(rank) + ")")
            }

            let num-items = computed.len()
            let item-height = 0.35
            let box-height = num-items * item-height + 0.15

            rect(
              (state-x - state-box-w - 0.4, state-y - box-height / 2),
              (state-x + state-box-w + 0.4, state-y + box-height / 2),
              fill: rgb("#f0fdf4"),
              stroke: (paint: state-color, thickness: 1pt),
              radius: 0.08,
            )

            for (idx, item) in computed.enumerate() {
              let y-pos = state-y + box-height / 2 - 0.22 - idx * item-height
              content(
                (state-x, y-pos),
                text(fill: state-text-color, size: 10pt, font: "Menlo")[#item],
              )
            }
          })
        }

        // Steps 5-6: After All2All, show own Q with all KVs
        for step in (5, 6) {
          only(step, {
            let computed = ()
            for kv-id in range(4) {
              computed.push("A(Q" + str(rank) + ",K" + str(kv-id) + "V" + str(kv-id) + ")")
            }

            let num-items = computed.len()
            let item-height = 0.35
            let box-height = num-items * item-height + 0.15

            let box-fill = if step == 6 { rgb("#faf5ff") } else { rgb("#f0fdf4") }
            let box-stroke = if step == 6 { final-color } else { state-color }
            let text-color = if step == 6 { rgb("#6b21a8") } else { state-text-color }

            rect(
              (state-x - state-box-w - 0.4, state-y - box-height / 2),
              (state-x + state-box-w + 0.4, state-y + box-height / 2),
              fill: box-fill,
              stroke: (paint: box-stroke, thickness: 1pt),
              radius: 0.08,
            )

            for (idx, item) in computed.enumerate() {
              let y-pos = state-y + box-height / 2 - 0.22 - idx * item-height
              content(
                (state-x, y-pos),
                text(fill: text-color, size: 10pt, font: "Menlo")[#item],
              )
            }
          })
        }
      }
    })
  ]
])


== Q-Passing Roofline


#table(
  columns: 2,
  inset: 15pt,
  [*FLOPS*], [ $2 T(P + T) D$],
  [*Q bytes*], [$T D e$],
  [*KV bytes*], [$2 (P + T) D (N_(K V)) / (N_H) e$],
)

*Computation*: $display((2 T(P + T) D) / N)$ FLOPs

*Communication* (unidirectional): $bold(T D e)$ bytes
#pagebreak()

Overlap condition:
$
  (2 T(P + T) D) /(C N) & >= (T D e) / BW \
                  P + T & >= N / 2 (C e) / BW
$

Doesn't depend on $N_H, N_(K V)$

#pagebreak()

// Pass-Q: (P+T)_min = N/2 * C * e / BW
// Note: C*e is the same for FP4 and FP8! (9e15 * 0.5 = 4.5e15 * 1 = 4.5e15)
#let passq_min(n, c, e, bw) = n / 2 * c * e / bw

#let passq_fp8_nvl = passq_min(8, blackwell_c_fp8, 1, blackwell_nvlink_bw)
#let passq_fp8_ib = passq_min(8, blackwell_c_fp8, 1, blackwell_ifb_bw)


Minimum $(P + T)$ for communication overlap (8 $times$ B200):

#align(center)[
  #set text(font: "Menlo", size: 14pt)
  #cetz.canvas({
    import cetz.draw: *

    let data = (
      ([NVLink], passq_fp8_nvl),
      ([IFB   ], passq_fp8_ib),
    )

    set-style(barchart: (bar-width: 0.6))
    chart.barchart(
      size: (8, 4),
      label-key: 0,
      value-key: 1,
      data,
      x-label: [$(P + T)_min$ (tokens)],
      x-tick-step: 20000,
      x-format: x => text(size: 12pt)[#(plot.formats.sci(x))],
      bar-style: idx => {
        let bar-colors = (
          colors.blue, // FP4 NVLink
          colors.blue-light, // FP8 NVLink
          colors.purple, // FP4 IB
          colors.purple-light, // FP8 IB
        )
        (fill: bar-colors.at(idx), stroke: none)
      },
    )
  })
]

Note: $C dot e$ is the same for FP4 and FP8, so model doesn't matter!

Requires balanced KVs across ranks (round-robin during decode)

== This Paper Kinda Stinks

They have a bunch of math mistakes. Ex:
#image("og_paper_bad_math.png", width: 100%)

#pause

This is _so wrong it hurts._

#pause

They also don't actually solve their inequalities!

#pause

Let's do it right.

== All-to-All cost

All-to-All time in a ring is roughly #footnote[
  #link("https://jax-ml.github.io/scaling-book/sharding/#our-final-communication-primitive-the-alltoall")[
    #underline("See this derivation")
  ]
]
$
  T_"all2all" = (T D e) / (4 BW)
$

All-to-All time in NVL72 is just $(T D e) / (N BW)$

Need to check if $T_"all2all" < (T_"kv,comm" - T_"kv,compute")$

== All-to-All cost

$
  T_"kv,comm" - T_"kv,compute" & = 2 (P + T) D (N_(K V)) / (N_H) e / BW - (2 (P + T) T D) / (N C) \
                               & = 2(P+T) D ( N_(K V) / N_H e / BW- T / (N C)) \
              (T D e) / (4 BW) & < T_"kv,comm" - T_"kv,compute" \
$

Ends up being quadratic... hand it to `sympy`, solve for $T$ and plot


== $T_max$ vs Prefix Length

#let t_max(P, N, n_kv, n_h, bw, c, e) = {
  // From: t_all2all < t_kv_comm - t_kv_compute
  // Coefficients of quadratic α·T² + β·T + γ < 0
  let alpha = 2 / (c * N)
  let beta = 2 * P / (c * N) + e / (4 * bw) - 2 * n_kv * e / (bw * n_h)
  let gamma = -2 * n_kv * P * e / (bw * n_h)
  let discriminant = beta * beta - 4 * alpha * gamma
  (-beta + calc.sqrt(discriminant)) / (2 * alpha)
}

Maximum $T$ where Pass-Q is faster than Pass-KV (8 $times$ B200 IFB):


// Generate data points for P from 1 to 1M (log scale sampling)
#let p_values = nt.logspace(2, 6, 100)

#let llama_b200_ifb = p_values.map(p => {
  (p, t_max(p, 8, llama_nkv, llama_nh, blackwell_ifb_bw, blackwell_c_fp8, 1))
})

#let gptoss_b200_ifb = p_values.map(p => {
  (p, t_max(p, 8, gptoss_nkv, gptoss_nh, blackwell_ifb_bw, blackwell_c_fp8, 1))
})

#align(center)[
  #tmax-plot((
    (llama_b200_ifb, colors.blue, [Llama-405B ($N_H$/$N_(K V)$ = 16)]),
    (gptoss_b200_ifb, colors.red, [GPT-OSS ($N_H$/$N_(K V)$ = 8)]),
  ))
]

Below $T_max$, use All-to-All (Pass-Q). Above $T_max$, use Pass-KV.


== What about NVL72?
All-to-all cost is $(T D e) / (bold(N) dot 4 dot BW)$ and $BW$ is 900GB/s v.s. 200GB/s.

#let t_max_nvl72(P, N, n_kv, n_h, bw, c, e) = {
  // From: t_all2all < t_kv_comm - t_kv_compute (NVL72 version)
  // Only difference: β has e/(4·BW·N) instead of e/(4·BW)
  let alpha = 2 / (c * N)
  let beta = 2 * P / (c * N) - 2 * n_kv * e / (bw * n_h) + e / (4 * bw * N)
  let gamma = -2 * n_kv * P * e / (bw * n_h)
  let discriminant = beta * beta - 4 * alpha * gamma
  (-beta + calc.sqrt(discriminant)) / (2 * alpha)
}
#let p_values = nt.logspace(2, 6, 100)

// NVL72 with NVLink bandwidth (900GB/s)
#let llama_nvl72 = p_values.map(p => {
  (p, t_max_nvl72(p, 72, llama_nkv, llama_nh, blackwell_nvlink_bw, blackwell_c_fp8, 1))
})

#let gptoss_nvl72 = p_values.map(p => {
  (p, t_max_nvl72(p, 72, gptoss_nkv, gptoss_nh, blackwell_nvlink_bw, blackwell_c_fp8, 1))
})

#let llama_tmin_nvl_line = hline_x.map(x => (x, llama_tmin_fp8_nvl_b200))
#let gptoss_tmin_nvl_line = hline_x.map(x => (x, gptoss_tmin_fp4_nvl_b200))

#align(center)[
  #tmax-plot((
    (llama_nvl72, colors.blue, [Llama-405B (NVL72)]),
    (gptoss_nvl72, colors.red, [GPT-OSS (NVL72)]),
    (llama_tmin_nvl_line, colors.blue-light, [Llama $T_min$ (NVL)]),
    (gptoss_tmin_nvl_line, colors.orange, [GPT-OSS $T_min$ (NVL)]),
  ))
]

With NVL72's lower all-to-all cost, Pass-Q is optimal up to larger $T$.

But we only fail to hide communication for fairly small $T$.

== Meta's Hopper Performance

#image("ttft_comparison.pdf", width: 100%)

pass-Q is slighly better than pass-KV for very low miss rates

Does the math say the same?

== Validating Meta's Hopper Performance
4 x H100 with 200GB/s (unidirectional) IFB

#let p_values = nt.logspace(2, 6, 5)

#let llama_h100 = p_values.map(p => {
  (p, t_max(p, 4, llama_nkv, llama_nh, h100_ifb_bw, h100_fp8_flops, 1))
})

#let gptoss_h100 = p_values.map(p => {
  (p, t_max(p, 4, gptoss_nkv, gptoss_nh, h100_ifb_bw, h100_fp8_flops, 1))
})

#let hline_x = (100, 1000000)
#let llama_tmin_ib_line = hline_x.map(x => (x, llama_tmin_fp8_ib_h100))
#let gptoss_tmin_ib_line = hline_x.map(x => (x, gptoss_tmin_fp4_ib_h100))
#align(center)[
  #tmax-plot((
    (gptoss_h100, colors.orange, [GPT-OSS (H100 IFB)]),
    (llama_h100, colors.blue, [Llama-405B (H100 IFB)]),
    (llama_tmin_ib_line, colors.blue-light, [Llama-405B $T_min$ (IFB)]),
    (gptoss_tmin_ib_line, colors.red, [GPT-OSS $T_min$ (IFB)]),
  ))
]
#only(1)[
For H100 IFB at P=128K, $T_"max" approx 5K approx 4\%$ miss rate! Empirically $approx 5\%$!
]
#only(2)[
  $T_"min"$ is 1250 for both models... Well above the 4k where Meta sees a difference
]

== Putting it all together
All $N = 8$

#align(center)[
  #tmax-plot((
    (gptoss_b200_ifb, colors.red, [GPT-OSS (B200 IFB)]),
    (gptoss_h100, colors.orange, [GPT-OSS (H100 IFB)]),
    (llama_b200_ifb, colors.purple, [Llama-405B (B200 IFB)]),
    (llama_h100, colors.blue, [Llama-405B (H100 IFB)]),
    (llama_nvl72, colors.blue-light, [Llama-405B (NVL72)]),
    (gptoss_nvl72, colors.green, [GPT-OSS (NVL72)]),
  ))
]

NVL72 pass-Q may be optimal, but $T$ which isn't hideable is small.

== Extra: DSv3 MLA

// MLA coefficients from solve_ineq.py:
// α = 2*H_q*(C_rope + C_v)/(C*N)
// β = 2*H_q*P*(C_rope + C_v)/(C*N) - C_v*e/BW + D*e/(4*BW)
// γ = -C_v*P*e/BW
#let dsv3_d = 7168  // model dimension for Q communication

#let t_max_mla(P, N, c_v, h_q, c_k, d, bw, c, e) = {
  let alpha = 2 * h_q * (c_k + c_v) / (c * N)
  let beta = 2 * h_q * P * (c_k + c_v) / (c * N) - c_v * e / bw + d * e / (4 * bw)
  let gamma = -c_v * P * e / bw
  let discriminant = beta * beta - 4 * alpha * gamma
  (-beta + calc.sqrt(discriminant)) / (2 * alpha)
}

#let dsv3_mla_ifb = p_values.map(p => {
  (p, t_max_mla(p, 8, dsv3_cv, dsv3_hq, dsv3_c, dsv3_d, blackwell_ifb_bw, blackwell_c_fp8, 1))
})

// Horizontal lines for T_min (constant y across x range)
#let hline_x = (100, 1000000)
#let dsv3_tmin_line = hline_x.map(x => (x, dsv3_tmin_fp4_ib_b200))
#let llama_tmin_line = hline_x.map(x => (x, llama_tmin_fp4_ib_b200))

#align(center)[
  #tmax-plot(
    (
      (dsv3_mla_ifb, colors.green, [DSv3 MLA $T_max$]),
      (dsv3_tmin_line, colors.green-light, [DSv3 MLA $T_min$]),
      (llama_b200_ifb, colors.blue, [Llama-405B $T_max$]),
      (llama_tmin_line, colors.blue-light, [Llama-405B $T_min$]),
    ),
    y-min: 50,
  )
]

MLA: smaller KV cache → lower $T_min$, higher $T_max$ → Pass-Q is never necessary.
