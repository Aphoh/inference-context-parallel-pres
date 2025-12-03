#import "@preview/touying:0.5.2": *
#import "@preview/cetz:0.2.2"
#import "@preview/fletcher:0.5.1" as fletcher: node, edge, diagram
#import themes.university: *

#show: university-theme.with(
  aspect-ratio: "16-9",
  config-common(handout: false),
  config-info(
    title: [Flash Attention 1],
    author: [William Arnold],
    date: datetime.today(),
    institution: [KAIST],
    logo: emoji.school,
  ),
)
#let cetz-canvas = touying-reducer.with(reduce: cetz.canvas, cover: cetz.draw.hide.with(bounds: true))
#let fletcher-diagram = touying-reducer.with(reduce: fletcher.diagram, cover: fletcher.hide)

#let smat(..args) = { math.mat(delim: "[", ..args) }
#let bvec(a) = { math.accent(math.bold(a), math.arrow)}
#title-slide()

== Overview

1. GPU Architecture
2. Attention Overview
3. Online Softmax
4. Flash Attention
5. Discussion

== GPU Architecture

#grid(columns: (auto, auto),
[
  #align(center)[#image("a100mem.jpg")]
], [
  #text(size: 20pt, [
  #pause
  SM: 'Stream Multiprocessor' #pause

  L1/SRAM: Fast and tiny #pause

  L2 fast-ish and small #pause

  HBM: Slow and big #pause

  How does an SM do compute?
  ])
]
)

== GPU Architecture

#grid(columns: (auto, auto),
[
  #align(center)[#image("sm.jpg")]
], [
  #text(size: 20pt, [
  #pause
  Each Warp is 32 executing 'threads' #pause

  Warps can use tensor cores #pause

  Each tensor core can do one 16x16x16 matrix multiply _every 16 cycles_ #pause

  That's $(16*16*16)/16 = 256$ FLOPs/cycle!#pause

  Threads/warps can communicate only through SRAM #pause

  Goal: break your problem into pieces of roughly `size(SRAM)` #pause

  Goal: keep the tensor cores fed
  
  ])
]
)

== GPU Architecture: example, matrix multiply

#grid(columns: (3fr, 3fr),
[
  #align(center)[#image("tiled.svg")]
], [
  #text(size: 20pt, [
    #pause
    Each tile is given to an SM #pause

    Each SM breaks the tile into 16x16x16 pieces #pause

    Each piece is assigned to a warp #pause

    Each warp does it's 16x16x16 matrix multiply, accumulating into SRAM #pause

    Otherwise, we'd have to go from  $"SRAM" ->  "HBM" -> "SRAM"$ for each 16x16x16 piece!

  ])
]
)

== GPU Architecture takeaways

#slide(repeat: 3, self => [
  #let (uncover, only, alternatives) = utils.methods(self)

  #only("1-")[
    1. Break your problem into pieces that can fit into SRAM
  ]

  #only("2-")[
    2. Use tensor cores whenever you can
  ]

  #only("3-")[
    3. Minimize read/writes to HBM
  ]

])

== Attention Overview
Queries, Keys, Values: $[S, D]$
$
Q = mat(delim: "[",
  dots.h, q_1, dots.h;
  dots.h, q_2, dots.h;
  dots.v, dots.v, dots.v;
  dots.h, q_H, dots.h;
),
K = mat(delim: "[",
  dots.h, k_1, dots.h;
  dots.h, k_2, dots.h;
  dots.v, dots.v, dots.v;
  dots.h, k_H, dots.h;
), 
V = mat(delim: "[",
  dots.h, v_1, dots.h;
  dots.h, v_2, dots.h;
  dots.v, dots.v, dots.v;
  dots.h, v_H, dots.h;
)
$
#pause

Output: $"Softmax"(Q K^T) V$ 
#pause

What does this look like in the GPU?

== Naive Attention Computation


#slide(repeat: 8, self => [
  #let (uncover, only, alternatives) = utils.methods(self)

  #for i in range(1, 8) {
    only(str(i))[ #image("naive/" + str(i) + ".jpg") ]
  }
  #only("8")[ #image("naive/7.jpg") ]
  #place(bottom + center,  dy: -1in,
    uncover("8")[Can we do less IO?]
   )
])

== Attention: Single query

Assume query is a single vector $bvec(q)$
$
bvec(a) = bvec(q) K^T pause &= smat(
  dots.h, bvec(q), dots.h;
)
smat(
  dots.v, dots.v, dots.h, dots.v;
  bvec(k)_2, bvec(k)_3, dots.h, bvec(k)_S;
  dots.v, dots.v, dots.h, dots.v;
  
)  \
pause
&= smat(
bvec(q) dot bvec(k)_1, bvec(q) dot bvec(k)_2, dots.h, bvec(q) dot bvec(k)_S;
) \
&= smat(
a_1, a_2, dots.h, a_S;
) \
pause 
"Softmax"(bvec(a)) &= smat((exp(a_1 - m)) / Z, (exp(a_2 - m))/Z, ... , (exp(a_S - m)) / Z) \
"where" m &= max_j a_j, Z = sum_(j=1)^S exp(a_j - m)

$
== Attention: Softmax scans $bvec(a)$ 3 times

1. Compute 
$
m = max_j a_j
$ 
#pause
2. Compute 
$
Z = sum_(j=1)^S exp(a_1 - m)
$
#pause
3. Compute
$
"Softmax"(a) = smat((exp(a_1 - m)) / Z, (exp(a_2 - m))/Z, ... , (exp(a_S - m)) / Z) \
$

== Attention: Softmax in two scans
We can do better! #pause

This is called "Online Softmax", published by NVIDIA #footnote[Online normalizer calculation for softmax, https://arxiv.org/abs/1805.02867]

#pause
Want to scan over $bvec(a)$ only once, computing $m, Z$ at the same time


#align(start)[$forall i in {1..S}$]
$
m_i &= max(m_(i-1), a_i) #pause arrow.l m_S "will be the max" \ #pause
Z_i &= sum_(j=1)^i e^(a_j - m_i) #pause arrow.l "requires summing everything up..." \ #pause
& "Can we express" Z_i "in terms of" Z_(i-1) "?" \ 
$ 



== Attention: Softmax in two reads
$
Z_i &= sum_(j=1)^i e^(a_j - m_i) pause \
    &= sum_(j=1)^(i-1) e^(a_j - m_i) + e^(a_i - m_i) "(pull out" i"-th term)" \ pause
    &= sum_(j=1)^(i-1) e^(a_j - bold(m_(i-1) + m_(i-1)) - m_i) + e^(a_i - m_i) \ pause
    &= (sum_(j=1)^(i-1) e^(a_j - m_(i-1)))e^(m_(i-1) - m_i) + e^(a_i - m_i) \ pause
    &= Z_(i-1)e^(m_(i-1) - m_i) + e^(a_i - m_i) 
$

== Attention: Softmax in two reads

1. #align(start)[$forall i in {1..S}$]
$
m_i &= max(m_(i-1), a_i) \ #pause
Z_i &= Z_(i-1)e^(m_(i-1) - m_i) + e^(a_i - m_i) \
$
#pause
2. 
$
"Softmax"(bvec(a)) = smat(dots.h, exp(a_i - m_S) / Z_S, dots.h)
$

== Attention in two read/writes
Can we do this _while we read $Q,K,V$_? #pause
$
smat(
  dots.h, bvec(q), dots.h;
)
smat(
  dots.v, dots.v, dots.h, dots.v;
  bvec(k)_1, bvec(k)_2, dots.h, bvec(k)_S;
  dots.v, dots.v, dots.h, dots.v;
  
) arrow.r smat(dots.h, bvec(x), dots.h)
$ #pause

$forall i in {1..S}$
$
x_i &= bvec(q) dot bvec(k_i) \ #pause
m_i &= max(m_(i-1), x_i) \ #pause
Z_i &= Z_(i-1)e^(m_(i-1) - m_i) + e^(a_i - m_i) \ 
$

== Attention in two read/writes

Now apply softmax _as we compute_ $"Softmax"(bvec(q)K^T)V$

#pause
$
smat(
  x_1, x_2, dots.h, x_S;
)
smat(
  dots.h, bvec(v)_1, dots.h;
  dots.h, bvec(v)_2, dots.h;
  dots.v, dots.v, dots.v;
  dots.h, bvec(v)_H, dots.h;
) arrow.r
smat(dots.h, bvec(o), dots.h)
$
#pause

$forall i in {1.. S}$
$
w_i &= exp(x_i - m_S) / Z_S \ pause
bvec(o_i) &= o_(i-1) + w_i bvec(v_i) <- "Final output at " bvec(o)_S
$


== Attention in two read/writes

Overall looks like 

$
forall i in {1..S} \
x_i &= bvec(q) dot bvec(k_i) <- "Write to hbm" \ 
m_i &= max(m_(i-1), x_i) \ 
Z_i &= Z_(i-1)e^(m_(i-1) - m_i) + e^(a_i - m_i) \ 

forall i in {1.. S} \

w_i &= exp(x_i - m_S) / Z_S <- "Read " x_i " from hbm" \ 
bvec(o_i) &= o_(i-1) + w_i bvec(v_i)
$


== Attention: can we get it in one?

#pause Problem: need $w_i$ to compute $bvec(o_i)$...

#pause Solution: Can we adjust $bvec(o_i)$ as we scan over everything?

#pause Try the same trick as before
#pause
$
bvec(o_i)' &:= sum_(j=1)^i exp(x_j - m_i) / Z_i bvec(v_j), "  replacing" m_S -> m_i, Z_S -> Z_i \ pause
           &= sum_(j=1)^(i-1) exp(x_j - m_i) / Z_i bvec(v_j) + exp(x_i - m_i) / Z_i bvec(v_i)\
$

== Attention: can we get it in one?

$
bvec(o_i)' &= sum_(j=1)^(i-1) exp(x_j - m_i) / Z_i bvec(v_j) + exp(x_i - m_i) / Z_i bvec(v_i) \ pause
           &= sum_(j=1)^(i-1) 
            exp(x_j - m_i) / Z_i 
            exp(x_j - m_(i-1)) / exp(x_j - m_(i-1)) 
            Z_(i-1) / Z_(i-1)
            bvec(v_j) + dots.h \ pause
            
           &= sum_(j=1)^(i-1) 
            exp(x_j - m_(i-1)) / Z_(i-1)
            exp(x_j - m_i) / exp(x_j - m_(i-1)) 
            Z_(i-1) / Z_i
            bvec(v_j) + dots.h \ pause

           &= (sum_(j=1)^(i-1) 
            exp(x_j - m_(i-1)) / Z_(i-1)
            bvec(v_j)
              )
            exp(m_(i-1) - m_i)
            Z_(i-1) / Z_i
             + dots.h \ pause

          &= bvec(o)'_(i-1)
            exp(m_(i-1) - m_i)
            (Z_(i-1) \/ Z_i)
            + (exp(x_i - m_i) \/ Z_i) bvec(v_i) 
          
$

== Attention: putting it all together

$forall i in {1..S}$
$
x_i &= bvec(q) dot bvec(k_i) \
m_i &= max(m_(i-1), x_i) \
Z_i &= Z_(i-1)e^(m_(i-1) - m_i) + e^(a_i - m_i) \
pause
bvec(o)'_i &= bvec(o)'_(i-1)
            e^(m_i - m_(i-1))
            Z_(i-1) / Z_i
            + e^(x_i - m_i) / Z_i bvec(v_i)
$
Once we hit $i=S$, $bvec(o)'_S$ has the correct output for $bvec(q)$!

== Multiple queries at once

#pause
#grid(columns: (2fr, 1fr), 
  image("Screenshot 2024-09-23 at 10.18.58 PM.png"),
  [
    #pause
    Pick \# of queries/keys that fit into sram #pause
 
    HBM usage scales with the number of $bvec(q)$ we compute at once, not with $S$ #pause

    Important for long sequence modeling! #pause
  ],
)

== Performance

#align(center)[
  #image("flash-attn-figure.png")
]
