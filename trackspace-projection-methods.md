# Projection methods for trackspace 2-D layout

## Recommendation

**The best fit is a frozen, weight-conditioned parametric projection—not simply a parametric model trained once on the current weighted vectors.** I would shortlist **ParamRepulsor and Parametric UMAP**, with a **t-SNE-distilled encoder** as another practical option.

However, **a smooth kernel extension of your existing t-SNE layout deserves a prototype before committing to neural-network training**. Your dismissal of the Nyström/kernel family is too strong: its weight Jacobian can be **exact, nonlinear in the weights, and particularly cheap with your additive squared-distance construction**.

Neither family automatically guarantees t-SNE-quality neighbourhood preservation across changing weights. That is the empirical acceptance criterion, rather than a property established by choosing an algorithm.

The critical distinctions are:

| Operation | Smooth movement? | Responds to the new metric correctly? |
|---|---|---|
| Independently refit a projection at each weight setting | Not reliably | Optimizes the new metric, but may switch solutions |
| Pass reweighted vectors through an encoder trained at one setting | Usually, subject to preprocessing | Not necessarily; the new inputs can be out of distribution |
| Pass reweighted vectors into an old nearest-neighbour-based transform | Not reliably | Generally compares against **old-weight reference vectors** |
| Use a frozen encoder **trained across weight settings** | Yes, by construction with appropriate activations | Can learn the required response; must validate |
| Recompute smooth affinities to fixed landmarks using the **new metric on both sides** | Yes | Correct new affinities, but a frozen reference geometry |

The last two are the most useful architectures for this application.

## 1. The distinction your requirements need to make explicit

Let the normalized source blocks be $u_{is}$, and write

$$
x_i(w)=\bigoplus_s \sqrt{w_s}\,u_{is}.
$$

Before any cross-source semantic correction,

$$
D_{ij}^2(w)=\sum_s w_s\|u_{is}-u_{js}\|^2.
$$

There are two different derivatives:

**Derivative of a frozen mapping**

$$
\frac{\partial}{\partial w_s}f_{\theta_0}(x_i(w)).
$$

**Derivative of the mapping you would learn again under the new metric**

$$
\frac{d}{dw_s}f_{\theta(w)}(x_i(w))
=
J_xf\,\frac{\partial x_i}{\partial w_s}
+
J_\theta f\,\frac{\partial\theta}{\partial w_s}.
$$

An ordinary frozen parametric model gives the first term—not the second. This is entirely useful for interaction, but **differentiability alone does not establish that the movement preserves the newly requested neighbourhoods**.

Parametric UMAP explicitly learns a neural mapping instead of optimizing each point’s coordinates independently; parametric t-SNE makes the analogous substitution for a neighbourhood-preservation objective. Neither construction, by itself, means the model has learned how to respond to previously unseen source-weight settings. ([Parametric UMAP documentation](https://umap-learn.readthedocs.io/en/latest/parametric_umap.html))

### A separate problem with an old reference index

Passing newly weighted vectors into an existing transform typically evaluates a source contribution like

$$
\left\|\sqrt{w_s}\,u_{is}-\sqrt{w_{0s}}\,u_{js}\right\|^2,
$$

whereas the requested metric is

$$
w_s\|u_{is}-u_{js}\|^2.
$$

Those are different quantities. **Freeze the reference’s 2-D coordinates, not its high-dimensional metric weights.**

This is why a purpose-built weight-aware extension is materially different from repeatedly calling `.transform()` on an estimator fitted at $w_0$.

## 2. Candidate assessment

Here, “smooth” refers to changing source weights while retaining the fitted projection artifact—not to independently retraining it.

| Method | Frozen out-of-sample insertion | Cheap weight derivative | Assessment for this application |
|---|---|---|---|
| **ParamRepulsor / ParamPaCMAP** | Yes, through an encoder | Automatic differentiation | **Strong candidate**, especially with weight-conditioned training. Targets a documented weakness of parametric methods: lost local detail. |
| **Parametric UMAP** | Yes, through an encoder | Automatic differentiation | **Strong baseline.** Needs the same weight-conditioning treatment. |
| **Parametric t-SNE / t-SNE-distilled encoder** | Yes, through an encoder | Automatic differentiation | Attractive when matching your current t-SNE presentation is important. Distillation and direct parametric t-SNE are different training objectives. |
| **ivis** | Yes, through an encoder | Automatic differentiation | Valid candidate, but ivis is a **triplet-trained Siamese method**, not inherently an autoencoder or t-SNE implementation. |
| **PaCMAP / LocalMAP** | Implemented through inherited or direct transform machinery | No robust, ready-made derivative of metric-dependent layouts | Useful layout generators; **not a drop-in fix for smooth sliders**. |
| **TriMap** | No native out-of-sample transform in the standard `TRIMAP` implementation inspected | Not provided | Poor match without adding a separate mapping layer. |
| **Ordinary UMAP** | Yes | Transform includes neighbour selection and optimization | Good insertion support, but not the smooth weight-aware function you need. |
| **UMAP/t-SNE + smooth landmark extension** | Yes | **Closed-form** for suitable kernels | **Strong low-engineering-cost prototype**, especially for local dragging. |
| **Diffusion maps + Nyström** | Yes | Closed-form with frozen spectral quantities and a smooth kernel | Mechanically suitable; 2-D microgenre fidelity is the main risk. |
| **PHATE** | Interpolation path exists, with a warning in the implementation | Possible for a controlled extension, not the full refit pipeline | Lower priority: extra diffusion/MDS machinery without solving the central issue. |

The neural capabilities are supported by the ParamRepulsor implementation, Parametric UMAP documentation, and ivis API. ([ParamRepulsor implementation](https://raw.githubusercontent.com/hyhuang00/ParamRepulsor/main/src/parampacmap/parampacmap.py)) PaCMAP exposes `transform`, and `LocalMAP` inherits from PaCMAP; the standard TriMap implementation inspected has no corresponding transform method. ([PaCMAP implementation](https://raw.githubusercontent.com/YingfanWang/PaCMAP/master/source/pacmap/pacmap.py)) UMAP’s transform performs neighbour lookup and layout optimization rather than a simple analytical mapping. ([UMAP implementation documentation](https://umap-learn.readthedocs.io/en/latest/_modules/umap/umap_.html))

A detail worth catching: PHATE’s documentation says it does not embed unseen observations, but its implementation contains an `extend_to_data` → `interpolate` path, accompanied by a warning discouraging transformation of a new data matrix. I would not classify it as either “no transform whatsoever” or “unqualified production-quality transform support.” ([PHATE implementation documentation](https://phate.readthedocs.io/en/stable/_modules/phate/phate.html))

### Why add ParamRepulsor?

Its NeurIPS 2024 paper directly investigates the assumption that parametric versions preserve local structure like their nonparametric counterparts. The authors find substantial local-detail losses in their experiments and introduce stronger repulsion and hard-negative mining to address them. That is unusually relevant to your microgenre requirement. It does **not** demonstrate parity with t-SNE on your music features. ([ParamRepulsor paper](https://arxiv.org/abs/2411.15894))

The reference implementation is PyTorch-based and explicitly provides CPU, CUDA and Apple M-series/MPS installation options. That makes it a more immediately compatible research starting point than adding TensorFlow solely for ivis or the reference Parametric UMAP implementation. ([ParamRepulsor README](https://github.com/hyhuang00/ParamRepulsor/blob/main/README.md))

## 3. Best long-term architecture: a weight-conditioned encoder

I would define the layout as

$$
y_i(w)=F_\theta(z_i,w),
$$

where $z_i$ contains fixed per-source representations, and **$\theta$ stays frozen during both dragging and insertion**.

The important addition is training over weight settings:

$$
\mathcal L(\theta)
=
\mathbb E_{w\sim Q}
\left[
\mathcal L_{\text{neighbourhood}}
\big(F_\theta(Z,w),P(w)\big)
\right]
+
\lambda\,\mathcal L_{\text{stability}}.
$$

Here $P(w)$ describes neighbourhoods under your requested weighted metric. This is a proposed adaptation of the parametric methods, **not an existing flag in those packages**.

### What training should actually do

For each sampled weight setting, train attraction and repulsion against neighbourhoods appropriate to that setting. Use the **same global weight setting for both endpoints of each training pair or triplet**.

Do not train once against a graph at $w_0$, vary the encoder inputs, and assume that this teaches metric adaptation. It mostly teaches robustness to input changes while retaining the old graph.

A practical training distribution would cover normal slider combinations, source-ablation settings, and local perturbations around common presets. A deterministic union of candidate neighbours from multiple sources/settings can reduce graph-building work, but its recall must be measured: it is not guaranteed to contain every important neighbour throughout the weight cube.

For spatial consistency, use one jointly trained model and optionally a modest reference-layout penalty. Independent teacher layouts that are merely Procrustes-aligned can still contain incompatible cluster rearrangements; aligning rotation and translation does not fix those.

### t-SNE distillation is also useful

You can train an encoder against an existing t-SNE layout:

$$
\mathcal L_{\text{distill}}
=
\sum_i\|F_\theta(z_i,w_0)-y_i^{\text{tSNE}}\|^2.
$$

That gives a convenient starting chart, but it is only supervision at $w_0$. To obtain useful slider behaviour, add metric-dependent neighbourhood training across weights, or carefully constructed, mutually consistent targets at additional settings.

I would start with **coordinate distillation plus a neighbourhood objective**, rather than require a decoder. Reconstruction is optional machinery, not a prerequisite for a differentiable projection.

### Jacobian and frame cost

For a frozen conditional encoder,

$$
J_{is}=\frac{\partial F_\theta(z_i,w)}{\partial w_s}.
$$

There are only five input directions of interest. Use five Jacobian-vector products, or a batched per-track Jacobian—not a Jacobian with respect to every coordinate of the entire feature matrix. PyTorch exposes forward-mode JVPs directly. ([PyTorch `torch.func.jvp` documentation](https://docs.pytorch.org/docs/2.12/generated/torch.func.jvp.html))

Then render

$$
Y_{\text{preview}}(w_0+\Delta w)
=
Y(w_0)+J(w_0)\Delta w.
$$

At $N=20{,}000$, $S=5$, float32 storage for that Jacobian is only

$$
20{,}000\times2\times5\times4
=
800{,}000\text{ bytes}.
$$

**The drag frame can therefore be a tiny array operation.** The expensive operation is refreshing the Jacobian, which can follow your separate “a few seconds” budget.

For implementation, call the underlying network with gradients enabled. ParamRepulsor’s public transform currently goes through `torch.inference_mode()`, so that wrapper is not the function to differentiate directly. ([ParamRepulsor implementation](https://raw.githubusercontent.com/hyhuang00/ParamRepulsor/main/src/parampacmap/parampacmap.py))

## 4. The kernel alternative is stronger than your brief suggests

A Nyström extension is linear in its **kernel features or reference coefficients**, not generally linear in the original input or in source weights. The Gaussian-kernel diffusion-map extension is a concrete example. ([Diffusion-map Nyström documentation](https://search.r-project.org/CRAN/refmans/diffusionMap/html/nystrom.html))

Your source decomposition makes a particularly convenient smooth extension possible.

### An explicit construction

Choose fixed reference tracks or landmarks $\ell$, and cache

$$
\Delta_{i\ell s}
=
\|u_{is}-u_{\ell s}\|^2.
$$

Define

$$
D_{i\ell}^2(w)
=
\sum_s w_s\Delta_{i\ell s},
$$

and smooth normalized affinities

$$
a_{i\ell}(w)
=
\frac{\exp(-D_{i\ell}^2(w)/\tau)}
{\sum_m\exp(-D_{im}^2(w)/\tau)}.
$$

Let

$$
g_i(w)=\sum_\ell a_{i\ell}(w)c_\ell.
$$

The fixed $c_\ell\in\mathbb R^2$ can be landmark coordinates from t-SNE/UMAP, or fitted regression coefficients. For a true diffusion-map Nyström extension, the coefficients incorporate the appropriate eigenfunction/eigenvalue factors.

**Calling this a “smooth landmark extension of t-SNE/UMAP” is more accurate than calling it “Nyström UMAP.”** Nyström has a specific spectral meaning; arbitrary interpolation of optimized UMAP coordinates is not automatically that method.

### Exact derivative

With fixed $\tau$, reference coefficients and support,

$$
\boxed{
\frac{\partial g_i}{\partial w_s}
=
-\frac{1}{\tau}
\sum_\ell
a_{i\ell}(w)\,
\Delta_{i\ell s}
\big(c_\ell-g_i(w)\big)
}
$$

This follows directly by differentiating the normalized exponential weights.

It is an **exact derivative of this nonlinear extension**, not a finite-difference approximation. It also avoids differentiating $\sqrt{w_s}$, because the squared distances are already affine in $w$.

I checked this expression against central differences on a small synthetic normalized-block example; the relative derivative error was approximately $3\times10^{-10}$. That is a correctness check of the formula, **not** a music-quality or Apple Silicon performance benchmark.

### What it gives you

Once distances are cached, evaluating all points and all source derivatives costs approximately $O(NMS)$, with $M$ reference candidates per point. No repeated pass through all 1,819 features is necessary for existing tracks.

New tracks require their source-to-reference distances, then the same formula. Existing points remain untouched.

The full nonlinear map can also be evaluated again after a drag, rather than relying indefinitely on the first-order approximation.

### What it does not give you

**Frozen reference coordinates cannot freely reorganize their own geometry to match a radically different metric.** Barycentric interpolation can average across separate islands, shrink structure, and underrepresent rare microgenres without suitable references.

Thus I see two uses:

**Local drag preview.** Preserve the exact original layout with a residual:

$$
\widetilde y_i(w)
=
y_i^0+g_i(w)-g_i(w_0).
$$

This guarantees $\widetilde y_i(w_0)=y_i^0$, with an analytical movement field. It is an honest local surrogate, not a claim to have reoptimized t-SNE.

**Standalone projection.** Fit coefficients or a richer local regression to reproduce the reference chart, then validate both held-out insertion and weight changes. This could be sufficient without a neural model, but the fidelity question remains empirical.

For either use, **do not rebuild hard top-$k$ support at every perturbation**. Use all selected landmarks or a sufficiently broad frozen candidate set. Hard support changes reintroduce discontinuities; frozen support trades some adaptivity for smoothness.

Also, when finite-differencing any transform, subtract its own baseline:

$$
g(w_0+h)-g(w_0),
$$

not $g(w_0+h)-Y_{\text{original fit}}$, unless those baselines are known to match.

## 5. t-SNE itself is not mathematically undifferentiable

Your experiment is strong evidence against **independent-refit finite differences**, but non-convexity does not imply that every local solution branch lacks a derivative.

For a suitably regular, converged solution of

$$
\nabla_Y L(Y,w)=0,
$$

implicit differentiation gives

$$
H\,\frac{\partial Y}{\partial w_s}
=
-\frac{\partial^2L}{\partial Y\,\partial w_s},
$$

after handling coordinate symmetries and other degeneracies.

A recent research paper applies this approach to t-SNE, explicitly addressing the Hessian’s translation/rotation null space. Its full-matrix implementation has a substantial computational bottleneck; the authors discuss iterative products as a direction for scaling. ([Frontiers in Bioinformatics paper](https://www.frontiersin.org/journals/bioinformatics/articles/10.3389/fbinf.2025.1719516/full))

For your problem, only five right-hand sides are needed, so a matrix-free, regularized sensitivity solver is more plausible than constructing the paper’s full input Jacobian. Nevertheless, it remains a research implementation with conditioning and branch-stability problems.

**I would rank it behind conditional neural projection and the kernel prototype**, not dismiss it as mathematically impossible.

Similarly, `openTSNE.transform` is not globally guaranteed continuous: its API uses nearest-neighbour-based initialization and iterative placement. It keeps reference coordinates fixed and ignores interactions between newly inserted points, which is useful for insertion but does not establish smooth metric adaptation. ([openTSNE API documentation](https://opentsne.readthedocs.io/en/stable/api/index.html))

## 6. Constraints that need handling regardless of algorithm

### Zero weights

For your current input parameterization,

$$
\frac{\partial x_i}{\partial w_s}
=
\frac{u_{is}}{2\sqrt{w_s}}
$$

within the relevant block. This is singular at zero.

The map can remain continuous, but a generic encoder can have a very large slope near source activation. ReLU networks add their own derivative boundaries.

Two clean approaches are:

- Differentiate in amplitude coordinates $a_s=\sqrt{w_s}$, with the UI interpretation made explicit.
- Use a conditional encoder with smooth dependence on $w$, while retaining $\sum_s w_s\Delta_s$ as the metric defining training targets.

For the second, a gated representation such as $[w_s z_{is}]_s$, together with $w$, also prevents a zero-weight source from leaking directly into the output. The gating is an encoder design choice; it does not redefine the target distance.

All-zero weights produce no meaningful neighbourhood geometry. Reject that state or define a deterministic fallback.

### PCA and semantic transforms are part of the derivative

A differentiable final stage does not make a pipeline smooth if you independently recompute PCA, folder directions, landmark choices or normalization statistics at every setting.

Freeze those fitted quantities within a **chart revision**, or explicitly account for their changes.

There is an additional issue with your rank-$k$ semantic correction: the additive distance shortcut survives a block-separable correction, but **not generally a cross-source transform**.

For a fixed linear transform $A$,

$$
D_A^2(w)=
\big(x_i(w)-x_j(w)\big)^\top
A^\top A
\big(x_i(w)-x_j(w)\big).
$$

Cross-block entries introduce terms proportional to $\sqrt{w_sw_t}$. Cache/evaluate those low-rank terms or differentiate the actual transform; do not silently keep using only $\sum_s w_s\Delta_s$.

Also avoid refitting per-feature standardization after applying source weights: for positive weights, dividing by each newly weighted column’s standard deviation can cancel the intended source scaling.

### Determinism needs a reference-artifact identity

“Same current dataset and weights” is not enough if one layout came from a fresh fit and another came from inserting points into an older chart.

A workable contract is

```text
chart_revision =
  hash(reference data, preprocessing, semantic transform,
       model/landmark artifact, algorithm configuration)

layout_revision =
  hash(chart_revision, sorted track-feature revisions,
       source weights, transform configuration)
```

A layout produced by insertion can then be deterministic **conditional on the same chart**, while the original coordinates remain fixed.

Persist the chart artifact beyond the layout LRU. Otherwise, eviction followed by a fresh fit can change the meaning of a supposedly identical layout request.

**Do not derive the random seed from the changing weight-dependent layout hash.** That would introduce a completely new random stream for each tiny slider movement.

Exact reproducibility also needs a defined runtime scope. PyTorch does not promise identical results across versions, platforms or CPU/GPU execution; UMAP documents nondeterminism from multithreaded optimization. Pin the environment and use deterministic execution where required, rather than treating a seed as sufficient. ([PyTorch reproducibility documentation](https://docs.pytorch.org/docs/2.12/notes/randomness.html))

### Fixed insertion has an unavoidable adaptation limit

A frozen chart can place genuinely new tracks, but cannot automatically reserve new space for an entirely new microgenre by moving old islands.

Treat **insertion** and **explicit chart rebuilding** as different operations. Neither a neural encoder nor Nyström removes this distinction.

## 7. Scale and compute budget

These are operation-count and storage estimates, not measured timings on your machine.

### Shortlisted serving paths

| Design at 20k tracks | Main work for all-point evaluation | Important storage/work |
|---|---|---|
| Full-input MLP, $1819\to128\to64\to2$ | About **4.82 billion multiply-accumulates** | Raw float32 features: about **145.5 MB** |
| Compressed conditional MLP, $133\to128\to64\to2$ | About **0.51 billion multiply-accumulates** | Five JVP directions add several forward-equivalent passes |
| Kernel extension, 512 landmarks, five sources | $O(NMS)$ after distance caching | Source distances: about **204.8 MB** |
| Kernel extension, 128 frozen candidates per track | $O(NMS)$ after distance caching | Source distances: about **51.2 MB**, plus indices |
| Cached Jacobian preview | $O(2NS)$ | **0.8 MB** Jacobian |

A fixed per-source compression or cached learned source representation can therefore make a substantial difference. Validate its neighbour recall first: a projection cannot recover microgenre information discarded by its input compression.

### Cold fitting

For the neural candidates, training cost is graph construction plus repeated network evaluation over sampled pairs/triplets. Training across weights increases that work; it is not a cheap refit operation.

For PaCMAP, LocalMAP, TriMap and UMAP, sparse pair/graph optimization can scale well with $N$, but the high-dimensional neighbour search remains important. Do not assume an $O(N\log N)$ guarantee for approximate search in this feature space.

For landmark diffusion methods, a straightforward dense implementation moves the expensive spectral work to an $M\times M$ problem, with $O(M^2)$ storage and potentially $O(M^3)$ decomposition work. PHATE additionally performs diffusion-potential and MDS operations; its implementation uses matrix powers of the diffusion operator. ([PHATE implementation documentation](https://phate.readthedocs.io/en/stable/_modules/phate/phate.html))

**Nothing here justifies predicting cold-fit timings from your existing t-SNE numbers.** The useful conclusion is that the shortlisted serving paths eliminate refitting from interaction altogether.

### CLAP/MPS contention

My default deployment would be **CPU serving for projection and Jacobian refreshes**, with optional MPS training performed separately from interactive inference.

Avoiding your Python `inference_lock` does not prevent two MPS workloads from competing for the same hardware. A separate CPU projection worker gives a cleaner isolation boundary than choosing another Metal library and assuming contention disappears.

## 8. Evaluation that would settle the choice

I would compare three implementations first:

| Prototype | Purpose |
|---|---|
| Existing openTSNE layout + smooth kernel/residual extension | Establish the cheapest viable slider behaviour |
| Weight-conditioned ParamRepulsor | Test stronger parametric local preservation on CPU/MPS |
| Weight-conditioned Parametric UMAP or t-SNE-distilled encoder | Establish whether the simpler/reference objective is sufficient |

Measure four things independently.

**Neighbourhood quality.** Compare high-dimensional versus 2-D neighbour recall and trustworthiness at several scales, such as $k=10,30,100$. Evaluate both your intended weighted distance and the actual PCA/cosine geometry used by clustering where they differ. Leiden-label purity can be secondary; it cannot replace within-microgenre neighbourhood preservation.

**Weight generalization.** Hold out weight combinations, not just tracks. Include source-off settings, small local changes and large changes that genuinely alter nearest neighbours. A stationary map is perfectly smooth but fails this test.

**Derivative correctness.** Compare $J\Delta w$ against differences of the **same frozen mapping** at decreasing step sizes. For a smooth implementation, first-order residuals should decrease quadratically until numerical error dominates. Separately measure whether the resulting neighbourhood changes are useful.

**Operational behaviour.** Test 10k and 20k tracks, insertion batches of a few dozen, row permutations, cold reloads and different insertion groupings. Verify old coordinate bytes remain unchanged on insertion, and measure complete latency including preprocessing and data transfer.

## Bottom line

**Yes: conditional parametric projections and smooth kernel/landmark extensions can provide the computational and continuity properties you need.** Their microgenre fidelity across the relevant weight range must be demonstrated on your data.

My implementation order would be:

1. **Prototype a weight-aware smooth extension around the existing t-SNE chart.** It exploits your additive distances and gives an exact, cheap Jacobian.
2. **Evaluate a weight-conditioned ParamRepulsor encoder** for a projection that can learn larger metric-dependent reorganizations.
3. Keep Parametric UMAP or t-SNE distillation as the comparison—not ordinary PaCMAP/LocalMAP refitting.

The two changes I would make to the research brief are: **“parametric” must include training for weight variation**, and **“kernel/Nyström” must not be dismissed as merely a linear approximation**. Those distinctions matter more here than which nonparametric algorithm generates the initial chart.
