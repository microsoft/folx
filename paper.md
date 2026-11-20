---
title: 'folx: Forward Laplacian for JAX'
tags:
  - Python
  - JAX
  - automatic differentiation
  - Laplacian
  - variational Monte Carlo
  - neural wave functions
authors:
  - name: Nicholas Gao
    orcid: 0000-0002-1487-2657
    affiliation: "1, 3, 4"
  - name: Adam Foster
    orcid: 0000-0003-1624-7481
    affiliation: "2"
  - name: P. Bernát Szabó
    orcid: 0000-0003-1824-8322
    affiliation: "2"
  - name: Jonas Köhler
    orcid: 0000-0002-7256-2892
    affiliation: "1, 4"
  - name: Frank Noé
    orcid: 0000-0003-4169-9324
    affiliation: "2"
  - name: Jan Hermann
    orcid: 0000-0002-2779-0749
    affiliation: "2"
affiliations:
  - name: CuspAI
    index: 1
  - name: Microsoft Research AI for Science
    index: 2
  - name: Work done while at Technical University of Munich
    index: 3
  - name: Work done while at Microsoft Research AI for Science
    index: 4
date: 13 November 2026 # TODO: submission date
bibliography: resources/paper.bib
---

# Summary

The Laplacian $\Delta f = \sum_{i=1}^n \partial^2 f / \partial r_i^2$ of a function $f:\mathbb{R}^n \to \mathbb{C}$ measures how much a function deviates from its local average and is a critical component in computational simulations of partial differential equations.
In modern auto differentiable frameworks, one computes the Laplacian one input at a time, i.e., with $n$ computational passes, one for each row of the Hessian.
We present `folx`, a Python library for computing exact Laplacians of JAX [@bradburyJAXComposableTransformations2018] programs.
Following the forward Laplacian algorithm [@liComputationalFrameworkNeural2024], we decompose a program into elementary operations $f = f_L \circ \dots \circ f_1$ and propagate the value $f(x)$, the Jacobian $\nabla f(x) = (\partial f / \partial x_1, \dots, \partial f / \partial x_n)$, and the Laplacian $\Delta f(x)$ through one operation at a time.
`folx` applies to unmodified real- or complex-valued functions with arbitrary outputs and automatically exploits Jacobian sparsity at compile time.

![Time per Laplacian for $N$ electrons of (a) $\log|\psi|$ of a FermiNet-style wave function and (b) a deep set $\rho(\sum_i \phi(\mathbf{r}_i))$ with a per-electron MLP $\phi$ (batches of 512 samples, float32, NVIDIA RTX 5070 Ti, see `resources/scripts/ferminet.py` and `deepset.py`). Dense `folx` represents dense implementations such as fwdlap.\label{fig:scaling}](resources/figures/scaling.png){width="100%"}

# Statement of need
Neural wave functions are trained by minimizing the expectation of the local energy $E_L:\mathbb{R}^{N \times 3} \to \mathbb{C}$ [@pfauInitioSolutionManyelectron2020; @hermannDeepneuralnetworkSolutionElectronic2020; @hermannAbinitioQuantumChemistry2022]
\begin{equation}
E_L(\mathbf{r}) = -\frac{1}{2}\frac{\Delta \psi(\mathbf{r})}{\psi(\mathbf{r})} + V(\mathbf{r})
\label{eq:local-energy}
\end{equation}
over samples of the positions $\mathbf{r} \in \mathbb{R}^{N \times 3}$ of $N \in \mathbb{N}$ electrons, where $\psi:\mathbb{R}^{N \times 3} \to \mathbb{C}$ is the wave function and $V:\mathbb{R}^{N \times 3} \to \mathbb{R}$ the potential energy [@foulkesQuantumMonteCarlo2001].
The kinetic term in \autoref{eq:local-energy} requires $\Delta \psi$ for every sample at every optimization step, which dominated the cost of neural-network VMC.
@liComputationalFrameworkNeural2024 introduced the *forward Laplacian*, which propagates the value, Jacobian, and Laplacian jointly, and enabled sparsity propagation, accelerating neural-network VMC by more than 20$\times$.
However, their implementation requires a wrapped array API, and sparsity has to be handled manually.

With `folx`, we provide the forward Laplacian as a functional in the spirit of `jax.grad` or `jax.vmap` that applies to any JAX code, and we exploit Jacobian sparsity while remaining exact.
Beyond the kinetic energy in VMC, `folx` applies wherever Laplacians appear, e.g., in pseudo-Hamiltonians [@fuEmpoweringNeuralNetworkbased2025], high-dimensional physics-informed neural networks [@raissiPhysicsinformedNeuralNetworks2019; @shiStochasticTaylorDerivative2024], PDE solvers [@jnoCode2026], and stochastic differential equations [@zhaoTME2021].
On FermiNet-style wave functions, `folx` is 1.7–6.7$\times$ faster than a series of Hessian-vector products, and on a deep set over electrons up to 190$\times$ (\autoref{fig:scaling}).
Sparsity accounts for up to 3.4$\times$ and 70$\times$ of these speedups, respectively.

# State of the field
When we started `folx`, the only forward Laplacian implementation was LapJAX [@lapjax2023], the implementation of @liComputationalFrameworkNeural2024, which wraps a subset of `jax.numpy` to accept tuples of value, Jacobian, and Laplacian, such that frameworks like Flax or Haiku require dependency injection, and sparsity has to be handled manually.
Since this wrapped array API caused too many integration problems with existing JAX code, we built `folx` as a new package that traces programs instead of extending LapJAX.

Alternatively, automatic differentiation frameworks such as JAX or PyTorch [@paszkePyTorchImperativeStyle2019] trace programs and compute Laplacians exactly from the full Hessian.
Later, @dangelCollapsingTaylorMode2025 showed that collapsing Taylor mode differentiation [@bettencourtTaylormodeAutomaticDifferentiation2019] recovers the forward Laplacian and extends it to other linear differential operators.
Developed concurrently with `folx`, fwdlap [@chenFwdlap2023] implements the forward Laplacian as a custom JAX tracer analogous to `jax.jvp` and thus applies to unmodified JAX code, but cannot propagate sparsity.
To date, no other package provides automatic Jacobian sparsity propagation, where most of the complexity of `folx` lies.

# Software design

We decompose a twice-differentiable $f = f_L \circ \dots \circ f_1$ into elementary operations $f_l:\mathbb{R}^{m_{l-1}} \to \mathbb{R}^{m_l}$ with intermediates $\mathbf{x}_l = f_l(\mathbf{x}_{l-1})$, starting from the input $\mathbf{x}_0 \in \mathbb{R}^n$.
Given the Jacobian $\nabla \mathbf{x}_{l-1} \in \mathbb{R}^{m_{l-1} \times n}$ and Laplacian $\Delta \mathbf{x}_{l-1} \in \mathbb{R}^{m_{l-1}}$ of an intermediate with respect to the input, we obtain those of the next intermediate via
\begin{equation}
\nabla \mathbf{x}_l = J_{f_l}(\mathbf{x}_{l-1}) \, \nabla \mathbf{x}_{l-1},
\quad
\Delta \mathbf{x}_l = J_{f_l}(\mathbf{x}_{l-1}) \, \Delta \mathbf{x}_{l-1} + \operatorname{tr}\left(\nabla \mathbf{x}_{l-1}^\top \, H_{f_l}(\mathbf{x}_{l-1}) \, \nabla \mathbf{x}_{l-1}\right),
\label{eq:fwd-lapl}
\end{equation}
where $J_{f_l}(\mathbf{x}_{l-1}) \in \mathbb{R}^{m_l \times m_{l-1}}$ and $H_{f_l}(\mathbf{x}_{l-1}) \in \mathbb{R}^{m_l \times m_{l-1} \times m_{l-1}}$ denote the Jacobian and Hessian of $f_l$ with respect to its own input $\mathbf{x}_{l-1}$, and the trace is taken per output component [@liComputationalFrameworkNeural2024].
Starting from $\nabla \mathbf{x}_0 = I \in \mathbb{R}^{n \times n}$ and $\Delta \mathbf{x}_0 = 0 \in \mathbb{R}^n$, the final triplet $(\mathbf{x}_L, \nabla \mathbf{x}_L, \Delta \mathbf{x}_L)$ equals $(f, \nabla f, \Delta f)$ at $\mathbf{x}_0$.
Even without sparsity, this is $\approx 2\times$ faster than $n$ Hessian-vector products due to the symmetry of $H_{f_l}$, we evaluate only half of the second-order terms [@liComputationalFrameworkNeural2024].
We treat complex arrays as pairs of real arrays and flatten arbitrarily shaped arrays and pytrees into vectors.

![Forward Laplacian evaluation in `folx`. (a) We trace an unmodified JAX function into (b) its jaxpr. (c) For each equation, we map its input triplets of value, Jacobian, and Laplacian to a new triplet via its forward Laplacian rule, starting from $(\mathbf{r}, I, 0)$, where $J_v$ and $H_v$ denote the Jacobian and Hessian of the primitive computing $v$ with respect to its input (\autoref{eq:fwd-lapl}), which are diagonal for elementwise functions. Rules are derived from annotations (blue) or implemented as custom rules (orange).\label{fig:design}](resources/figures/design.png){width="100%"}

**Functional.**
In our primary entry point `forward_laplacian(f)`, we trace `f` into a jaxpr (\autoref{fig:design}), an intermediate representation of the computational graph in terms of a set of primitives, and evaluate it with a custom interpreter that computes the triplet $(g(\mathbf{x}), \nabla g(\mathbf{x}), \Delta g(\mathbf{x}))$ for each primitive $g$.
Since every JAX program traces to the same primitives, we support models written in, e.g., Flax or Haiku without modification.
Unlike JAX, we perform constant folding at this stage such that indices are statically known for sparsity propagation.
Since the result is again a JAX function, `folx` composes freely with all of JAX's transformations like `jax.jit`, `jax.vmap`, and `jax.vjp`.

**Compile-time sparsity.**
Materializing full Jacobians increases memory and compute by up to a factor of $n$.
However, in many functions intermediates only depend on a known subset of the input, e.g., in neural wave functions, early intermediates depend on only three of the $n$ coordinates.
We therefore store the Jacobian of every intermediate $\mathbf{x} \in \mathbb{R}^m$ either densely, $\nabla \mathbf{x} \in \mathbb{R}^{m \times n}$, or sparsely (\autoref{fig:sparsity}a).
A sparse Jacobian consists of values $D_{\mathbf{x}} \in \mathbb{R}^{m \times d}$ and static input indices $\mathcal{I}_{\mathbf{x}} \in \{-1, 0, \dots, n-1\}^{m \times d}$, where $d \ll n$ is the maximum number of inputs that any entry of $\mathbf{x}$ depends on, and we pad rows with fewer dependencies with index $-1$ and zero values.
The dense Jacobian follows as $(\nabla \mathbf{x})_{ij} = \sum_{l=0}^{d-1} (D_{\mathbf{x}})_{il} \, \delta_{j, (\mathcal{I}_{\mathbf{x}})_{il}}$.
We determine these indices once during tracing by propagating them through every operation (\autoref{fig:sparsity}b–d), such that the compiled program only contains dense operations on $D_{\mathbf{x}}$.

![Compile-time sparsity for four electrons with three coordinates each, colored by electron. (a) A sparse Jacobian $\nabla \mathbf{x}$ ($m = 4$, $n = 12$) is stored as values $D_{\mathbf{x}}$ ($d = 3$) and static indices $\mathcal{I}_{\mathbf{x}}$. (b) Elementwise functions preserve the pattern. (c) For concatenations, we stack rows and pad them to the largest $d$. (d) For reductions, we merge the index sets of the reduced entries.\label{fig:sparsity}](resources/figures/sparsity.png){width="100%"}

Users enable sparsity with a threshold $d_{\max}$ that bounds $d$, e.g., $d_{\max} = 6$ in \autoref{fig:scaling}.
Once an intermediate depends on more than $d_{\max}$ inputs, or data-dependent indexing prevents static analysis, we automatically switch to dense Jacobians.
For functions applied independently to each of $N$ electrons, this reduces the time and memory complexity by a factor of $\mathcal{O}(N)$.
For remaining memory limits, we process large batches in sequential chunks with `folx.batched_vmap`, trading parallelism for memory.

**Propagation rules.**
Instead of hand-deriving \autoref{eq:fwd-lapl} for each primitive, we obtain its Jacobian and Hessian terms from JAX's automatic differentiation and only annotate its structure (\autoref{fig:design}c).
Annotations such as elementwise, linear, or product structure simplify the Hessian term and determine how sparsity patterns change (\autoref{fig:sparsity}).
This way, we cover 94 JAX primitives and 9 named functions such as `slogdet` (`folx` 0.2.31) and fall back to a generic dense rule for all others.
Users can extend `folx` at three levels: annotating a new function, replacing only its Hessian term, or registering a rule that maps input triplets $(\mathbf{x}, \nabla \mathbf{x}, \Delta \mathbf{x})$ directly to output triplets, as DeepQMC does for attention.

# Research impact statement

**Integrations.**
23 GitHub repositories depend on `folx`, including FermiNet [@spencerFermiNet2020], DeepQMC [@schatzleDeepQMCOpensourceSoftware2023], DeepErwin [@scherbelaSolvingElectronicSchrodinger2022], Neural Pfaffians [@gaoNeuralPfaffiansSolving2024], Excited Pfaffians [@gaoExcitedPfaffiansGeneralized2026], Orbformer [@fosterInitioFoundationModel2025], STDE [@stdeCode2024], and jNO [@jnoCode2026].

**Publications.**
`folx` has seen broad adoption in neural-network VMC, with at least 14 papers using it since 2023.
Applications include excited states [@pfauAccurateComputationQuantum2024; @schatzleAbinitioSimulationExcitedstate2025; @gaoExcitedPfaffiansGeneralized2026; @jeongVariationalMonteCarlo2026], generalized and foundation-model wave functions [@gaoNeuralPfaffiansSolving2024; @fosterInitioFoundationModel2025], large molecules [@scherbelaAccurateAbinitioNeuralnetwork2025], sign equivariance [@gaoRepresentingElectronicWave2024], high-pressure hydrogen [@linteauNeuralWaveFunctions2025], electron–nuclear wave functions [@zhangSchrodingerNetUniversalNeural2025; @zhangSmartMachineLearning2026], anyons [@perezfadonAnyonicNeuralNetwork2026], normalizing flows [@cochranEquivariantContinuousNormalizing2026], lattice field theory [@bedaqueNeuralWavefunctionsQuantum2026], and automatic differentiation research [@dangelCollapsingTaylorMode2025].

**Community.**
As of November 2026, `folx` has 32 versions on PyPI since December 2023, receives about 15,000 downloads per month, and has 110 stars and 25 forks on GitHub.
Thirteen users outside the original developers have opened 22 issues and pull requests, and nine developers from the Technical University of Munich, CuspAI, Microsoft, Freie Universität Berlin, EPFL, and École Polytechnique have contributed code, including fixes for new JAX releases.

# AI usage disclosure

The software design and initial implementation were done by humans.
AI (Claude, Codex) was used for optimizing specific functions and expanding test coverage.
The authors reviewed, edited, and validated all AI-assisted outputs.

# Acknowledgements

Parts of the software were developed while N.G. was employed at Microsoft Research AI for Science and the Technical University of Munich.
We thank Clemens Giuliani, David Linteau, Giuseppe Carleo, Dan Fiedler, and Filippo Vicentini for their contributions.

# References
