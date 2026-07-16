# Generalized Macdonald Functions

<!-- NOTE: repo/library name is under review (currently a placeholder,
     "gmplib", pending a rename away from the GNU MP / libgmp collision).
     Swap the title, clone URL, and any badges once a final name is chosen. -->

A [SageMath](https://www.sagemath.org/) library for computing **Generalized Macdonald functions** (GMPs) — simultaneous eigenfunctions of the horizontal generators of the elliptic Hall algebra acting on the Fock space of N-tuples of partitions (the level-N horizontal representation).

These functions generalize the classical Macdonald P-functions (recovered at N = 1) and arise naturally in:
- The refined topological vertex and 5d instanton partition functions
- The AGT correspondence between gauge theory and 2d CFT
- The geometry of Hilbert schemes of points on surfaces
- Representation theory of the quantum toroidal algebra $U_{q,t}(\widehat{\widehat{\mathfrak{gl}}}_1)$

The construction follows:
> J.-E. Bourgine, L. Cassia, A. Stoyan,
> *Generalized Macdonald functions and quantum toroidal gl(1) algebra* (2025),
> [arXiv:2508.19704](https://arxiv.org/abs/2508.19704).

---

## Mathematical background

The library works with **multi-partitions** $\boldsymbol{\lambda} = (\lambda^{(0)}, \ldots, \lambda^{(N-1)})$ — N-tuples of integer partitions. The Generalized Macdonald function $P_{\boldsymbol{\lambda}}$ is the unique eigenfunction of the zero-mode operator $x^{+}_0$:

$$\rho^{(N,0)}_{u_0,\dots,u_{N-1}}(x^{+}_0) \cdot P_{\boldsymbol{\lambda}} = \mathrm{eigenvalue}(\boldsymbol{\lambda})\cdot P_{\boldsymbol{\lambda}}, \qquad \mathrm{eigenvalue}(\boldsymbol{\lambda}) = \sum_{i=0}^{N-1} u_i \, x_{\lambda^{(i)}}$$

expanded in the tensor product of ordinary Macdonald P-functions. The deformation parameters satisfy $q_1 q_2 q_3 = 1$ with $q_1 = q$, $q_2 = 1/t$, $q_3 = t/q$.

Two independent algorithms are available for computing $P_{\boldsymbol{\lambda}}$:
1. **Kernel method** (`GMP`) — $P_{\boldsymbol{\lambda}}$ is the null vector of the eigenvalue-equation matrix, found via `GMPC`.
2. **Magnus expansion** (`magnus_exp`) — an iterative refinement starting from the pure tensor $P_{\lambda^{(0)}} \otimes \cdots \otimes P_{\lambda^{(N-1)}}$, converging in finitely many steps.

Both should agree; comparing them is a useful correctness check on a new installation.

---

## Installation and requirements

- [SageMath](https://www.sagemath.org/) ≥ 9.0
- [SymPy](https://www.sympy.org/)

No additional installation is needed beyond cloning the repository:

```bash
git clone https://github.com/lucacassia/gmplib
cd gmplib
```

---

## Quick start

```python
sage: load("gmplib.py")

# Compute the GMP for the bi-partition ((2,1), (1)) at N=2
sage: G = GMP(([2,1], [1]))

# Verify the eigenfunction equation x^{+}_0 G = eigenvalue * G
sage: testEigenfunction(([2,1], [1]))
True

# Cross-check against the alternative Magnus-expansion algorithm
sage: G == magnus_exp(([2,1], [1]))
True

# Verify the elementary Pieri rule
sage: pieriTest(([2], [1]))
True

# Expand G in the GMP basis
sage: to_gmp(G)
{((2, 1), (1)): 1}
```

---

## Key functions

### Setup and base ring

The library initialises a base fraction field over $\mathbb{Q}$ with parameters `q, t, r, z, Q` and 30 equivariant/spectral weights each `u0,...,u29` and `v0,...,v29`, and pre-builds the symmetric function bases `Sym`, `McdP`, `McdQ`, `Ht` (modified Macdonald), `p`, `e`, `h`, `s` on that field.

### Computing GMPs

| Function | Description |
|----------|-------------|
| `GMP(lam)` | $P_{\boldsymbol{\lambda}}$ in the tensor product of Macdonald P-bases, via the kernel method |
| `magnus_exp(lam)` | $P_{\boldsymbol{\lambda}}$ via Magnus-expansion iterative refinement (alternative to `GMP`) |
| `GMQ(lam)` | Dual GMP w.r.t. the scalar product `scalar_Z` |
| `tildeGMP(lam)` | Spherically normalised GMP: $P_{\boldsymbol{\lambda}}$ divided by the product of principal specialisations |
| `barGMP(lam)` | *(legacy)* GMP normalised by its own evaluation at the epsilon arguments |
| `GMK(lam)` | Inhomogeneous Generalised Macdonald K-function, for checking the generalised GHT identity |
| `GMPast(lam)` | Starred (inhomogeneous) GMP $P^*_{\boldsymbol{\lambda}}$ |
| `iGMP(lam)` | Inverted GMP with reversed spectral parameters |

### GMP coefficients and basis conversion

| Function | Description |
|----------|-------------|
| `GMPC(lam, mu)` | Change-of-basis coefficient $C(\boldsymbol{\lambda}, \boldsymbol{\mu})$, cached in `GMPC_cache` |
| `GMMatrixElement(lam, nu)` | Row of the eigenvalue-equation matrix (built from `xplus_fast`) |
| `to_gmp(x)` | Decompose a tensor-product symmetric function in the GMP basis, via the scalar product `scalar_Z` |
| `to_gmp2(x)` | Same, via solving a linear system — may be faster when many coefficients are needed |

### The fast combinatorial engine

Polynomial-time implementations of the vertex-operator action, built directly from Pieri coefficients and Young-diagram combinatorics (`Partition.up()`/`.down()`) rather than repeated symbolic skewing. `GMMatrixElement`, and therefore `GMP`/`GMPC`, use these by default.

| Function | Description |
|----------|-------------|
| `x_fast(sgn, k, x)` | Single-copy operator $x^{\pm}_k$ (level 1), via direct path enumeration on Young diagrams |
| `psi_fast(k, x)` | Single-copy Cartan current $\varphi^{\pm}_k$ |
| `x_fast_on_tensor(i, sgn, k, x)` / `psi_fast_on_tensor(i, k, x)` | Apply the corresponding single-copy operator to tensor factor `i`, identity elsewhere |
| `LAM_fast(i, sgn, k, x)` | Mode-$k$ component of $\Lambda^{(i)}(z)$ / $\Lambda^{*(i)}(z)$, the coproduct pieces of $x^+(z)$ / $x^-(z)$ |
| `drinfeld(sgn, k, x)` | Level-$N$ implementation of the Drinfeld currents, `sum_i u_i^{sgn} * LAM_fast(i,sgn,k,x)` |
| `x_comb(sgn, k, x)` | Reference recursive definition of `x_fast` via commutators — exponential-time, useful for cross-checking `x_fast` on small cases |
| `mMcdP(lam)` | $P_{\lambda^{(0)}} \otimes \cdots \otimes P_{\lambda^{(N-1)}}$, the pure tensor of ordinary Macdonald P-functions |

### Pieri rules and elementary coefficients

| Function | Description |
|----------|-------------|
| `pieriTest(lam)` | Verify the elementary $e_1$-Pieri rule for `GMP(lam)` |
| `pieriTestDual(nu)` | Verify the dual (skew) Pieri rule |
| `psi_prime_PE(nu, lam)` | Elementary Pieri vertex coefficient $\psi'_{\nu/\lambda}$ |
| `psi2_prime_PE(nu, lam)` | Elementary dual Pieri vertex coefficient $\psi''_{\nu/\lambda}$ |
| `alpha_N(i, nu, lam)` / `alpha2_N(i, nu, lam)` | N-component (dual) Pieri transition coefficients, accounting for non-cocommutativity of the coproduct |
| `e1mul(x)` / `e1del(x)` | Single-copy elementary Pieri operator (`e[1] * x`) and its dual (`x.skew_by(e[1])`), in closed combinatorial form |
| `part_plus(m, mu)` / `part_minus(m, mu)` | All partitions reachable from `mu` by adding/removing `m` boxes |
| `pieri_set_plus(m, lam)` / `pieri_set_minus(m, nu)` | Multi-partition analogues of the above |

### Vertex operators and the quantum toroidal algebra action

| Function | Description |
|----------|-------------|
| `xplus(k, x)` / `xminus(k, x)` | Full N-component raising/lowering generator, symbolic (non-fast) reference implementation |
| `xplus_k(k, x)` / `xminus_k(k, x)` | Single-copy vertex operator mode, symbolic reference implementation |
| `LAM(i, k, x)` / `LAMast(i, k, x)` | Symbolic reference implementation of the twisted per-factor operator (cf. `LAM_fast`) |
| `testEigenfunction(mu)` | Verify $x^{+}_0 P_{\boldsymbol{\mu}} = \mathrm{eigenvalue}(\boldsymbol{\mu}) \, P_{\boldsymbol{\mu}}$ |
| `framing(x, power)` / `framing_on_tensor(x, power)` | Apply the framing operator (single-copy / tensor product) |
| `Delta(z, x, power, dual)` / `Delta_on_tensor(...)` | Apply the $\Delta$ operator used in building instanton partition functions |

### Nekrasov factors and instanton partition functions

| Function | Description |
|----------|-------------|
| `Ylam(lam, z)` | Y-operator factor $\mathrm{PE}(-x_{2d}(\lambda)/z)$ |
| `PSIlam(lam, z)` | $\Psi$-operator factor $\mathrm{PE}((1-q_3)\,x_{2d}(\lambda)/z)$ |
| `Nekrasov(lam, mu, z)` | Nekrasov bifundamental factor (plethystic-exponential form) |
| `NekrasovJEB(lam, mu, z)` | Same, in the normalisation of arXiv:2508.19704 |
| `tNek(lam, mu, z)` | Normalised Nekrasov factor $\tilde{N}_{\lambda,\mu}(z)$ |

### Content, eigenvalue, and plethystic utilities

| Function | Description |
|----------|-------------|
| `PE(x)` | Plethystic exponential of a Laurent polynomial |
| `DET(x)` | Determinant factor $\prod_m z^{c_m}$ of a Laurent polynomial |
| `epsilon(lam, power)` | Epsilon specialisation argument $\varepsilon_\lambda$ (Macdonald–Koornwinder duality) |
| `mcdp_at_eps(lam)` | $P_\lambda(\varepsilon_\emptyset)$, the principal specialisation of the ordinary Macdonald P-function, cached |
| `eigenvalue(lam, power)` | Eigenvalue of the generalized Macdonald operator on `GMP(lam)` |
| `chi2d(lam, power)` | Content sum $\chi_\lambda = \sum_{(i,j)\in\lambda} q_1^{j} q_2^{i}$ |
| `x2d(lam, power)` | $x_\lambda = 1-(1-q_1)(1-q_2)\chi_\lambda = \varepsilon_\lambda/\varepsilon_\emptyset$ |
| `w(mu)` | Macdonald weight appearing in the Cauchy identity for modified Macdonald functions |
| `A(lam)` / `R(lam)` | Addable / removable contents of a partition |
| `b_lambda(lam)` / `blam(lam)` | Macdonald b-factor and inverse self-pairing $1/\langle P_\lambda,P_\lambda\rangle_{q,t}$ |

### Multi-symmetric function utilities

| Function | Description |
|----------|-------------|
| `generators(N, basis)` | The N standard degree-1 generators of an N-fold tensor product |
| `mPoly(mpart, parent)` | Lift an N-tuple of partitions to a pure tensor in a given basis |
| `mPartitions(N, k)` | All N-tuples of partitions of total weight `k` |
| `coercion_safe(x, f)` / `coercion_on_tensor(x, parent)` | Basis coercion, scalar-safe / factor-wise on a tensor product |
| `skew_on_tensor(x, y)` / `omega_on_tensor(x)` | Factor-wise skewing / omega involution on a tensor product |
| `degree_on_tensor(x)` | Maximum total degree of a multi-symmetric function |
| `scalar_on_tensor_qt(x, y)` / `scalar_Z(f, g)` | Factor-wise Macdonald $(q,t)$-scalar product / the twisted GMP scalar product of §3.2.3 of arXiv:2508.19704 |
| `counit_on_tensor(x)` / `level(n, x)` | Counit / projection onto the degree-`n` component |
| `evalArg(x, arg)` / `evaluate_on_tensor(x, arg)` / `diag_plethysm(x, arg_l)` | Plethystic evaluation, single-copy / factor-wise |
| `vectors_with_int(N, k)` / `generate_combinations(N, d, min_val)` | N-tuples of non-negative integers summing to a target |
| `is3d(mu)` / `is_reduced(lam)` / `subpart(lam)` | Plane-partition check / trailing-component check / sub-diagrams of a partition |

### Output formatting

| Function | Description |
|----------|-------------|
| `to_math(x)` | Format a multi-symmetric function (power-sum basis) as a Mathematica-style sum string |
| `to_math_l(x)` | Format the coefficients of a multi-symmetric function as a Mathematica list |
| `part_to_str(mu)` / `mpart_to_str(mu)` | Mathematica-style string for a partition / multi-partition |

### Cache management

GMP coefficients are computed on demand and cached in `GMPC_cache`. For long computations, persist the cache between sessions:

```python
sage: save_cache("my_session")       # saves to my_session.sobj
sage: d = load_cache("my_session.sobj")
sage: gmp_init(d)                    # restore on next session
sage: cache_to_dict()                # inspect the current cache as a plain dict
```

---


## Jupyter notebooks

The repository includes several notebooks demonstrating the library:

| Notebook | Content |
|----------|---------|
| `GHT.ipynb` | Explicit checks of the GHT identity |
| `cauchy.ipynb` | Cauchy identity for GMPs |
| `conjecture3_1.ipynb` | Checks of the conjectured identity 3.1 |
| `level-2.ipynb` | Example computations of GMPs at level two |
| `pieri.ipynb` | Pieri rules and dual Pieri rules |
| `benchmark_kernel_vs_magnus.ipynb` | Performance comparison of different algorithms for computing GMPs |

---

## License

Distributed under the [GNU General Public License v3](LICENSE).
