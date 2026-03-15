# Generalized Macdonald Functions

A [SageMath](https://www.sagemath.org/) library for computing **Generalized Macdonald functions** (GMPs) — simultaneous eigenfunctions of the elliptic Hall algebra acting on an N-fold tensor product of Fock spaces.

These functions generalize the classical Macdonald P/Q-functions (recovered at N = 1) and arise naturally in:
- The refined topological vertex and 5d instanton partition functions
- The AGT correspondence between gauge theory and 2d CFT
- The geometry of Hilbert schemes of points on surfaces
- Representation theory of the quantum toroidal algebra $U_{q,t}(\widehat{\widehat{\mathfrak{gl}}}_1)$

The construction follows:
> Bourgine J.-E., Cassia L., Stoyan A. *Generalized Macdonald functions and quantum toroidal gl(1) algebra* (2025). [arXiv:2508.19704](https://arxiv.org/abs/2508.19704)

---

## Mathematical background

The library works with **multi-partitions** $\boldsymbol{\lambda} = (\lambda^{(0)}, \ldots, \lambda^{(N-1)})$ — N-tuples of integer partitions. The Generalized Macdonald function $G_{\boldsymbol{\lambda}}$ is defined as the unique eigenfunction of the zero-mode operator $x^+(0)$:

$$x^+_0\, G_{\boldsymbol{\lambda}} = \mathrm{eigenvalue}(\boldsymbol{\lambda})\, G_{\boldsymbol{\lambda}}, \qquad \mathrm{eigenvalue}(\boldsymbol{\lambda}) = \sum_{i=1}^{N} u_i\, x_{\lambda^{(i)}}$$

expanded in the tensor product of Macdonald P-functions. The deformation parameters satisfy $q_1 q_2 q_3 = 1$ with $q_1 = q$, $q_2 = 1/t$, $q_3 = t/q$.

---

## Installation and requirements

- [SageMath](https://www.sagemath.org/) ≥ 9.0
- [SymPy](https://www.sympy.org/) (for `partitions` iterator)

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

# Verify the eigenfunction equation x^+_0 G = eigenvalue * G
sage: testEigenfunction(([2,1], [1]))
True

# Verify the elementary Pieri rule
sage: pieri(([2], [1]))
True

# Expand G in the GMP basis
sage: to_gmp(G)
{((2, 1), (1)): 1}
```

---

## Key functions

### Computing GMPs

| Function | Description |
|----------|-------------|
| `GMP(lam)` | Compute $G_{\boldsymbol{\lambda}}$ in the tensor product of Macdonald P-bases |
| `GMQ(lam)` | Compute the dual GMP $\tilde{G}_{\boldsymbol{\lambda}}$ (Q-type) |
| `tildeGMP(lam)` | Normalised GMP: $G_{\boldsymbol{\lambda}}$ divided by the product of principal specialisations |
| `barGMP(lam)` | GMP normalised by its own evaluation at the epsilon arguments |
| `GMK(lam)` | Generalised Macdonald K-function |
| `GMPast(lam)` | Dual (starred) GMP $G^*_{\boldsymbol{\lambda}}$ |
| `iGMP(lam)` | Inverted GMP with reversed spectral parameters |

### GMP coefficients

| Function | Description |
|----------|-------------|
| `GMPC(lam, mu)` | Change-of-basis coefficient $C(\boldsymbol{\lambda}, \boldsymbol{\mu})$ |
| `GMMatrixElement(lam, nu)` | Row of the eigenvalue-equation matrix |
| `to_gmp(x)` | Decompose a tensor-product symmetric function in the GMP basis (scalar product method) |
| `to_gmp2(x)` | Same, via linear system (may be faster for many coefficients) |

### Pieri rules

| Function | Description |
|----------|-------------|
| `pieri(lam)` | Verify the $e_1$ Pieri rule for $G_{\boldsymbol{\lambda}}$ |
| `pieriTest(m, lam)` | Verify the degree-$m$ Pieri rule |
| `pieriTestDual(m, nu)` | Verify the dual (skew) Pieri rule |
| `pieri_set(m, lam)` | Multi-partitions reachable from $\boldsymbol{\lambda}$ by adding $m$ boxes |
| `pieri_set_minus(m, nu)` | Multi-partitions reachable by removing $m$ boxes |
| `psi_prime_PE(nu, lam)` | Elementary Pieri vertex coefficient $\psi'(\nu, \lambda)$ |
| `phi_prime_PE(nu, lam)` | Homogeneous Pieri vertex coefficient $\phi'(\nu, \lambda)$ |
| `psi2_prime_PE(nu, lam)` | Dual Pieri vertex coefficient $\psi_2'(\nu, \lambda)$ |

### Vertex operators and algebra action

| Function | Description |
|----------|-------------|
| `xplus(k, x)` | Full N-component raising generator $x^+_k$ |
| `xminus(k, x)` | Full N-component lowering generator $x^-_k$ |
| `LAM(i, k, x)` | Twisted positive operator $\Lambda_{i,k}$ (i-th component) |
| `LAMast(i, k, x)` | Twisted negative operator $\Lambda^*_{i,k}$ |
| `xplus_k(k, x)` | Single-factor raising mode |
| `xminus_k(k, x)` | Single-factor lowering mode |
| `a_k(k, x)` | Creation/annihilation operator $a_k$ |

### Scalar products

| Function | Description |
|----------|-------------|
| `scalar_N(f, g)` | N-component GMP scalar product $\langle f, g \rangle_N$ |
| `scalar_N_prime(f, g)` | Primed variant $\langle f, g \rangle'_N$ |
| `scalar_on_tensor_qt(x, y)` | Factor-wise Macdonald $(q,t)$-scalar product |

### Utilities

| Function | Description |
|----------|-------------|
| `PE(x)` | Plethystic exponential |
| `DET(x)` | Determinant factor of a Laurent polynomial |
| `epsilon(part)` | Epsilon specialisation argument for a partition |
| `eigenvalue(lam)` | Eigenvalue of $G_{\boldsymbol{\lambda}}$ under $x^+_0$ |
| `chi2d(lam)` | content sum of a 2d partition |
| `x2d(lam)` | Normalised 2d content polynomial |
| `mPartitions(N, k)` | All N-tuples of partitions of total weight $k$ |

### Cache management

GMP coefficients are computed on demand and cached. For long computations, persist the cache between sessions:

```python
sage: save_cache("my_session")       # saves to my_session.sobj
sage: d = load_cache("my_session.sobj")
sage: gmp_init(d)                    # restore on next session
```

---

## Jupyter notebooks

The repository includes several notebooks demonstrating the library:

| Notebook | Content |
|----------|---------|
| `GHT.ipynb` | Intertwiners and the GHT identity for N = 1, 2, 3 |
| `cauchy.ipynb` | Cauchy identity for GMPs |
| `conjecture.ipynb` | Checks of conjectured identities |
| `level-2.ipynb` | Level-2 representations |
| `pieri.ipynb` | Pieri rules and dual Pieri rules |

---

## License

Distributed under the [GNU General Public License v3](LICENSE).
