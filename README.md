# Generalized Macdonald Functions

A [SageMath](https://www.sagemath.org/) library for computing **Generalized Macdonald functions** (GMPs) — simultaneous eigenfunctions of the elliptic Hall algebra acting on an N-fold tensor product of Fock spaces.

These functions generalize the classical Macdonald P/Q-functions (recovered at N = 1) and arise naturally in:
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

The library works with **multi-partitions** $\boldsymbol{\lambda} = (\lambda^{(0)}, \ldots, \lambda^{(N-1)})$ — N-tuples of integer partitions. The Generalized Macdonald function $P_{\boldsymbol{\lambda}}$ is defined as the unique eigenfunction of the zero-mode operator $x^{+}_0$:

$$\rho^{(N,0)}_{u_0,\dots,u_{N-1}}(x^{+}_0) \cdot P_{\boldsymbol{\lambda}} = \mathrm{eigenvalue}(\boldsymbol{\lambda})\cdot P_{\boldsymbol{\lambda}}, \qquad \mathrm{eigenvalue}(\boldsymbol{\lambda}) = \sum_{i=0}^{N-1} u_i x_{\lambda^{(i)}}$$

expanded in the tensor product of ordinary Macdonald P-functions. The deformation parameters satisfy $q_1 q_2 q_3 = 1$ with $q_1 = q$, $q_2 = 1/t$, $q_3 = t/q$.

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

# Verify the eigenfunction equation x^{+}_0 G = eigenvalue * G
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
| `GMP(lam)` | Compute $P_{\boldsymbol{\lambda}}$ in the tensor product of Macdonald P-bases |
| `GMQ(lam)` | Compute the dual GMP w.r.t. the appropriate scalar product |
| `tildeGMP(lam)` | Spherically normalised GMP: $P_{\boldsymbol{\lambda}}$ divided by the product of principal specialisations |
| `barGMP(lam)` | GMP normalised by its own evaluation at the epsilon arguments |
| `GMK(lam)` | Auxiliary inhomogeneous Generalised Macdonald K-function |
| `GMPast(lam)` | Starred (inhomogenous) GMP $P^*_{\boldsymbol{\lambda}}$ |
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
| `pieriTest(lam)` | Verify the $e_1$-Pieri rule |
| `pieriTestDual(nu)` | Verify the dual (skew) Pieri rule |
| `psi_prime_PE(nu, lam)` | Elementary Pieri vertex coefficient $\psi'_{\nu/\lambda}$ |

### Vertex operators and algebra action

| Function | Description |
|----------|-------------|
| `xplus(k, x)` | Full N-component raising generator $x^{+}_k$ |
| `xminus(k, x)` | Full N-component lowering generator $x^{-}_k$ |
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
| `epsilon(part)` | Epsilon specialisation argument for a partition, $\epsilon_\lambda$ |
| `eigenvalue(lam)` | Eigenvalue of $x^{+}_0$ acting on $P_{\boldsymbol{\lambda}}$ |
| `chi2d(lam)` | Content sum of a 2d partition, $\chi_\lambda = \sum_{(i,j)\in\lambda}q_1^{j-1}q_2^{i-1}$ |
| `x2d(lam)` | Defined as $x_\lambda=1-(1-q_1)(1-q_2)\chi_\lambda=\frac{\epsilon_\lambda}{\epsilon_\emptyset}$ |
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
| `GHT.ipynb` | Explicit checks of the GHT identity |
| `cauchy.ipynb` | Cauchy identity for GMPs |
| `conjecture3_1.ipynb` | Checks of the conjectured identity 3.1 |
| `level-2.ipynb` | Example computations of GMPs at level two |
| `pieri.ipynb` | Pieri rules and dual Pieri rules |

---

## License

Distributed under the [GNU General Public License v3](LICENSE).
