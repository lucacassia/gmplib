r"""
Generalized Macdonald Functions
================================

A SageMath library for computing **Generalized Macdonald functions** (GMPs) —
simultaneous eigenfunctions of the horizontal generators of the elliptic Hall algebra action on the Fock space
of N-tuples of partitions, i.e. the level-N horizontal representation.

These functions generalize both the classical Macdonald P-functions (recovered at N=1)
and the fixed-point basis of the moduli space of instantons. They appear naturally
in the study of the refined topological vertex, AGT correspondence, and the geometry
of Hilbert schemes of points.

The construction is based on the action of the quantum toroidal algebra
U_{q,t}(gl_1^{hat hat}), following the conventions of:

    Bourgine J.-E., Cassia L., Stoyan A. "Generalized Macdonald functions and quantum toroidal gl(1) algebra" (2025).
    arXiv:2508.19704

Setup
-----
The library initialises a base fraction field over QQ with parameters::

    q, t       -- Macdonald deformation parameters
    r, z, Q    -- auxiliary parameters
    u0,...,u29 -- equivariant weights (spectral parameters) for the N factors
    v0,...,v29 -- auxiliary spectral parameters

The three roots of unity of the elliptic deformation are::

    q1 = q,   q2 = 1/t,   q3 = t/q,   with  q1*q2*q3 = 1.

The following symmetric function bases are pre-built on the base field:

    Sym    -- symmetric functions over the base field
    McdP   -- Macdonald P-basis
    McdQ   -- Macdonald Q-basis
    Ht     -- modified Macdonald Ht-basis (q=q1, t=q2)
    p      -- power-sum basis
    e      -- elementary symmetric function basis
    h      -- homogeneous symmetric function basis
    s      -- Schur basis

Caching
-------
GMP coefficients are expensive to compute. The library maintains a global dict
``GMPC_cache`` that maps pairs of multi-partitions to their GMP coefficients.
Use ``save_cache`` / ``load_cache`` to persist results between sessions.

Examples
--------
Compute the GMP for the bi-partition ((2,1), (1)) at N=2::

    sage: load("gmplib.py")
    sage: poly = GMP(([2,1], [1]))
    sage: testEigenfunction(([2,1], [1]))
    True

Verify the Pieri rule::

    sage: pieri(([2], [1]))
    True

References
----------
- Macdonald, I.G. "Symmetric Functions and Hall Polynomials" (2nd ed., 1995).
- Schiffmann-Vasserot, "The elliptic Hall algebra and the equivariant K-theory
  of the Hilbert scheme of A^2" (2013).
- Feigin et al., "Quantum continuous gl_\infty" (2011).
"""

#*****************************************************************************
#       Copyright (C) 2024 Luca Cassia <luca.cassia@tuta.io>,
#
#  Distributed under the terms of the GNU General Public License (GPL)
#
#    This code is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    General Public License for more details.
#
#  The full text of the GPL is available at:
#
#                  http://www.gnu.org/licenses/
#*****************************************************************************

# ---------------------------------------------------------------------------
# Base ring, symbols, and symmetric function bases
# ---------------------------------------------------------------------------

import time
from sage.all import *

N = 30
param = list('u%d'%i for i in range(N)) + list('v%d'%i for i in range(N)) + ['q','t','r','z','Q']

Sym = SymmetricFunctions(FractionField(PolynomialRing(QQ,param)))
ring = Sym.base_ring()
(q,t,r,z,Q) = [ring.gens()[i] for i in range(2*N,2*N+5)]
u = list(ring(var('u%d' % i)) for i in range(N))
v = list(ring(var('v%d' % i)) for i in range(N))

# The three deformation parameters satisfying q1*q2*q3 = 1
q1=q
q2=1/t
q3=t/q
kappa1=(1-q1)*(1-q2)*(1-q3)

L = LaurentPolynomialRing(QQ,param)
F = L.fraction_field()

# Symmetric function bases
McdP = Sym.macdonald().P()
McdQ = Sym.macdonald().Q()
Ht = Sym.macdonald(q=q1,t=q2).Ht()
p = Sym.powersum()
e = Sym.e()
h = Sym.homogeneous()
s = Sym.Schur()

# Global cache for GMP change-of-basis coefficients
GMPC_cache = dict({})


# ---------------------------------------------------------------------------
# Cache persistence
# ---------------------------------------------------------------------------

from sage.misc.persist import load, save

def gmp_init(dd=dict({})):
    """
    Initialise (or reset) the global GMP coefficient cache.

    Parameters
    ----------
    dd : dict, optional
        A pre-populated cache dictionary, typically loaded from disk via
        ``load_cache``.  Defaults to an empty dict, which clears the cache.

    Examples
    --------
    ::

        sage: gmp_init()           # clear cache
        sage: d = load_cache("my_cache")
        sage: gmp_init(d)          # restore saved cache
    """
    global GMPC_cache
    GMPC_cache = dd

def save_cache(file_name):
    """
    Persist the current GMP coefficient cache to a SageMath ``.sobj`` file.

    The file is written using :func:`sage.misc.persist.save`.  The ``.sobj``
    extension is appended automatically by Sage.

    Parameters
    ----------
    file_name : str
        Path (without the ``.sobj`` extension) where the cache will be saved.

    Examples
    --------
    ::

        sage: save_cache("gmp_level2")
        Saved cache to file: gmp_level2.sobj
    """
    global GMPC_cache
    try:
        save(GMPC_cache, filename=file_name)
        print("Saved cache to file:", file_name+".sobj")
    except:
        print("Saving failed!")

def cache_to_dict():
    """
    Return the current GMP coefficient cache as a plain Python dict.

    Returns
    -------
    dict
        The global ``GMPC_cache`` dictionary mapping pairs of multi-partitions
        to rational functions in ``q, t``.
    """
    global GMPC_cache
    return GMPC_cache

def load_cache(file_name):
    """
    Load a previously saved GMP coefficient cache from a ``.sobj`` file.

    Parameters
    ----------
    file_name : str
        Full path to the ``.sobj`` file (including the extension).

    Returns
    -------
    dict
        The loaded cache dictionary, or an empty dict if the file is not found.

    Notes
    -----
    This function returns the loaded dict but does **not** install it as the
    active cache.  Pass the result to :func:`gmp_init` to activate it::

        sage: d = load_cache("gmp_level2.sobj")
        sage: gmp_init(d)
    """
    try:
        obj = load(file_name)
        print("Loaded cache from file:", file_name)
        return obj
    except:
        print("Cache file not found. Starting from scratch.")
        return dict({})


# ---------------------------------------------------------------------------
# Multi-partition combinatorics
# ---------------------------------------------------------------------------

from itertools import product,permutations
from sympy.utilities.iterables import partitions

def partitions_up_to_k(k):
    """
    Generate all integer partitions of 0, 1, ..., k (in Sage Partition form).

    Parameters
    ----------
    k : int
        Maximum total weight.

    Yields
    ------
    Partition
        Sage ``Partition`` objects in non-decreasing order of weight.
    """
    for n in range(k+1):
        for partition in Partitions(n):
            yield partition

def vectors_with_partitions(N,k):
    """
    Generate all N-tuples of partitions whose total weight equals k.

    Parameters
    ----------
    N : int
        Number of components (length of each tuple).
    k : int
        Required total weight, i.e. ``sum(sum(mu) for mu in result) == k``.

    Returns
    -------
    filter
        An iterator over N-tuples of Sage ``Partition`` objects satisfying
        the weight constraint.
    """
    return filter(lambda vpart: sum(map(sum, vpart)) == k, product(partitions_up_to_k(k),repeat=N))

def vectors_with_int(N,k):
    """
    Generate all N-tuples of non-negative integers summing to k.

    Parameters
    ----------
    N : int
        Length of each tuple.
    k : int
        Required total.

    Returns
    -------
    filter
        An iterator over N-tuples of integers in ``{0,...,k}`` with sum ``k``.
    """
    return filter(lambda vint: sum(vint) == k, product(range(k+1),repeat=N))

def generate_combinations(N, d, min_val=0):
    """
    Generate all N-tuples of integers in ``[min_val, d]`` summing to d.

    Parameters
    ----------
    N : int
        Length of each tuple.
    d : int
        Required total.
    min_val : int, optional
        Minimum value for each component (default 0).

    Returns
    -------
    filter
        An iterator over valid N-tuples.
    """
    possible_values = range(min_val, d + 1)
    all_combinations = product(possible_values, repeat=N)
    valid_combinations = filter(lambda comb: sum(comb) == d, all_combinations)    
    return valid_combinations

def mPartitions(N,k):
    """
    Return all N-tuples of partitions (multi-partitions) of total weight k.

    Alias for :func:`vectors_with_partitions`.

    Parameters
    ----------
    N : int
        Number of components.
    k : int
        Total weight.

    Returns
    -------
    list
    """
    return list(vectors_with_partitions(N,k))

def mPoly(mpart,parent):
    """
    Lift an N-tuple of partitions to an element of a tensor product of
    symmetric function rings.

    Parameters
    ----------
    mpart : tuple of Partition
        An N-tuple ``(mu_0, ..., mu_{N-1})`` of partitions.
    parent : symmetric function basis
        The basis to use for each tensor factor (e.g. ``p``, ``McdP``).

    Returns
    -------
    element of ``parent^{tensor N}``
        The pure tensor ``parent(mu_0) x ... x parent(mu_{N-1})``.
    """
    return tensor(list(map(lambda mu: coercion_safe(parent(mu),p),mpart)))

def is_reduced(lam):
    """
    Check whether a multi-partition is *reduced* (last component is non-empty).

    Parameters
    ----------
    lam : tuple of list
        Multi-partition as a tuple of partition lists.

    Returns
    -------
    bool
        ``True`` if ``lam[-1] != []``.
    """
    return lam[-1] != []

def is3d(mu):
    """
    Check whether an N-tuple of partitions forms a valid plane partition.

    A tuple ``(mu_0, ..., mu_{N-1})`` is a plane partition (3d Young diagram)
    if each partition contains the next one: ``mu_{i+1} subset mu_i`` for all i.

    Parameters
    ----------
    mu : tuple of list
        N-tuple of partition lists.

    Returns
    -------
    bool
        ``True`` if the containment condition holds for all consecutive pairs.
    """
    N = len(mu)
    res = True
    for i in range(N-1):
        res = res and Partition(mu[i+1]).contains(Partition(mu[i]))
        if res == False:
            return res
    return res

def subpart(lam):
    """
    Return all partitions contained in ``lam`` (sub-diagrams of the Young diagram).

    Parameters
    ----------
    lam : list or Partition
        A partition.

    Returns
    -------
    filter
        Iterator over all partitions ``mu`` with ``mu subset lam``.
    """
    deg = sum(lam)
    return filter(lambda x:Partition(lam).contains(x),partitions_up_to_k(deg))


# ---------------------------------------------------------------------------
# Scalar utilities
# ---------------------------------------------------------------------------

def nfactor(x):
    """
    Factor ``x`` if non-zero, otherwise return 0.

    Parameters
    ----------
    x : ring element

    Returns
    -------
    ring element
        ``factor(x)`` if ``x != 0``, else ``0``.
    """
    if x==0:
        return x
    else:
        return factor(x)

def Kronecker_delta(x,y):
    """
    Kronecker delta function.

    Parameters
    ----------
    x, y : any comparable

    Returns
    -------
    int
        1 if ``x == y``, else 0.
    """
    return 1 if x == y else 0


# ---------------------------------------------------------------------------
# Content and eigenvalue functions
# ---------------------------------------------------------------------------

def epsilon(part,power=1):
    r"""
    Compute the epsilon specialisation argument for a partition.

    The epsilon map encodes the generating series of box contents of ``part``:

        epsilon(part) = sum_{i} q^{part[i]} * t^{-i-1}  +  t^{-len(part)-1} / (1 - 1/t)

    Evaluating a Macdonald P-function at ``epsilon(mu)`` gives the specialisation
    appearing in the Macdonald-Koornwinder duality.

    Parameters
    ----------
    part : Partition
    power : int, optional
        Rescaling exponent applied to both ``q`` and ``t`` (default 1).

    Returns
    -------
    rational function
    """
    res = sum(q**(part[i])*t**(-i-1) for i in range(len(part))) + t**(-len(part)-1)/(1-1/t)
    return res.subs(q=q**power,t=t**power)

def eigenvalue(lam):
    r"""
    Compute the eigenvalue of the GMP ``G_lam`` under the zero-mode ``x^{+}_0``.

    For an N-tuple of partitions ``lam = (lam_0,...,lam_{N-1})``:

        eigenvalue(lam) = sum_{i=0}^{N-1}  u_i * x2d(lam_i)

    Parameters
    ----------
    lam : tuple of list
        N-tuple of partitions.

    Returns
    -------
    rational function
        Rational function of q, t, u_i.
    """
    return sum(u[i]*x2d(lam[i]) for i in range(len(lam)))

def w(mu):
    r"""
    Compute the Macdonald weight ``w(mu)`` appearing in the Cauchy identity
    for modified Macdonald functions.

    This is the coefficient of ``Ht(mu)`` when expanding
    ``e_{|mu|}[ X / (1-q1)(1-q2) ]``.

    Parameters
    ----------
    mu : list or Partition

    Returns
    -------
    rational function
        Rational function of q, t.
    """
    return 1/e[sum(mu)](Ht[1]/(1-q1)/(1-q2)).coefficient(mu)

def chi2d(lam,power=1):
    r"""
    Compute the 2d content polynomial of a partition. Also the K-theoretic
    character of the rank ``|lam|`` tautological sheaf over the Hilbert scheme of
    points on A^2 pulled-back to the fixed-point ``lam``.

    For a partition ``lam``, this sums over all boxes (i, j) in the Young diagram:

        chi2d(lam) = sum_{(i,j) \in lam}  q2^i * q1^j

    With ``power != 1``, the deformation parameters are rescaled:
    ``q1 -> q1^power``, ``q2 -> q2^power``.

    Parameters
    ----------
    lam : list or Partition
    power : int, optional
        (default 1)

    Returns
    -------
    ring element

    See Also
    --------
    x2d == 1 - (1-q1)(1-q2)*chi2d.
    """
    return sum(sum((q2**i*q1**j)**power for j in range(lam[i])) for i in range(len(lam)))

def x2d(lam,power=1):
    r"""
    Compute the polynomial (K-theoretic character) defined as:

        x2d(lam) = 1 - (1 - q1) * (1 - q2) * chi2d(lam)

    This is the building block of eigenvalues and vertex operator coefficients.
    It also satisfies:

        x2d(lam) = A(lam) - q/t * R(lam)

    where ``A(lam)`` is the sum of content of boxes which can be added to ``lam``
    and ``R(lam)`` is the sum of contents of boxes which can be removed.

    Parameters
    ----------
    lam : list or Partition
    power : int, optional
        (default 1)

    Returns
    -------
    ring element

    See Also
    --------
    chi2d, eigenvalue
    """
    return 1-(1-q1**power)*(1-q2**power)*chi2d(lam,power)


# ---------------------------------------------------------------------------
# Plethystic exponential and determinant
# ---------------------------------------------------------------------------

def PE(x):
    r"""
    Compute the plethystic exponential of a Laurent polynomial.

    For ``x = sum_m c_m * z^m`` (with no constant term):

        PE(x) = prod_m  (1 - z^m)^{-c_m}

    Parameters
    ----------
    x : element of the Laurent polynomial ring ``L``

    Returns
    -------
    rational function

    Notes
    -----
    Central building block for vertex operators, Nekrasov functions,
    and Pieri coefficients throughout the library.
    """
    x = L(x)
    return ring(prod((1-monomial)**-coeff for coeff,monomial in x))

def DET(x):
    r"""
    Compute the determinant factor ``prod_m z^{c_m}`` of a Laurent polynomial.

    For ``x = sum_m c_m * z^m`` returns ``prod_m z^{c_m}``.

    Parameters
    ----------
    x : element of the Laurent polynomial ring ``L``

    Returns
    -------
    ring element

    Notes
    -----
    Used in the framing operator and normalisation of Nekrasov functions.
    """
    x = L(x)
    return prod(z**c for c,z in x)


# ---------------------------------------------------------------------------
# Vertex operators and Nekrasov factors
# ---------------------------------------------------------------------------

def Ylam(lam,z):
    r"""
    Compute the Y-operator factor ``PE(-x2d(lam)/z)``.

    Parameters
    ----------
    lam : list or Partition
    z : ring element

    Returns
    -------
    rational function
    """
    return PE(-x2d(lam)/z)

def PSIlam(lam,z):
    r"""
    Compute the PSI-operator factor ``PE((1-q3)*x2d(lam)/z)``.

    Parameters
    ----------
    lam : list or Partition
    z : ring element

    Returns
    -------
    rational function
    """
    return PE((1-q3)*x2d(lam)/z)

def Nekrasov(lam,mu,z):
    r"""
    Compute the Nekrasov bifundamental factor (plethystic exponential form).

    Evaluates:

        PE( -z * (1 - x2d(lam)*x2d(mu)^\vee) / (1-q1)(1-q2) )

    This factor appears in the 5d instanton partition function of SYM.

    Parameters
    ----------
    lam, mu : list or Partition
    z : ring element
        Fugacity (Coulomb branch parameter ratio).

    Returns
    -------
    rational function

    See Also
    --------
    NekrasovJEB, tNek
    """
    return PE(-z*(1-x2d(lam)*x2d(mu,-1))/(1-q1)/(1-q2))

def NekrasovJEB(lam,mu,z):
    r"""
    Compute the Nekrasov factor in the JEB normalisation, i.e. the same as in the
    article arXiv:2508.19704.

    Parameters
    ----------
    lam, mu : list or Partition
    z : ring element

    Returns
    -------
    rational function

    See Also
    --------
    Nekrasov, tNek
    """
    return PE(z*(1-q1)*(1-q2)*chi2d(lam)*chi2d(mu,-1)-z/q3*chi2d(lam)-z*chi2d(mu,-1))

def tNek(lam,mu,z):
    r"""
    Compute the normalised Nekrasov factor, ``\tilde{N}_{\lam,\mu}(z)```.

    Defined as:

        tNek(lam, mu, z) = NekrasovJEB(lam, mu, z) / (-z/q3)^{|lam|} / DET(chi2d(lam))

    Parameters
    ----------
    lam, mu : list or Partition
    z : ring element

    Returns
    -------
    rational function
    """
    return NekrasovJEB(lam,mu,z) / (-z/q3)**sum(lam) / DET(chi2d(lam))


# ---------------------------------------------------------------------------
# Pieri coefficients
# ---------------------------------------------------------------------------

def psi_prime_PE(nu,lam):
    r"""
    Compute the elementary Pieri coefficient ``psi'(nu, lam)``.

    Controls the action of the elementary Pieri operator: the coefficient with
    which ``P_nu`` appears when ``e_1`` acts on ``P_lam``.
    Requires ``|nu| >= |lam|``; returns 0 if ``|nu| < |lam|``.

    Parameters
    ----------
    nu, lam : list or Partition

    Returns
    -------
    rational function

    See Also
    --------
    psi2_prime_PE, pieriTest
    """
    nu = Partition(nu)
    lam = Partition(lam)
    m = sum(nu)-sum(lam)
    if m == 0:
        rhs = 1 if nu == lam else 0
    else:
        if lam == []:
            rhs = PE(-chi2d([1]*m)/t - (1-1/t)*(chi2d(nu,-1)-chi2d(lam,-1))*(chi2d(nu)-chi2d(lam)) + (chi2d(nu,-1)-chi2d(lam,-1)) * x2d(lam) ) * (e[m](epsilon([]))) / McdP(nu)(epsilon([]))
        else:
            rhs = PE(-chi2d([1]*m)/t - (1-1/t)*(chi2d(nu,-1)-chi2d(lam,-1))*(chi2d(nu)-chi2d(lam)) + (chi2d(nu,-1)-chi2d(lam,-1)) * x2d(lam) ) * (e[m](epsilon([]))) * McdP(lam)(epsilon([])) / McdP(nu)(epsilon([]))
    return rhs

def psi2_prime_PE(nu,lam):
    r"""
    Compute the elementary dual Pieri coefficient ``psi2'(nu, lam)``.

    Controls the action of the elementary Pieri operator: the coefficient with
    which ``P_lam`` appears when ``e_1^\perp`` acts on ``P_nu``.
    Requires ``|nu| >= |lam|``; returns 0 if ``|nu| < |lam|``.

    Parameters
    ----------
    nu, lam : list or Partition

    Returns
    -------
    rational function

    See Also
    --------
    psi_prime_PE, pieriTestDual
    """
    nu = Partition(nu)
    lam = Partition(lam)
    m = sum(nu)-sum(lam)
    if m == 0:
        rhs = 1 if nu == lam else 0
    else:
        if lam == []:
            rhs = (-q) * PE(-1 - q3*(chi2d(nu,-1)-chi2d(lam,-1)) * x2d(nu) ) * (McdP[m](epsilon([]))) * McdP(nu)(epsilon([]))
        else:
            rhs = (-q) * PE(-1 - q3*(chi2d(nu,-1)-chi2d(lam,-1)) * x2d(nu) ) * (McdP[m](epsilon([]))) * McdP(nu)(epsilon([])) / McdP(lam)(epsilon([]))
    return rhs

def alpha_N(i,nu,lam):
    r"""
    Compute the N-component Pieri transition coefficient ``alpha_N(i, nu, lam)``.

    Accounts for the non-cocommutativity of the coproduct.
    This is the coefficient that witnesses the transition ``lam[i] -> nu``
    with ``|nu|=|lam[i]+1|``.

    Parameters
    ----------
    i : int
        Component index (0-indexed) at which the transition occurs.
    nu : Partition
        Target partition for component i.
    lam : tuple of Partitions
        Full N-tuple of source partitions.

    Returns
    -------
    rational function
    """
    N = len(lam)
    return PE( (1-q3)*(chi2d(nu,-1)-chi2d(lam[i],-1))*u[i]**-1*sum(u[j]*x2d(lam[j]) for j in range(i+1,N)) )

def alpha2_N(i,nu,lam):
    r"""
    Compute the dual N-component Pieri transition coefficient ``alpha2_N(i, nu, lam)``.

    Used in the dual Pieri rule.

    Parameters
    ----------
    i : int
        Component index (0-indexed) at which the transition `nu[i] -> lam`` occurs,
        with ``|lam|=|nu[i]-1|``.
    nu : tuple of Partitions
        Full N-tuple of source partitions.
    lam : Partition
        Target partition for component i.

    Returns
    -------
    ring element
    """
    N = len(nu)
    return PE( (1-q3)*(chi2d(nu[i],-1)-chi2d(lam,-1))*u[i]**-1*sum(u[j]*x2d(nu[j]) for j in range(i)) )


# ---------------------------------------------------------------------------
# Addition / removal operators on Young diagrams
# ---------------------------------------------------------------------------

def rtildeast(lam,z):
    r"""
    Return ``PE(-q3*x2d(lam)/z - 1)`` if ``z`` is a removable content of ``lam``.
    This is a normalized dual Pieri coefficient.

    Parameters
    ----------
    lam : Partition
    z : ring element

    Returns
    -------
    rational function
    """
    if z in R(lam):
        return PE(-q3*x2d(lam)/z-1)
    else:
        return 0

def rtilde(lam,z):
    r"""
    Return ``PE(x2d(lam)/z - 1)`` if ``z`` is an addable content of ``lam``.
    This is a normalized Pieri coefficient.

    Parameters
    ----------
    lam : Partition
    z : ring element

    Returns
    -------
    rational function
    """
    if z in A(lam):
        return PE(x2d(lam)/z-1)
    else:
        return 0

def tP(lam):
    r"""
    Compute the Macdonald P-function ``\tilde{P}_lam`` in the spherical normalization.

    Defined as the Schur expansion of ``McdP(lam)`` divided by its principal
    specialisation ``McdP(lam)(epsilon([]))``.

    Parameters
    ----------
    lam : Partition

    Returns
    -------
    element of ``s``
    """
    if lam == []:
        return s.one()
    else:
        return s(McdP(lam))/McdP(lam)(epsilon([]))

def b_lambda(lam):
    r"""
    Compute the Macdonald b-factor ``b_lam = c2(lam) / c1(lam)``.

    Parameters
    ----------
    lam : Partition

    Returns
    -------
    rational function
    """
    lam = Partition(lam)
    return McdP.c2(lam)/McdP.c1(lam)

def blam(lam):
    r"""
    Compute the inverse self-pairing ``1 / <P_lam, P_lam>_{q,t}``.

    Parameters
    ----------
    lam : Partition

    Returns
    -------
    rational function

    See Also
    --------
    b_lambda (they should be equal)
    """
    if lam == []:
        return 1
    return 1/McdP(lam).scalar_qt(McdP(lam))

def b_tilde(lam):
    r"""
    Compute the inverse self-pairing for the tilde Macdonald function.

    Returns ``1 / <tP_lam, tP_lam>_{q,t}``.

    Parameters
    ----------
    lam : Partition

    Returns
    -------
    rational function
    """
    if lam == []:
        return 1
    return 1/tP(lam).scalar_qt(tP(lam))

def A(lam):
    r"""
    Return the set of addable contents of a partition.

    An addable content is ``chi2d(nu) - chi2d(lam)`` for each partition ``nu``
    obtained from ``lam`` by adding one box.

    Parameters
    ----------
    lam : Partition

    Returns
    -------
    map
        Iterator over addable contents.
    """
    return map(lambda x:chi2d(x)-chi2d(lam),Partition(lam).up())

def R(lam):
    r"""
    Return the set of removable contents of a partition.

    A removable content is ``chi2d(lam) - chi2d(nu)`` for each partition ``nu``
    obtained from ``lam`` by removing one box.

    Parameters
    ----------
    lam : Partition

    Returns
    -------
    map
        Iterator over removable contents.
    """
    return map(lambda x:chi2d(lam)-chi2d(x),Partition(lam).down())
    
def a_k(k,x):
    r"""
    Apply the creation/annihilation operator ``a_k`` to a symmetric function.

    For ``k > 0`` this is an annihilation operator (skew by power-sum ``p_k``);
    for ``k < 0`` it is a creation operator (multiply by ``p_{|k|}``);
    for ``k = 0`` it is the identity.

    Concretely::

        a_0(x)  =  x
        a_k(x)  =  -(z^{-k}/k)(1-q1^k)(1-q3^k) * x.skew_by(p_k)   for k > 0
        a_k(x)  =  -(z^{|k|}/|k|)(1-q2^{|k|})(1-q3^{|k|}) * p_{|k|} * x  for k < 0

    Parameters
    ----------
    k : int
        Mode number.
    x : symmetric function

    Returns
    -------
    symmetric function
    """
    if k==0:
        return x
    if k>0:
        return -z**-k*(1-q1**k)*(1-q3**k)*x.skew_by(p[k])/k
    if k<0:
        k=-k
        return -z**-k*(1-q2**k)*(1-q3**k)*p[k]*x/k


# ---------------------------------------------------------------------------
# Multi-symmetric function utilities
# ---------------------------------------------------------------------------

def coercion_safe(x,f):
    r"""
    Coerce a symmetric function ``x`` into the basis ``f``, handling scalars.

    If ``x`` equals its counit (i.e. it is a scalar), returns
    ``x.counit() * f.one()`` to avoid coercion errors.

    Parameters
    ----------
    x : symmetric function
    f : symmetric function basis (e.g. ``p``, ``s``, ``McdP``)

    Returns
    -------
    element of ``f``
    """
    if x == x.counit():
        return x.counit() * f.one()
    return f(x-x.counit()) + x.counit()*f.one()

def skew_on_tensor(x,y):
    r"""
    Compute the factor-wise skewing of ``x`` by ``y``.

    For ``x``, ``y`` in ``Sym^{tensor N}``, applies the skew operation
    independently in each tensor factor.

    Parameters
    ----------
    x, y : elements of ``Sym^{tensor N}``

    Returns
    -------
    element of ``Sym^{tensor N}``
    """
    parent1 = x.parent().tensor_factors()
    parent2 = y.parent().tensor_factors()
    return sum( sum( c1 * c2 * tensor(map(lambda k1,p1,k2,p2: p1(k1).skew_by(p2(k2)),mp1,parent1,mp2,parent2)) for mp1,c1 in x) for mp2,c2 in y)

def omega_on_tensor(x):
    r"""
    Apply the omega involution factor-by-factor to a tensor-product element.

    The omega involution maps ``e_k <-> h_k``.

    Parameters
    ----------
    x : element of ``Sym^{tensor N}``

    Returns
    -------
    element of ``Sym^{tensor N}``
    """
    parent = x.parent().tensor_factors()
    return sum( coeff * tensor([parent[i](mpart[i]).omega() for i in range(len(parent))]) for mpart,coeff in x)

def coercion_on_tensor(x,parent):
    r"""
    Coerce a multi-symmetric function to a new list of bases.

    Parameters
    ----------
    x : element of ``Sym^{tensor N}``
    parent : list of symmetric function bases, length N
        Target basis for each tensor factor.

    Returns
    -------
    element of the tensor product of the ``parent`` factors
    """
    old_parent = x.parent().tensor_factors()
    return sum( coeff * tensor(map(lambda part,old_base,base: coercion_safe(old_base(part),base),mpart,old_parent,parent)) for mpart,coeff in x)

def degree_on_tensor(x):
    r"""
    Return the maximum total degree of a multi-symmetric function.

    The total degree of a pure tensor ``mu_0 x ... x mu_{N-1}`` is
    ``sum(sum(mu_i))``.

    Parameters
    ----------
    x : element of ``Sym^{tensor N}``

    Returns
    -------
    int
    """
    return max(map(lambda part: sum(map(sum,part)),x.support()))

def scalar_on_tensor_qt(x,y):
    r"""
    Compute the Macdonald (q,t)-scalar product of multi-symmetric functions ``x`` and ``y``.

    Uses the pairing ``<P_lam, Q_mu> = delta_{lam,mu}`` factor-by-factor.

    Parameters
    ----------
    x : element of ``Sym^{tensor N}``
    y : element of ``Sym^{tensor N}``

    Returns
    -------
    rational function
    """
    l = len(x.parent().tensor_factors())
    x = coercion_on_tensor(x,[McdP]*l)
    y = coercion_on_tensor(y,[McdQ]*l)
    return sum( sum( c1 * c2 * prod(map(lambda k1,k2: Kronecker_delta(k1,k2),lam,mu)) for lam,c1 in x) for mu,c2 in y)

def counit_on_tensor(x):
    r"""
    Evaluate the counit of each tensor factor and return the product.

    Parameters
    ----------
    x : element of ``Sym^{tensor N}``

    Returns
    -------
    rational function
    """
    parent = x.parent().tensor_factors()
    return sum( coeff * prod(pp(k).counit() for k,pp in zip(mu,parent)) for mu,coeff in x)

def counit_on_tensor2(x):
    r"""
    Extract the coefficient of the all-empty multi-partition ``([], ..., [])``.
    Same output as :func:`counit_on_tensor`

    Parameters
    ----------
    x : element of ``Sym^{tensor N}``

    Returns
    -------
    rational function
    """
    return sum( coeff * prod(Kronecker_delta(k,[]) for k in mu) for mu,coeff in x)

def level(n,x):
    r"""
    Project a multi-symmetric function onto its degree-n component.

    Parameters
    ----------
    n : int
        Total degree to project onto.
    x : element of ``Sym^{tensor N}``

    Returns
    -------
    element of ``Sym^{tensor N}``
    """
    N = len(x.parent().tensor_factors())
    x = coercion_on_tensor(x,[p]*N)
    dd = dict(x)
    return sum(dd[key]*mPoly(key,p) for key in dd if sum(map(sum,key))==n)

def e1t(N):
    r"""
    Compute the N-fold coproduct of ``e[1]`` in the spherical normalization.

    Returns ``(1/epsilon([]))`` times the sum of the N elementary degree-1 generators,
    one per tensor factor.

    Parameters
    ----------
    N : int
        Number of tensor factors.

    Returns
    -------
    element of ``Sym^{tensor N}``
    """
    X = [tensor([p[1] if j==i else p.one() for j in range(N)]) for i in range(N)]
    return sum(X)/epsilon([])

def evalArg(x,arg):
    r"""
    Evaluate a symmetric function ``x`` at the argument ``arg``.

    Handles the constant term correctly: ``x0 + (x - x0)(arg)``.

    Parameters
    ----------
    x : symmetric function
    arg : ring element or symmetric function

    Returns
    -------
    ring element or symmetric function
    """
    x0 = x.counit()
    x = x-x0
    return x0 + x(arg)

def evaluate_on_tensor(x,arg):
    r"""
    Evaluate a multi-symmetric function at a list of arguments.

    Factor i is independently evaluated at ``arg[i]``.

    Parameters
    ----------
    x : element of ``Sym^{tensor N}``
    arg : list of length N

    Returns
    -------
    ring element
    """
    N = len(x.parent().tensor_factors())
    x = coercion_on_tensor(x,[s]*N)
    return sum(coeff*prod(evalArg(s(mu[i]),arg[i]) for i in range(N)) for mu,coeff in x)

def diag_plethysm(x,arg_l):
    r"""
    Apply a list of plethystic substitutions, one per tensor factor. Differs from
    :func:`evaluate_on_tensor` in that its argument is a list of elements of ``Sym``
    and it returns another multi-symmetric function.

    Parameters
    ----------
    x : element of ``Sym^{tensor N}``
    arg_l : list of length N elements of Sym
        Plethystic substitution argument for each tensor factor.

    Returns
    -------
    element of ``Sym^{tensor N}``
    """
    parent = x.parent().tensor_factors()
    N = len(parent)
    x = coercion_on_tensor(x,[p]*N)
    return sum( coeff * tensor([p(mu)(a) for mu,a in zip(part,arg_l)]) for part,coeff in x)

def subsr(x):
    r"""
    Apply the substitution ``r -> -1`` to the coefficients of a tensor product element.
    (not very elegant workaround to implement the non-plethystic minus sign, usually denoted as `ϵ`)

    Parameters
    ----------
    x : element of ``Sym^{tensor N}``

    Returns
    -------
    element of ``Sym^{tensor N}``
    """
    parent = x.parent().tensor_factors()
    N = len(parent)
    return sum( coeff.subs(r=-1) * tensor(map(lambda part,base: base(part),mu,parent)) for mu,coeff in x)

def rev(x):
    r"""
    Reverse the order of tensor factors in a multi-symmetric function.

    Parameters
    ----------
    x : element of ``Sym^{tensor N}``

    Returns
    -------
    element of ``Sym^{tensor N}``
    """
    N = len(x.parent().tensor_factors())
    x = coercion_on_tensor(x,[p]*N)
    return sum(coeff*mPoly(reversed(mu),p) for mu,coeff in x)

def scalar_Z(f,g):
    r"""
    Compute the N-component GMP scalar product ``<f, g>_Z``.
    It implements the inner product defined in Sec.3.2.3 of arxiv:2508.19704.

    The scalar product with respect to which the GMPs are orthogonal. It
    differs from the naive tensor product of Macdonald scalar products by a
    twist that mixes the spectral parameters ``u_i`` via powers of ``q3``.

    Parameters
    ----------
    f : element of ``Sym^{tensor N}``
    g : element of ``Sym^{tensor N}``

    Returns
    -------
    rational function
    """
    ev = lambda k,x: p(k)(x) if k!=[] else 1
    N = len(g.parent().tensor_factors())
    X = [tensor([p[1] if j==i else p[[]] for j in range(N)]) for i in range(N)]
    g = coercion_on_tensor(g,[p]*N)
    g = sum(coeff * prod(ev(mu[k],X[N-k-1]-(1-q3)*sum(q3**(j-N+k)*X[j] for j in range(N-k,N))) for k in range(N)) for mu,coeff in g)
    return scalar_on_tensor_qt(f,g)

def scalar_Z_prime(f,g):
    r"""
    Compute the primed N-component GMP scalar product ``<f, g>_Z'``.

    Same as :func:`scalar_Z` but with the tensor factors of ``g`` reversed
    before taking the scalar product.

    Parameters
    ----------
    f : element of ``Sym^{tensor N}``
    g : element of ``Sym^{tensor N}``

    Returns
    -------
    rational function
    """
    ev = lambda k,x: p(k)(x) if k!=[] else 1
    N = len(g.parent().tensor_factors())
    X = [tensor([p[1] if j==i else p[[]] for j in range(N)]) for i in range(N)]
    g = coercion_on_tensor(g,[p]*N)
    g = sum(coeff * prod(ev(mu[k],X[N-k-1]-(1-q3)*sum(q3**(j-N+k)*X[j] for j in range(N-k,N))) for k in range(N)) for mu,coeff in g)
    g = rev(g)
    return scalar_on_tensor_qt(f,g)


# ---------------------------------------------------------------------------
# Vertex operators and the generalised Macdonald operator
# ---------------------------------------------------------------------------

def xplus_k(k,x):
    r"""
    Apply the operator ``x^{+}_k`` to a symmetric function.
    {!!! the label `k` might be inverted w.r.t. usual conventions !!!}

    Implements the vertex operator generating the quantum toroidal
    algebra action at level one.

    Parameters
    ----------
    k : int
        Mode number.
    x : symmetric function

    Returns
    -------
    symmetric function
    """
    parent = x.parent()
    degree = x.degree()
    x = coercion_safe(x,s)
    if k >= 0:
        res = s([k])((1-t**-1)*s([1]))*x + sum(s([j+1+k])((1-t**-1)*s([1]))*x.skew_by(s([j+1])(-(1-q)*s([1]))) for j in range(degree))
    else:
        res = x.skew_by(s([-k])(-(1-q)*s([1]))) + sum(s([i+1])((1-t**-1)*s([1]))*x.skew_by(s([i+1-k])(-(1-q)*s([1]))) for i in range(degree))
    return coercion_safe(res,parent)

def xminus_k(k,x):
    r"""
    Apply the operator ``x^{-}_k`` to a symmetric function.
    {!!! the label `k` might be inverted w.r.t. usual conventions !!!}

    The `minus` counterpart of :func:`xplus_k`.

    Parameters
    ----------
    k : int
        Mode number.
    x : symmetric function

    Returns
    -------
    symmetric function
    """
    parent = x.parent()
    degree = x.degree()
    x = coercion_safe(x,s)
    if k >= 0:
        res = s([k])((1-t)*s([1]))*x + sum(s([j+1+k])((1-t)*s([1]))*x.skew_by(s([j+1])(-(1-q**-1)*s([1]))) for j in range(degree))
    else:
        res = x.skew_by(s([-k])(-(1-q**-1)*s([1]))) + sum(s([i+1])((1-t)*s([1]))*x.skew_by(s([i+1-k])(-(1-q**-1)*s([1]))) for i in range(degree))
    return coercion_safe(res,parent)

# def M_k(k,x):
#     parent = x.parent()
#     degree = x.degree()
#     x = coercion_safe(x,s)
#     if k >= 0:
#         res = s([k])((1-q2)*s([1]))*x + sum(s([j+1+k])((1-q2)*s([1]))*x.skew_by(s([j+1])(-(1-q1)*s([1]))) for j in range(degree))
#     else:
#         res = x.skew_by(s([-k])(-(1-q1)*s([1]))) + sum(s([i+1])((1-q2)*s([1]))*x.skew_by(s([i+1-k])(-(1-q1)*s([1]))) for i in range(degree))
#     return coercion_safe(res,parent)

# def Mast_k(k,x):
#     parent = x.parent()
#     degree = x.degree()
#     x = coercion_safe(x,s)
#     if k >= 0:
#         res = s([k])(-(1-q2)*s([1]))*x + sum(s([j+1+k])(-(1-q2)*s([1]))*x.skew_by(s([j+1])((1-q1)*q3*s([1]))) for j in range(degree))
#     else:
#         res = x.skew_by(s([-k])((1-q1)*q3*s([1]))) + sum(s([i+1])(-(1-q2)*s([1]))*x.skew_by(s([i+1-k])((1-q1)*q3*s([1]))) for i in range(degree))
#     return coercion_safe(res,parent)

def xplus_on_tensor(i,k,x):
    r"""
    Apply the operator ``x^{+}_k`` to the i-th factor of a tensor product.

    All other factors are left unchanged.

    Parameters
    ----------
    i : int
        Index of the factor to act on (0-indexed).
    k : int
        Mode number.
    x : element of ``Sym^{tensor N}``

    Returns
    -------
    element of ``Sym^{tensor N}``
    """
    parent = x.parent().tensor_factors()
    N = len(parent)
    return sum( coeff * tensor( [parent[j](mpart[j]) for j in range(i)]
                               +[xplus_k(k,parent[i](mpart[i]))]
                               +[parent[j](mpart[j]) for j in range(i+1,N)] ) for mpart,coeff in x)

def xminus_on_tensor(i,k,x):
    r"""
    Apply the operator ``x^{-}_k`` to the i-th factor of a tensor product.

    Parameters
    ----------
    i : int
        Index of the factor to act on (0-indexed).
    k : int
        Mode number.
    x : element of ``Sym^{tensor N}``

    Returns
    -------
    element of ``Sym^{tensor N}``
    """
    parent = x.parent().tensor_factors()
    N = len(parent)
    return sum( coeff * tensor( [parent[j](mpart[j]) for j in range(i)]
                               +[xminus_k(k,parent[i](mpart[i]))]
                               +[parent[j](mpart[j]) for j in range(i+1,N)] ) for mpart,coeff in x)

def LAM(i,k,x):
    r"""
    Apply the twisted operator ``Lambda_{i,k}`` to a tensor product.

    This is the i-th component of the full raising generator, with a twist by
    the spectral parameters ``u_j`` for ``j < i``:

        Lambda_{i,k} = sum_b  s_{k-b}(twist) * x^{+}_b(i)

    where ``twist = (1-q2)(1-q3) * sum_{j<i} X_j``.

    Parameters
    ----------
    i : int
        Component index.
    k : int
        Mode number.
    x : element of ``Sym^{tensor N}``

    Returns
    -------
    element of ``Sym^{tensor N}``
    """
    deg = degree_on_tensor(x)
    parent = x.parent().tensor_factors()
    N = len(parent)
    X = [tensor([p[1] if j==i else p[[]] for j in range(N)]) for i in range(N)]
    twist = (1-q2)*(1-q3)*sum(X[j] for j in range(i))
    if twist == 0:
        return xplus_on_tensor(i,k,x)
    else:
        return sum( s([k-b])(twist) * xplus_on_tensor(i,b,x) for b in range(-deg,k)) + xplus_on_tensor(i,k,x)

def LAMast(i,k,x):
    r"""
    Apply the twisted operator ``Lambda^*_{i,k}`` to a tensor product.

    The dual counterpart of :func:`LAM`, with a twist by the spectral parameters
    ``u_j`` for ``j > i``.

    Parameters
    ----------
    i : int
        Component index.
    k : int
        Mode number.
    x : element of ``Sym^{tensor N}``

    Returns
    -------
    element of ``Sym^{tensor N}``
    """
    deg = degree_on_tensor(x)
    parent = x.parent().tensor_factors()
    N = len(parent)
    X = [tensor([p[1] if j==i else p[[]] for j in range(N)]) for i in range(N)]
    twist = -(1-q1**-1)*(1-q3**-1)*sum(q3**j*X[j] for j in range(i+1,N))
    if twist == 0:
        return xminus_on_tensor(i,k,x)
    else:
        return sum( q3**(-i*(k+b))*skew_on_tensor(xminus_on_tensor(i,k+b,x),s([b])(twist)) for b in range(1,deg+1)) + q3**(-i*k)*xminus_on_tensor(i,k,x)

def xplus(k,x):
    r"""
    Apply the full N-component raising generator ``x^{+}_k`` to a tensor product.

    This is the k-th mode of the quantum toroidal algebra current
    acting diagonally across all N Fock space factors:

        x^{+}_k = sum_{i=0}^{N-1}  u_i * Lambda_{i,k}

    The GMP ``G_lam`` is an eigenfunction of ``x^{+}_0`` with eigenvalue
    ``eigenvalue(lam)``.

    Parameters
    ----------
    k : int
        Mode number. Use ``k=0`` for the eigenvalue equation.
    x : element of ``Sym^{tensor N}``

    Returns
    -------
    element of ``Sym^{tensor N}``

    See Also
    --------
    xminus, testEigenfunction, eigenvalue
    """
    N = len(x.parent().tensor_factors())
    return sum(ring(u[i])*LAM(i,k,x) for i in range(N))

def xminus(k,x):
    r"""
    Apply the full N-component lowering generator ``x^{-}_k`` to a tensor product.

        x^{-}_k = sum_{i=0}^{N-1}  u_i^{-1} * Lambda^*_{i,k}

    Parameters
    ----------
    k : int
        Mode number.
    x : element of ``Sym^{tensor N}``

    Returns
    -------
    element of ``Sym^{tensor N}``

    See Also
    --------
    xplus
    """
    N = len(x.parent().tensor_factors())
    return sum(ring(1/u[i])*LAMast(i,k,x) for i in range(N))

def testEigenfunction(mu):
    r"""
    Verify that ``G_mu`` is an eigenfunction of ``x^{+}_0``.

    Checks that ``x^{+}_0 G_mu == eigenvalue(mu) * G_mu``.

    Parameters
    ----------
    mu : tuple of list
        Multi-partition labelling the GMP to test.

    Returns
    -------
    bool

    Examples
    --------
    ::

        sage: testEigenfunction(([2,1], [1]))
        True
    """
    poly = GMP(mu)
    return xplus(0,poly) == eigenvalue(mu)*poly

def framing(x,power=1):
    r"""
    Apply the framing operator to a symmetric function.

    Multiplies the Macdonald P-coefficient of each ``P_lam`` by
    ``DET(chi2d(lam))^power``.

    Parameters
    ----------
    x : symmetric function
    power : int, optional
        Framing power (default 1).

    Returns
    -------
    symmetric function
    """
    par = x.parent()
    x = coercion_safe(x,McdP)
    return coercion_safe(sum( DET(chi2d(lam))**power*coeff*McdP(lam) for lam,coeff in x),par)

def framing_on_tensor(x,power=1):
    r"""
    Apply the framing operator to each GMP component of a tensor product element.

    For ``x = sum_mu c_mu * G_mu``, returns
    ``sum_mu c_mu * prod_k DET(u_k * chi2d(mu_k))^power * G_mu``.

    Parameters
    ----------
    x : element of ``Sym^{tensor N}``
    power : int, optional
        Framing power (default 1).

    Returns
    -------
    element of ``Sym^{tensor N}``
    """
    dd = to_gmp(x)
    return sum(dd[mu] * prod(DET(u[k]*chi2d(mu[k]))**power for k in range(len(mu))) * GMP(mu) for mu in dd)

def Delta(z,x,power=1,dual=1):
    r"""
    Apply the Delta operator to a symmetric function.

    Multiplies the coefficient of each ``P_mu`` by ``PE(-power * z * chi2d(mu, dual))``.

    Parameters
    ----------
    z : ring element
    x : symmetric function
    power : int, optional
        (default 1)
    dual : int, optional
        Exponent in ``chi2d(mu, dual)`` (default 1).

    Returns
    -------
    symmetric function
    """
    par = x.parent()
    x = coercion_safe(x,McdP)
    return coercion_safe( sum(PE(-power*z*chi2d(mu,dual))*c*McdP(mu) for mu,c in x) , par)

def Delta_on_tensor(z,x,power=1,dual=1):
    r"""
    Apply the Delta operator to a multi-symmetric function by expanding to the GMP basis.

    Parameters
    ----------
    z : ring element
    x : element of ``Sym^{tensor N}``
    power : int, optional
    dual : int, optional

    Returns
    -------
    element of ``Sym^{tensor N}``
    """
    dd = to_gmp(x)
    return sum(dd[mu] * prod(PE(-power*z*u[k]*chi2d(mu[k],dual)) for k in range(len(mu))) * GMP(mu) for mu in dd)


# ---------------------------------------------------------------------------
# GMP coefficient computation
# ---------------------------------------------------------------------------

def GMMatrixElement(lam,nu):
    r"""
    Compute one row of the eigenvalue-equation matrix for the GMP ``G_lam``.

    The GMP is defined as a null vector of the matrix
    ``[x^{+}_0(P_nu) - eigenvalue(lam) * delta_{mu,nu}]_{mu,nu}``.
    This function returns the row indexed by ``lam``.

    Parameters
    ----------
    lam : tuple of Partitions
        Multi-partition labelling the GMP.
    nu : tuple of Partitions
        Multi-partition labelling the column.

    Returns
    -------
    list
        Ring elements, one per multi-partition of the same degree as ``lam``.
    """
    if len(lam) != len(nu):
        return 0
    degree = sum(map(sum,lam))
    poly = xplus(0,mPoly(nu,McdP))
    poly = coercion_on_tensor(poly,[McdP]*len(nu))
    return [poly[mu] - eigenvalue(lam)*Kronecker_delta(mu,nu) for mu in mPartitions(len(lam),degree)]

def GMPC(lam,mu):
    r"""
    Compute the GMP change-of-basis coefficient ``C(lam, mu)``.

    This is the coefficient of ``P_mu`` (tensor product of Macdonald P-functions)
    in the expansion of the GMP ``G_lam``:

        G_lam = sum_mu  C(lam, mu) * P_{mu_0} x ... x P_{mu_{N-1}}

    Computed by finding the kernel of the eigenvalue-equation matrix; cached
    in ``GMPC_cache`` for reuse.

    Parameters
    ----------
    lam : tuple of Partitions
        Multi-partition labelling the GMP.
    mu : tuple of Partitions
        Multi-partition labelling the P-basis element.

    Returns
    -------
    ring element
        Rational function of ``q, t, u_i``. Returns 0 if lengths or degrees differ.

    Notes
    -----
    Results are cached in ``GMPC_cache``.  Use :func:`save_cache` /
    :func:`load_cache` to persist across sessions.

    See Also
    --------
    GMP, save_cache, load_cache
    """
    if len(lam) != len(mu):
        return 0

    global GMPC_cache
    N = len(lam)
    mp1 = tuple(map(Partition,lam))
    mp2 = tuple(map(Partition,mu))
    degree1 = sum(map(sum,mp1))
    degree2 = sum(map(sum,mp2))

    if degree1 != degree2:
        return 0

    if degree1 == 0:
        return 1

    if N == 1:
        for mp3 in mPartitions(N,degree1):
            GMPC_cache[(mp1,mp3)] = Kronecker_delta(mp1,mp3)
        return GMPC_cache[(mp1,mp2)]

    if lam[-1] == []:
        if mu[-1] == []:
            return GMPC(lam[:-1],mu[:-1])
        else:
            return 0

    try:
        return GMPC_cache[(mp1,mp2)]

    except KeyError:
        mparts = list(mPartitions(N,degree2))
        M = matrix([GMMatrixElement(mp1,b) for b in mparts])
        null_vec = M.kernel().basis()[0]
        for i in range(len(mparts)):
            GMPC_cache[(mp1,mparts[i])] = null_vec[i]
        return GMPC_cache[(mp1,mp2)]

def GMP(lam):
    r"""
    Compute the Generalized Macdonald P-function ``G_lam``.

    The GMP for an N-tuple of partitions ``lam`` is the unique (up to scalar)
    simultaneous eigenfunction of the zero-mode ``x^{+}_0`` with eigenvalue
    ``eigenvalue(lam)``, expanded in the tensor product of Macdonald P-functions:

        G_lam = sum_mu  C(lam, mu) * P_{mu_0} x ... x P_{mu_{N-1}}

    At N=1 the GMP reduces to the standard Macdonald P-function ``P_lam``.

    Parameters
    ----------
    lam : tuple of Partitions
        N-tuple of partitions, e.g. ``([2,1], [1])`` for N=2.

    Returns
    -------
    element of ``Sym^{tensor N}``

    Examples
    --------
    ::

        sage: G = GMP(([2], [1]))
        sage: testEigenfunction(([2], [1]))
        True

    See Also
    --------
    GMQ, GMPC, tildeGMP, barGMP
    """
    degree = sum(map(sum,lam))
    N = len(lam)
    mparts = mPartitions(N,degree)
    return sum(mPoly(mu,McdP)*GMPC(lam,mu) for mu in mparts)

def GMQ(lam):
    r"""
    Compute the Generalized Macdonald Q-function ``GMQ_lam``.

    The dual of GMP w.r.t. the inner product `scalar_Z`

    Parameters
    ----------
    lam : tuple of Partitions

    Returns
    -------
    element of ``Sym^{tensor N}``

    See Also
    --------
    GMP, b_lambda
    """
    N = len(lam)
    rlam = tuple(reversed(lam))
    poly = sum(coeff.subs({u[i]:u[N-i-1] for i in range(N)}) * mPoly(mu,p) for mu,coeff in GMP(rlam))
    return poly * prod(b_lambda(lam[i]) for i in range(N))

def tildeGMP(lam):
    r"""
    Compute the normalised GMP ``tilde{G}_lam``.

    Divides ``G_lam`` by the product of principal specialisations:

        tilde{G}_lam = G_lam / prod_{k} P_{lam_k}(epsilon([]))

    Parameters
    ----------
    lam : tuple of Partitions

    Returns
    -------
    element of ``Sym^{tensor N}``
    """
    return GMP(lam)/prod(evalArg(McdP(lam[k]),epsilon([])) for k in range(len(lam)))

def barGMP(lam):
    r"""
    Compute the GMP in a different normalisation {legacy; to be removed}.

    Divides ``G_lam`` by its evaluation at ``(u_k * epsilon([]))``.

    Parameters
    ----------
    lam : tuple of Partitions

    Returns
    -------
    element of ``Sym^{tensor N}``
    """
    N = len(lam)
    poly = GMP(lam)
    return poly/evaluate_on_tensor(poly,[u[k]*epsilon([]) for k in range(N)])

def GMK(lam):
    r"""
    Compute the generalised (inhomogeneous) Macdonald K-function. Useful to check
    the generalised GHT.

    Applies framing and a diagonal plethysm with a shifted epsilon argument to
    the normalised GMP, then sets ``r -> -1``.

    Parameters
    ----------
    lam : tuple of Partitions

    Returns
    -------
    element of ``Sym^{tensor N}``
    """
    N = len(lam)
    return subsr(diag_plethysm(framing_on_tensor(tildeGMP(lam)),[q2*p[1]-r*epsilon([])*q3**k for k in range(N)]))

def iGMP(lam):
    r"""
    Compute the inverted GMP with reversed spectral parameters.

    Parameters
    ----------
    lam : tuple of Partitions

    Returns
    -------
    element of ``Sym^{tensor N}``
    """
    N = len(lam)
    ev = lambda part,x: p(part)(x) if part!=[] else 1
    return sum(coeff.subs({u[i]:u[N-i-1] for i in range(N)}) * mPoly(mu,p) for mu,coeff in GMP(tuple(reversed(lam))))

def GMPast(lam):
    r"""
    Compute the starred (inhomogeneous) GMP ``G^*_lam``.

    Applies framing, a diagonal plethysm with a framed epsilon argument, and
    then inverts the framing, with the substitution ``r -> -1``.

    Parameters
    ----------
    lam : tuple of Partitions

    Returns
    -------
    element of ``Sym^{tensor N}``
    """
    return subsr(framing_on_tensor(diag_plethysm(framing_on_tensor(GMP(lam)),[s[1]+r*epsilon([])*q3**k*s[0] for k in range(len(lam))]),-1))

def to_gmp(x):
    r"""
    Decompose a multi-symmetric function in the GMP basis.

    Returns the dictionary ``{lam: c_lam}`` such that
    ``x = sum_lam c_lam * G_lam``, computed via the scalar product
    ``<x, GMQ_mu>_Z``.

    Parameters
    ----------
    x : element of ``Sym^{tensor N}``

    Returns
    -------
    dict
        Keys are tuples of ``Partition`` objects; values are ring elements.
    """
    if x == 0:
        return dict({})
    deg = degree_on_tensor(x)
    N = len(x.parent().tensor_factors())
    dd = {}
    for n in range(deg+1):
        for mu in mPartitions(N,n):
            if n == 0:
                coeff = counit_on_tensor2(x)
            else:
                coeff = scalar_Z(x,GMQ(mu))
            if coeff != 0:
                dd[tuple(mu)] = coeff
    return dd

def to_gmp2(x):
    r"""
    Decompose a multi-symmetric function in the GMP basis (matrix method).

    Alternative to :func:`to_gmp` that solves a linear system rather than
    using the scalar product. May be faster when many coefficients are needed.

    Parameters
    ----------
    x : element of ``Sym^{tensor N}``

    Returns
    -------
    dict
        Keys are tuples of ``Partition`` objects; values are ring elements.
    """
    if x == 0:
        return dict({})
    deg = degree_on_tensor(x)
    N = len(x.parent().tensor_factors())
    x = coercion_on_tensor(x,[McdP]*N)
    dd = dict({})
    ll = flatten(list(list(mPartitions(N,n)) for n in range(deg+1)),max_level=1)
    A = Matrix([[GMPC(mu,lam) for mu in ll] for lam in ll])
    b = vector([x[mu] for mu in ll])
    sol = flatten(A.solve_right(b))
    for i in range(len(ll)):
        if sol[i] != 0:
            dd[ll[i]] = sol[i]
    return dd


# ---------------------------------------------------------------------------
# Pieri rules
# ---------------------------------------------------------------------------

def part_plus(m,mu):
    r"""
    Return all partitions obtainable from ``mu`` by adding ``m`` boxes.

    Parameters
    ----------
    m : int
        Number of boxes to add. Returns ``[]`` if ``m < 0``.
    mu : Partition

    Returns
    -------
    list of Partitions
    """
    if m<0:
        return list()
    return list(s(e[m]*s(mu)).support())

def part_minus(m,mu):
    r"""
    Return all partitions obtainable from ``mu`` by removing ``m`` boxes.

    Parameters
    ----------
    m : int
        Number of boxes to remove. Returns ``[]`` if ``m < 0``.
    mu : Partition

    Returns
    -------
    list of Partitions
    """
    if m<0:
        return list()
    return list(s(s(mu).skew_by(e[m])).support())

def pieri_set_plus(m,lam):
    r"""
    Compute the Pieri set for adding ``m`` boxes to ``lam``.

    Returns all multi-partitions ``nu`` of total weight ``|lam| + m`` such that
    each component ``nu_i`` is obtainable from ``lam_i`` by adding boxes.

    Parameters
    ----------
    m : int
        Number of boxes to add.
    lam : tuple of Partitions

    Returns
    -------
    filter
    """
    N = len(lam)
    return filter(lambda nu: all(nu[i] in part_plus(sum(nu[i])-sum(lam[i]),lam[i]) for i in range(N)), mPartitions(N,sum(map(sum,lam))+m))

def pieri_set_minus(m,nu):
    r"""
    Compute the dual Pieri set for removing ``m`` boxes from ``nu``.

    Returns all multi-partitions ``lam`` of total weight ``|nu| - m`` such that
    each component ``lam_i`` is obtainable from ``nu_i`` by removing boxes.

    Parameters
    ----------
    m : int
        Number of boxes to remove.
    nu : tuple of Partitions

    Returns
    -------
    filter
    """
    N = len(nu)
    return filter(lambda lam: all(lam[i] in part_minus(sum(nu[i])-sum(lam[i]),nu[i]) for i in range(N)), mPartitions(N,sum(map(sum,nu))-m))

def pieriTest(lam):
    r"""
    Verify the elementary Pieri rule for the GMP ``G_lam``.

    Checks that:

        e_1(X_0 + ... + X_{N-1}) * G_lam
        = sum_{i, nu: lam_i -> nu}  alpha_N(i, nu, lam) * psi'(nu, lam_i) * G_{lam with lam_i -> nu}

    Parameters
    ----------
    lam : tuple of Partitions

    Returns
    -------
    bool
    """
    N = len(lam)
    X = [tensor([p[1] if j==i else p[[]] for j in range(N)]) for i in range(N)]
    lhs = e[1](sum(X))*GMP(lam)
    rhs = sum( sum(alpha_N(i,nu,lam)*psi_prime_PE(nu,lam[i])*GMP([lam[j] for j in range(i)]+[nu]+[lam[j] for j in range(i+1,N)]) for nu in Partition(lam[i]).up() ) for i in range(N))
    return rhs==lhs

def pieriTestDual(nu):
    r"""
    Verify the dual (skew) Pieri rule for the GMP ``G_nu``.

    Parameters
    ----------
    nu : tuple of Partitions

    Returns
    -------
    bool
    """
    N = len(nu)
    X = [tensor([p[1] if j==i else p[[]] for j in range(N)]) for i in range(N)]
    lhs = skew_on_tensor(GMP(nu),e[1]((1-q)/(1-t)*sum(q3**(i)*X[i] for i in range(N))))
    rhs = sum( prod( alpha2_Z(i,nu,lam[i])*psi2_prime_PE(nu[i],lam[i]) for i in range(N)) * GMP(lam) for lam in pieri_set_minus(1,nu))
    return lhs==rhs


# ---------------------------------------------------------------------------
# Output / formatting utilities
# ---------------------------------------------------------------------------

def part_to_str(mu):
    r"""
    Convert a partition to a Mathematica-style string ``{a, b, c}``.

    Parameters
    ----------
    mu : Partition

    Returns
    -------
    str
        E.g. ``[2,1]`` -> ``"{2,1}"``, ``[]`` -> ``"{}"``.
    """
    if mu==[]:
        return "{}"
    ss = "{"
    for k in mu:
        ss += str(k)+","
    return ss[:-1]+"}"

def _part_to_str(mu):
    r"""
    Convert a multi-partition to a comma-separated string of ``{a,b}`` blocks.

    Internal helper used by :func:`to_math` (no outer braces).

    Parameters
    ----------
    mu : tuple or list of list

    Returns
    -------
    str
    """
    res = ""
    for k in mu:
        if k == []:
            res += "{},"
        else:
            res += "{"
            for s in k:
                res += str(s)+","
            res = res[:-1] + "},"
    return res[:-1]

def mpart_to_str(mu):
    r"""
    Convert a multi-partition to a Mathematica-style string ``{{a,b},{c},...}``.

    Parameters
    ----------
    mu : tuple or list of list

    Returns
    -------
    str
    """
    res = "{"
    for k in mu:
        if k == []:
            res += "{},"
        else:
            res += "{"
            for s in k:
                res += str(s)+","
            res = res[:-1] + "},"
    return res[:-1]+"}"

def to_math(x):
    r"""
    Format a multi-symmetric function as a Mathematica-style sum string.

    Parameters
    ----------
    x : element of ``Sym^{tensor N}``  (in the power-sum basis)

    Returns
    -------
    str
        A string like ``"coeff1*p[{2,1},{1}]+coeff2*p[{3},{}]+..."``.
    """
    res = ""
    for mu,coeff in x:
        res += str(factor(coeff))+"*p["+_part_to_str(mu)+"]+"
    return res[:-1]

def to_math_l(x):
    r"""
    Format the coefficients of a multi-symmetric function as a Mathematica list.

    Parameters
    ----------
    x : element of ``Sym^{tensor N}``

    Returns
    -------
    str
        A Mathematica list string ``"{c1, c2, ...}"``.
    """
    res = ""
    for mu,coeff in x:
        res += str(factor(coeff))+","
    return "{"+res[:-1]+"}"
