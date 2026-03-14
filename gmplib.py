r"""
Generalized Macdonald Functions

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

# Set up the base ring, the symbols and useful basis

import time
from sage.all import *
N = 30
param = list('u%d'%i for i in range(N)) + list('v%d'%i for i in range(N)) + ['q','t','r','z','Q']

Sym = SymmetricFunctions(FractionField(PolynomialRing(QQ,param)))
ring = Sym.base_ring()
(q,t,r,z,Q) = [ring.gens()[i] for i in range(2*N,2*N+5)]
u = list(ring(var('u%d' % i)) for i in range(N))
v = list(ring(var('v%d' % i)) for i in range(N))
q1=q
q2=1/t
q3=t/q
kappa1=(1-q1)*(1-q2)*(1-q3)

L = LaurentPolynomialRing(QQ,param)
F = L.fraction_field()

McdP = Sym.macdonald().P()
McdQ = Sym.macdonald().Q()
Ht = Sym.macdonald(q=q1,t=q2).Ht()
p = Sym.powersum()
e = Sym.e()
h = Sym.homogeneous()
s = Sym.Schur()

GMPC_cache = dict({})


# Functions to save and load cache to and from a file

from sage.misc.persist import load, save

def gmp_init(dd=dict({})):
    global GMPC_cache
    GMPC_cache = dd

def save_cache(file_name):
    global GMPC_cache
    try:
        save(GMPC_cache, filename=file_name)
        print("Saved cache to file:", file_name+".sobj")
    except:
        print("Saving failed!")

def cache_to_dict():
    global GMPC_cache
    return GMPC_cache

def load_cache(file_name):
    try:
        obj = load(file_name)
        print("Loaded cache from file:", file_name)
        return obj
    except:
        print("Cache file not found. Starting from scratch.")
        return dict({})


# Implement functions to list all multi-partitions of a certain total degree

from itertools import product,permutations
from sympy.utilities.iterables import partitions

def partitions_up_to_k(k):
    for n in range(k+1):
        for partition in Partitions(n):
            yield partition

def vectors_with_partitions(N,k):
    return filter(lambda vpart: sum(map(sum, vpart)) == k, product(partitions_up_to_k(k),repeat=N))

def vectors_with_int(N,k):
    return filter(lambda vint: sum(vint) == k, product(range(k+1),repeat=N))

def generate_combinations(N, d, min_val=0):
    possible_values = range(min_val, d + 1)
    all_combinations = product(possible_values, repeat=N)
    valid_combinations = filter(lambda comb: sum(comb) == d, all_combinations)    
    return valid_combinations

def mPartitions(N,k):
    return vectors_with_partitions(N,k)

def mPoly(mpart,parent):
    return tensor(list(map(lambda mu: coercion_safe(parent(mu),p),mpart)))

def is_reduced(lam):
    return lam[-1] != []

def is3d(mu):
    N = len(mu)
    res = True
    for i in range(N-1):
        res = res and Partition(mu[i+1]).contains(Partition(mu[i]))
        if res == False:
            return res
    return res

def subpart(lam):
    deg = sum(lam)
    return filter(lambda x:Partition(lam).contains(x),partitions_up_to_k(deg))


# Define useful functions

def nfactor(x):
    if x==0:
        return x
    else:
        return factor(x)

def Kronecker_delta(x,y):
    return 1 if x == y else 0

def epsilon(part,power=1):
    res = sum(q**(part[i])*t**(-i-1) for i in range(len(part))) + t**(-len(part)-1)/(1-1/t)
    return res.subs(q=q**power,t=t**power)

def eigenvalue(lam):
    return sum(u[i]*x2d(lam[i]) for i in range(len(lam)))

def w(mu):
    return 1/e[sum(mu)](Ht[1]/(1-q1)/(1-q2)).coefficient(mu)

def chi2d(lam,power=1):
    return sum(sum((q2**i*q1**j)**power for j in range(lam[i])) for i in range(len(lam)))

def x2d(lam,power=1):
    return 1-(1-q1**power)*(1-q2**power)*chi2d(lam,power)

def PE(x):
    x = L(x)
    return ring(prod((1-monomial)**-coeff for coeff,monomial in x))

def DET(x):
    x = L(x)
    return prod(z**c for c,z in x)

def Ylam(lam,z):
    return PE(-x2d(lam)/z)

def PSIlam(lam,z):
    return PE((1-q3)*x2d(lam)/z)

def Nekrasov(lam,mu,z):
    return PE(-z*(1-x2d(lam)*x2d(mu,-1))/(1-q1)/(1-q2))

def NekrasovJEB(lam,mu,z):
    return PE(z*(1-q1)*(1-q2)*chi2d(lam)*chi2d(mu,-1)-z/q3*chi2d(lam)-z*chi2d(mu,-1))

def tNek(lam,mu,z):
    return NekrasovJEB(lam,mu,z) / (-z/q3)**sum(lam) / DET(chi2d(lam))

def psi_prime_PE(nu,lam):
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

def phi_prime_PE(nu,lam):
    nu = Partition(nu)
    lam = Partition(lam)
    m = sum(nu)-sum(lam)
    if m == 0:
        rhs = 1 if nu == lam else 0
    else:
        if lam == []:
            rhs = PE(-chi2d([m])*q - (1-q)*(chi2d(nu,-1)-chi2d(lam,-1))*(chi2d(nu)-chi2d(lam)) + (chi2d(nu,-1)-chi2d(lam,-1)) * x2d(lam) ) * (McdP[m](epsilon([]))) / McdP(nu)(epsilon([]))
        else:
            rhs = PE(-chi2d([m])*q - (1-q)*(chi2d(nu,-1)-chi2d(lam,-1))*(chi2d(nu)-chi2d(lam)) + (chi2d(nu,-1)-chi2d(lam,-1)) * x2d(lam) ) * (McdP[m](epsilon([]))) * McdP(lam)(epsilon([])) / McdP(nu)(epsilon([]))
    return rhs

def psi2_prime_PE(nu,lam):
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
    N = len(lam)
    return PE( (1-q3)*(chi2d(nu,-1)-chi2d(lam[i],-1))*u[i]**-1*sum(u[j]*x2d(lam[j]) for j in range(i+1,N)) )

def alpha2_N(i,nu,lam):
    N = len(nu)
    return PE( (1-q3)*(chi2d(nu[i],-1)-chi2d(lam,-1))*u[i]**-1*sum(u[j]*x2d(nu[j]) for j in range(i)) )

def rtildeast(lam,z):
    if z in R(lam):
        return PE(-q3*x2d(lam)/z-1)
    else:
        return 0

def rtilde(lam,z):
    if z in A(lam):
        return PE(x2d(lam)/z-1)
    else:
        return 0

def tP(lam):
    if lam == []:
        return s.one()
    else:
        return s(McdP(lam))/McdP(lam)(epsilon([]))

def b_lambda(lam):
    lam = Partition(lam)
    return McdP.c2(lam)/McdP.c1(lam)

def blam(lam):
    if lam == []:
        return 1
    return 1/McdP(lam).scalar_qt(McdP(lam))

def b_tilde(lam):
    if lam == []:
        return 1
    return 1/tP(lam).scalar_qt(tP(lam))

def A(lam):
    return map(lambda x:chi2d(x)-chi2d(lam),Partition(lam).up())

def R(lam):
    return map(lambda x:chi2d(lam)-chi2d(x),Partition(lam).down())
    
def a_k(k,x):
    if k==0:
        return x
    if k>0:
        return -z**-k*(1-q1**k)*(1-q3**k)*x.skew_by(p[k])/k
    if k<0:
        k=-k
        return -z**-k*(1-q2**k)*(1-q3**k)*p[k]*x/k


# Define useful functions for manipulating multi-symmetric functions

def coercion_safe(x,f):
    if x == x.counit():
        return x.counit() * f.one()
    return f(x-x.counit()) + x.counit()*f.one()

def skew_on_tensor(x,y):
    parent1 = x.parent().tensor_factors()
    parent2 = y.parent().tensor_factors()
    return sum( sum( c1 * c2 * tensor(map(lambda k1,p1,k2,p2: p1(k1).skew_by(p2(k2)),mp1,parent1,mp2,parent2)) for mp1,c1 in x) for mp2,c2 in y)

def omega_on_tensor(x):
    parent = x.parent().tensor_factors()
    return sum( coeff * tensor([parent[i](mpart[i]).omega() for i in range(len(parent))]) for mpart,coeff in x)

def coercion_on_tensor(x,parent):
    old_parent = x.parent().tensor_factors()
    return sum( coeff * tensor(map(lambda part,old_base,base: coercion_safe(old_base(part),base),mpart,old_parent,parent)) for mpart,coeff in x)

def degree_on_tensor(x):
    return max(map(lambda part: sum(map(sum,part)),x.support()))

def scalar_on_tensor_qt(x,y):
    l = len(x.parent().tensor_factors())
    x = coercion_on_tensor(x,[McdP]*l)
    y = coercion_on_tensor(y,[McdQ]*l)
    return sum( sum( c1 * c2 * prod(map(lambda k1,k2: Kronecker_delta(k1,k2),lam,mu)) for lam,c1 in x) for mu,c2 in y)

def counit_on_tensor(x):
    parent = x.parent().tensor_factors()
    return sum( coeff * prod(pp(k).counit() for k,pp in zip(mu,parent)) for mu,coeff in x)

def counit_on_tensor2(x):
    return sum( coeff * prod(Kronecker_delta(k,[]) for k in mu) for mu,coeff in x)

def level(n,x):
    N = len(x.parent().tensor_factors())
    x = coercion_on_tensor(x,[p]*N)
    dd = dict(x)
    return sum(dd[key]*mPoly(key,p) for key in dd if sum(map(sum,key))==n)

def e1t(N):
    X = [tensor([p[1] if j==i else p.one() for j in range(N)]) for i in range(N)]
    return sum(X)/epsilon([])

def evalArg(x,arg):
    x0 = x.counit()
    x = x-x0
    return x0 + x(arg)

def evaluate_on_tensor(x,arg):
    N = len(x.parent().tensor_factors())
    x = coercion_on_tensor(x,[s]*N)
    return sum(coeff*prod(evalArg(s(mu[i]),arg[i]) for i in range(N)) for mu,coeff in x)

def diag_plethysm(x,arg_l):
    parent = x.parent().tensor_factors()
    N = len(parent)
    x = coercion_on_tensor(x,[p]*N)
    return sum( coeff * tensor([p(mu)(a) for mu,a in zip(part,arg_l)]) for part,coeff in x)

def subsr(x):
    parent = x.parent().tensor_factors()
    N = len(parent)
    return sum( coeff.subs(r=-1) * tensor(map(lambda part,base: base(part),mu,parent)) for mu,coeff in x)

def rev(x):
    N = len(x.parent().tensor_factors())
    x = coercion_on_tensor(x,[p]*N)
    return sum(coeff*mPoly(reversed(mu),p) for mu,coeff in x)

def scalar_N(f,g):
    ev = lambda k,x: p(k)(x) if k!=[] else 1
    N = len(g.parent().tensor_factors())
    X = [tensor([p[1] if j==i else p[[]] for j in range(N)]) for i in range(N)]
    g = coercion_on_tensor(g,[p]*N)
    g = sum(coeff * prod(ev(mu[k],X[N-k-1]-(1-q3)*sum(q3**(j-N+k)*X[j] for j in range(N-k,N))) for k in range(N)) for mu,coeff in g)
    return scalar_on_tensor_qt(f,g)

def scalar_N_prime(f,g):
    ev = lambda k,x: p(k)(x) if k!=[] else 1
    N = len(g.parent().tensor_factors())
    X = [tensor([p[1] if j==i else p[[]] for j in range(N)]) for i in range(N)]
    g = coercion_on_tensor(g,[p]*N)
    g = sum(coeff * prod(ev(mu[k],X[N-k-1]-(1-q3)*sum(q3**(j-N+k)*X[j] for j in range(N-k,N))) for k in range(N)) for mu,coeff in g)
    g = rev(g)
    return scalar_on_tensor_qt(f,g)


# Define the vertex operators and the generalized Macdonald operator

def xplus_k(k,x):
    parent = x.parent()
    degree = x.degree()
    x = coercion_safe(x,s)
    if k >= 0:
        res = s([k])((1-t**-1)*s([1]))*x + sum(s([j+1+k])((1-t**-1)*s([1]))*x.skew_by(s([j+1])(-(1-q)*s([1]))) for j in range(degree))
    else:
        res = x.skew_by(s([-k])(-(1-q)*s([1]))) + sum(s([i+1])((1-t**-1)*s([1]))*x.skew_by(s([i+1-k])(-(1-q)*s([1]))) for i in range(degree))
    return coercion_safe(res,parent)

def xminus_k(k,x):
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
    parent = x.parent().tensor_factors()
    N = len(parent)
    return sum( coeff * tensor( [parent[j](mpart[j]) for j in range(i)]
                               +[xplus_k(k,parent[i](mpart[i]))]
                               +[parent[j](mpart[j]) for j in range(i+1,N)] ) for mpart,coeff in x)

def xminus_on_tensor(i,k,x):
    parent = x.parent().tensor_factors()
    N = len(parent)
    return sum( coeff * tensor( [parent[j](mpart[j]) for j in range(i)]
                               +[xminus_k(k,parent[i](mpart[i]))]
                               +[parent[j](mpart[j]) for j in range(i+1,N)] ) for mpart,coeff in x)

def LAM(i,k,x):
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
    N = len(x.parent().tensor_factors())
    return sum(ring(u[i])*LAM(i,k,x) for i in range(N))

def xminus(k,x):
    N = len(x.parent().tensor_factors())
    return sum(ring(1/u[i])*LAMast(i,k,x) for i in range(N))

# def xminus2(k,x):
#     N = len(x.parent().tensor_factors())
#     return sum(ring(1/u[i])*LAMast(i,k,x)*q3**(-i*k) for i in range(N))

def testEigenfunction(mu):
    poly = GMP(mu)
    return xplus(0,poly) == eigenvalue(mu)*poly

def framing(x,power=1):
    par = x.parent()
    x = coercion_safe(x,McdP)
    return coercion_safe(sum( DET(chi2d(lam))**power*coeff*McdP(lam) for lam,coeff in x),par)

def framing_on_tensor(x,power=1):
    dd = to_gmp(x)
    return sum(dd[mu] * prod(DET(u[k]*chi2d(mu[k]))**power for k in range(len(mu))) * GMP(mu) for mu in dd)

def Delta(z,x,power=1,dual=1):
    par = x.parent()
    x = coercion_safe(x,McdP)
    return coercion_safe( sum(PE(-power*z*chi2d(mu,dual))*c*McdP(mu) for mu,c in x) , par)

def Delta_on_tensor(z,x,power=1,dual=1):
    dd = to_gmp(x)
    return sum(dd[mu] * prod(PE(-power*z*u[k]*chi2d(mu[k],dual)) for k in range(len(mu))) * GMP(mu) for mu in dd)


# Compute matrix elements of the change of basis from tensor products of McdP to GMP

def GMMatrixElement(lam,nu):
    if len(lam) != len(nu):
        return 0
    degree = sum(map(sum,lam))
    poly = xplus(0,mPoly(nu,McdP))
    poly = coercion_on_tensor(poly,[McdP]*len(nu))
    return [poly[mu] - eigenvalue(lam)*Kronecker_delta(mu,nu) for mu in mPartitions(len(lam),degree)]

def GMPC(lam,mu):
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
    degree = sum(map(sum,lam))
    N = len(lam)
    mparts = mPartitions(N,degree)
    return sum(mPoly(mu,McdP)*GMPC(lam,mu) for mu in mparts)

def specialize_u(x,power=1,reverse=False):
    parent = x.parent().tensor_factors()
    N = len(parent)
    if reverse:
        return sum( coeff.subs({u[k]:ring(q3**(power*k)) for k in range(N)}) * tensor(map(lambda part,base: base(part),mu,parent)) for mu,coeff in x)
    else:
        return sum( coeff.subs({u[k]:ring(q3**(power*(N-k-1))) for k in range(N)}) * tensor(map(lambda part,base: base(part),mu,parent)) for mu,coeff in x)

def GMQ(lam):
    N = len(lam)
    rlam = tuple(reversed(lam))
    poly = sum(coeff.subs({u[i]:u[N-i-1] for i in range(N)}) * mPoly(mu,p) for mu,coeff in GMP(rlam))
    return poly * prod(b_lambda(lam[i]) for i in range(N))

def tildeGMP(lam):
    return GMP(lam)/prod(evalArg(McdP(lam[k]),epsilon([])) for k in range(len(lam)))

def barGMP(lam):
    N = len(lam)
    poly = GMP(lam)
    return poly/evaluate_on_tensor(poly,[u[k]*epsilon([]) for k in range(N)])

def GMK(lam):
    N = len(lam)
    return subsr(diag_plethysm(framing_on_tensor(tildeGMP(lam)),[q2*p[1]-r*epsilon([])*q3**k for k in range(N)]))

def iGMP(lam):
    N = len(lam)
    ev = lambda part,x: p(part)(x) if part!=[] else 1
    return sum(coeff.subs({u[i]:u[N-i-1] for i in range(N)}) * mPoly(mu,p) for mu,coeff in GMP(tuple(reversed(lam))))

def GMPast(lam):
    return subsr(framing_on_tensor(diag_plethysm(framing_on_tensor(GMP(lam)),[s[1]+r*epsilon([])*q3**k*s[0] for k in range(len(lam))]),-1))

def to_gmp(x):
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
                coeff = scalar_N(x,GMQ(mu))
            if coeff != 0:
                dd[tuple(mu)] = coeff
    return dd

def to_gmp2(x):
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


# Pieri stuff

def pieri(lam):
    N = len(lam)
    X = [tensor([p[1] if j==i else p[[]] for j in range(N)]) for i in range(N)]
    lhs = e[1](sum(X))*GMP(lam)
    rhs = sum( sum(alpha_N(i,nu,lam)*psi_prime_PE(nu,lam[i])*GMP([lam[j] for j in range(i)]+[nu]+[lam[j] for j in range(i+1,N)]) for nu in Partition(lam[i]).up() ) for i in range(N))
    return rhs==lhs

def part_plus(m,mu):
    if m<0:
        return list()
    return list(s(e[m]*s(mu)).support())

def part_minus(m,mu):
    if m<0:
        return list()
    return list(s(s(mu).skew_by(e[m])).support())

def pieri_set(m,lam):
    N = len(lam)
    return filter(lambda nu: all(nu[i] in part_plus(sum(nu[i])-sum(lam[i]),lam[i]) for i in range(N)), mPartitions(N,sum(map(sum,lam))+m))

def pieri_set_minus(m,nu):
    N = len(nu)
    return filter(lambda lam: all(lam[i] in part_minus(sum(nu[i])-sum(lam[i]),nu[i]) for i in range(N)), mPartitions(N,sum(map(sum,nu))-m))

def pieriTest(m,lam):
    N = len(lam)
    X = [tensor([p[1] if j==i else p[[]] for j in range(N)]) for i in range(N)]
    lhs = e[m](sum(X))*GMP(lam)
    rhs = sum( prod(alpha_N(i,nu[i],lam)*psi_prime_PE(nu[i],lam[i]) for i in range(N)) * GMP(nu) for nu in pieri_set(m,lam))
    return lhs==rhs

def pieriTestDual(m,nu):
    N = len(nu)
    X = [tensor([p[1] if j==i else p[[]] for j in range(N)]) for i in range(N)]
    lhs = skew_on_tensor(GMP(nu),e[m]((1-q)/(1-t)*sum(q3**(i)*X[i] for i in range(N))))
    rhs = sum( prod( alpha2_N(i,nu,lam[i])*psi2_prime_PE(nu[i],lam[i]) for i in range(N)) * GMP(lam) for lam in pieri_set_minus(m,nu))
    return lhs==rhs


# MacMahon stuff

def Pnm(n,m,N):
    X = [tensor([p[1] if j==i else p.one() for j in range(N)]) for i in range(N)]
    return (1-q/t)*sum(q3**((m-1)*(N-k-1))*sum(sum((-q/t)**(i-b)*s([b]+[1]*(i-b)) for b in range(m,i+1))((1-t)*X[k])*splits(n-i,k,N) for i in range(m,n+1)) for k in range(N))

def splits(n,k,N):
    if n < 0:
        return 0
    if n == 0:
        return mPoly([[]]*N,p)
    if n > 0:
        if k >= N or k < 0:
            return 0
        else:
            X = [tensor([p[1] if j==i else p.one() for j in range(N)]) for i in range(N)]
            poly = (1-q/t)*(1-t)*sum(X[j] for j in range(k))
            if poly == 0:
                return 0
            else:
                return s[n](poly)


# Screening stuff

def psi_plus(poly,n):
    degree = degree_on_tensor(poly)
    N = 2
    X = [tensor([p[1] if j==i else p[[]] for j in range(N)]) for i in range(N)]
    A = -(1-t)/(1-q)*(q3*X[0]-X[1])
    B = (X[0]-X[1])*q1*q2
    if n > 0:
        return s[n](A)*poly + sum(s[n+i](A)*skew_on_tensor(poly,s[i](B)) for i in range(1,degree+1))
    if n == 0:
        return poly + sum(s[i](A)*skew_on_tensor(poly,s[i](B)) for i in range(1,degree+1))
    else:
        return skew_on_tensor(poly,s[-n](B)) + sum(s[n+i](A)*skew_on_tensor(poly,s[i](B)) for i in range(-n+1,degree+1))


# Output stuff

def part_to_str(mu):
    if mu==[]:
        return "{}"
    ss = "{"
    for k in mu:
        ss += str(k)+","
    return ss[:-1]+"}"

def _part_to_str(mu):
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
    res = ""
    for mu,coeff in x:
        res += str(factor(coeff))+"*p["+_part_to_str(mu)+"]+"
    return res[:-1]

def to_math_l(x):
    res = ""
    for mu,coeff in x:
        res += str(factor(coeff))+","
    return "{"+res[:-1]+"}"
