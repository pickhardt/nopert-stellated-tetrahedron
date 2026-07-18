"""Exact-form order-4 remainder bound (Lemma C4), majorant method, pure Fraction.

h(delta) = nt.(in_j - q_a)/|nt|; all factors entire in delta. Majorants
(coefficientwise, nonneg coefficients; constants rounded soundly:
nu DOWN, W/RJ/D0/B/t* UP):

  NT   = nu + W_ab*(e^d - 1)                       [shift row: (e^d-1-d)/d]
  DIFF = D0 + RJ*(e^{(B+1)d} - 1) + W_ja*(e^d - 1) + t*_ d
  U    = (2 nu W_ab E + W_ab^2 E^2)/nu^2,  E = e^d-1  [resp. shifted]
  H    = NT*DIFF/(nu*(1-U))

|h - P3|(delta) <= delta^4 * C4 for 0 <= delta <= rho0, where
  C4 = sum_{k=4..K} c_k(H) rho0^{k-4} + tail,
  tail = H(r)/r^4 * (rho0/r)^{K-3}/(1-rho0/r)   (nonneg coefficients => c_k <= H(r)/r^k).
"""
from fractions import Fraction as Fr
import math

K = 12

def series_exp(a, K=K):
    """coefficients of e^{a d} - up to K; a Fraction."""
    c = [Fr(1)]
    for k in range(1, K+1): c.append(c[-1]*a/k)
    return c
def smul(A, B, K=K):
    C = [Fr(0)]*(K+1)
    for i, ai in enumerate(A):
        if ai == 0: continue
        for j, bj in enumerate(B):
            if i+j > K: break
            C[i+j] += ai*bj
    return C
def sadd(A, B): return [a+b for a,b in zip(A,B)]
def sscale(A, c): return [a*c for a in A]
def sgeom(U, K=K):
    """1/(1-U) with U[0]==0."""
    assert U[0] == 0
    R = [Fr(1)] + [Fr(0)]*K
    P = [Fr(1)] + [Fr(0)]*K
    for m in range(1, K+1):
        P = smul(P, U, K)
        R = sadd(R, P)
        if all(p == 0 for p in P): break
    return R

def C4_row(nu, D0, RJ, W_ab, W_ja, B, tstar, rho0=Fr(1,1000), shift=0):
    """nu: Fraction LOWER bound on sqrt(ns0) [shift row: on |c_1(nt)|];
    others Fraction UPPER bounds. Returns Fraction upper bound on C4."""
    E = series_exp(Fr(1));  E1 = list(E); E1[0] -= 1              # e^d - 1
    if shift == 0:
        Em = E1
    else:
        Em = [E1[k+1] for k in range(K)] + [Fr(0)]                # (e^d-1-d)/d shifted
        Em[0] = Fr(0)  # (e^d-1)/d = 1 + d/2 + ...; subtract the 1 (constant into nu? no:)
        # (e^d - 1 - d)/d : coefficients c_k = 1/(k+1)! for k>=1, c_0 = 0
        Em = [Fr(0)] + [Fr(1)/math.factorial(k+1) for k in range(1, K+1)]
        Em = [Fr(e) if not isinstance(e, Fr) else e for e in Em]
    NT = sadd([nu]+[Fr(0)]*K, sscale(Em, W_ab))
    EB = series_exp(B+1); EB1 = list(EB); EB1[0] -= 1
    DIFF = sadd(sadd(sscale(EB1, RJ), sscale(E1, W_ja)), [D0, tstar]+[Fr(0)]*(K-1))
    U = sadd(sscale(Em, 2*W_ab/nu), sscale(smul(Em, Em), (W_ab/nu)**2))
    H = smul(smul(NT, DIFF), sgeom(U))
    H = sscale(H, Fr(1)/nu)
    C4 = sum(H[k]*rho0**(k-4) for k in range(4, K+1))
    # tail via evaluation at r: need H(r) upper; coefficients nonneg so evaluate
    # the majorant functions at r with outward float rounding (x1.01 inflation)
    r = Fr(1,20)
    er = Fr(math.ceil(math.exp(float(r))*10**12), 10**12)
    ebr = Fr(math.ceil(math.exp(float((B+1)*r))*10**12), 10**12)
    E1r = er-1
    Emr = E1r if shift == 0 else (er-1-r)/r
    NTr = nu + W_ab*Emr
    DIFFr = D0 + RJ*(ebr-1) + W_ja*E1r + tstar*r
    Ur = 2*W_ab*Emr/nu + (W_ab*Emr/nu)**2
    assert Ur < 1
    Hr = NTr*DIFFr/(nu*(1-Ur))
    q = rho0/r
    tail = Hr/r**4 * q**(K-3)/(1-q)
    return C4 + tail

def fr_up(x, den=10**9):   return Fr(math.ceil(x*den), den)
def fr_down(x, den=10**9): return Fr(math.floor(x*den), den)

if __name__ == '__main__':
    s3 = math.sqrt(3)
    for nu2, name in [(323/400,'nu2=323/400'), (1123/400,'nu2=1123/400'), (8,'nu2=8')]:
        for B in (3.39, 2.15, 0.15):
            v = C4_row(nu=fr_down(math.sqrt(nu2)), D0=fr_up(2*s3), RJ=fr_up(s3),
                       W_ab=fr_up(2*s3), W_ja=fr_up(2*s3), B=fr_up(B),
                       tstar=fr_up(0.0285*math.sqrt(2)))
            print('%-14s B=%.2f  C4 <= %.1f' % (name, B, float(v)))
