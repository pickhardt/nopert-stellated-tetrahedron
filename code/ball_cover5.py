"""Ball-covering v4 — PROVEN Lipschitz bounds (see LAMBDA_BOUNDS_PROOF.md).
Rotation: Λ_s=√3, Λ_dt=√3·s_hi, Λ_dp=√3·s_hi (proven, norm-preservation + ‖dexp‖≤1).
u₂: Λ_th,Λ_ph = |Ω|·Σλ[(2√3/L_i)|q_a−p_j| + 2√3], with L_i (edge len), |q_a−p_j| (pokeout),
|Ω|≤1/|e₃×u₂| ENCLOSED per box via fast_interval. All rigorous ⇒ G.lo>0 is a proof.
Much tighter near the pinch (Λ_dt=√3 s→0) ⇒ far fewer boxes than the empirical L=3."""
import numpy as np, math, time, json, sys
from fractions import Fraction as Fr
import fast_interval as F
import ball_cover as BC
from collections import deque
M=0.866034; smax=math.pi; MINW=5e-5; aa=Fr(11,20); R3=math.sqrt(3.0)
def _amax(iv): return max(abs(iv.lo),abs(iv.hi))
def _amin_pos(iv):  # rigorous lower bound on |value|, assuming interval doesn't straddle 0
    if iv.lo>0: return iv.lo
    if iv.hi<0: return -iv.hi
    return 0.0
POKE=2*R3    # proven upper bound |q_a − p_j| ≤ |q_a|+|p_j| ≤ 2√3 (norm-preservation, |π|≤1)
def u2_lip(box,eo,lam):
    """FULLY PROVEN Λ_th=Λ_ph = |Ω|·Σλ[(2√3/L_i)·POKE + 2√3], with L_i and |Ω| ENCLOSED over the
    box (interval outer shadow, no rodrigues). |Ω| ≤ 1/|e3×u2|. Loose (~5×) but rigorous."""
    th,thw,ph,phw,dt,dtw,dp,dpw,s,sw=box
    W=F.frame(F.iu2(th-thw,th+thw,ph-phw,ph+phw)); V=F.verts(aa)
    qo=[F.proj(F.applyW(W,v)) for v in V]
    u2=W[2]; exu=(u2[0].sqr()+u2[1].sqr()).sqrt(); Om=1.0/max(exu.lo,1e-6)
    S=0.0
    for k,w in lam:
        ei,vj=k//8,k%8; p_,r_,sgn=eo[ei]
        d0=qo[r_][0]-qo[p_][0]; d1=qo[r_][1]-qo[p_][1]
        Li=(d0.sqr()+d1.sqr()).sqrt().lo; Li=max(Li,1e-6)      # lower bound on edge length over box
        S+=w*((2*R3/Li)*POKE+2*R3)
    Lam=Om*S
    return Lam,Lam
def Gpoint(th,ph,dt,dp,s,eo,lam):
    return F.G_lambda_mv(F.iu2(th,th,ph,ph),F.idir(dt,dt,dp,dp),s,0.0,aa,eo,lam)
USE_AA=True     # tighten the u2 Lipschitz constants with affine arithmetic when the cheap bound fails
def certify(box,eo,lam):
    th,thw,ph,phw,dt,dtw,dp,dpw,s,sw=box
    Gc=Gpoint(th,ph,dt,dp,s,eo,lam); s_hi=s+sw
    rot_dir=(R3*s_hi)*dtw+(R3*s_hi)*dpw            # crude dir term R3*s*(dtw+dpw) -- measured ~3-8x loose
    rot_s=R3*sw                                    # s term (not s-amplified; kept crude, small)
    Lth,Lph=u2_lip(box,eo,lam)                     # proven worst-case product u2 bound (~37x loose)
    margin=Gc.lo-(Lth*thw+Lph*phw+rot_dir+rot_s)
    # GATED AFFINE-ARITHMETIC RETRY: the worst-case u2 AND dir bounds are loose. Only when the cheap
    # bound FAILS and the (tight) s-term alone leaves positive room (rot_s<Gc.lo, so tighter u2+dir
    # terms could still certify) do we pay for the affine-arithmetic Lambdas. Each AA bound is a proven
    # upper bound on sup_cell|dG/dparam| (glam_lip: theta/phi; glam_lip_dir: dt/dp), so min stays sound.
    if margin<=0 and USE_AA and rot_s<Gc.lo:
        import g_deriv_aa as GA, far_rot_lip as RL
        # LAZY: tighten u2, then dir, then s, recomputing margin after each and stopping as soon as it
        # clears. Each term only shrinks (min of proven bounds), so margin is monotone -> early-out is
        # exact (same cert decision as applying all three), and skips whole AA passes on the common case.
        if thw>0.0 or phw>0.0:
            try:
                La,Lp=GA.glam_lip(box,eo,lam,aa); Lth=min(Lth,La); Lph=min(Lph,Lp)
                margin=Gc.lo-(Lth*thw+Lph*phw+rot_dir+rot_s)
            except (ZeroDivisionError,ValueError): pass
        if margin<=0 and (dtw>0.0 or dpw>0.0):
            try:
                Ldt,Ldp=RL.glam_lip_dir(box,eo,lam,aa)
                rot_dir=min(rot_dir, Ldt*dtw+Ldp*dpw)
                margin=Gc.lo-(Lth*thw+Lph*phw+rot_dir+rot_s)
            except (ZeroDivisionError,ValueError): pass
        if margin<=0 and sw>0.0:
            try:
                rot_s=min(rot_s, RL.glam_lip_s(box,eo,lam,aa)*sw)
                margin=Gc.lo-(Lth*thw+Lph*phw+rot_dir+rot_s)
            except (ZeroDivisionError,ValueError): pass
    return margin
def subdiv(box):
    L=list(box);dims=[1,3,5,7,9];wi=max(dims,key=lambda z:L[z]);ci=wi-1
    a=L[:];b=L[:];a[ci]=L[ci]-L[wi]/2;a[wi]=L[wi]/2;b[ci]=L[ci]+L[wi]/2;b[wi]=L[wi]/2
    return tuple(a),tuple(b)
def cover_region(th0,th1,ph0,ph1, nth=6,nph=12,nd=3,ndp=6,ns=3, tlimit=0, log=False):
    stack=deque()
    for a in range(nth):
        thc=th0+(th1-th0)*(a+0.5)/nth; thw=(th1-th0)/nth/2
        for b in range(nph):
            phc=ph0+(ph1-ph0)*(b+0.5)/nph; phw=(ph1-ph0)/nph/2
            for c in range(nd):
                dtc=math.pi*(c+0.5)/nd; dtw=math.pi/nd/2
                for e in range(ndp):
                    dpc=2*math.pi*(e+0.5)/ndp; dpw=2*math.pi/ndp/2
                    for g in range(ns):
                        lo=0.15*(smax/0.15)**(g/ns); hi=0.15*(smax/0.15)**((g+1)/ns)
                        stack.append(((thc,thw,phc,phw,dtc,dtw,dpc,dpw,(lo+hi)/2,(hi-lo)/2),None,None))
    cert=fail=nbox=nlp=0; t0=time.time()
    while stack:
        if tlimit and time.time()-t0>tlimit: break
        box,lam,eo=stack.pop(); nbox+=1
        u2c=BC.u2center(box[0],box[2])
        if eo is None:
            eo=BC.edge_order(u2c); lam=None
            if eo is None:
                if max(box[1],box[3],box[5],box[7],box[9])<MINW: fail+=1
                else:
                    a,b=subdiv(box); stack.append((a,None,None)); stack.append((b,None,None))
                continue
        s_start=BC.dlo(u2c)/M
        if box[8]+box[9]<=s_start: continue
        # HULL-COMBINATORICS STABILITY: edge_order must be provably valid over the whole u₂-cell
        if not hull_stable(box,eo):
            if max(box[1],box[3])<MINW: fail+=1
            else:
                Lb=list(box); wi=1 if box[1]>=box[3] else 3; ci=wi-1
                a=Lb[:];b=Lb[:];a[ci]=Lb[ci]-Lb[wi]/2;a[wi]=Lb[wi]/2;b[ci]=Lb[ci]+Lb[wi]/2;b[wi]=Lb[wi]/2
                stack.append((tuple(a),None,None)); stack.append((tuple(b),None,None))  # u₂ changed
            continue
        if lam is not None and certify(box,eo,lam)>0: cert+=1; continue
        nlp+=1; mu,lam2=BC.lam_lp(u2c, box[8]*BC.dircenter(box[4],box[6]))
        if lam2 is None or mu<=0:
            if max(box[1],box[3],box[5],box[7],box[9])<MINW: fail+=1
            else:
                a,b=subdiv(box); stack.append((a,None,eo)); stack.append((b,None,eo))
            continue
        if certify(box,eo,lam2)>0: cert+=1
        elif max(box[1],box[3],box[5],box[7],box[9])<MINW: fail+=1
        else:
            a,b=subdiv(box)
            eoa=None if a[0]!=box[0] or a[2]!=box[2] else eo; eob=None if b[0]!=box[0] or b[2]!=box[2] else eo
            stack.append((a,lam2 if eoa else None,eoa)); stack.append((b,lam2 if eob else None,eob))
        if log and nbox%20000==0: print(f"    {nbox} cert={cert} fail={fail} stack={len(stack)} {nbox/(time.time()-t0):.0f}/s",flush=True)
    return cert,fail,nbox,nlp,(len(stack)==0)
if __name__=='__main__':
    print("v4 PROVEN Λ — tiny region convergence (compare to v3's 92k boxes, not done):")
    for name,(a,b,c,d) in {'tiny(mid)':(0.585,0.595,0.795,0.805)}.items():
        t0=time.time(); cert,fail,nbox,nlp,done=cover_region(a,b,c,d, nth=1,nph=1,nd=2,ndp=3,ns=2, tlimit=170,log=True)
        print(f"  {name}: cert={cert} fail={fail} nbox={nbox} done={done} {time.time()-t0:.0f}s")

# ---------- HULL-COMBINATORICS STABILITY (interval turn tests over the u₂-cell) ----------
def hull_stable(box,eo):
    """Verify the center hull `eo` is the true outer-shadow hull over the ENTIRE u₂-cell.
    Hull combinatorics are SCALE-INVARIANT ⇒ use the UNNORMALIZED frame (f̃1=e3×u₂, f̃2=u₂×f̃1;
    |f̃2|=|f̃1| since |u₂|=1) so q̃=(f̃2·v, −f̃1·v) are polynomials in u₂ (no division ⇒ tight). The
    turn signs of q̃ equal those of the true shadow. If every non-endpoint vertex is STRICTLY inward
    of every center edge over the whole cell, edge_order is valid there ⇒ certificate sound."""
    th,thw,ph,phw=box[0],box[1],box[2],box[3]
    u2=F.iu2(th-thw,th+thw,ph-phw,ph+phw)
    f1=F.cross([F.FI(0.0),F.FI(0.0),F.FI(1.0)],u2); f2=F.cross(u2,f1); V=F.verts(aa)
    qt=[[F.dot(f2,v), F.FI(0.0)-F.dot(f1,v)] for v in V]
    for (p_,r_,sgn) in eo:
        d0=qt[r_][0]-qt[p_][0]; d1=qt[r_][1]-qt[p_][1]
        n0=F.FI(0.0)-d1; n1=d0
        if sgn<0: n0=F.FI(0.0)-n0; n1=F.FI(0.0)-n1
        for k in range(8):
            if k==p_ or k==r_: continue
            side=n0*(qt[k][0]-qt[p_][0])+n1*(qt[k][1]-qt[p_][1])
            if side.lo<=0: return False
    return True

def in_F(th,thw,ph,phw):
    """True if the (θ,φ)-cell MIGHT intersect the T_d fundamental domain F = {u_x≥u_y≥u_z, u_y+u_z≥0}.
    Returns False only if the cell is PROVABLY entirely outside F (some constraint's interval hi < 0),
    so every point of F lies in a processed cell (over-covering boundary neighbours is harmless)."""
    u=F.iu2(th-thw,th+thw,ph-phw,ph+phw)
    if (u[0]-u[1]).hi < 0: return False
    if (u[1]-u[2]).hi < 0: return False
    if (u[1]+u[2]).hi < 0: return False
    return True
