from __future__ import annotations

"""VF-11 research-only range/certifiability engine.

NO_PRODUCT_AUTHORITY_MINTED.

Target architecture:
bilinear triplane -> LayerNorm -> Linear -> SiLU -> Linear -> SiLU -> Linear(sdf)

The algebraic bounds are conservative, but v1 uses ordinary IEEE-754 torch
arithmetic rather than a directed-rounding kernel. Therefore emitted states are
research certifiability evidence only; they are not shipping proof.
"""

from dataclasses import dataclass
import math
from typing import Iterable

import torch
from torch import nn


@dataclass(frozen=True)
class Interval:
    lo: torch.Tensor
    hi: torch.Tensor

    def __post_init__(self) -> None:
        if self.lo.shape != self.hi.shape:
            raise ValueError("INTERVAL_SHAPE_MISMATCH")
        if bool(torch.any(self.lo > self.hi)):
            raise ValueError("INTERVAL_ORDER_INVALID")

    @staticmethod
    def exact(x: torch.Tensor) -> "Interval":
        return Interval(x, x)


@dataclass(frozen=True)
class CellCertificate:
    state: str
    field_lo: float
    field_hi: float
    directional_lo: float | None
    directional_hi: float | None
    center_gradient_norm: float | None
    reason: str
    numerically_rigorous: bool = False


def _exact_like(value, ref: torch.Tensor) -> Interval:
    x = torch.as_tensor(value, dtype=ref.dtype, device=ref.device)
    return Interval.exact(x)


def iadd(a: Interval, b: Interval) -> Interval:
    return Interval(a.lo + b.lo, a.hi + b.hi)


def ineg(a: Interval) -> Interval:
    return Interval(-a.hi, -a.lo)


def isub(a: Interval, b: Interval) -> Interval:
    return iadd(a, ineg(b))


def imul(a: Interval, b: Interval) -> Interval:
    p = torch.stack((a.lo*b.lo, a.lo*b.hi, a.hi*b.lo, a.hi*b.hi), 0)
    return Interval(torch.amin(p, 0), torch.amax(p, 0))


def iscale(a: Interval, value: float) -> Interval:
    return imul(a, _exact_like(value, a.lo))


def isum(a: Interval) -> Interval:
    return Interval(torch.sum(a.lo), torch.sum(a.hi))


def imean(a: Interval) -> Interval:
    return Interval(torch.mean(a.lo), torch.mean(a.hi))


def ipow_positive(a: Interval, exponent: float) -> Interval:
    if bool(torch.any(a.lo <= 0)):
        raise ValueError("POSITIVE_INTERVAL_REQUIRED")
    if exponent >= 0:
        return Interval(a.lo.pow(exponent), a.hi.pow(exponent))
    return Interval(a.hi.pow(exponent), a.lo.pow(exponent))


def linear_bounds(x: Interval, dx: Interval, layer: nn.Linear) -> tuple[Interval, Interval]:
    w = layer.weight.detach().to(dtype=torch.float64, device=x.lo.device)
    b = (torch.zeros(w.shape[0], dtype=torch.float64, device=x.lo.device)
         if layer.bias is None else layer.bias.detach().to(dtype=torch.float64, device=x.lo.device))
    wp, wn = torch.clamp(w, min=0), torch.clamp(w, max=0)
    y = Interval(wp@x.lo + wn@x.hi + b, wp@x.hi + wn@x.lo + b)
    dy = Interval(wp@dx.lo + wn@dx.hi, wp@dx.hi + wn@dx.lo)
    return y, dy


_SILU_MIN_X = -1.2784645427610737
_SILU_DERIV_CRIT = 2.3993572805154677


def _silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def _silu_prime(x: torch.Tensor) -> torch.Tensor:
    s = torch.sigmoid(x)
    return s + x*s*(1.0-s)


def silu_bounds(x: Interval, dx: Interval) -> tuple[Interval, Interval]:
    flo, fhi = _silu(x.lo), _silu(x.hi)
    ylo, yhi = torch.minimum(flo, fhi), torch.maximum(flo, fhi)
    xm = torch.full_like(x.lo, _SILU_MIN_X)
    inside = (x.lo <= _SILU_MIN_X) & (x.hi >= _SILU_MIN_X)
    ylo = torch.where(inside, torch.minimum(ylo, _silu(xm)), ylo)

    plo, phi = _silu_prime(x.lo), _silu_prime(x.hi)
    dlo, dhi = torch.minimum(plo, phi), torch.maximum(plo, phi)
    for c in (-_SILU_DERIV_CRIT, _SILU_DERIV_CRIT):
        xc = torch.full_like(x.lo, c)
        pc = _silu_prime(xc)
        hit = (x.lo <= c) & (x.hi >= c)
        dlo = torch.where(hit, torch.minimum(dlo, pc), dlo)
        dhi = torch.where(hit, torch.maximum(dhi, pc), dhi)
    return Interval(ylo, yhi), imul(Interval(dlo, dhi), dx)


def _variance_bounds(x: Interval) -> Interval:
    if x.lo.ndim != 1:
        raise ValueError("LAYERNORM_EXPECTS_1D_INTERVAL")
    n = int(x.lo.numel())
    dlo = x.lo[:, None] - x.hi[None, :]
    dhi = x.hi[:, None] - x.lo[None, :]
    cross = (dlo <= 0) & (dhi >= 0)
    qlo = torch.where(cross, torch.zeros_like(dlo), torch.minimum(dlo*dlo, dhi*dhi))
    qhi = torch.maximum(dlo*dlo, dhi*dhi)
    mask = torch.triu(torch.ones((n,n), dtype=torch.bool, device=x.lo.device), diagonal=1)
    return Interval(torch.sum(qlo[mask])/(n*n), torch.sum(qhi[mask])/(n*n))


def layernorm_bounds(x: Interval, dx: Interval, layer: nn.LayerNorm) -> tuple[Interval, Interval]:
    if x.lo.ndim != 1 or tuple(layer.normalized_shape) != (x.lo.numel(),):
        raise ValueError("UNSUPPORTED_LAYERNORM_SHAPE")
    n = int(x.lo.numel())
    mu, dmu = imean(x), imean(dx)
    c = isub(x, Interval(mu.lo.expand_as(x.lo), mu.hi.expand_as(x.hi)))
    dc = isub(dx, Interval(dmu.lo.expand_as(dx.lo), dmu.hi.expand_as(dx.hi)))

    var = _variance_bounds(x)
    ve = Interval(var.lo + float(layer.eps), var.hi + float(layer.eps))
    inv = ipow_positive(ve, -0.5)
    vard = iscale(isum(imul(c, dc)), 2.0/n)
    dinv = iscale(imul(ipow_positive(ve, -1.5), vard), -0.5)

    invv = Interval(inv.lo.expand_as(c.lo), inv.hi.expand_as(c.hi))
    dinvv = Interval(dinv.lo.expand_as(c.lo), dinv.hi.expand_as(c.hi))
    y = imul(c, invv)
    dy = iadd(imul(dc, invv), imul(c, dinvv))
    if layer.elementwise_affine:
        gamma = layer.weight.detach().to(dtype=torch.float64, device=x.lo.device)
        beta = layer.bias.detach().to(dtype=torch.float64, device=x.lo.device)
        y = iadd(imul(y, Interval.exact(gamma)), Interval.exact(beta))
        dy = imul(dy, Interval.exact(gamma))
    return y, dy


def _pixel(g: float, size: int) -> float:
    return ((float(g)+1.0)*float(size)-1.0)*0.5


def _segments(lo: float, hi: float, size: int) -> list[tuple[int,float,float]]:
    if not (-1.0 <= lo <= hi <= 1.0):
        raise ValueError("CELL_OUTSIDE_NORMALIZED_DOMAIN")
    plo, phi = _pixel(lo,size), _pixel(hi,size)
    if plo < 0.0 or phi > float(size-1):
        raise ValueError("TRIPLANE_BORDER_REGION")
    if phi == plo:
        k = min(size-2, max(0, int(math.floor(plo))))
        return [(k, plo-k, plo-k)]
    start = max(0, int(math.floor(plo)))
    stop = min(size-2, int(math.floor(math.nextafter(phi, -math.inf))))
    rows = []
    for k in range(start, stop+1):
        a, b = max(plo,float(k)), min(phi,float(k+1))
        if a <= b:
            rows.append((k,a-k,b-k))
    if not rows:
        raise ValueError("NO_INTERIOR_TRIPLANE_SEGMENT")
    return rows


def _bilinear(v00, v10, v01, v11, *, u0,u1,v0,v1, du,dv, scale) -> tuple[Interval,Interval]:
    vals, ders = [], []
    for u in (u0,u1):
        for v in (v0,v1):
            vals.append(v00*(1-u)*(1-v) + v10*u*(1-v) + v01*(1-u)*v + v11*u*v)
            gu = (v10-v00)*(1-v) + (v11-v01)*v
            gv = (v01-v00)*(1-u) + (v11-v10)*u
            ders.append(float(du)*scale*gu + float(dv)*scale*gv)
    vv, dd = torch.stack(vals), torch.stack(ders)
    return Interval(torch.amin(vv,0),torch.amax(vv,0)), Interval(torch.amin(dd,0),torch.amax(dd,0))


def _union(rows: Iterable[Interval]) -> Interval:
    rows = list(rows)
    if not rows:
        raise ValueError("EMPTY_INTERVAL_UNION")
    return Interval(torch.amin(torch.stack([r.lo for r in rows]),0),
                    torch.amax(torch.stack([r.hi for r in rows]),0))


def triplane_bounds(planes: torch.Tensor, cell_lo: torch.Tensor, cell_hi: torch.Tensor,
                    direction: torch.Tensor) -> tuple[Interval,Interval]:
    if planes.ndim != 5 or planes.shape[0] != 1 or planes.shape[1] != 3:
        raise ValueError("PLANES_MUST_BE_1x3xCxHxW")
    if planes.shape[-1] != planes.shape[-2]:
        raise ValueError("SQUARE_TRIPLANES_REQUIRED")
    p = planes.detach().to(dtype=torch.float64)
    device = p.device
    lo = torch.as_tensor(cell_lo,dtype=torch.float64,device=device).reshape(3)
    hi = torch.as_tensor(cell_hi,dtype=torch.float64,device=device).reshape(3)
    d = torch.as_tensor(direction,dtype=torch.float64,device=device).reshape(3)
    if bool(torch.any(lo > hi)):
        raise ValueError("INVALID_CELL")

    size, scale = int(p.shape[-1]), float(p.shape[-1])*0.5
    parts, dparts = [], []
    for plane_id,(au,av) in enumerate(((0,1),(0,2),(1,2))):
        us, vs = _segments(float(lo[au]),float(hi[au]),size), _segments(float(lo[av]),float(hi[av]),size)
        vals, ders = [], []
        plane = p[0,plane_id]
        for iu,u0,u1 in us:
            for iv,v0,v1 in vs:
                vr,dr = _bilinear(plane[:,iv,iu], plane[:,iv,iu+1],
                                  plane[:,iv+1,iu], plane[:,iv+1,iu+1],
                                  u0=u0,u1=u1,v0=v0,v1=v1,
                                  du=float(d[au]),dv=float(d[av]),scale=scale)
                vals.append(vr); ders.append(dr)
        parts.append(_union(vals)); dparts.append(_union(ders))
    return (Interval(torch.cat([r.lo for r in parts]),torch.cat([r.hi for r in parts])),
            Interval(torch.cat([r.lo for r in dparts]),torch.cat([r.hi for r in dparts])))


def _validate(field: nn.Module):
    body, head = getattr(field,"body",None), getattr(field,"sdf_head",None)
    if not isinstance(body,nn.Sequential) or len(body) != 5:
        raise ValueError("UNSUPPORTED_SURFACE_FIELD_BODY")
    if not (isinstance(body[0],nn.LayerNorm) and isinstance(body[1],nn.Linear)
            and isinstance(body[2],nn.SiLU) and isinstance(body[3],nn.Linear)
            and isinstance(body[4],nn.SiLU) and isinstance(head,nn.Linear)):
        raise ValueError("UNSUPPORTED_SURFACE_FIELD_ARCHITECTURE")
    return body[0],body[1],body[3],head


def propagate_current_field(field: nn.Module, x: Interval, dx: Interval) -> tuple[Interval,Interval]:
    ln,l1,l2,head = _validate(field)
    x,dx = layernorm_bounds(x,dx,ln)
    x,dx = linear_bounds(x,dx,l1); x,dx = silu_bounds(x,dx)
    x,dx = linear_bounds(x,dx,l2); x,dx = silu_bounds(x,dx)
    x,dx = linear_bounds(x,dx,head)
    if x.lo.numel() != 1:
        raise ValueError("SDF_HEAD_NOT_SCALAR")
    return x,dx


def center_direction(field: nn.Module, planes: torch.Tensor, lo, hi) -> tuple[torch.Tensor,float]:
    center = ((torch.as_tensor(lo,device=planes.device)+torch.as_tensor(hi,device=planes.device))*0.5).to(planes.dtype)
    q = center.reshape(1,1,3).detach().requires_grad_(True)
    out = field(planes,q)
    sdf = out["sdf"] if isinstance(out,dict) else out
    grad = torch.autograd.grad(sdf.reshape(-1)[0],q)[0].reshape(3)
    norm = float(torch.linalg.vector_norm(grad).detach().cpu())
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("DEGENERATE_CENTER_GRADIENT")
    return (grad/torch.linalg.vector_norm(grad)).detach(), norm


def certify_cell(field: nn.Module, planes: torch.Tensor, lo, hi) -> CellCertificate:
    _validate(field)
    zero = torch.zeros(3,dtype=planes.dtype,device=planes.device)
    try:
        fx,dfx = triplane_bounds(planes,lo,hi,zero)
        fr,_ = propagate_current_field(field,fx,dfx)
    except ValueError as exc:
        return CellCertificate("UNKNOWN",float("nan"),float("nan"),None,None,None,str(exc))

    flo,fhi = float(fr.lo.item()),float(fr.hi.item())
    if flo > 0.0 or fhi < 0.0:
        return CellCertificate("PROVEN_EMPTY",flo,fhi,None,None,None,"FIELD_INTERVAL_EXCLUDES_ZERO")

    try:
        d,gn = center_direction(field,planes,lo,hi)
        fx,dfx = triplane_bounds(planes,lo,hi,d)
        _,dr = propagate_current_field(field,fx,dfx)
    except ValueError as exc:
        return CellCertificate("UNKNOWN",flo,fhi,None,None,None,str(exc))

    dlo,dhi = float(dr.lo.item()),float(dr.hi.item())
    if dlo > 0.0 or dhi < 0.0:
        return CellCertificate("PROVEN_REGULAR",flo,fhi,dlo,dhi,gn,
                               "FIXED_DIRECTION_DERIVATIVE_STRICTLY_ONE_SIGNED")
    return CellCertificate("UNKNOWN",flo,fhi,dlo,dhi,gn,
                           "DIRECTIONAL_DERIVATIVE_INTERVAL_CONTAINS_ZERO")
