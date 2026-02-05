# -*- coding: utf-8 -*-
# btc_momentum_onlyshortstrategy/strategies/helpers_risk.py
"""
Risikohjelpere tilpasset ST/ZL/WT-rammeverket.

Mål
----
- Være robuste mot manglende/ukjente parametre (for nye strategier).
- Ikke avhenge av gamle regime-/momentumvariabler.
- Beholde *samme funksjonsnavn/signatur* slik at eksisterende kall ikke brekkes.

Begreper (for ctx)
------------------
- ctx.p.fut_contract_size : kontrakt-størrelse i USD pr. kontrakt (float, default 1.0)
- ctx.p.im_utilization    : maks andel av wallet som kan bindes i IM (float, default 1.0)
- ctx.p.maker_fee         : maker fee (desimal, f.eks. 0.0002). (opsjonell)
- ctx.p.taker_fee         : taker fee (desimal, f.eks. 0.0005). (opsjonell)
- ctx.p.min_qty           : minste lot-størrelse i kontrakter (int, default 1)
- ctx.p.maint_rate        : vedlikeholdsmargin (desimal, default 0.004)
- ctx.p.maint_safety_mult : sikkerhetsmultipl. for maint (default 1.0)
- ctx.p.use_mark_price    : om mark-pris skal benyttes (bool, default True)
- ctx.wallet_coin         : USDT (eller tilsvarende) kontantbeholdning
- ctx.reserved_im_coin    : bundet IM
- ctx.unreal_pnl_coin     : urealisert PnL

Merk
----
- «Gearing»- og «ATR-risk sizing»-funksjoner beholdes, men er gjort
  *parametergeneriske* og trygge – de fungerer fint uten noen spesifikke
  strategi-innstillinger på ctx.p.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple


# ---------------------------------------------------------------------
# Generelle hjelpetjenester
# ---------------------------------------------------------------------
def infer_bpd(interval: str) -> int:
    """Antall bar per dag fra '1m'/'15m'/'1h'/'4h'/'1d'."""
    s = str(interval).strip().lower()
    try:
        if s.endswith("m"):
            m = max(1, int(s[:-1]))
            return max(1, 1440 // m)
        if s.endswith("h"):
            h = max(1, int(s[:-1]))
            return max(1, 24 // h)
        if s in ("1d", "d", "day", "daily"):
            return 1
        return 1440  # konservativ default
    except Exception:
        return 1440


# ---------------------------------------------------------------------
# Pris & kontrakt
# ---------------------------------------------------------------------
def mark_usd(ctx) -> float:
    """
    Returner mark- eller close-pris i USD (lineær kontrakt).
    Fallbacks er robuste og vil ikke kaste.
    """
    # preferer mark
    try:
        use_mark = bool(getattr(ctx.p, "use_mark_price", True))
    except Exception:
        use_mark = True

    try:
        if use_mark and hasattr(ctx.fut, "mark"):
            px = float(ctx.fut.mark[0])
            if px > 0 and math.isfinite(px):
                return px
    except Exception:
        pass

    # fallback: close
    try:
        px = float(ctx.fut.close[0])
        if px > 0 and math.isfinite(px):
            return px
    except Exception:
        pass

    # siste utvei – unngå crash
    return float(getattr(ctx.fut, "close")[0])


def _contract_size(ctx) -> float:
    try:
        cs = float(getattr(ctx.p, "fut_contract_size", 1.0))
        return cs if math.isfinite(cs) and cs > 0 else 1.0
    except Exception:
        return 1.0


def notional_usd(ctx, ct_abs: int, px_usd: float) -> float:
    """Notional-verdi i USD for |ct_abs| kontrakter."""
    cs = _contract_size(ctx)
    return float(max(0, int(ct_abs))) * cs * float(px_usd)


def risk_per_contract_usd(_ctx, stop_usd: float, _px_usd: float) -> float:
    """$-risiko pr. kontrakt gitt stop-avstand i USD (lineær: cs * stop)."""
    # cs leses ikke her for å holde API stabilt; behold enkel definisjon
    return float(stop_usd)


def im_needed_wallet(ctx, ct_abs: int, px_usd: float, leverage: Optional[float] = None) -> float:
    """IM i wallet-valuta for |ct_abs| kontrakter ved gitt/levert gearing."""
    lev = float(leverage if leverage is not None else getattr(ctx, "_active_leverage", 1.0))
    lev = max(1.0, lev)
    return notional_usd(ctx, abs(ct_abs), px_usd) / lev


def fee_cost_wallet(_ctx, notional_usd_val: float, _px_usd: float, fee_rate: float) -> float:
    """Avgiftskost i wallet-valuta for én side: notional * fee_rate."""
    return float(max(0.0, notional_usd_val)) * float(max(0.0, fee_rate))


def maint_needed_wallet(ctx, ct_signed: int, px_usd: float) -> float:
    """Vedlikeholdsmargin (wallet)."""
    if ct_signed == 0 or px_usd <= 0:
        return 0.0
    notion = notional_usd(ctx, abs(ct_signed), px_usd)
    try:
        rate = float(getattr(ctx.p, "maint_rate", 0.004))
    except Exception:
        rate = 0.004
    try:
        mult = float(getattr(ctx.p, "maint_safety_mult", 1.0))
    except Exception:
        mult = 1.0
    rate = max(0.0, rate)
    mult = max(0.0, mult)
    return notion * rate * mult


def equity_usd(ctx, _px_usd: float) -> float:
    """
    'Equity' i wallet-valuta: wallet + reservert IM + uPnL.
    (For lineære kontrakter i USDT – ingen pris-konverter.)
    """
    try:
        wallet = float(getattr(ctx, "wallet_coin", 0.0))
        im_res = float(getattr(ctx, "reserved_im_coin", 0.0))
        upnl = float(getattr(ctx, "unreal_pnl_coin", 0.0))
        return wallet + im_res + upnl
    except Exception:
        return 0.0


# ---------------------------------------------------------------------
# Loss-cap basis (robuste, parametergeneriske)
# ---------------------------------------------------------------------
def _loss_cap_mode(ctx, long_side: bool) -> str:
    if long_side:
        val = getattr(ctx.p, "hard_loss_cap_mode_long", None)
        if val is None:
            val = getattr(ctx.p, "hard_loss_cap_mode", "equity")
    else:
        val = getattr(ctx.p, "hard_loss_cap_mode", "equity")
    try:
        return str(val).lower()
    except Exception:
        return "equity"


def loss_cap_basis_usd_short(ctx, *, ct_abs: int, px_usd: float, leverage: float) -> float:
    """
    Basis for hard-loss budsjett (SHORT):
      - 'im'       -> initial margin
      - 'notional' -> notional verdi
      - ellers     -> gjeldende equity (>=0)
    """
    mode = _loss_cap_mode(ctx, long_side=False)
    if mode == "im":
        return im_needed_wallet(ctx, ct_abs, px_usd, leverage=leverage)
    if mode == "notional":
        return notional_usd(ctx, ct_abs, px_usd)
    return max(0.0, equity_usd(ctx, px_usd))


def loss_cap_basis_usd_long(ctx, *, ct_abs: int, px_usd: float, leverage: float) -> float:
    """Som over, men for LONG (modus-lookup bruker *_long først, ellers felles)."""
    mode = _loss_cap_mode(ctx, long_side=True)
    if mode == "im":
        return im_needed_wallet(ctx, ct_abs, px_usd, leverage=leverage)
    if mode == "notional":
        return notional_usd(ctx, ct_abs, px_usd)
    return max(0.0, equity_usd(ctx, px_usd))


# ---------------------------------------------------------------------
# (Valgfritt) ATR-basert sizing – beholdt, men robust og generisk
# ---------------------------------------------------------------------
def size_from_atr_risk(
    ctx,
    *,
    equity_usd: float,
    price_usd: float,
    atr_usd: float,
) -> int:
    """
    Beregn kontrakter fra fast $-risiko ved ATR-stop, med IM-begrensning og lot.
    Robuste defaults brukes hvis ctx.p mangler felter.

    Returnerer ABS kontrakter (int).
    """
    try:
        atr_mult = float(getattr(ctx.p, "atr_stop_mult", 2.0))
    except Exception:
        atr_mult = 2.0
    try:
        risk_frac = float(getattr(ctx.p, "risk_per_trade", 0.005))  # 0.5% av equity
    except Exception:
        risk_frac = 0.005
    try:
        im_util = float(getattr(ctx.p, "im_utilization", 1.0))
    except Exception:
        im_util = 1.0
    try:
        lot = int(max(1, int(getattr(ctx.p, "min_qty", 1))))
    except Exception:
        lot = 1

    stop_usd = max(1e-12, atr_mult * max(atr_usd, 0.0))
    risk_per_ct = max(1e-12, stop_usd)  # lineær: ~$ pr kontrakt ved stop
    target_risk = max(0.0, risk_frac * max(0.0, equity_usd))
    raw_ct = int(math.floor(target_risk / risk_per_ct))

    # IM-cap
    wallet_free = max(0.0, float(getattr(ctx, "wallet_coin", 0.0)))
    im_cap_wallet = im_util * wallet_free
    im_per_ct = im_needed_wallet(ctx, 1, price_usd, leverage=float(getattr(ctx, "_active_leverage", 1.0)))
    cap_ct = int(math.floor(im_cap_wallet / max(im_per_ct, 1e-12))) if im_per_ct > 0 else raw_ct

    ct = max(0, min(raw_ct, cap_ct))
    return int((ct // lot) * lot)


# ---------------------------------------------------------------------
# (Valgfritt) Gearing-styrke & sizing – generiske og trygge
# ---------------------------------------------------------------------
def signal_strength(
    _ctx,
    *,
    adx_val: float = 0.0,
    thrust_atr: float = 0.0,
    regime_ok: bool = True,
    vol_ok: bool = True,
    adx_floor: float = 15.0,
    adx_ceiling: float = 30.0,
    thrust_ref_atr: float = 1.0,
    w_adx: float = 0.34,
    w_thrust: float = 0.33,
    w_regvol: float = 0.33,
) -> float:
    """
    Kombinerer ADX, thrust (i ATR) og regime/vol til styrke∈[0,1].
    *Alltid* robust – kan brukes med defaults uten ctx.p.
    """
    # ADX komponent
    a0, a1 = float(adx_floor), float(adx_ceiling)
    if a1 <= a0:
        s_adx = 0.0
    else:
        s_adx = max(0.0, min(1.0, (float(adx_val) - a0) / (a1 - a0)))

    # Thrust komponent
    s_thr = max(0.0, min(1.0, float(thrust_atr) / max(1e-9, float(thrust_ref_atr))))

    # Regime/Vol komponent
    s_rv = 1.0 if (regime_ok and vol_ok) else (0.5 if (regime_ok or vol_ok) else 0.0)

    w_sum = max(1e-9, float(w_adx) + float(w_thrust) + float(w_regvol))
    z = (float(w_adx) * s_adx + float(w_thrust) * s_thr + float(w_regvol) * s_rv) / w_sum
    return max(0.0, min(1.0, z))


def size_from_gearing(
    ctx,
    *,
    equity_usd: float,
    price_usd: float,
    strength: float,
) -> Tuple[int, float, float]:
    """
    Oversett styrke∈[0,1] til (contracts_abs, leverage_used, alloc_used).
    - alloc skaleres lineært mellom alloc_min..alloc_max (defaults 0.5..0.95)
    - lev   skaleres lineært mellom 1.0..lev_max (default lev_max=2.0)
    Respekterer IM-utilization og min lot.

    Return: (ct_abs, lev, alloc)
    """
    # Defaults hvis ctx.p mangler felt
    alloc_min = float(getattr(ctx.p, "alloc_min", 0.50))
    alloc_max = float(getattr(ctx.p, "alloc_max", 0.95))
    lev_min = float(getattr(ctx.p, "lev_min", 1.0))
    lev_max = float(getattr(ctx.p, "lev_max", getattr(ctx.p, "max_leverage", 2.0)))
    im_util = float(getattr(ctx.p, "im_utilization", 1.0))
    lot = int(max(1, int(getattr(ctx.p, "min_qty", 1))))

    s = max(0.0, min(1.0, float(strength)))
    alloc = max(0.0, min(1.0, alloc_min + s * (alloc_max - alloc_min)))
    lev = max(1.0, lev_min + s * (lev_max - lev_min))

    wallet_free = max(0.0, float(getattr(ctx, "wallet_coin", 0.0)))
    im_cap_wallet = im_util * wallet_free
    target_im_wallet = min(alloc * wallet_free, im_cap_wallet)

    # Notional-mål og kontrakter
    cs = _contract_size(ctx)
    denom = cs * float(price_usd)
    notional_target = target_im_wallet * lev
    ct_abs = int(math.floor(notional_target / max(denom, 1e-12)))
    ct_abs = int((ct_abs // lot) * lot)

    # Sjekk IM per kontrakt
    if ct_abs > 0:
        im_per_ct = im_needed_wallet(ctx, 1, price_usd, leverage=lev)
        cap_ct = int(math.floor(im_cap_wallet / max(im_per_ct, 1e-12))) if im_per_ct > 0 else ct_abs
        ct_abs = max(0, min(ct_abs, cap_ct))
        if ct_abs == 0 and im_per_ct <= im_cap_wallet:
            ct_abs = lot

    return ct_abs, float(lev), float(alloc)
