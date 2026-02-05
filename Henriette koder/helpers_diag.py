# -*- coding: utf-8 -*-
# btc_momentum_onlyshortstrategy/strategies/helpers_diag.py
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# =============================================================================
# Internal utilities
# =============================================================================

def _finalize_df_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliserer timestamp-kolonne, kaster bool -> 0/1 og runder floats
    før skriving til CSV. Helt generisk – brukt av alle write_*()-funksjoner.
    """
    if df is None or df.empty:
        return df
    out = df.copy()

    # datetime → __dt__
    dt_candidates = [c for c in out.columns if c.lower() in ("dt", "datetime", "date", "timestamp", "time")]
    if dt_candidates:
        dtcol = dt_candidates[0]
        out["__dt__"] = pd.to_datetime(out[dtcol], errors="coerce", utc=True)
    elif isinstance(out.index, pd.DatetimeIndex):
        out["__dt__"] = pd.to_datetime(out.index, errors="coerce", utc=True)

    # bool -> 0/1
    for c in out.columns:
        if out[c].dtype == bool:
            out[c] = out[c].astype(int)

    # float avrunding
    for c in out.select_dtypes(include=[float, np.floating]).columns:
        out[c] = out[c].round(6)

    return out


def _as_bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Hent bool-serie for kolonnenavn, eller False-serie om ikke finnes."""
    if (col in df.columns) and (df[col].dtype == bool or df[col].dropna().isin([0, 1, True, False]).all()):
        return df[col].astype(bool)
    if col in df.columns:
        # fallback: sannhetsverdi for ikke-NaN/ikke-null
        return df[col].notna()
    return pd.Series(False, index=df.index)


def _ttl(s: pd.Series, bars: int) -> pd.Series:
    """TTL-vindu: True dersom minst ett True i siste 'bars'."""
    if s is None or len(s) == 0:
        return s
    return (s.astype(int).rolling(int(max(1, bars)), min_periods=1).max() >= 1).fillna(False)


def _pick_first_present(df: pd.DataFrame, names: Iterable[str]) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    return None


# =============================================================================
# ST/ZL/WT – bygg diagnostikk fra sammenslått dataframe (orkestratorens df_merged)
# =============================================================================

def build_rangefilter_requirements(
    df_merged: pd.DataFrame,
    *,
    signal_expiry: int = 3,
) -> pd.DataFrame:
    """
    Avleder Range Filter-relaterte vinduer, bekreftelser og «klargjort» flagg.

    Retur-DataFrame er lett å dumpe til CSV og å analysere i etterkant:

      • `rf_window_*`  : TB-impuls (trendline breakout) aktiv på denne baren.
      • `rf_wait_*`    : «hvor mange barer igjen» i et vindu (her: signal_expiry - progress).
      • `rf_ready_*`   : EMA-filter og TB-impuls er sanne samtidig.
      • `rf_confirm_*` : faktiske bekreftelser (identiske med modul-kolonnene).
      • `rf_expired_*` : TB-vinduer som stengte uten bekreftelse (u/brukt ⇒ False).
      • `rf_window_progress_*` : antall barer siden vinduet åpnet (0 = start-baren).
    """
    if df_merged is None or df_merged.empty:
        return pd.DataFrame(
            columns=[
                "dt",
                "px",
                "rf_lead_long",
                "rf_lead_short",
                "rf_window_long",
                "rf_window_short",
                "rf_wait_long",
                "rf_wait_short",
                "rf_ready_long",
                "rf_ready_short",
                "rf_ema_ok_long",
                "rf_ema_ok_short",
                "rf_tb_buysignal",
                "rf_tb_sellsignal",
                "rf_confirm_long",
                "rf_confirm_short",
                "rf_confirm_long_edge",
                "rf_confirm_short_edge",
                "rf_expired_long",
                "rf_expired_short",
                "rf_state",
                "rf_window_progress_long",
                "rf_window_progress_short",
            ]
        )

    df = df_merged.copy().sort_index()
    idx = df.index

    def _num_series(name: str) -> pd.Series:
        if name not in df.columns:
            return pd.Series(0, index=idx, dtype=int)
        return pd.to_numeric(df[name], errors="coerce").fillna(0).astype(int)

    def _float_series(name: str) -> pd.Series:
        if name not in df.columns:
            return pd.Series(np.nan, index=idx, dtype=float)
        return pd.to_numeric(df[name], errors="coerce")

    px = _float_series("close")

    lead_long = _as_bool_series(df, "rf_lead_long")
    lead_short = _as_bool_series(df, "rf_lead_short")

    tb_buy = _as_bool_series(df, "rf_tb_buysignal")
    tb_sell = _as_bool_series(df, "rf_tb_sellsignal")

    window_long = tb_buy
    window_short = tb_sell

    ema_ok_long = (
        _as_bool_series(df, "rf_ema_ok_long")
        if "rf_ema_ok_long" in df.columns
        else (_float_series("close") > _float_series("rf_ema")).fillna(False)
    )
    ema_ok_short = (
        _as_bool_series(df, "rf_ema_ok_short")
        if "rf_ema_ok_short" in df.columns
        else (_float_series("close") < _float_series("rf_ema")).fillna(False)
    )

    confirm_long = _as_bool_series(df, "rf_confirm_long")
    confirm_short = _as_bool_series(df, "rf_confirm_short")
    confirm_long_edge = _as_bool_series(df, "rf_confirm_long_edge")
    confirm_short_edge = _as_bool_series(df, "rf_confirm_short_edge")

    progress_long = pd.Series(np.nan, index=idx, dtype=float)
    progress_short = pd.Series(np.nan, index=idx, dtype=float)
    if not window_long.empty:
        grp_long = (window_long != window_long.shift()).cumsum()
        progress_long = (
            window_long.groupby(grp_long).cumcount().where(window_long, np.nan).astype(float)
        )
    if not window_short.empty:
        grp_short = (window_short != window_short.shift()).cumsum()
        progress_short = (
            window_short.groupby(grp_short).cumcount().where(window_short, np.nan).astype(float)
        )

    wait_long = np.where(window_long, np.maximum(int(signal_expiry) - (progress_long + 1), 0), 0)
    wait_short = np.where(window_short, np.maximum(int(signal_expiry) - (progress_short + 1), 0), 0)
    wait_long = pd.Series(wait_long, index=idx, dtype=int)
    wait_short = pd.Series(wait_short, index=idx, dtype=int)

    expired_long = pd.Series(False, index=idx, dtype=bool)
    expired_short = pd.Series(False, index=idx, dtype=bool)

    ready_long = window_long & ema_ok_long & tb_buy
    ready_short = window_short & ema_ok_short & tb_sell

    out = pd.DataFrame(
        {
            "dt": pd.to_datetime(idx),
            "px": px.astype(float),
            "rf_lead_long": lead_long.astype(bool),
            "rf_lead_short": lead_short.astype(bool),
            "rf_window_long": window_long.astype(bool),
            "rf_window_short": window_short.astype(bool),
            "rf_wait_long": wait_long.astype(int),
            "rf_wait_short": wait_short.astype(int),
            "rf_ready_long": ready_long.astype(bool),
            "rf_ready_short": ready_short.astype(bool),
            "rf_ema_ok_long": ema_ok_long.astype(bool),
            "rf_ema_ok_short": ema_ok_short.astype(bool),
            "rf_tb_buysignal": tb_buy.astype(bool),
            "rf_tb_sellsignal": tb_sell.astype(bool),
            "rf_confirm_long": confirm_long.astype(bool),
            "rf_confirm_short": confirm_short.astype(bool),
            "rf_confirm_long_edge": confirm_long_edge.astype(bool),
            "rf_confirm_short_edge": confirm_short_edge.astype(bool),
            "rf_expired_long": expired_long.astype(bool),
            "rf_expired_short": expired_short.astype(bool),
            "rf_state": _num_series("rf_state"),
            "rf_window_progress_long": progress_long,
            "rf_window_progress_short": progress_short,
        },
        index=idx,
    )

    return out


def build_stzlwt_requirements(
    df_merged: pd.DataFrame,
    *,
    ttl_bars: int = 3,
    **_legacy_kwargs,
) -> pd.DataFrame:
    """
    Bakoverkompatibel alias – ruter gamle kall over til Range Filter-implementasjonen.
    """
    return build_rangefilter_requirements(df_merged, signal_expiry=ttl_bars)


def build_signal_marks_df(signal_marks: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Konverterer orchestratorens signal_marks -> DataFrame for enkel filtrering/skriving.
    Forventer poster {'dt','type','price'}.
    """
    if not signal_marks:
        return pd.DataFrame(columns=["dt", "type", "price"])
    df = pd.DataFrame(signal_marks)
    if "dt" in df.columns:
        df["dt"] = pd.to_datetime(df["dt"], errors="coerce", utc=True)
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df.sort_values("dt")


# =============================================================================
# «Skrivere» – aksepterer enten DataFrame direkte eller et 'ctx' med rader lagret
# =============================================================================

def write_diagnostics_csv(ctx_or_df: Union[pd.DataFrame, Any], path: str) -> None:
    """
    Skriv en diagnostikkserie til CSV.
    - Hvis første argument er DataFrame: bruks direkte.
    - Hvis objekt med attribute 'diag_rows': skriver den.
    """
    try:
        if isinstance(ctx_or_df, pd.DataFrame):
            df = ctx_or_df
        else:
            df = pd.DataFrame(getattr(ctx_or_df, "diag_rows", []))

        if not df.empty:
            df = _finalize_df_for_csv(df)
            df.to_csv(path, index=False)
            logger.info("Wrote diagnostics to %s (%d rows)", path, len(df))
        else:
            logger.info("No diagnostics rows; skipping %s", path)
    except Exception:
        logger.exception("Failed to write diagnostics CSV")


def write_requirements_csv(ctx_or_df: Union[pd.DataFrame, Any], path: str) -> None:
    """
    Skriv «entry requirements»/cases til CSV.
    - Hvis DataFrame: bruks direkte.
    - Hvis objekt med attribute 'req_rows': skriver den.
    """
    try:
        if isinstance(ctx_or_df, pd.DataFrame):
            df = ctx_or_df
        else:
            df = pd.DataFrame(getattr(ctx_or_df, "req_rows", []))

        if not df.empty:
            df = _finalize_df_for_csv(df)
            df.to_csv(path, index=False)
            logger.info("Wrote requirements to %s (%d rows)", path, len(df))
        else:
            logger.info("No requirement rows; skipping %s", path)
    except Exception:
        logger.exception("Failed to write requirements CSV")


# =============================================================================
# «Blocks» / audit – behold API, men gjør det generisk
# =============================================================================

def audit_block(ctx_or_container: Any, *, dt, side: str, blockers: Dict[str, Optional[float]], px: float) -> None:
    """
    Registrer ett 'block'-event for revisjon. Strukturen er enkel og strateginøytral.
    - ctx_or_container kan være vilkårlig objekt vi putter 'block_rows' på,
      eller en dict med nøkkel 'block_rows'.
    """
    try:
        rec: Dict[str, Any] = {
            "dt": pd.to_datetime(dt),
            "side": str(side),
            "px": float(px),
        }
        for k, v in (blockers or {}).items():
            rec[f"{k}_gap"] = None if v is None else float(v)

        # lagre på container
        if isinstance(ctx_or_container, dict):
            ctx_or_container.setdefault("block_rows", []).append(rec)
        else:
            if not hasattr(ctx_or_container, "block_rows"):
                setattr(ctx_or_container, "block_rows", [])  # type: ignore[attr-defined]
            ctx_or_container.block_rows.append(rec)  # type: ignore[attr-defined]
    except Exception:
        logger.exception("audit_block failed")


def write_blocks_csv(ctx_or_container: Any, path: str) -> None:
    try:
        if isinstance(ctx_or_container, dict):
            rows = ctx_or_container.get("block_rows", [])
        else:
            rows = getattr(ctx_or_container, "block_rows", [])
        df = pd.DataFrame(rows)
        if not df.empty:
            df = _finalize_df_for_csv(df)
            df.to_csv(path, index=False)
            logger.info("Wrote blocks to %s (%d rows)", path, len(df))
        else:
            logger.info("No blocks to write; skipping %s", path)
    except Exception:
        logger.exception("Failed to write blocks CSV")


# =============================================================================
# Bakoverkompatible «stubber» (beholder navn – refererer IKKE til gamle indikatorer)
# =============================================================================

def vol_weighted_mom(_ctx: Any, _lookback_days: int) -> float:
    """
    Historisk volum-momentberegning. Ikke relevant for Range Filter-orkestratoren.
    Returnerer 0.0 for å ikke forstyrre eksisterende kall.
    """
    return 0.0


def snapshot_diag(*args, **kwargs) -> None:
    """
    Historisk funksjon for å pushe «diag_rows» inn i en kontekst.
    I Range Filter-rammeverket anbefales: bygg DataFrame med
    build_rangefilter_requirements() og skriv den med write_diagnostics_csv().
    Denne stubben gjør ingen skade (no-op).
    """
    return None


def maybe_snapshot_periodic(*args, **kwargs) -> None:
    """
    Historisk periodic snapshot. Beholdes som no-op for å ikke brekke kall.
    """
    return None


def snapshot_requirements(
    ctx_or_df: Union[Any, pd.DataFrame],
    *,
    dt: Optional[pd.Timestamp] = None,
    px: Optional[float] = None,
    df_merged: Optional[pd.DataFrame] = None,
    ttl_bars: int = 3,
) -> Optional[pd.DataFrame]:
    """
    Bakovervennlig «snapshot»-inngang:
      - Hvis df_merged er gitt: returnerer en ferdig krav/«cases»-DataFrame.
      - Hvis ctx_or_df er en DataFrame: tolkes som df_merged og returneres transformert.
      - Ellers: no-op (returnerer None) – bruk det nye APIet direkte.
    """
    try:
        if df_merged is not None:
            return build_rangefilter_requirements(df_merged, signal_expiry=ttl_bars)
        if isinstance(ctx_or_df, pd.DataFrame):
            return build_rangefilter_requirements(ctx_or_df, signal_expiry=ttl_bars)
    except Exception:
        logger.exception("snapshot_requirements failed")
    return None
