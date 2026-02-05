# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

def compute_vol_target_weights(
    df: pd.DataFrame, 
    sigma_target: float = 0.10, 
    n_days: int = 60, 
    w_max: float = 1.0,
    annualization_factor: int = 252
) -> pd.Series:
    """
    Beregner vekter for Volatility Targeting.
    
    UTEN SHIFT (Convention A).
    Håndterer warmup med 0.
    
    Args:
        df: DataFrame med minst 'close' kolonne.
        sigma_target: Målvolatilitet (annualisert).
        n_days: Vindu for volatilitet.
        w_max: Maksimal eksponering.
        annualization_factor: 252 for aksjer, 365 for krypto.
        
    Returns:
        pd.Series: Serie med vekter w_t.
    """
    # Beregn daglig avkastning
    returns = df['close'].pct_change()
    
    # Rullerende standardavvik
    rolling_std = returns.rolling(window=n_days).std()
    
    # Annualisert volatilitet
    annual_vol = rolling_std * np.sqrt(annualization_factor)
    
    # Bytte ut 0/NaN med stor verdi før divisjon for å unngå Inf
    annual_vol_safe = annual_vol.replace(0, np.nan)
    
    # Beregn vekter (UTEN SHIFT her)
    weights = sigma_target / annual_vol_safe
    
    # Apply limit/cap
    weights = weights.clip(upper=w_max)
    
    # Håndter warmup perioder med 0 (ikke i markedet før vi har vol-estimat)
    return weights.fillna(0)
