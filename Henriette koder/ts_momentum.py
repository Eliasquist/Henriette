# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

def generate_ts_momentum_signals(df: pd.DataFrame, lookback_l: int, allow_short: bool = False) -> pd.Series:
    """
    Simple Time-Series Momentum Signal.
    Based on cumulative % return over a single lookback window (L).
    No thresholds or filters added to maintain academic transparency.
    """
    close = df['close']
    
    # Calculate Momentum (Simple % return over L bars)
    cum_return = close / close.shift(lookback_l) - 1
    
    # Signal: +1 if return > 0, else 0 (or -1 if allow_short)
    if not allow_short:
        signal = (cum_return > 0).astype(int)
    else:
        signal = np.sign(cum_return).fillna(0).astype(int)
        
    return signal.fillna(0)
