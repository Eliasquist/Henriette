
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from managed_vs_unmanaged_run import run_ts_comparison

def load_data_robust(filepath):
    """Robust loader handling both YF (timestamp index) and external (open_time col) data."""
    # 1. Try YFinance standard (Index=timestamp)
    try:
        df = pd.read_csv(filepath)
        # Check for timestamp or Date column
        date_col = None
        for col in ['timestamp', 'Date', 'date']:
            if col in df.columns:
                date_col = col
                break
        
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], utc=True)
            df.set_index(date_col, inplace=True)
            if 'close' in df.columns:
                 return df['close'].resample('1D').last().to_frame()
    except Exception as e:
        # print(f"Debug YF load: {e}") 
        pass
        
    try:
        # 2. Try Original Crypto Format
        df_raw = pd.read_csv(filepath)
        if 'open_time' in df_raw.columns:
            s = df_raw['open_time']
            if np.issubdtype(s.dtype, np.number):
                df_raw['timestamp'] = pd.to_datetime(s, unit='ms', utc=True)
            else:
                df_raw['timestamp'] = pd.to_datetime(s, utc=True)
            df_raw.set_index('timestamp', inplace=True)
            return df_raw['close'].resample('1D').last().to_frame()
    except Exception as e:
        print(f"Failed to load {filepath}: {e}")
        return pd.DataFrame()

def calculate_natural_volatility(df, is_crypto=False):
    ann_factor = 365 if is_crypto else 252
    return df['close'].pct_change().std() * np.sqrt(ann_factor)

def optimize_target_vol(df, name, is_crypto, lookback=120):
    # Expanded sweep from low to super-high vol
    targets = np.linspace(0.40, 1.50, 20)
    best_sharpe = -100
    best_vol = 0
    
    for t in targets:
        try:
            res = run_ts_comparison(
                df, f"{name}_{t:.2f}", 
                lookback_l=lookback, sigma_target=t, n_days_vol=120, 
                w_max=2.0, fee_bps=5.0 if is_crypto else 1.0, 
                is_crypto=is_crypto, is_inverse=False,
                rebalance_k_unmanaged=22, rebalance_k_voltarget=11
            )
            # Use 'voltarget' stats if available
            stats = res.attrs['stats']['voltarget']
            if stats['Sharpe'] > best_sharpe:
                best_sharpe = stats['Sharpe']
                best_vol = t
        except:
            pass
            
    return best_vol, best_sharpe

def test_law():
    assets = [
        {"name": "ETH", "file": "data/ETHUSDT_spot_ohlcv.csv", "is_crypto": True, "lookback": 120},
        {"name": "BTC", "file": "data/BTC_daily.csv", "is_crypto": True, "lookback": 120},
        {"name": "SOL", "file": "data/SOL_daily.csv", "is_crypto": True, "lookback": 120},
        {"name": "NVDA", "file": "data/NVDA_daily.csv", "is_crypto": False, "lookback": 60}, # Stocks need faster L
        {"name": "MSTR", "file": "data/MSTR_daily.csv", "is_crypto": False, "lookback": 60},
        {"name": "TSLA", "file": "data/TSLA_daily.csv", "is_crypto": False, "lookback": 60}
    ]
    
    print("\n" + "="*80)
    print("TESTING THE UNIVERSAL SCALING LAW: 'Optimal Target Volatility = 1.25x Natural Volatility'")
    print("Hypothesis: Strategy performance peaks when TargetVol is ~25% higher than Natural Vol")
    print("="*80)
    print(f"{'Asset':<6} | {'NatVol':<8} | {'Pred.Target (1.25x)':<20} | {'Actual Optimal':<15} | {'Ratio':<6} | {'Verdict'}")
    print("-" * 80)
    
    consistent_count = 0
    
    for a in assets:
        try:
            df = load_data_robust(a["file"])
            if df.empty:
                print(f"{a['name']:<6} | FAIL LOAD")
                continue
                
            # 1. Measure
            nat_vol = calculate_natural_volatility(df, a["is_crypto"])
            
            # 2. Predict
            pred_target = nat_vol * 1.25
            
            # 3. Test (Optimize)
            actual_opt, sharpe = optimize_target_vol(df, a["name"], a["is_crypto"], a["lookback"])
            
            # 4. Verify
            ratio = actual_opt / nat_vol if nat_vol > 0 else 0
            is_match = 1.10 <= ratio <= 1.40 # Acceptance corridor
            match_str = "YES" if is_match else "NO"
            
            if is_match: consistent_count += 1
            
            print(f"{a['name']:<6} | {nat_vol:<8.1%} | {pred_target:<20.1%} | {actual_opt:<15.1%} | {ratio:<6.2f} | {match_str}")
            
        except Exception as e:
            print(f"Error {a['name']}: {e}")

    print("-" * 80)
    print(f"Consistency Score: {consistent_count}/{len(assets)} assets obey the 1.25x Law.")

if __name__ == "__main__":
    test_law()
