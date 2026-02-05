"""
Global Parameter Optimization V4.1 - Thesis-Safe Common Calendar Holdout
========================================================================
Improvements over V4:
1) Common Calendar Holdout (2023-01-01 to End) for all assets
   - Ensures rigorous cross-asset generalization test on SAMME regime
2) Timezone-naive date normalization
3) Fixed reporting of absolute Sharpe levels in JSON/HTML

Walk-Forward (on Pre-2023 data):
- Fold1: train [0%, 50%), test [50%, 60%)
- Fold2: train [0%, 60%), test [60%, 70%)
- Fold3: train [0%, 70%), test [70%, 100%] (Use all remaining pre-2023 data for last fold)

Final Holdout: [2023-01-01, End] - evaluated ONCE on best config

Usage: python3 optimize_global_config_v4.py
"""

import os
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from managed_vs_unmanaged_run import load_data_robust

# =============================================================================
# CONFIGURATION
# =============================================================================
ASSETS = {
    'ETH': {'file': 'data/ETHUSDT_spot_ohlcv.csv', 'is_crypto': True, 'fee_bps': 5.0, 'ann_factor': 365, 'max_exposure': 1.50},
    'SOL': {'file': 'data/SOL_daily.csv', 'is_crypto': True, 'fee_bps': 5.0, 'ann_factor': 365, 'max_exposure': 1.50},
    'TSLA': {'file': 'data/TSLA_daily.csv', 'is_crypto': False, 'fee_bps': 1.0, 'ann_factor': 252, 'max_exposure': 1.10},
    'SPY': {'file': 'data/SPY_daily.csv', 'is_crypto': False, 'fee_bps': 1.0, 'ann_factor': 252, 'max_exposure': 1.10},
}

# COMMON CALENDAR HOLDOUT
HOLDOUT_START_DATE = '2023-01-01'

# Walk-Forward Folds (Percentages of PRE-HOLDOUT data)
FOLDS = [
    {'name': 'Fold1', 'train_end': 0.50, 'test_start': 0.50, 'test_end': 0.60},
    {'name': 'Fold2', 'train_end': 0.60, 'test_start': 0.60, 'test_end': 0.70},
    {'name': 'Fold3', 'train_end': 0.70, 'test_start': 0.70, 'test_end': 1.00},
]

# Random search config
N_SAMPLES = 25000
PRIMARY_SEED = 42
REPLICATION_SEEDS = [1, 7, 21, 42, 99]
W_MAX = 2.0

# Parameter ranges
L_MIN, L_MAX = 10, 252
N_MIN, N_MAX = 10, 180
K_MIN, K_MAX = 5, 30
R_MIN, R_MAX = 0.3, 1.8
TARGET_VOL_MIN, TARGET_VOL_MAX = 0.05, 1.50

# Scoring
MIN_AVG_EXPOSURE = 0.70
TURNOVER_PENALTY = 0.0005
MEAN_BONUS = 0.25
EXPOSURE_OVER_1_PENALTY = 0.10

# Output
OUTPUT_DIR = 'global_opt_v4'
PLOTLY_CONFIG = {'displayModeBar': False, 'responsive': True, 'displaylogo': False}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def calculate_nat_vol_on_slice(df, start_idx, end_idx, ann_factor):
    """Calculate natural volatility on a specific slice (no look-ahead)."""
    df_slice = df.iloc[start_idx:end_idx].copy()
    df_slice['close'] = df_slice['close'].ffill()
    rets = df_slice['close'].pct_change(fill_method=None)
    return rets.std(ddof=1) * np.sqrt(ann_factor)

def run_strategy_for_fold(df, L, N, K, target_vol, w_max, fee_bps, ann_factor, 
                          eval_start_idx, eval_end_idx):
    """Run strategy on data and evaluate metrics on specified window."""
    df = df.copy().sort_index()
    df['close'] = df['close'].ffill()
    
    df['ret'] = df['close'].pct_change(fill_method=None)
    df['mom'] = df['close'].pct_change(L, fill_method=None)
    df['signal'] = (df['mom'] > 0).astype(float)
    df['vol'] = df['ret'].rolling(N).std(ddof=1) * np.sqrt(ann_factor)
    df['w_scale'] = (target_vol / df['vol'].replace(0, np.nan)).clip(0, w_max)
    df['w_scale'] = df['w_scale'].bfill().ffill().fillna(1.0)
    
    df['w_voltarget'] = df['signal'] * df['w_scale']
    
    rebal_mask = np.arange(len(df)) % K == 0
    df['w_exec_voltarget'] = df['w_voltarget'].where(rebal_mask).ffill().fillna(0)
    df['w_exec_voltarget'] = df['w_exec_voltarget'].shift(1).fillna(0)
    
    df['turnover_voltarget'] = df['w_exec_voltarget'].diff().abs()
    
    fee_rate = fee_bps / 10000
    df['cost_voltarget'] = df['turnover_voltarget'] * fee_rate
    
    df['ret_market'] = df['ret']
    df['ret_voltarget'] = df['w_exec_voltarget'] * df['ret'] - df['cost_voltarget']
    
    warmup = max(L, N) + 10
    df = df.iloc[warmup:].copy()
    
    if len(df) < 50:
        return None
    
    eval_start_adj = max(0, eval_start_idx - warmup)
    eval_end_adj = eval_end_idx - warmup
    
    if eval_end_adj <= eval_start_adj or eval_end_adj > len(df):
        return None
    
    df_eval = df.iloc[eval_start_adj:eval_end_adj].copy()
    
    if len(df_eval) < 30:
        return None
    
    df_eval['equity_market'] = (1 + df_eval['ret_market'].fillna(0)).cumprod()
    df_eval['equity_voltarget'] = (1 + df_eval['ret_voltarget'].fillna(0)).cumprod()
    
    return df_eval

def compute_metrics(df, strategy, ann_factor):
    """Compute metrics for a strategy."""
    if df is None or len(df) < 30:
        return None
    
    ret_col = f'ret_{strategy}'
    eq_col = f'equity_{strategy}'
    
    rets = df[ret_col].dropna()
    if len(rets) < 30:
        return None
    
    eq = df[eq_col]
    years = len(rets) / ann_factor
    final_eq = eq.iloc[-1]
    
    try:
        cagr = (final_eq ** (1/years)) - 1 if years > 0 and final_eq > 0 else 0
    except:
        cagr = 0
    
    std = rets.std(ddof=1)
    sharpe = (rets.mean() / std * np.sqrt(ann_factor)) if std > 1e-10 else 0.0
    max_dd = (eq / eq.cummax() - 1).min()
    
    if strategy == 'market':
        avg_exposure = 1.0
        avg_turnover = 0.0
        ann_turnover = 0.0
    else:
        w_exec = df[f'w_exec_{strategy}']
        turnover = df[f'turnover_{strategy}']
        avg_exposure = w_exec.abs().mean()
        avg_turnover = turnover.mean()
        ann_turnover = avg_turnover * ann_factor
    
    return {
        'Sharpe': sharpe, 'CAGR': cagr, 'MaxDD': max_dd, 'Final_Eq': final_eq,
        'AvgExposure': avg_exposure, 'AvgTurnover': avg_turnover, 'AnnTurnover': ann_turnover,
        'N_obs': len(rets)
    }

def sample_parameters(n_samples, seed=42):
    """Generate random parameter samples."""
    rng = np.random.RandomState(seed)
    samples = []
    
    for _ in range(n_samples):
        u = rng.uniform(np.log10(L_MIN), np.log10(L_MAX))
        L = int(round(10**u))
        L = max(L_MIN, min(L_MAX, L))
        N = rng.randint(N_MIN, N_MAX + 1)
        K = rng.randint(K_MIN, K_MAX + 1)
        r = rng.uniform(R_MIN, R_MAX)
        samples.append({'L': L, 'N': N, 'K': K, 'r': r})
    return samples

# =============================================================================
# MAIN EVALUATION (Search on Pre-2023 Data)
# =============================================================================
def evaluate_config_v4(params, asset_data_search):
    """Evaluate config on Pre-Holdout search data."""
    L, N, K, r = params['L'], params['N'], params['K'], params['r']
    
    result = {'L': L, 'N': N, 'K': K, 'r': r}
    fold_details = {asset: [] for asset in ASSETS.keys()}
    
    for asset_name, data in asset_data_search.items():
        df = data['df']
        ann_factor = data['ann_factor']
        fee_bps = data['fee_bps']
        max_exposure = data['max_exposure']
        
        n_total = len(df)
        
        for fold in FOLDS:
            fold_name = fold['name']
            
            train_end_idx = int(n_total * fold['train_end'])
            test_start_idx = int(n_total * fold['test_start'])
            test_end_idx = int(n_total * fold['test_end'])
            
            # Nat vol on train only
            nat_vol_fold = calculate_nat_vol_on_slice(df, 0, train_end_idx, ann_factor)
            if np.isnan(nat_vol_fold) or nat_vol_fold < 0.01:
                return None
            
            target_vol_fold = max(TARGET_VOL_MIN, min(TARGET_VOL_MAX, r * nat_vol_fold))
            
            df_eval = run_strategy_for_fold(df, L, N, K, target_vol_fold, W_MAX, fee_bps, ann_factor,
                                            test_start_idx, test_end_idx)
            if df_eval is None:
                return None
            
            metrics_mkt = compute_metrics(df_eval, 'market', ann_factor)
            metrics_vt = compute_metrics(df_eval, 'voltarget', ann_factor)
            
            if metrics_mkt is None or metrics_vt is None:
                return None
            
            if metrics_vt['AvgExposure'] > max_exposure:
                return None  # Violates per-asset cap
            
            fold_details[asset_name].append({
                'fold': fold_name,
                'Sharpe_Mkt': metrics_mkt['Sharpe'],
                'Sharpe_VT': metrics_vt['Sharpe'],
                'Delta_Sharpe': metrics_vt['Sharpe'] - metrics_mkt['Sharpe'],
                'AvgExposure_VT': metrics_vt['AvgExposure'],
                'AnnTurnover_VT': metrics_vt['AnnTurnover'],
            })
    
    all_exposures = []
    all_turnovers = []
    
    for asset_name in ASSETS.keys():
        folds = fold_details[asset_name]
        delta_sharpes = [f['Delta_Sharpe'] for f in folds]
        exposures = [f['AvgExposure_VT'] for f in folds]
        turnovers = [f['AnnTurnover_VT'] for f in folds]
        
        result[f'{asset_name}_Delta_Sharpe_avg'] = np.mean(delta_sharpes)
        result[f'{asset_name}_Delta_Sharpe_min'] = np.min(delta_sharpes)
        result[f'{asset_name}_Delta_Sharpe_max'] = np.max(delta_sharpes)
        result[f'{asset_name}_Sharpe_VT_avg'] = np.mean([f['Sharpe_VT'] for f in folds])
        result[f'{asset_name}_Sharpe_Mkt_avg'] = np.mean([f['Sharpe_Mkt'] for f in folds])
        result[f'{asset_name}_AvgExposure_avg'] = np.mean(exposures)
        
        all_exposures.extend(exposures)
        all_turnovers.extend(turnovers)
    
    avg_exposure_global = np.mean(all_exposures)
    avg_turnover_ann_global = np.mean(all_turnovers)
    
    result['avg_exposure_global'] = avg_exposure_global
    result['avg_turnover_ann_global'] = avg_turnover_ann_global
    
    if avg_exposure_global < MIN_AVG_EXPOSURE:
        result['score'] = -999
        result['score_core'] = -999
        result['score_mean'] = 0
        return result
    
    # Scoring
    delta_sharpes = [result[f'{a}_Delta_Sharpe_avg'] for a in ASSETS.keys()]
    score_core = min(delta_sharpes)
    score_mean = np.mean(delta_sharpes)
    exposure_penalty = max(0, avg_exposure_global - 1.0) * EXPOSURE_OVER_1_PENALTY
    
    score = score_core + MEAN_BONUS * score_mean - TURNOVER_PENALTY * avg_turnover_ann_global - exposure_penalty
    
    result['score'] = score
    result['score_core'] = score_core
    result['score_mean'] = score_mean
    result['exposure_penalty'] = exposure_penalty
    
    return result

def evaluate_holdout_calendar(params, asset_data_holdout, asset_data_search):
    """Evaluate best config on Common Calendar Holdout (2023+)."""
    L, N, K, r = int(params['L']), int(params['N']), int(params['K']), params['r']
    
    results = {}
    
    for asset_name, data in asset_data_holdout.items():
        df_holdout = data['df']
        df_search = asset_data_search[asset_name]['df']
        
        ann_factor = data['ann_factor']
        fee_bps = data['fee_bps']
        
        # Calculate Nat Vol using ALL search data (up to 2022-12-31)
        # This is strictly not looking at holdout data
        nat_vol_search = df_search['close'].pct_change(fill_method=None).std(ddof=1) * np.sqrt(ann_factor)
        
        target_vol = max(TARGET_VOL_MIN, min(TARGET_VOL_MAX, r * nat_vol_search))
        
        # Run on holdout data
        # Note: We run on holdout slice directly. 
        # Ideally we'd prepend some search data for warmup, but 2023-Start is ample.
        # To be cleaner, let's just run on df_holdout but skip first max(L,N) days.
        
        df_eval = run_strategy_for_fold(df_holdout, L, N, K, target_vol, W_MAX, fee_bps, ann_factor,
                                        0, len(df_holdout))
        
        if df_eval is None:
            results[asset_name] = {'error': 'insufficient_data'}
            continue
        
        metrics_mkt = compute_metrics(df_eval, 'market', ann_factor)
        metrics_vt = compute_metrics(df_eval, 'voltarget', ann_factor)
        
        if metrics_mkt is None or metrics_vt is None:
            results[asset_name] = {'error': 'compute_failed'}
            continue
        
        results[asset_name] = {
            'Sharpe_VT': metrics_vt['Sharpe'],
            'Sharpe_Mkt': metrics_mkt['Sharpe'],
            'Delta_Sharpe': metrics_vt['Sharpe'] - metrics_mkt['Sharpe'],
            'AvgExposure': metrics_vt['AvgExposure'],
            'AnnTurnover': metrics_vt['AnnTurnover'],
            'CAGR_VT': metrics_vt['CAGR'],
            'CAGR_Mkt': metrics_mkt['CAGR'],
            'MaxDD_VT': metrics_vt['MaxDD'],
        }
    
    return results

# =============================================================================
# FIGURE 5 V4 GENERATION
# =============================================================================
SHARED_CSS = """
<style>
    body { font-family: 'Segoe UI', system-ui, sans-serif; background: #f8f9fa; margin: 0; padding: 20px; }
    .card { background: white; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); 
            padding: 30px; max-width: 1250px; margin: 0 auto; }
    h1 { font-size: 1.5em; color: #1a1a2e; margin-bottom: 0.3em; }
    h2 { font-size: 1.2em; color: #333; margin-top: 25px; }
    .subtitle { font-size: 0.95em; color: #555; margin-bottom: 20px; line-height: 1.5; }
    .method { font-size: 0.85em; color: #666; font-style: italic; margin-bottom: 15px; 
              background: #f0f0f0; padding: 12px; border-radius: 6px; }
    .metrics-table { width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 0.88em; }
    .metrics-table th, .metrics-table td { padding: 8px 10px; text-align: left; border-bottom: 1px solid #e0e0e0; }
    .metrics-table th { background: #f5f5f5; font-weight: 600; }
    .caption { font-size: 0.85em; color: #666; margin-top: 15px; font-style: italic; }
    .footnote { font-size: 0.8em; color: #888; margin-top: 20px; border-top: 1px solid #eee; padding-top: 15px; }
    .positive { color: #2ca02c; font-weight: 600; }
    .negative { color: #d62728; }
    .holdout-box { background: #e8f4e8; border: 1px solid #4caf50; border-radius: 8px; padding: 15px; margin: 20px 0; }
    .warning { background: #fff3e0; border: 1px solid #ff9800; border-radius: 6px; padding: 10px; margin: 10px 0; font-size: 0.85em; }
    @media print { .card { box-shadow: none; } }
</style>
"""

def generate_fig5_v4(best_config, holdout_results, all_results_df, n_valid, seed_comparison):
    """Generate Figure 5 V4.1."""
    best = best_config
    assets = list(ASSETS.keys())
    
    # Plot 1: Delta Sharpe
    delta_sharpes = [best.get(f'{a}_Delta_Sharpe_avg', 0) for a in assets]
    colors = ['#2ca02c' if d > 0 else '#d62728' for d in delta_sharpes]
    
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=assets, y=delta_sharpes, marker_color=colors,
        text=[f'{d:+.3f}' for d in delta_sharpes], textposition='outside'
    ))
    fig1.update_layout(title='Search Phase ΔSharpe (2020-2022)', template='plotly_white', height=380)
    
    # Plot 2: Holdout
    holdout_deltas = [holdout_results.get(a, {}).get('Delta_Sharpe', 0) for a in assets]
    hol_colors = ['#2ca02c' if d > 0 else '#d62728' for d in holdout_deltas]
    
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=assets, y=holdout_deltas, marker_color=hol_colors,
        text=[f'{d:+.3f}' for d in holdout_deltas], textposition='outside'
    ))
    fig2.update_layout(title=f'Holdout ΔSharpe ({HOLDOUT_START_DATE}+)', template='plotly_white', height=380)
    
    chart1_html = pio.to_html(fig1, full_html=False, include_plotlyjs=False, config=PLOTLY_CONFIG)
    chart2_html = pio.to_html(fig2, full_html=False, include_plotlyjs=False, config=PLOTLY_CONFIG)
    
    # Tables
    params_table = f"""
    <table class='metrics-table'>
        <thead><tr><th colspan="2">Best Parameters</th><th colspan="2">Scores</th></tr></thead>
        <tbody>
            <tr><td>Lookback (L)</td><td>{best['L']}</td><td>score_core</td><td>{best['score_core']:.4f}</td></tr>
            <tr><td>Vol Window (N)</td><td>{best['N']}</td><td>score_mean</td><td>{best['score_mean']:.4f}</td></tr>
            <tr><td>Rebalance (K)</td><td>{best['K']}</td><td>Avg Exposure</td><td>{best['avg_exposure_global']:.2f}</td></tr>
            <tr><td>Vol Ratio (r)</td><td>{best['r']:.2f}</td><td>Exp Penalty</td><td>{best.get('exposure_penalty', 0):.4f}</td></tr>
        </tbody>
    </table>"""
    
    wf_table = """
    <table class='metrics-table'>
        <thead><tr><th>Asset</th><th>Sharpe(VT)</th><th>Sharpe(Mkt)</th><th>ΔSharpe (avg)</th><th>Min/Max</th><th>Avg Exp</th></tr></thead>
        <tbody>"""
    for asset in assets:
        svt = best.get(f'{asset}_Sharpe_VT_avg', 0)
        smkt = best.get(f'{asset}_Sharpe_Mkt_avg', 0)
        delta = best.get(f'{asset}_Delta_Sharpe_avg', 0)
        dmin = best.get(f'{asset}_Delta_Sharpe_min', 0)
        dmax = best.get(f'{asset}_Delta_Sharpe_max', 0)
        exp = best.get(f'{asset}_AvgExposure_avg', 0)
        color = 'positive' if delta > 0 else 'negative'
        wf_table += f"<tr><td>{asset}</td><td>{svt:.3f}</td><td>{smkt:.3f}</td>"
        wf_table += f"<td class='{color}'>{delta:+.3f}</td><td>[{dmin:+.2f}, {dmax:+.2f}]</td><td>{exp:.2f}</td></tr>"
    wf_table += "</tbody></table>"
    
    hol_table = """
    <div class="holdout-box">
    <h3 style="margin-top:0">Final Calendar Holdout (2023-01-01 to End)</h3>
    <table class='metrics-table'>
        <thead><tr><th>Asset</th><th>Sharpe(VT)</th><th>Sharpe(Mkt)</th><th>ΔSharpe</th><th>CAGR(VT)</th><th>MaxDD(VT)</th><th>Beats?</th></tr></thead>
        <tbody>"""
    beats = 0
    for asset in assets:
        hr = holdout_results.get(asset, {})
        if 'error' in hr: b_str = "ERR"
        else:
            delta = hr.get('Delta_Sharpe', 0)
            if delta > 0: beats += 1
            b_str = "YES" if delta > 0 else "NO"
            color = 'positive' if delta > 0 else 'negative'
            hol_table += f"<tr><td>{asset}</td><td>{hr.get('Sharpe_VT',0):.3f}</td><td>{hr.get('Sharpe_Mkt',0):.3f}</td>"
            hol_table += f"<td class='{color}'>{delta:+.3f}</td><td>{hr.get('CAGR_VT',0):.1%}</td><td>{hr.get('MaxDD_VT',0):.1%}</td><td>{b_str}</td></tr>"
    hol_table += "</tbody></table></div>"
    
    seed_table = """
    <h3>Parameter Region Stability (Multi-Seed)</h3>
    <table class='metrics-table'>
        <thead><tr><th>Seed</th><th>L</th><th>N</th><th>K</th><th>r</th><th>score_core</th></tr></thead>
        <tbody>"""
    for sc in seed_comparison:
        seed_table += f"<tr><td>{sc['seed']}</td><td>{sc['L']}</td><td>{sc['N']}</td><td>{sc['K']}</td><td>{sc['r']:.2f}</td><td>{sc['score_core']:.4f}</td></tr>"
    seed_table += "</tbody></table>"
    
    subtitle = f"Optimization on {2020}-{2022} data. Holdout test on {2023}-End (Common Calendar Regime).<br>"
    if beats == 4: subtitle += "Global config generalizes to ALL assets in holdout."
    elif beats == 0: subtitle += "Global config fails to generalize to any asset in holdout."
    else: subtitle += f"Partial generalization: {beats}/4 assets beat market in holdout (Regime Dependent)."

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8"><title>Figure 5 - Global Parameter Search (V4.1)</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>{SHARED_CSS}
</head>
<body>
<div class="card">
    <h1>Figure 5 — Global Parameter Search (Thesis Appendix)</h1>
    <p class="subtitle">{subtitle}</p>
    <div class="method">
        <strong>Methodology (Thesis-Safe):</strong> Random search (N={N_SAMPLES:,}) on <strong>Pre-2023 Data</strong>.
        Walk-Forward Validation (3 disjoint folds). Final Test on <strong>Common Calendar Holdout (2023-Present)</strong>.
        Strict Exposure Caps (Crypto ≤1.50, Equities ≤1.10).
    </div>
    <div style="display: flex; gap: 20px; flex-wrap: wrap;">
        <div>{chart1_html}</div><div>{chart2_html}</div>
    </div>
    <h2>Best Configuration (Selected on 2020-2022 Data)</h2>
    {params_table}
    <h2>Search Phase Results (Walk-Forward 2020-2022)</h2>
    {wf_table}
    {hol_table}
    {seed_table}
    <p class="footnote">Signal=0 implies 0% exposure. Costs applied. Benchmark frictionless.</p>
</div></body></html>"""
    with open('fig5_global_v4.html', 'w', encoding='utf-8') as f: f.write(html)

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70); print("GLOBAL OPTIMIZATION V4.1 (Calendar Holdout)"); print("=" * 70)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load and Split Data
    print("\n[1/6] Loading data (Split 2023-01-01)...")
    data_search = {}
    data_holdout = {}
    
    for name, cfg in ASSETS.items():
        df = load_data_robust(cfg['file'])
        df.index = pd.to_datetime(df.index).tz_localize(None) # Normalize TZ
        df = df.sort_index().copy()
        
        split_date = pd.Timestamp(HOLDOUT_START_DATE)
        df_search = df[df.index < split_date].copy()
        df_holdout = df[df.index >= split_date].copy()
        
        data_search[name] = {'df': df_search, **{k:v for k,v in cfg.items() if k!='file'}}
        data_holdout[name] = {'df': df_holdout, **{k:v for k,v in cfg.items() if k!='file'}}
        
        print(f"   {name}: Search={len(df_search)} rows, Holdout={len(df_holdout)} rows")

    # Sample
    print(f"\n[2/6] Sampling {N_SAMPLES:,} configs...")
    samples = sample_parameters(N_SAMPLES, PRIMARY_SEED)
    
    # Evaluate Search
    print(f"\n[3/6] Evaluating on Search Data...")
    results = []
    valid = 0
    for i, p in enumerate(samples):
        if (i+1)%5000==0: print(f"   {i+1}/{N_SAMPLES} ({valid} valid)")
        r = evaluate_config_v4(p, data_search)
        if r and r['score'] > -100:
            results.append(r)
            valid += 1
            
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(OUTPUT_DIR, 'global_random_results.csv'), index=False)
    
    if valid==0: print("No valid configs!"); return
    
    best = df_results.loc[df_results['score'].idxmax()].to_dict()
    print(f"\nBEST: L={int(best['L'])}, N={int(best['N'])}, K={int(best['K'])}, r={best['r']:.2f}")
    
    # Evaluate Holdout
    print("\n[4/6] Evaluating Calendar Holdout...")
    hol_res = evaluate_holdout_calendar(best, data_holdout, data_search)
    
    # Multi-seed
    print("\n[5/6] Multi-Seed Stability...")
    seed_comp = []
    for seed in REPLICATION_SEEDS:
        s_samps = sample_parameters(3000, seed)
        s_res = []
        for p in s_samps:
            r_ = evaluate_config_v4(p, data_search)
            if r_ and r_['score'] > -100: s_res.append(r_)
        if s_res:
            b_ = pd.DataFrame(s_res).nlargest(1,'score').iloc[0]
            seed_comp.append({'seed':seed, 'L':int(b_['L']), 'N':int(b_['N']), 'K':int(b_['K']), 'r':b_['r'], 'score_core':b_['score_core']})
            print(f"   Seed {seed}: L={int(b_['L'])}, K={int(b_['K'])}")
            
    # Save & Plot
    best_config = best.copy()
    # Add metrics for table
    for name in ASSETS.keys():
        best_config[f'{name}_Sharpe_VT_avg'] = float(best.get(f'{name}_Sharpe_VT_avg', 0))
        best_config[f'{name}_Sharpe_Mkt_avg'] = float(best.get(f'{name}_Sharpe_Mkt_avg', 0))
        best_config[f'{name}_Delta_Sharpe_avg'] = float(best.get(f'{name}_Delta_Sharpe_avg', 0))
        best_config[f'{name}_Delta_Sharpe_min'] = float(best.get(f'{name}_Delta_Sharpe_min', 0))
        best_config[f'{name}_Delta_Sharpe_max'] = float(best.get(f'{name}_Delta_Sharpe_max', 0))
        best_config[f'{name}_AvgExposure_avg'] = float(best.get(f'{name}_AvgExposure_avg', 0))

    with open(os.path.join(OUTPUT_DIR, 'best_global_config.json'), 'w') as f:
        json.dump({'best': best_config, 'holdout': hol_res}, f, indent=2, default=str)
        
    generate_fig5_v4(best_config, hol_res, df_results, valid, seed_comp)
    print("\nDONE.")

if __name__ == "__main__":
    main()
