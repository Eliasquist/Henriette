"""
Thesis Robustness Checks: Stress Testing & Statistical Inference
================================================================
This script performs two critical validations for the Master's Thesis:
1. Cost Sensitivity Stress Test:
   - Varying Transaction Fees (10, 25, 50 bps)
   - Varying Funding Spreads (0%, 2%, 4% above Cash Rate)
   - Checks if Sharpe > Buy&Hold persists under realistic institutional costs.

2. Paired Block Bootstrap Significance Test (Rigorous):
   - Resamples return series pairwise (Strategy, Market) using Circular Block Bootstrap.
   - Block Size = 10 (approx 2 weeks of autocorrelation protection).
   - Generates P-values and 95% Confidence Intervals for ΔSharpe.

Usage: python3 thesis_stress_test.py
"""

import numpy as np
import pandas as pd
from btc_momentum_onlyshortstrategy.strategies.ts_momentum import generate_ts_momentum_signals
from btc_momentum_onlyshortstrategy.strategies.vol_targeting import compute_vol_target_weights
from managed_vs_unmanaged_run import load_data_robust

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIGS = {
    'ETH': {
        'file': 'data/ETHUSDT_spot_ohlcv.csv', 
        'is_crypto': True, 
        'L': 120, 'N': 170, 'K': 11, 'Target': 0.95, 'Start': '2020-01-01'
    },
    'SOL': {
        'file': 'data/SOL_daily.csv', 
        'is_crypto': True,
        'L': 15, 'N': 172, 'K': 12, 'Target': 1.35, 'Start': '2021-01-01' # using approx Global Opt setting
    },
    'SPY': {
        'file': 'data/SPY_daily.csv',
        'is_crypto': False,
        'L': 252, 'N': 60, 'K': 21, 'Target': 0.15, 'Start': '2000-01-01'
    }
}

STRESS_FEES = [10, 25, 50] # bps
STRESS_FUNDING = [0.00, 0.02, 0.04] # spread annual

BOOTSTRAP_SAMPLES = 10000
BLOCK_SIZE = 12 # ~2 weeks to preserve short-term serial correlation

# =============================================================================
# 1. CORE STRATEGY ENGINE
# =============================================================================
def run_strategy_stress(df, config, fee_bps, funding_spread):
    """Run strategy with explicit Fee and Funding inputs."""
    df = df.copy()
    L, N, K_vt, sigma_target = config['L'], config['N'], config['K'], config['Target']
    ann_factor = 365 if config['is_crypto'] else 252
    
    # Identical logic to managed_vs_unmanaged_run.py
    # 1. Signals
    df['sig_daily'] = generate_ts_momentum_signals(df, L, allow_short=False)
    
    # 2. Vol Scaling
    df['vt_scaling_daily'] = compute_vol_target_weights(df, sigma_target, N, 2.0, ann_factor)
    
    # 3. Warmup Drop (CRITICAL MATCH)
    warmup = max(L, N) + 10
    df = df.iloc[warmup:].copy()
    
    # 4. Rebalance logic
    df['rebalance'] = False
    df.iloc[::K_vt, df.columns.get_loc('rebalance')] = True
    
    # 5. Target Weights
    df['target_w'] = np.where(df['rebalance'], df['sig_daily'] * df['vt_scaling_daily'], np.nan)
    df['target_w'] = df['target_w'].ffill().fillna(0.0)
    
    # 6. Executed Weights (Shift 1)
    df['w_exec'] = df['target_w'].shift(1).fillna(0.0)
    
    # 7. Costs
    # Turnover
    turnover = df['w_exec'].diff().abs().fillna(0)
    cost_turnover = turnover * (fee_bps / 10000)
    
    # Funding
    # Point A: "Funding applies to total gross exposure (w)"
    cost_funding = df['w_exec'].abs() * (funding_spread / ann_factor)
    
    # 8. Returns
    # managed_vs_unmanaged_run uses 'ret_daily' which is close.pct_change()
    df['ret'] = df['close'].pct_change()
    df['strat_gross'] = df['w_exec'] * df['ret']
    df['strat_net'] = df['strat_gross'] - cost_turnover - cost_funding
    
    return df[['strat_net', 'ret']].rename(columns={'ret': 'mkt_ret'})

def get_sharpe(rets, ann_factor):
    """Calculate annualized Sharpe ratio with ddof=1 (sample std)."""
    rets = np.asarray(rets)
    if len(rets) < 10:
        return 0.0
    sd = np.std(rets, ddof=1)
    if sd < 1e-9:
        return 0.0
    return (rets.mean() / sd) * np.sqrt(ann_factor)

# =============================================================================
# 2. PAIRED BLOCK BOOTSTRAP (RIGOROUS INFERENCE)
# =============================================================================
def paired_block_bootstrap(ret_strat, ret_bench, n_samples=10000, block_size=12, ann_factor=365):
    """
    Test H0: Sharpe(Strat) <= Sharpe(Bench).
    Uses Circular Block Bootstrap to preserve serial correlation.
    
    Returns:
      - p_value: Prob(SharpeDiff <= 0)
      - ci_lower: 2.5th percentile of Sharpe Diff distribution
      - ci_upper: 97.5th percentile of Sharpe Diff distribution
      - diff_dist: validation distribution array
    """
    # Align Data
    data = pd.DataFrame({'s': ret_strat, 'b': ret_bench}).dropna()
    n = len(data)
    if n < 50: return np.nan, np.nan, np.nan, []
    
    s_vals = data['s'].values
    b_vals = data['b'].values
    
    observed_diff = get_sharpe(s_vals, ann_factor) - get_sharpe(b_vals, ann_factor)
    
    # Pre-compute block indices (Circular)
    # Total blocks per sample
    num_blocks = int(np.ceil(n / block_size))
    
    bootstrap_diffs = []
    rng = np.random.default_rng(42)
    
    # --- Block Bootstrap Loop ---
    for _ in range(n_samples):
        # 1. Choose random start indices for blocks
        start_indices = rng.integers(0, n, size=num_blocks)
        
        # 2. Construct resampled time series
        # Using numpy advanced indexing is tricky with wrapping, let's do it explicitly with indices
        # Generate full index array for this resample
        resampled_indices = []
        for start in start_indices:
            # Create block of indices [start, start+1, ..., start+L-1] % n
            block = np.arange(start, start + block_size) % n
            resampled_indices.extend(block)
        
        # Trim to original length n (since ceil might over-fill)
        resampled_indices = np.array(resampled_indices[:n])
        
        # 3. Get bootstrap samples
        s_boot = s_vals[resampled_indices]
        b_boot = b_vals[resampled_indices]
        
        # 4. Calculate Sharpe Diff
        # Sharpe = (mean / std(ddof=1)) * sqrt(Ann)
        # Using ddof=1 for sample std (consistency with get_sharpe)
        sh_s = (s_boot.mean() / (np.std(s_boot, ddof=1) + 1e-9)) * np.sqrt(ann_factor)
        sh_b = (b_boot.mean() / (np.std(b_boot, ddof=1) + 1e-9)) * np.sqrt(ann_factor)
        
        bootstrap_diffs.append(sh_s - sh_b)
        
    bootstrap_diffs = np.array(bootstrap_diffs)
    
    # Shift Distribution for H0 Test?
    # No, typically for "Significance of Outperformance", we check how much of the bootstrap distribution
    # of the statistic (Diff) is below zero. This corresponds to the probability that the "True" parameter is <= 0.
    # Ref: Efron & Tibshirani. 
    p_value = (bootstrap_diffs <= 0).mean()
    
    ci_lower = np.percentile(bootstrap_diffs, 2.5)
    ci_upper = np.percentile(bootstrap_diffs, 97.5)
    
    return p_value, ci_lower, ci_upper, bootstrap_diffs

# =============================================================================
# 3. FIGURE 6 GENERATION (Updated for Multi-Asset Forest Plot)
# =============================================================================
def generate_fig6(full_results_df):
    """
    Generate Figure 6:
    Panel A: Heatmap of Cost Sensitivity (ETH).
    Panel B: Forest Plot of Bootstrap Results (ETH, SOL, SPY).
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # --- Prepare Data ---
    # Panel A: ETH Stress
    stress_eth = full_results_df[
        (full_results_df['Type'] == 'Stress') & 
        (full_results_df['Asset'] == 'ETH')
    ]
    pivot = stress_eth.pivot(index='Funding', columns='Fee', values='Sharpe')
    pivot = pivot.sort_index(ascending=False)
    
    fees = pivot.columns.tolist()
    funds = pivot.index.tolist()
    z_vals = pivot.values
    
    # Panel B: Bootstrap Forest Plot
    boot_res = full_results_df[full_results_df['Type'] == 'Bootstrap'].reset_index()
    # Ordered Assets
    boot_res['Asset'] = pd.Categorical(boot_res['Asset'], categories=['ETH', 'SOL', 'SPY'], ordered=True)
    boot_res = boot_res.sort_values('Asset')
    
    assets = boot_res['Asset'].tolist()
    deltas = boot_res['Obs_Delta'].tolist()
    # Error bars (relative to delta)
    err_low = [d - r['CI_Lower'] for d, r in zip(deltas, boot_res.to_dict('records'))]
    err_high = [r['CI_Upper'] - d for d, r in zip(deltas, boot_res.to_dict('records'))]
    colors = ['#2ca02c' if d > 0 else '#d62728' for d in deltas]
    
    # --- Create Plot ---
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("<b>(A) Cost Sensitivity 'Matrix of Pain' (ETH)</b><br><i>Sharpe Ratio vs Funding & Fees</i>", 
                       "<b>(B) Statistical Significance (95% CI)</b><br><i>Paired Block Bootstrap (N=10,000)</i>"),
        horizontal_spacing=0.15
    )
    
    # Panel A: Heatmap
    fig.add_trace(go.Heatmap(
        z=z_vals, x=fees, y=funds,
        colorscale='RdYlGn', zmin=0.8, zmax=1.3,
        colorbar=dict(title='Sharpe', x=0.45, thickness=10),
        text=z_vals, texttemplate="%{z:.2f}",
        textfont={"size":12}
    ), row=1, col=1)
    
    # Panel B: Forest Plot
    fig.add_trace(go.Scatter(
        x=deltas, y=assets,
        mode='markers',
        marker=dict(color=colors, size=15, symbol='square'),
        error_x=dict(
            type='data',
            symmetric=False,
            array=err_high,
            arrayminus=err_low,
            color='black',
            thickness=1.5,
            width=5
        ),
        name='Delta Sharpe'
    ), row=1, col=2)
    
    # Add Zero Line to Panel B
    fig.add_vline(x=0, line_dash="dash", line_color="gray", row=1, col=2)
    
    # Layout
    fig.update_layout(
        title_text="<b>Figure 6 — Robustness & Inference: Multi-Asset Validation</b>",
        height=500, width=1100,
        template='plotly_white',
        showlegend=False
    )
    
    # Axes A
    fig.update_xaxes(title_text="Transaction Fee (bps)", type='category', row=1, col=1)
    fig.update_yaxes(title_text="Funding Spread (Annual %)", tickformat=".0%", row=1, col=1)
    
    # Axes B
    fig.update_xaxes(title_text="Δ Sharpe Ratio (Strategy - Market)", row=1, col=2)
    fig.update_yaxes(title_text="", row=1, col=2, type='category')
    
    fig.write_html("fig6_robustness.html")
    print("Figure 6 saved to fig6_robustness.html")

# =============================================================================
# 4. RUNNER
# =============================================================================
def main():
    print("="*80)
    print("THESIS ROBUSTNESS CHECK: PAIRED BLOCK BOOTSTRAP & STRESS TESTS")
    print("="*80)
    
    full_results_list = []
    
    # Loop over all assets
    for asset in ['ETH', 'SOL', 'SPY']: # Ordered
        print(f"\n>>> PROCESSING ASSET: {asset}")
        cfg = CONFIGS[asset]
        
        # Load Data
        df = load_data_robust(cfg['file'])
        df = df[df.index >= cfg['Start']]
        if df.empty: 
            print("Skipping (No Data)")
            continue
            
        ann_factor = 365 if cfg['is_crypto'] else 252
        
        # 1. Stress Test Matrix
        print(f"   [1] Running Cost Sensitivity Matrix...")
        # Compute market Sharpe from same sample window as strategy (after warmup)
        baseline_res = run_strategy_stress(df, cfg, 0, 0.0).dropna()
        market_sharpe = get_sharpe(baseline_res['mkt_ret'], ann_factor)
        
        for fee in STRESS_FEES:
            for fund in STRESS_FUNDING:
                res_df = run_strategy_stress(df, cfg, fee, fund)
                net_ret = res_df['strat_net'].dropna()
                s = get_sharpe(net_ret, ann_factor)
                
                full_results_list.append({
                    'Type': 'Stress',
                    'Asset': asset, 
                    'Fee': fee, 
                    'Funding': fund, 
                    'Sharpe': s, 
                    'Delta': s - market_sharpe
                })
        
        # 2. Bootstrap Inference
        print(f"   [2] Running Paired Block Bootstrap (N={BOOTSTRAP_SAMPLES})...")
        res_base = run_strategy_stress(df, cfg, 10, 0.0).dropna()
        
        p_val, ci_low, ci_high, dist = paired_block_bootstrap(
            res_base['strat_net'], res_base['mkt_ret'], 
            BOOTSTRAP_SAMPLES, BLOCK_SIZE, ann_factor
        )
        
        obs_diff = get_sharpe(res_base['strat_net'], ann_factor) - get_sharpe(res_base['mkt_ret'], ann_factor)
        
        print(f"      Observed Delta: {obs_diff:+.3f}")
        print(f"      P-Value: {p_val:.1%} (Prob <= 0)")
        print(f"      95% CI: [{ci_low:+.3f}, {ci_high:+.3f}]")
        
        full_results_list.append({
            'Type': 'Bootstrap',
            'Asset': asset,
            'Obs_Delta': obs_diff,
            'P_Value': p_val,
            'CI_Lower': ci_low,
            'CI_Upper': ci_high
        })
        
        # Note: Funding on Total Exposure (w) is conservative for perpetuals.
        # For margin/spot, funding on (w-1) would be more lenient.
        # This choice is documented in thesis_text/stress_test_table.md.
    
    # Save & Generate Figure
    df_out = pd.DataFrame(full_results_list)
    df_out.to_csv("thesis_robustness_final.csv", index=False)
    
    generate_fig6(df_out)
    
    print("\n" + "="*80)
    print("FINAL THESIS TABLE (Paste-Ready)")
    print("="*80)
    
    print("| Asset | Obs. ΔSharpe | Bootstrap P-Value | 95% Confidence Interval |")
    print("| :--- | :---: | :---: | :---: |")
    
    for asset in ['ETH', 'SOL', 'SPY']:
        row = df_out[(df_out['Type'] == 'Bootstrap') & (df_out['Asset'] == asset)]
        if not row.empty:
            r = row.iloc[0]
            sig = "**" if r['P_Value'] < 0.05 else ""
            print(f"| {asset} | {r['Obs_Delta']:+.2f} | {sig}{r['P_Value']:.1%}{sig} | [{r['CI_Lower']:+.2f}, {r['CI_Upper']:+.2f}] |")
            
if __name__ == "__main__":
    main()
