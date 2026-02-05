"""
Thesis Figure Generator V4 (Thesis-Safe)
Generates 4 separate HTML figure pages for master thesis screenshots.
Follows strict consistency: chart ↔ table ↔ caption alignment.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from managed_vs_unmanaged_run import load_data_robust

# =============================================================================
# SHARED CSS TEMPLATE
# =============================================================================
SHARED_CSS = """
<style>
    * { box-sizing: border-box; }
    body {
        font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
        max-width: 1200px;
        margin: 0 auto;
        padding: 30px 20px;
        background: #fff;
        color: #333;
        line-height: 1.6;
    }
    .figure-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 25px 30px;
        background: #fafafa;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    h1.figure-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #1a1a2e;
        margin: 0 0 20px 0;
        border-bottom: 2px solid #333;
        padding-bottom: 10px;
    }
    .subtitle {
        font-size: 1.0rem;
        color: #555;
        margin-bottom: 20px;
        font-style: italic;
    }
    .chart-container {
        margin: 20px 0;
    }
    .metrics-table {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
        font-size: 0.95rem;
    }
    .metrics-table th {
        background: #f0f0f0;
        padding: 12px 15px;
        text-align: left;
        border: 1px solid #ddd;
        font-weight: 600;
    }
    .metrics-table td {
        padding: 10px 15px;
        border: 1px solid #ddd;
        text-align: right;
    }
    .metrics-table td:first-child {
        text-align: left;
        font-weight: 500;
    }
    .footnote {
        font-size: 0.85rem;
        color: #666;
        margin-top: 10px;
        padding: 8px 10px;
        background: #f5f5f5;
        border-left: 3px solid #999;
    }
    .caption {
        font-size: 0.9rem;
        color: #555;
        font-style: italic;
        margin-top: 15px;
        padding: 10px;
        background: #f8f8f8;
        border-left: 3px solid #999;
    }
    @media print {
        body { padding: 10px; }
        .figure-card { box-shadow: none; border: 1px solid #ccc; }
    }
</style>
"""

PLOTLY_CONFIG = {'displayModeBar': False, 'responsive': True, 'displaylogo': False}

# =============================================================================
# STRATEGY RUNNER (with exposure/turnover tracking)
# =============================================================================
def run_strategy(df, lookback_l, sigma_target, n_days_vol, w_max, fee_bps, 
                 is_crypto, rebalance_k, include_costs=True):
    """
    Run TSMOM + VolTarget strategy with detailed metrics.
    Returns DataFrame with equity curves and stats dict.
    """
    ann_factor = 365 if is_crypto else 252
    df = df.copy().sort_index()
    
    # Forward-fill close prices to handle gaps (weekends, holidays)
    # This is standard practice for price data; we then use fill_method=None for pct_change
    df['close'] = df['close'].ffill()
    
    # Returns (using forward-filled close)
    df['ret'] = df['close'].pct_change(fill_method=None)
    
    # Momentum signal (1 = long, 0 = cash)
    df['mom'] = df['close'].pct_change(lookback_l, fill_method=None)
    df['signal'] = (df['mom'] > 0).astype(float)
    
    # Rolling volatility
    df['vol'] = df['ret'].rolling(n_days_vol).std() * np.sqrt(ann_factor)
    
    # Sanity check: vol should not be mostly NaN
    vol_valid_pct = df['vol'].notna().mean()
    if vol_valid_pct < 0.5:
        raise ValueError(f"Volatility series is {100*(1-vol_valid_pct):.0f}% NaN; check input data frequency/quality.")
    
    # Target weight (vol scaling)
    df['w_target'] = (sigma_target / df['vol'].replace(0, np.nan)).clip(0, w_max)
    # No-leakage NaN handling: fillna(1.0) for initial undefined region, then ffill for later gaps
    df['w_target'] = df['w_target'].fillna(1.0)  # Initial NaNs get default weight
    df['w_target'] = df['w_target'].ffill()       # Later gaps get forward-filled
    
    # Apply signal
    df['w_unmanaged'] = df['signal']
    df['w_voltarget'] = df['signal'] * df['w_target']
    
    # Rebalancing: update weights every K days
    rebal_mask = np.arange(len(df)) % rebalance_k == 0
    df['w_exec_unmanaged'] = df['w_unmanaged'].where(rebal_mask).ffill().fillna(0)
    df['w_exec_voltarget'] = df['w_voltarget'].where(rebal_mask).ffill().fillna(0)
    
    # Shift weights (decision at t, return at t+1)
    df['w_exec_unmanaged'] = df['w_exec_unmanaged'].shift(1).fillna(0)
    df['w_exec_voltarget'] = df['w_exec_voltarget'].shift(1).fillna(0)
    
    # Turnover
    df['turnover_unmanaged'] = df['w_exec_unmanaged'].diff().abs()
    df['turnover_voltarget'] = df['w_exec_voltarget'].diff().abs()
    
    # Transaction costs (on strategies only, not market)
    fee_rate = fee_bps / 10000
    df['cost_unmanaged'] = df['turnover_unmanaged'] * fee_rate if include_costs else 0
    df['cost_voltarget'] = df['turnover_voltarget'] * fee_rate if include_costs else 0
    
    # Strategy returns
    df['ret_market'] = df['ret']  # Buy & Hold (frictionless - NO costs)
    df['ret_unmanaged'] = df['w_exec_unmanaged'] * df['ret'] - df['cost_unmanaged']
    df['ret_voltarget'] = df['w_exec_voltarget'] * df['ret'] - df['cost_voltarget']
    
    # Drop warmup
    warmup = max(lookback_l, n_days_vol) + 10
    df = df.iloc[warmup:].copy()
    
    # Equity curves
    df['equity_market'] = (1 + df['ret_market']).cumprod()
    df['equity_unmanaged'] = (1 + df['ret_unmanaged']).cumprod()
    df['equity_voltarget'] = (1 + df['ret_voltarget']).cumprod()
    
    # Drawdowns
    for col in ['market', 'unmanaged', 'voltarget']:
        eq = df[f'equity_{col}']
        df[f'dd_{col}'] = eq / eq.cummax() - 1
    
    # Calculate stats
    def calc_stats(ret_series, eq_series, w_exec_series, turnover_series):
        years = len(ret_series) / ann_factor
        final_eq = eq_series.iloc[-1]
        cagr = (final_eq ** (1/years)) - 1
        sharpe = ret_series.mean() / ret_series.std(ddof=1) * np.sqrt(ann_factor) if ret_series.std() > 0 else 0
        max_dd = (eq_series / eq_series.cummax() - 1).min()
        avg_exposure = w_exec_series.abs().mean()
        avg_turnover = turnover_series.mean()
        ann_turnover = avg_turnover * ann_factor
        return {
            'Final_Eq': final_eq,
            'CAGR': cagr,
            'Sharpe': sharpe,
            'MaxDD': max_dd,
            'Avg_Exposure': avg_exposure,
            'Avg_Turnover': avg_turnover,
            'Ann_Turnover': ann_turnover
        }
    
    stats = {
        'market': calc_stats(df['ret_market'], df['equity_market'], 
                            pd.Series(1.0, index=df.index), pd.Series(0.0, index=df.index)),
        'unmanaged': calc_stats(df['ret_unmanaged'], df['equity_unmanaged'],
                               df['w_exec_unmanaged'], df['turnover_unmanaged']),
        'voltarget': calc_stats(df['ret_voltarget'], df['equity_voltarget'],
                               df['w_exec_voltarget'], df['turnover_voltarget'])
    }
    
    return df, stats

# =============================================================================
# CHART BUILDERS
# =============================================================================
def create_equity_drawdown_chart(df, strategies, log_scale=True, height=800):
    """Create 2-row subplot: equity curve + drawdown"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.65, 0.35],
        subplot_titles=("Equity Curve", "Drawdown")
    )
    
    colors = {'market': '#888888', 'unmanaged': '#1f77b4', 'voltarget': '#2ca02c', 
              'voltarget_best': '#2ca02c', 'voltarget_safe': '#ff7f0e'}
    names = {'market': 'Buy & Hold', 'unmanaged': 'Unmanaged (Signal)', 'voltarget': 'VolTarget',
             'voltarget_best': 'VolTarget (Best Sharpe)', 'voltarget_safe': 'VolTarget (Risk-Adj)'}
    
    for strat in strategies:
        color = colors.get(strat, '#333')
        name = names.get(strat, strat)
        
        fig.add_trace(
            go.Scatter(x=df.index, y=df[f'equity_{strat}'], 
                      name=name, line=dict(color=color, width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df[f'dd_{strat}'], 
                      name=name, line=dict(color=color, width=1.5),
                      showlegend=False),
            row=2, col=1
        )
    
    fig.update_yaxes(title_text="Equity", type="log" if log_scale else "linear", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    
    fig.update_layout(
        height=height,
        width=1150,
        template="plotly_white",
        margin=dict(l=60, r=30, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    
    return pio.to_html(fig, full_html=False, include_plotlyjs=False, config=PLOTLY_CONFIG)

def create_scaling_chart(sweeps, height=500):
    """Create Sharpe vs Ratio chart for multiple assets"""
    fig = go.Figure()
    
    colors = {'ETH': '#ff7f0e', 'SOL': '#9467bd', 'TSLA': '#1f77b4'}
    
    for name, sweep in sweeps.items():
        fig.add_trace(go.Scatter(
            x=sweep['ratio'], y=sweep['sharpe'],
            mode='lines+markers', name=name,
            line=dict(color=colors.get(name, '#333'), width=2.5),
            marker=dict(size=6)
        ))
    
    # Reference line at 1.25
    fig.add_vline(x=1.25, line_width=2, line_dash="dash", line_color="#2ca02c",
                  annotation_text="Reference: 1.25x", annotation_position="top right")
    
    fig.update_layout(
        height=height,
        width=1150,
        template="plotly_white",
        xaxis_title="Target Volatility / Natural Volatility Ratio",
        yaxis_title="Sharpe Ratio",
        margin=dict(l=60, r=30, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    
    return pio.to_html(fig, full_html=False, include_plotlyjs=False, config=PLOTLY_CONFIG)

# =============================================================================
# TABLE BUILDER (6 rows standard)
# =============================================================================
def create_metrics_table_6row(stats, strategies=['market', 'unmanaged', 'voltarget'], 
                               headers=None):
    """Create HTML table with standard 6 metrics"""
    if headers is None:
        headers = ['Metric'] + [{'market': 'Buy & Hold', 'unmanaged': 'Unmanaged', 'voltarget': 'VolTarget'}[s] for s in strategies]
    
    rows = [
        ['Final Equity ($1k start)'] + [f"${stats[s]['Final_Eq']*1000:,.0f}" for s in strategies],
        ['CAGR'] + [f"{stats[s]['CAGR']:.1%}" for s in strategies],
        ['Sharpe Ratio'] + [f"{stats[s]['Sharpe']:.2f}" for s in strategies],
        ['Max Drawdown'] + [f"{stats[s]['MaxDD']:.1%}" for s in strategies],
        ['Avg Exposure'] + [f"{stats[s]['Avg_Exposure']:.2f}" for s in strategies],
        ['Turnover (daily / ann.)'] + [f"{stats[s]['Avg_Turnover']:.4f} / {stats[s]['Ann_Turnover']:.1f}" for s in strategies],
    ]
    
    html = "<table class='metrics-table'><thead><tr>"
    for h in headers:
        html += f"<th>{h}</th>"
    html += "</tr></thead><tbody>"
    for row in rows:
        html += "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
    html += "</tbody></table>"
    return html

def create_scaling_table(sweep_results):
    """Create table for scaling law results"""
    headers = ['Asset', 'Natural Vol', 'Optimal Target', 'Ratio', 'Sharpe', 'Max DD']
    
    html = "<table class='metrics-table'><thead><tr>"
    for h in headers:
        html += f"<th>{h}</th>"
    html += "</tr></thead><tbody>"
    
    for name, data in sweep_results.items():
        html += f"<tr><td>{name}</td>"
        html += f"<td>{data['nat_vol']:.1%}</td>"
        html += f"<td>{data['opt_target']:.0%}</td>"
        html += f"<td>{data['ratio']:.2f}x</td>"
        html += f"<td>{data['sharpe']:.2f}</td>"
        html += f"<td>{data['maxdd']:.1%}</td></tr>"
    
    html += "</tbody></table>"
    return html

# =============================================================================
# PAGE BUILDER
# =============================================================================
def build_html_page(title, subtitle, chart_html, table_html, footnote, caption):
    """Assemble complete HTML page"""
    subtitle_html = f"<p class='subtitle'>{subtitle}</p>" if subtitle else ""
    footnote_html = f"<p class='footnote'>{footnote}</p>" if footnote else ""
    
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
    {SHARED_CSS}
</head>
<body>
    <div class="figure-card">
        <h1 class="figure-title">{title}</h1>
        {subtitle_html}
        <div class="chart-container">
            {chart_html}
        </div>
        {table_html}
        {footnote_html}
        <p class="caption">{caption}</p>
    </div>
</body>
</html>"""

# =============================================================================
# SENSITIVITY SWEEP
# =============================================================================
def run_sensitivity_sweep(df, name, nat_vol, is_crypto, l_val, n_val, fee_bps):
    """Run parameter sweep for scaling law analysis"""
    results = []
    ratios = np.linspace(0.5, 2.0, 15)
    
    for r in ratios:
        target_vol = nat_vol * r
        target_vol = max(0.05, min(target_vol, 1.5))
        try:
            _, stats = run_strategy(
                df, lookback_l=l_val, sigma_target=target_vol, n_days_vol=n_val,
                w_max=2.0, fee_bps=fee_bps, is_crypto=is_crypto, rebalance_k=10
            )
            results.append({
                'ratio': r, 
                'target_vol': target_vol, 
                'sharpe': stats['voltarget']['Sharpe'],
                'maxdd': stats['voltarget']['MaxDD']
            })
        except Exception as e:
            print(f"Error in sweep {name} r={r}: {e}")
    
    return pd.DataFrame(results)

def calculate_natural_volatility(df, is_crypto):
    """Calculate natural volatility on full sample"""
    ann_factor = 365 if is_crypto else 252
    rets = df['close'].pct_change(fill_method=None)
    return rets.std(ddof=1) * np.sqrt(ann_factor)

# =============================================================================
# CONSISTENCY CHECK
# =============================================================================
def verify_consistency(name, df, stats):
    """Verify chart and table consistency"""
    print(f"\n   Consistency Check: {name}")
    for strat in ['market', 'unmanaged', 'voltarget']:
        if f'equity_{strat}' in df.columns:
            chart_val = df[f'equity_{strat}'].iloc[-1]
            table_val = stats[strat]['Final_Eq']
            match = abs(chart_val - table_val) < 0.001
            status = "OK" if match else "MISMATCH"
            print(f"      {strat}: Chart={chart_val:.4f}, Table={table_val:.4f} [{status}]")

# =============================================================================
# MAIN GENERATOR
# =============================================================================
def generate_thesis_figures():
    print("=" * 70)
    print("THESIS FIGURE GENERATOR V4 (Thesis-Safe)")
    print("=" * 70)
    
    FOOTNOTE = "Signal=0 implies 0% exposure and 0% cash return. Costs apply to strategies only as turnover cost on executed weights."
    
    # Load data
    print("\n[1/5] Loading data...")
    df_eth = load_data_robust("data/ETHUSDT_spot_ohlcv.csv")
    df_spy = load_data_robust("data/SPY_daily.csv")
    df_tsla = load_data_robust("data/TSLA_daily.csv")
    df_sol = load_data_robust("data/SOL_daily.csv")
    
    # Calculate natural volatilities
    vol_eth = calculate_natural_volatility(df_eth, True)
    vol_spy = calculate_natural_volatility(df_spy, False)
    vol_tsla = calculate_natural_volatility(df_tsla, False)
    vol_sol = calculate_natural_volatility(df_sol, True)
    
    print(f"   ETH Natural Vol: {vol_eth:.1%}")
    print(f"   SPY Natural Vol: {vol_spy:.1%}")
    print(f"   TSLA Natural Vol: {vol_tsla:.1%}")
    print(f"   SOL Natural Vol: {vol_sol:.1%}")
    
    # =========================================================================
    # FIGURE 1: ETH
    # =========================================================================
    print("\n[2/5] Generating Figure 1 (ETH)...")
    res_eth, stats_eth = run_strategy(
        df_eth, lookback_l=120, sigma_target=0.95, n_days_vol=170,
        w_max=2.0, fee_bps=5.0, is_crypto=True, rebalance_k=10
    )
    
    chart_eth = create_equity_drawdown_chart(res_eth, ['market', 'unmanaged', 'voltarget'], log_scale=True)
    table_eth = create_metrics_table_6row(stats_eth)
    
    html_eth = build_html_page(
        title="Figure 1 — ETH: TSMOM + Volatility Targeting (K=10)",
        subtitle="Long-only momentum strategy with volatility-scaled position sizing.",
        chart_html=chart_eth,
        table_html=table_eth,
        footnote=FOOTNOTE,
        caption="Strategy results include proportional turnover costs; the buy-and-hold benchmark is frictionless."
    )
    with open("fig1_eth.html", "w", encoding="utf-8") as f:
        f.write(html_eth)
    print("   -> fig1_eth.html written")
    verify_consistency("ETH", res_eth, stats_eth)
    
    # =========================================================================
    # FIGURE 2: SPY (Fixed Narrative)
    # =========================================================================
    print("\n[3/5] Generating Figure 2 (SPY)...")
    res_spy, stats_spy = run_strategy(
        df_spy, lookback_l=252, sigma_target=0.20, n_days_vol=60,
        w_max=2.0, fee_bps=1.0, is_crypto=False, rebalance_k=10
    )
    
    # Determine narrative direction (metric-driven)
    vt_beats_market = stats_spy['voltarget']['Final_Eq'] > stats_spy['market']['Final_Eq']
    avg_exposure = stats_spy['voltarget']['Avg_Exposure']
    ann_turnover = stats_spy['voltarget']['Ann_Turnover']
    
    # Debug output
    print(f"   SPY Metrics: Avg_Exposure={avg_exposure:.2f}, Ann_Turnover={ann_turnover:.1f}")
    
    # Metric-driven subtitle (no "fails/underperforms" language)
    if vt_beats_market and avg_exposure > 1.05:
        subtitle = (f"VolTarget exceeds Buy & Hold in this sample, but with materially higher "
                   f"average exposure ({avg_exposure:.2f}x). The incremental return is largely "
                   f"attributable to exposure scaling on a drifting asset rather than robust "
                   f"trend-timing alpha; results are parameter- and cost-sensitive.")
        title_spy = "Figure 2 — SPY: Exposure-Driven Outperformance (K=10)"
    elif vt_beats_market and avg_exposure <= 1.05:
        subtitle = (f"VolTarget shows modest improvement over Buy & Hold (Avg Exposure: {avg_exposure:.2f}x). "
                   f"The scaling effect is limited; incremental benefit is sensitive to parameters and costs.")
        title_spy = "Figure 2 — SPY: Modest Improvement (K=10)"
    else:
        subtitle = ("VolTarget does not exceed Buy & Hold in this sample. "
                   "Incremental performance appears limited and sensitive to parameterization "
                   "and turnover costs under the long-only, cash=0 assumption.")
        title_spy = "Figure 2 — SPY: Limited Incremental Benefit (K=10)"
    
    chart_spy = create_equity_drawdown_chart(res_spy, ['market', 'unmanaged', 'voltarget'], log_scale=False)
    table_spy = create_metrics_table_6row(stats_spy)
    
    html_spy = build_html_page(
        title=title_spy,
        subtitle=subtitle,
        chart_html=chart_spy,
        table_html=table_spy,
        footnote=FOOTNOTE,
        caption="Equity capital markets show different dynamics than crypto; improvements should be interpreted cautiously."
    )
    with open("fig2_spy.html", "w", encoding="utf-8") as f:
        f.write(html_spy)
    print("   -> fig2_spy.html written")
    verify_consistency("SPY", res_spy, stats_spy)
    
    # =========================================================================
    # FIGURE 3: TSLA (Using Best from Optimization)
    # =========================================================================
    print("\n[4/5] Generating Figure 3 (TSLA)...")
    
    # Load optimization results
    try:
        df_opt = pd.read_csv("TSLA_OPTIMIZATION_FULL.csv")
        print(f"   Loaded {len(df_opt)} optimization results")
        
        # Best by Sharpe
        best_sharpe = df_opt.nlargest(1, 'Sharpe').iloc[0]
        print(f"   Best Sharpe: L={int(best_sharpe['L'])}, σ={best_sharpe['sigma']:.2f}, "
              f"N={int(best_sharpe['N'])}, K={int(best_sharpe['K'])} -> Sharpe={best_sharpe['Sharpe']:.2f}")
        
        # Best risk-adjusted (MaxDD > -60%)
        filtered = df_opt[df_opt['MaxDD'] > -0.60]
        if len(filtered) > 0:
            best_safe = filtered.nlargest(1, 'Sharpe').iloc[0]
        else:
            filtered = df_opt[df_opt['MaxDD'] > -0.70]
            best_safe = filtered.nlargest(1, 'Sharpe').iloc[0] if len(filtered) > 0 else best_sharpe
        print(f"   Best Risk-Adj: L={int(best_safe['L'])}, σ={best_safe['sigma']:.2f}, "
              f"N={int(best_safe['N'])}, K={int(best_safe['K'])} -> Sharpe={best_safe['Sharpe']:.2f}, MaxDD={best_safe['MaxDD']:.1%}")
        
        # Run both configs
        res_tsla_best, stats_tsla_best = run_strategy(
            df_tsla, lookback_l=int(best_sharpe['L']), sigma_target=best_sharpe['sigma'],
            n_days_vol=int(best_sharpe['N']), w_max=2.0, fee_bps=1.0, 
            is_crypto=False, rebalance_k=int(best_sharpe['K'])
        )
        
        res_tsla_safe, stats_tsla_safe = run_strategy(
            df_tsla, lookback_l=int(best_safe['L']), sigma_target=best_safe['sigma'],
            n_days_vol=int(best_safe['N']), w_max=2.0, fee_bps=1.0, 
            is_crypto=False, rebalance_k=int(best_safe['K'])
        )
        
        # PATCH 3: Align on common evaluation window (fair comparison)
        common_idx = res_tsla_best.index.intersection(res_tsla_safe.index)
        res_tsla_best = res_tsla_best.loc[common_idx].copy()
        res_tsla_safe = res_tsla_safe.loc[common_idx].copy()
        print(f"   Common evaluation period: {common_idx[0].date()} to {common_idx[-1].date()} ({len(common_idx)} days)")
        
        # REBASE equity to 1.0 at common start (for fair visual comparison)
        rebase_factor_market = res_tsla_best['equity_market'].iloc[0]
        rebase_factor_best = res_tsla_best['equity_voltarget'].iloc[0]
        rebase_factor_safe = res_tsla_safe['equity_voltarget'].iloc[0]
        
        res_tsla_best['equity_market_rebased'] = res_tsla_best['equity_market'] / rebase_factor_market
        res_tsla_best['equity_voltarget_rebased'] = res_tsla_best['equity_voltarget'] / rebase_factor_best
        res_tsla_safe['equity_voltarget_rebased'] = res_tsla_safe['equity_voltarget'] / rebase_factor_safe
        
        # Recompute drawdowns on rebased series
        res_tsla_best['dd_market_rebased'] = res_tsla_best['equity_market_rebased'] / res_tsla_best['equity_market_rebased'].cummax() - 1
        res_tsla_best['dd_voltarget_rebased'] = res_tsla_best['equity_voltarget_rebased'] / res_tsla_best['equity_voltarget_rebased'].cummax() - 1
        res_tsla_safe['dd_voltarget_rebased'] = res_tsla_safe['equity_voltarget_rebased'] / res_tsla_safe['equity_voltarget_rebased'].cummax() - 1
        
        # RECOMPUTE stats on the rebased/trimmed window
        ann_factor = 252
        def recompute_stats(eq_series, ret_series, w_exec_series, turnover_series):
            years = len(eq_series) / ann_factor
            final_eq = eq_series.iloc[-1]
            cagr = (final_eq ** (1/years)) - 1
            sharpe = ret_series.mean() / ret_series.std(ddof=1) * np.sqrt(ann_factor) if ret_series.std() > 0 else 0
            max_dd = (eq_series / eq_series.cummax() - 1).min()
            avg_exposure = w_exec_series.abs().mean()
            avg_turnover = turnover_series.mean()
            ann_turnover = avg_turnover * ann_factor
            return {
                'Final_Eq': final_eq,
                'CAGR': cagr,
                'Sharpe': sharpe,
                'MaxDD': max_dd,
                'Avg_Exposure': avg_exposure,
                'Avg_Turnover': avg_turnover,
                'Ann_Turnover': ann_turnover
            }
        
        # Recompute using rebased equity and trimmed returns
        stats_tsla_market = recompute_stats(
            res_tsla_best['equity_market_rebased'], res_tsla_best['ret_market'],
            pd.Series(1.0, index=res_tsla_best.index), pd.Series(0.0, index=res_tsla_best.index)
        )
        stats_tsla_best_rebased = recompute_stats(
            res_tsla_best['equity_voltarget_rebased'], res_tsla_best['ret_voltarget'],
            res_tsla_best['w_exec_voltarget'], res_tsla_best['turnover_voltarget']
        )
        stats_tsla_safe_rebased = recompute_stats(
            res_tsla_safe['equity_voltarget_rebased'], res_tsla_safe['ret_voltarget'],
            res_tsla_safe['w_exec_voltarget'], res_tsla_safe['turnover_voltarget']
        )
        
        # Debug output
        print(f"   TSLA Best: Avg_Exposure={stats_tsla_best_rebased['Avg_Exposure']:.2f}, Ann_Turnover={stats_tsla_best_rebased['Ann_Turnover']:.1f}")
        print(f"   TSLA Safe: Avg_Exposure={stats_tsla_safe_rebased['Avg_Exposure']:.2f}, Ann_Turnover={stats_tsla_safe_rebased['Ann_Turnover']:.1f}")
        
        # Merge for plotting (using REBASED series)
        res_tsla = res_tsla_best.copy()
        res_tsla['equity_market'] = res_tsla_best['equity_market_rebased']
        res_tsla['equity_voltarget_best'] = res_tsla_best['equity_voltarget_rebased']
        res_tsla['dd_voltarget_best'] = res_tsla_best['dd_voltarget_rebased']
        res_tsla['dd_market'] = res_tsla_best['dd_market_rebased']
        res_tsla['equity_voltarget_safe'] = res_tsla_safe['equity_voltarget_rebased']
        res_tsla['dd_voltarget_safe'] = res_tsla_safe['dd_voltarget_rebased']
        
        # Custom chart
        fig_tsla = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
            row_heights=[0.65, 0.35], subplot_titles=("Equity Curve", "Drawdown")
        )
        
        # Equity
        fig_tsla.add_trace(go.Scatter(x=res_tsla.index, y=res_tsla['equity_market'],
                                      name='Buy & Hold', line=dict(color='#888', width=2)), row=1, col=1)
        fig_tsla.add_trace(go.Scatter(x=res_tsla.index, y=res_tsla['equity_voltarget_best'],
                                      name=f'VolTarget (Best: L={int(best_sharpe["L"])})', 
                                      line=dict(color='#2ca02c', width=2.5)), row=1, col=1)
        fig_tsla.add_trace(go.Scatter(x=res_tsla.index, y=res_tsla['equity_voltarget_safe'],
                                      name=f'VolTarget (Risk-Adj: L={int(best_safe["L"])})', 
                                      line=dict(color='#ff7f0e', width=2)), row=1, col=1)
        
        # Drawdown
        fig_tsla.add_trace(go.Scatter(x=res_tsla.index, y=res_tsla['dd_market'],
                                      name='Buy & Hold', line=dict(color='#888', width=1.5), showlegend=False), row=2, col=1)
        fig_tsla.add_trace(go.Scatter(x=res_tsla.index, y=res_tsla['dd_voltarget_best'],
                                      name='VolTarget (Best)', line=dict(color='#2ca02c', width=1.5), showlegend=False), row=2, col=1)
        fig_tsla.add_trace(go.Scatter(x=res_tsla.index, y=res_tsla['dd_voltarget_safe'],
                                      name='VolTarget (Risk-Adj)', line=dict(color='#ff7f0e', width=1.5), showlegend=False), row=2, col=1)
        
        fig_tsla.update_yaxes(title_text="Equity", type="linear", row=1, col=1)
        fig_tsla.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)
        fig_tsla.update_xaxes(title_text="Date", row=2, col=1)
        fig_tsla.update_layout(
            height=800, width=1150, template="plotly_white",
            margin=dict(l=60, r=30, t=40, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        
        chart_tsla = pio.to_html(fig_tsla, full_html=False, include_plotlyjs=False, config=PLOTLY_CONFIG)
        
        # Table (using REBASED stats on common window)
        table_tsla_html = "<table class='metrics-table'><thead><tr>"
        table_tsla_html += f"<th>Metric</th><th>Buy & Hold</th><th>Best Sharpe (L={int(best_sharpe['L'])})</th><th>Risk-Adj (L={int(best_safe['L'])})</th>"
        table_tsla_html += "</tr></thead><tbody>"
        
        rows = [
            ['Final Equity ($1k)', f"${stats_tsla_market['Final_Eq']*1000:,.0f}",
             f"${stats_tsla_best_rebased['Final_Eq']*1000:,.0f}",
             f"${stats_tsla_safe_rebased['Final_Eq']*1000:,.0f}"],
            ['CAGR', f"{stats_tsla_market['CAGR']:.1%}",
             f"{stats_tsla_best_rebased['CAGR']:.1%}",
             f"{stats_tsla_safe_rebased['CAGR']:.1%}"],
            ['Sharpe', f"{stats_tsla_market['Sharpe']:.2f}",
             f"{stats_tsla_best_rebased['Sharpe']:.2f}",
             f"{stats_tsla_safe_rebased['Sharpe']:.2f}"],
            ['Max DD', f"{stats_tsla_market['MaxDD']:.1%}",
             f"{stats_tsla_best_rebased['MaxDD']:.1%}",
             f"{stats_tsla_safe_rebased['MaxDD']:.1%}"],
            ['Avg Exposure', f"{stats_tsla_market['Avg_Exposure']:.2f}",
             f"{stats_tsla_best_rebased['Avg_Exposure']:.2f}",
             f"{stats_tsla_safe_rebased['Avg_Exposure']:.2f}"],
            ['Turnover (daily / ann.)', "-",
             f"{stats_tsla_best_rebased['Avg_Turnover']:.4f} / {stats_tsla_best_rebased['Ann_Turnover']:.1f}",
             f"{stats_tsla_safe_rebased['Avg_Turnover']:.4f} / {stats_tsla_safe_rebased['Ann_Turnover']:.1f}"],
        ]
        for row in rows:
            table_tsla_html += "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
        table_tsla_html += "</tbody></table>"
        
        # Stats for subtitle (use rebased market sharpe)
        beats_market_sharpe = len(df_opt[df_opt['Sharpe'] > stats_tsla_market['Sharpe']])
        pct_beats = 100 * beats_market_sharpe / len(df_opt)
        
        subtitle_tsla = (f"TSLA exhibits strong parameter sensitivity; shorter lookbacks (L={int(best_sharpe['L'])}) "
                        f"tend to perform better in this sample. Only {pct_beats:.1f}% of {len(df_opt)} tested configurations "
                        f"exceed the market's Sharpe ratio. Improvements are associated with higher exposure and turnover trade-offs.")
        
    except FileNotFoundError:
        print("   Warning: TSLA_OPTIMIZATION_FULL.csv not found, using default params")
        res_tsla, stats_tsla = run_strategy(
            df_tsla, lookback_l=60, sigma_target=0.80, n_days_vol=120,
            w_max=2.0, fee_bps=1.0, is_crypto=False, rebalance_k=10
        )
        chart_tsla = create_equity_drawdown_chart(res_tsla, ['market', 'voltarget'], log_scale=False)
        table_tsla_html = create_metrics_table_6row(stats_tsla, ['market', 'voltarget'])
        subtitle_tsla = "Default configuration shown. Run TSLA optimization to see best parameters."
    
    html_tsla = build_html_page(
        title="Figure 3 — TSLA: Parameter Sensitivity Study (2160 Configurations)",
        subtitle=subtitle_tsla,
        chart_html=chart_tsla,
        table_html=table_tsla_html,
        footnote=FOOTNOTE,
        caption="High-volatility single stocks require careful calibration; results should not be generalized."
    )
    with open("fig3_tsla.html", "w", encoding="utf-8") as f:
        f.write(html_tsla)
    print("   -> fig3_tsla.html written")
    
    # PATCH 5: TSLA Consistency Check (using rebased stats)
    try:
        print("\n   Consistency Check: TSLA (rebased)")
        market_chart = res_tsla['equity_market'].iloc[-1]
        market_table = stats_tsla_market['Final_Eq']
        best_chart = res_tsla['equity_voltarget_best'].iloc[-1]
        best_table = stats_tsla_best_rebased['Final_Eq']
        safe_chart = res_tsla['equity_voltarget_safe'].iloc[-1]
        safe_table = stats_tsla_safe_rebased['Final_Eq']
        
        print(f"      market: Chart={market_chart:.4f}, Table={market_table:.4f} [{'OK' if abs(market_chart-market_table)<0.01 else 'MISMATCH'}]")
        print(f"      voltarget_best: Chart={best_chart:.4f}, Table={best_table:.4f} [{'OK' if abs(best_chart-best_table)<0.01 else 'MISMATCH'}]")
        print(f"      voltarget_safe: Chart={safe_chart:.4f}, Table={safe_table:.4f} [{'OK' if abs(safe_chart-safe_table)<0.01 else 'MISMATCH'}]")
    except:
        pass
    
    # =========================================================================
    # FIGURE 4: SCALING LAW
    # =========================================================================
    print("\n[5/5] Generating Figure 4 (Scaling Law)...")
    
    sweep_eth = run_sensitivity_sweep(df_eth, "ETH", vol_eth, True, 120, 170, 5.0)
    sweep_sol = run_sensitivity_sweep(df_sol, "SOL", vol_sol, True, 120, 120, 5.0)
    sweep_tsla = run_sensitivity_sweep(df_tsla, "TSLA", vol_tsla, False, 20, 90, 1.0)  # Using best L=20
    
    chart_scaling = create_scaling_chart({'ETH': sweep_eth, 'SOL': sweep_sol, 'TSLA': sweep_tsla})
    
    # Find optimal for each
    def find_optimal(sweep, nat_vol):
        best = sweep.loc[sweep['sharpe'].idxmax()]
        return {
            'nat_vol': nat_vol,
            'opt_target': best['target_vol'],
            'ratio': best['ratio'],
            'sharpe': best['sharpe'],
            'maxdd': best['maxdd']
        }
    
    scaling_results = {
        'ETH': find_optimal(sweep_eth, vol_eth),
        'SOL': find_optimal(sweep_sol, vol_sol),
        'TSLA': find_optimal(sweep_tsla, vol_tsla)
    }
    
    table_scaling = create_scaling_table(scaling_results)
    
    html_scaling = build_html_page(
        title="Figure 4 — Sensitivity: Sharpe vs Target Vol Ratio",
        subtitle="Optimal target volatility relative to natural volatility across asset classes.",
        chart_html=chart_scaling,
        table_html=table_scaling,
        footnote="Natural volatility is estimated on the full sample as std(returns) × sqrt(ann_factor).",
        caption="The 1.25x reference line is a visual aid based on observed clustering, not a universal claim."
    )
    with open("fig4_scaling.html", "w", encoding="utf-8") as f:
        f.write(html_scaling)
    print("   -> fig4_scaling.html written")
    
    print("\n" + "=" * 70)
    print("ALL FIGURES GENERATED SUCCESSFULLY")
    print("=" * 70)
    print("\nOutput files:")
    print("  - fig1_eth.html")
    print("  - fig2_spy.html")
    print("  - fig3_tsla.html")
    print("  - fig4_scaling.html")

if __name__ == "__main__":
    generate_thesis_figures()
