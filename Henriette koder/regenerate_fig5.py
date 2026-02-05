
import json
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd

# Load existing results
with open('global_opt_v4/best_global_config.json', 'r') as f:
    data = json.load(f)

best = data['best']
holdout_results = data['holdout']
ASSETS = ['ETH', 'SOL', 'TSLA', 'SPY']
HOLDOUT_START_DATE = '2023-01-01'
PLOTLY_CONFIG = {'displayModeBar': False, 'responsive': True, 'displaylogo': False}

# Style
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
    @media print { .card { box-shadow: none; } }
</style>
"""

# Plot 1: Delta Sharpe (Search)
delta_sharpes = [best.get(f'{a}_Delta_Sharpe_avg', 0) for a in ASSETS]
colors = ['#2ca02c' if d > 0 else '#d62728' for d in delta_sharpes]

fig1 = go.Figure()
fig1.add_trace(go.Bar(
    x=ASSETS, y=delta_sharpes, marker_color=colors,
    text=[f'{d:+.3f}' for d in delta_sharpes], textposition='outside'
))
fig1.update_layout(title='Search Phase ΔSharpe (Pre-2023)', template='plotly_white', height=380)

# Plot 2: Holdout
holdout_deltas = [holdout_results.get(a, {}).get('Delta_Sharpe', 0) for a in ASSETS]
hol_colors = ['#2ca02c' if d > 0 else '#d62728' for d in holdout_deltas]

fig2 = go.Figure()
fig2.add_trace(go.Bar(
    x=ASSETS, y=holdout_deltas, marker_color=hol_colors,
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
for asset in ASSETS:
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

# Enhanced Holdout Table
hol_table = """
<div class="holdout-box">
<h3 style="margin-top:0">Final Calendar Holdout (2023-01-01 to End)</h3>
<table class='metrics-table'>
    <thead><tr><th>Asset</th><th>Sharpe(VT)</th><th>Sharpe(Mkt)</th><th>ΔSharpe</th><th>Avg Exp</th><th>Turnover</th><th>MaxDD</th><th>Rank</th></tr></thead>
    <tbody>"""
beats = 0
for asset in ASSETS:
    hr = holdout_results.get(asset, {})
    if 'error' in hr: b_str = "ERR"
    else:
        delta = hr.get('Delta_Sharpe', 0)
        if delta > 0: beats += 1
        b_str = "PASS" if delta > 0 else "FAIL"
        color = 'positive' if delta > 0 else 'negative'
        exp = hr.get('AvgExposure', 0)
        turn = hr.get('AnnTurnover', 0)
        hol_table += f"<tr><td>{asset}</td><td>{hr.get('Sharpe_VT',0):.3f}</td><td>{hr.get('Sharpe_Mkt',0):.3f}</td>"
        hol_table += f"<td class='{color}'>{delta:+.3f}</td><td>{exp:.2f}x</td><td>{turn:.1f}x</td><td>{hr.get('MaxDD_VT',0):.1%}</td><td>{b_str}</td></tr>"
hol_table += "</tbody></table></div>"

subtitle = f"Optimization on Pre-2023 data [ETH:2020+, TSLA:2010+, SPY:2000+]. Holdout: 2023-Present.<br>"
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
        <strong>Methodology (Thesis-Safe):</strong> Random search (N=25,000) on <strong>Pre-2023 Data</strong>.
        Walk-Forward Validation (3 disjoint folds). Final Test on <strong>Common Calendar Holdout (2023-Present)</strong>.
        Strict Selection Caps (Crypto ≤1.50, Equities ≤1.10).
    </div>
    <div style="display: flex; gap: 20px; flex-wrap: wrap;">
        <div>{chart1_html}</div><div>{chart2_html}</div>
    </div>
    <h2>Best Configuration (Selected on 2020-2022 Data)</h2>
    {params_table}
    <h2>Search Phase Results (Walk-Forward Pre-2023)</h2>
    {wf_table}
    {hol_table}
    <p class="footnote">Signal=0 implies 0% exposure. Costs applied. Benchmark frictionless. Exp caps applied during selection; holdout allows regime drift up to 2.0x.</p>
</div></body></html>"""

with open('fig5_global_v4.html', 'w', encoding='utf-8') as f: f.write(html)
print("Regenerated fig5_global_v4.html with Exposure columns.")
