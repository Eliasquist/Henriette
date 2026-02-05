"""
TSLA Comprehensive Parameter Optimization
Tests 1000+ combinations to find optimal strategy configuration.
"""

import pandas as pd
import numpy as np
from itertools import product
from generate_thesis_figures import run_strategy, load_data_robust, calculate_natural_volatility

def run_tsla_optimization():
    print("=" * 70)
    print("TSLA COMPREHENSIVE PARAMETER OPTIMIZATION")
    print("=" * 70)
    
    # Load data
    print("\n[1/3] Loading TSLA data...")
    df_tsla = load_data_robust("data/TSLA_daily.csv")
    nat_vol = calculate_natural_volatility(df_tsla, False)
    print(f"   Natural Volatility: {nat_vol:.1%}")
    print(f"   Data Range: {df_tsla.index.min()} to {df_tsla.index.max()}")
    print(f"   Total Days: {len(df_tsla)}")
    
    # Parameter grid
    L_values = [20, 40, 60, 80, 100, 120, 150, 180, 252]  # 9 values
    sigma_values = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20]  # 10 values
    N_values = [30, 60, 90, 120, 150, 180]  # 6 values
    K_values = [5, 10, 15, 20]  # 4 values
    
    total_combos = len(L_values) * len(sigma_values) * len(N_values) * len(K_values)
    print(f"\n[2/3] Running {total_combos} parameter combinations...")
    print(f"   L (Lookback): {L_values}")
    print(f"   σ* (Target Vol): {sigma_values}")
    print(f"   N (Vol Window): {N_values}")
    print(f"   K (Rebalance): {K_values}")
    
    results = []
    tested = 0
    
    for L, sigma, N, K in product(L_values, sigma_values, N_values, K_values):
        try:
            _, stats = run_strategy(
                df_tsla, 
                lookback_l=L, 
                sigma_target=sigma, 
                n_days_vol=N,
                w_max=2.0, 
                fee_bps=1.0, 
                is_crypto=False, 
                rebalance_k=K
            )
            
            results.append({
                'L': L,
                'sigma': sigma,
                'N': N,
                'K': K,
                'Final_Eq': stats['voltarget']['Final_Eq'],
                'CAGR': stats['voltarget']['CAGR'],
                'Sharpe': stats['voltarget']['Sharpe'],
                'MaxDD': stats['voltarget']['MaxDD'],
                'Avg_Exposure': stats['voltarget']['Avg_Exposure'],
                'Turnover': stats['voltarget']['Avg_Turnover'],
                'Market_Eq': stats['market']['Final_Eq'],
                'Market_Sharpe': stats['market']['Sharpe']
            })
            
            tested += 1
            if tested % 100 == 0:
                print(f"   Tested {tested}/{total_combos} combinations...")
                
        except Exception as e:
            print(f"   Error at L={L}, σ={sigma}, N={N}, K={K}: {e}")
    
    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    print(f"\n[3/3] Analysis Complete. {len(df_results)} valid configurations.")
    
    # Save full results
    df_results.to_csv("TSLA_OPTIMIZATION_FULL.csv", index=False)
    print(f"\n   Full results saved to: TSLA_OPTIMIZATION_FULL.csv")
    
    # Top 20 by Sharpe
    print("\n" + "=" * 70)
    print("TOP 20 BY SHARPE RATIO")
    print("=" * 70)
    top_sharpe = df_results.nlargest(20, 'Sharpe')
    for i, row in top_sharpe.iterrows():
        print(f"   L={row['L']:3d}, σ={row['sigma']:.2f}, N={row['N']:3d}, K={row['K']:2d} | "
              f"Sharpe={row['Sharpe']:.2f}, CAGR={row['CAGR']:.1%}, MaxDD={row['MaxDD']:.1%}, Exposure={row['Avg_Exposure']:.2f}")
    
    # Top 20 by Final Equity
    print("\n" + "=" * 70)
    print("TOP 20 BY FINAL EQUITY")
    print("=" * 70)
    top_eq = df_results.nlargest(20, 'Final_Eq')
    for i, row in top_eq.iterrows():
        print(f"   L={row['L']:3d}, σ={row['sigma']:.2f}, N={row['N']:3d}, K={row['K']:2d} | "
              f"Final=${row['Final_Eq']*1000:,.0f}, CAGR={row['CAGR']:.1%}, Sharpe={row['Sharpe']:.2f}")
    
    # Best risk-adjusted (Sharpe > 0.5 AND MaxDD > -60%)
    print("\n" + "=" * 70)
    print("BEST RISK-ADJUSTED (Sharpe > 0.5 AND MaxDD > -60%)")
    print("=" * 70)
    filtered = df_results[(df_results['Sharpe'] > 0.5) & (df_results['MaxDD'] > -0.60)]
    if len(filtered) > 0:
        best_risk = filtered.nlargest(10, 'Sharpe')
        for i, row in best_risk.iterrows():
            print(f"   L={row['L']:3d}, σ={row['sigma']:.2f}, N={row['N']:3d}, K={row['K']:2d} | "
                  f"Sharpe={row['Sharpe']:.2f}, MaxDD={row['MaxDD']:.1%}, CAGR={row['CAGR']:.1%}")
    else:
        print("   No configurations meet criteria. Relaxing MaxDD to -70%...")
        filtered = df_results[(df_results['Sharpe'] > 0.4) & (df_results['MaxDD'] > -0.70)]
        best_risk = filtered.nlargest(10, 'Sharpe')
        for i, row in best_risk.iterrows():
            print(f"   L={row['L']:3d}, σ={row['sigma']:.2f}, N={row['N']:3d}, K={row['K']:2d} | "
                  f"Sharpe={row['Sharpe']:.2f}, MaxDD={row['MaxDD']:.1%}, CAGR={row['CAGR']:.1%}")
    
    # Parameter sensitivity summary
    print("\n" + "=" * 70)
    print("PARAMETER SENSITIVITY SUMMARY")
    print("=" * 70)
    
    print("\n   Average Sharpe by L (Lookback):")
    for L in L_values:
        avg = df_results[df_results['L'] == L]['Sharpe'].mean()
        print(f"      L={L:3d}: {avg:.3f}")
    
    print("\n   Average Sharpe by σ* (Target Vol):")
    for sigma in sigma_values:
        avg = df_results[df_results['sigma'] == sigma]['Sharpe'].mean()
        print(f"      σ={sigma:.2f}: {avg:.3f}")
    
    print("\n   Average Sharpe by K (Rebalance):")
    for K in K_values:
        avg = df_results[df_results['K'] == K]['Sharpe'].mean()
        print(f"      K={K:2d}: {avg:.3f}")
    
    # Market comparison
    print("\n" + "=" * 70)
    print("MARKET BENCHMARK COMPARISON")
    print("=" * 70)
    market_eq = df_results['Market_Eq'].iloc[0]
    market_sharpe = df_results['Market_Sharpe'].iloc[0]
    beats_market = len(df_results[df_results['Final_Eq'] > market_eq])
    better_sharpe = len(df_results[df_results['Sharpe'] > market_sharpe])
    
    print(f"   Market Final Equity: ${market_eq*1000:,.0f}")
    print(f"   Market Sharpe: {market_sharpe:.2f}")
    print(f"   Configs beating Market Equity: {beats_market}/{len(df_results)} ({100*beats_market/len(df_results):.1f}%)")
    print(f"   Configs with better Sharpe: {better_sharpe}/{len(df_results)} ({100*better_sharpe/len(df_results):.1f}%)")
    
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    
    return df_results

if __name__ == "__main__":
    run_tsla_optimization()
