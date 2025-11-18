#!/usr/bin/env python3
"""Diagnostic script to check for numerical issues and investigate value loss."""

import json
import sys
import numpy as np
from modal_deploy.check_results import app, check_training_results


def diagnose_training():
    """Run comprehensive diagnostics on training metrics."""
    
    print('='*80)
    print('TRAINING DIAGNOSTICS')
    print('='*80)
    print()
    
    with app.run():
        result = check_training_results.remote()
        
        if not result.get('all_entries'):
            print('❌ No metrics found. Is training running?')
            return
        
        metrics = result['all_entries']
        full_scale = [m for m in metrics if m.get('trajectories_generated', 0) >= 10000]
        
        if len(full_scale) == 0:
            print('❌ No full-scale iterations found')
            return
        
        # Get unique iterations (some iterations may have multiple metric entries)
        unique_iterations = sorted(set([m.get('iteration', 0) for m in full_scale]))
        
        print(f'✓ Found {len(full_scale)} full-scale metric entries')
        print(f'✓ Unique iterations: {len(unique_iterations)}')
        print(f'✓ Latest iteration: {max(unique_iterations)}')
        print()
        
        # Use only the latest entry per iteration to avoid duplicates
        latest_per_iter = {}
        for m in full_scale:
            it = m.get('iteration', 0)
            if it not in latest_per_iter or m.get('timestamp', '') > latest_per_iter[it].get('timestamp', ''):
                latest_per_iter[it] = m
        
        # Extract metrics from unique iterations
        sorted_iters = sorted(latest_per_iter.keys())
        iterations = sorted_iters
        value_losses = [latest_per_iter[it].get('value_loss', 0) for it in sorted_iters]
        policy_losses = [latest_per_iter[it].get('policy_loss', 0) for it in sorted_iters]
        
        # 1. Check for NaN/Inf values
        print('='*80)
        print('1. NUMERICAL STABILITY CHECK')
        print('='*80)
        
        nan_inf_value = sum(1 for v in value_losses if not np.isfinite(v))
        nan_inf_policy = sum(1 for v in policy_losses if not np.isfinite(v))
        
        if nan_inf_value > 0:
            print(f'⚠ WARNING: {nan_inf_value} value losses are NaN/Inf')
            nan_indices = [i for i, v in enumerate(value_losses) if not np.isfinite(v)]
            print(f'   Iterations with NaN/Inf value loss: {[iterations[i] for i in nan_indices[:10]]}')
        else:
            print('✓ All value losses are finite')
        
        if nan_inf_policy > 0:
            print(f'⚠ WARNING: {nan_inf_policy} policy losses are NaN/Inf')
            nan_indices = [i for i, v in enumerate(policy_losses) if not np.isfinite(v)]
            print(f'   Iterations with NaN/Inf policy loss: {[iterations[i] for i in nan_indices[:10]]}')
        else:
            print('✓ All policy losses are finite')
        
        print()
        
        # 2. Value loss analysis
        print('='*80)
        print('2. VALUE LOSS ANALYSIS')
        print('='*80)
        
        # Filter out NaN/Inf
        valid_value_losses = [v for v in value_losses if np.isfinite(v)]
        valid_iterations = [iterations[i] for i, v in enumerate(value_losses) if np.isfinite(v)]
        
        if len(valid_value_losses) == 0:
            print('❌ No valid value losses found')
            return
        
        print(f'First iteration:  {valid_value_losses[0]:.2f}')
        print(f'Last iteration:   {valid_value_losses[-1]:.2f}')
        print(f'Min value loss:    {min(valid_value_losses):.2f}')
        print(f'Max value loss:    {max(valid_value_losses):.2f}')
        print(f'Mean value loss:   {np.mean(valid_value_losses):.2f}')
        print(f'Median value loss: {np.median(valid_value_losses):.2f}')
        print()
        
        # Check growth rate
        if len(valid_value_losses) > 10:
            early_avg = np.mean(valid_value_losses[:10])
            late_avg = np.mean(valid_value_losses[-10:])
            growth_factor = late_avg / early_avg if early_avg > 0 else float('inf')
            
            print(f'Early avg (first 10):  {early_avg:.2f}')
            print(f'Late avg (last 10):    {late_avg:.2f}')
            print(f'Growth factor:         {growth_factor:.2f}x')
            
            if growth_factor > 1000:
                print('⚠ CRITICAL: Value loss has grown by >1000x')
                print('   This suggests numerical instability or unbounded values')
            elif growth_factor > 100:
                print('⚠ WARNING: Value loss has grown significantly')
                print('   Monitor closely - may indicate issues')
            else:
                print('✓ Value loss growth is within reasonable bounds')
        
        print()
        
        # Check for exponential growth
        if len(valid_value_losses) > 20:
            recent = valid_value_losses[-20:]
            # Check if recent values are growing exponentially
            log_values = [np.log(v) if v > 0 else 0 for v in recent]
            if len([v for v in log_values if v > 0]) > 10:
                # Fit linear trend to log values
                x = np.arange(len(log_values))
                valid_log = [(x[i], log_values[i]) for i in range(len(log_values)) if log_values[i] > 0]
                if len(valid_log) > 5:
                    x_vals = [v[0] for v in valid_log]
                    y_vals = [v[1] for v in valid_log]
                    slope = np.polyfit(x_vals, y_vals, 1)[0]
                    
                    print(f'Recent log-scale slope: {slope:.4f}')
                    if slope > 0.1:
                        print('⚠ WARNING: Exponential growth detected in value loss')
                        print('   This suggests values are growing unbounded')
                    elif slope > 0.05:
                        print('⚠ CAUTION: Rapid growth in value loss')
                    else:
                        print('✓ Growth rate is manageable')
        
        print()
        
        # 3. Policy loss analysis
        print('='*80)
        print('3. POLICY LOSS ANALYSIS')
        print('='*80)
        
        valid_policy_losses = [v for v in policy_losses if np.isfinite(v)]
        
        print(f'First iteration:  {valid_policy_losses[0]:.4f}')
        print(f'Last iteration:   {valid_policy_losses[-1]:.4f}')
        print(f'Min policy loss:  {min(valid_policy_losses):.4f}')
        print(f'Max policy loss:  {max(valid_policy_losses):.4f}')
        print(f'Mean policy loss: {np.mean(valid_policy_losses):.4f}')
        print()
        
        improvement = ((valid_policy_losses[0] - valid_policy_losses[-1]) / valid_policy_losses[0] * 100) if valid_policy_losses[0] > 0 else 0
        print(f'Overall improvement: {improvement:.1f}%')
        
        if valid_policy_losses[-1] < 0.1:
            print('✓ Policy loss is excellent (< 0.1)')
        elif valid_policy_losses[-1] < 0.2:
            print('✓ Policy loss is good (< 0.2)')
        elif valid_policy_losses[-1] < 0.3:
            print('⚠ Policy loss is moderate (0.2-0.3)')
        else:
            print('⚠ Policy loss is high (> 0.3)')
        
        print()
        
        # 4. Correlation analysis
        print('='*80)
        print('4. VALUE vs POLICY LOSS CORRELATION')
        print('='*80)
        
        if len(valid_value_losses) == len(valid_policy_losses):
            # Check if high value loss correlates with policy degradation
            correlation = np.corrcoef(valid_value_losses, valid_policy_losses)[0, 1]
            print(f'Correlation coefficient: {correlation:.4f}')
            
            if correlation > 0.5:
                print('⚠ WARNING: Strong positive correlation')
                print('   High value loss may be affecting policy learning')
            elif correlation > 0.3:
                print('⚠ CAUTION: Moderate positive correlation')
            elif correlation < -0.3:
                print('✓ Negative correlation - value loss increase may not affect policy')
            else:
                print('✓ Weak correlation - value loss and policy loss are independent')
            
            # Check recent trend
            if len(valid_value_losses) >= 20:
                recent_val = valid_value_losses[-10:]
                recent_pol = valid_policy_losses[-10:]
                recent_corr = np.corrcoef(recent_val, recent_pol)[0, 1]
                print(f'Recent correlation (last 10): {recent_corr:.4f}')
        
        print()
        
        # 5. Recommendations
        print('='*80)
        print('5. RECOMMENDATIONS')
        print('='*80)
        
        recommendations = []
        
        if max(valid_value_losses) > 1e6:
            recommendations.append('CRITICAL: Value loss exceeds 1M - investigate immediately')
            recommendations.append('  → Check counterfactual value computation')
            recommendations.append('  → Verify value network predictions are bounded')
            recommendations.append('  → Consider value clipping or normalization')
        
        if growth_factor > 100 if 'growth_factor' in locals() else False:
            recommendations.append('WARNING: Value loss growing rapidly')
            recommendations.append('  → Check for unbounded counterfactual values')
            recommendations.append('  → Consider reducing learning rate for value network')
            recommendations.append('  → Add gradient clipping (already implemented ✓)')
        
        if valid_policy_losses[-1] < 0.2:
            recommendations.append('✓ Policy loss is improving well - continue training')
        
        if correlation > 0.5 if 'correlation' in locals() else False:
            recommendations.append('WARNING: Value loss may be affecting policy')
            recommendations.append('  → Monitor policy loss closely')
            recommendations.append('  → Consider fixing value loss issues')
        
        if not recommendations:
            recommendations.append('✓ No critical issues detected')
            recommendations.append('  → Continue monitoring')
            recommendations.append('  → Policy loss improvement is the key metric')
        
        for rec in recommendations:
            print(rec)
        
        print()
        print('='*80)
        print('SUMMARY')
        print('='*80)
        print(f'Total iterations: {len(full_scale)}')
        print(f'Policy loss: {valid_policy_losses[0]:.4f} → {valid_policy_losses[-1]:.4f} ({improvement:.1f}% improvement)')
        print(f'Value loss: {valid_value_losses[0]:.2f} → {valid_value_losses[-1]:.2f}')
        
        if valid_policy_losses[-1] < valid_policy_losses[0]:
            print('✓ Training is progressing - policy loss improving')
        else:
            print('⚠ Policy loss not improving - investigate')
        
        if max(valid_value_losses) > 1e6:
            print('⚠ Value loss is extremely high - investigate numerical stability')
        elif max(valid_value_losses) > 1e4:
            print('⚠ Value loss is high - monitor closely')
        else:
            print('✓ Value loss is within reasonable range')


if __name__ == '__main__':
    diagnose_training()

