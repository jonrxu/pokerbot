import modal
from modal_deploy.config import checkpoint_volume, image
from evaluation.comprehensive_eval import eval_app, evaluate_checkpoints_modal

if __name__ == "__main__":
    print("="*80)
    print("CHECKPOINT COMPARISON: Phase 3 vs Old Phase 2 (Iter 999)")
    print("="*80)
    
    # List of iterations to test against old best model (999)
    test_iters = [0, 5, 10, 19]
    old_iter = 999
    
    results = {}
    
    with eval_app.run():
        for iter1 in test_iters:
            name = f"Iter {iter1} vs Old Best (Iter {old_iter})"
            print(f"\n{name}")
            print("-" * 80)
            try:
                result = evaluate_checkpoints_modal.remote(iter1, old_iter, 2000)
                results[f"{iter1}_vs_{old_iter}"] = result
                
                win_rate = result['iteration1_win_rate']
                payoff = result['iteration1_avg_payoff']
                
                print(f"  Iter {iter1} Win Rate: {win_rate:.2%}")
                print(f"  Iter {iter1} Avg Payoff: {payoff:.2f} chips/game")
                print(f"  Old Best Win Rate: {result['iteration2_win_rate']:.2%}")
                
                if win_rate > 0.55:
                    print(f"  ✓ Iter {iter1} is significantly better (>55% win rate)")
                elif win_rate > 0.50:
                    print(f"  ⚠ Iter {iter1} is slightly better (>50% win rate)")
                elif win_rate < 0.45:
                    print(f"  ✗ Iter {iter1} is WORSE (<45% win rate)")
                else:
                    print(f"  ≈ Results are roughly even (45-55% win rate)")
                    
            except Exception as e:
                print(f"  ✗ Error: {e}")
    
    print("\n" + "="*80)
    print("SUMMARY: New Run vs Old Best")
    print("="*80)
    print(f"{'Matchup':<30} {'New Win Rate':<20} {'New Payoff':<20}")
    print("-" * 80)
    
    for iter1 in test_iters:
        key = f"{iter1}_vs_{old_iter}"
        if key in results:
            res = results[key]
            wr = res['iteration1_win_rate']
            pay = res['iteration1_avg_payoff']
            print(f"{key:<30} {wr:<20.2%} {pay:<20.2f}")
        else:
            print(f"{key:<30} {'N/A':<20} {'N/A':<20}")
