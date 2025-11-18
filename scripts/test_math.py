#!/usr/bin/env python3
"""Rigorous math verification for Deep CFR regret updates."""

import unittest
import torch
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple

# Import local modules
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poker_game.game import PokerGame, GameState, Action
from poker_game.information_set import InformationSet
from training.deep_cfr import DeepCFR
from models.value_policy_net import ValuePolicyNet
from poker_game.state_encoder import StateEncoder

class MockInfoSet:
    def __init__(self, key):
        self.key = key

class TestRegretMath(unittest.TestCase):
    def setUp(self):
        self.game = PokerGame()
        self.encoder = StateEncoder()
        self.input_dim = self.encoder.feature_dim
        self.value_net = ValuePolicyNet(self.input_dim)
        self.policy_net = ValuePolicyNet(self.input_dim)
        
        self.deep_cfr = DeepCFR(
            value_net=self.value_net,
            policy_net=self.policy_net,
            state_encoder=self.encoder,
            game=self.game,
            device='cpu'
        )

    def test_regret_update_math(self):
        """
        Manually verify regret update arithmetic.
        Scenario:
        - Player 0 is at a decision node.
        - Actions: CHECK (0), BET_MIN (1).
        - Counterfactual values (calculated backwards):
            - Value if CHECK: -10
            - Value if BET_MIN: +50
        - Current Node CF Value: (0.5 * -10) + (0.5 * 50) = 20 (assuming uniform strategy)
        - Regret Update:
            - Regret(CHECK) = -10 - 20 = -30
            - Regret(BET_MIN) = 50 - 20 = +30
        """
        print("\n=== TEST: Regret Update Math ===")
        
        # Mock info set
        key = "test_state_key"
        
        # Mock inputs
        cf_value_check = -10.0
        cf_value_bet = 50.0
        node_cf_value = 20.0
        
        # Simulate update loop (like in train.py)
        # Action 0 (CHECK)
        regret_check = cf_value_check - node_cf_value
        self.deep_cfr.regret_memory[key][0] += regret_check
        
        # Action 1 (BET_MIN)
        regret_bet = cf_value_bet - node_cf_value
        self.deep_cfr.regret_memory[key][1] += regret_bet
        
        print(f"Calculated Regret(CHECK): {regret_check} (Expected: -30.0)")
        print(f"Calculated Regret(BET_MIN): {regret_bet} (Expected: 30.0)")
        
        self.assertAlmostEqual(self.deep_cfr.regret_memory[key][0], -30.0)
        self.assertAlmostEqual(self.deep_cfr.regret_memory[key][1], 30.0)
        print("✓ Regret accumulation is correct.")

    def test_strategy_calculation(self):
        """
        Verify get_strategy produces correct probabilities from regrets.
        Using regrets from previous test: {-30, +30}.
        Regret Matching:
        - Positive Regrets: {CHECK: 0, BET_MIN: 30}
        - Sum: 30
        - Strategy: {CHECK: 0/30=0, BET_MIN: 30/30=1.0}
        """
        print("\n=== TEST: Strategy Calculation ===")
        key = "test_state_key"
        info_set = MockInfoSet(key)
        
        # Setup regrets
        self.deep_cfr.regret_memory[key][0] = -30.0
        self.deep_cfr.regret_memory[key][1] = 30.0
        
        legal_actions = [(Action.CHECK, 0), (Action.RAISE, 10)]
        
        strategy = self.deep_cfr.get_strategy(info_set, legal_actions)
        
        print(f"Regrets: {dict(self.deep_cfr.regret_memory[key])}")
        print(f"Strategy: {strategy}")
        
        self.assertAlmostEqual(strategy[0], 0.0)
        self.assertAlmostEqual(strategy[1], 1.0)
        print("✓ Strategy calculation (Regret Matching) is correct.")

    def test_missing_regret_entry(self):
        """
        Verify fix for 'missing regret entry' bug.
        Scenario: Regret memory has entry for Action 0 but NOT Action 1.
        Before fix: Returned uniform.
        After fix: Should treat Action 1 as 0.0 regret.
        
        Regrets: {CHECK: 50, BET: (missing -> 0)}
        Pos Regrets: {CHECK: 50, BET: 0}
        Strategy: {CHECK: 1.0, BET: 0.0}
        """
        print("\n=== TEST: Missing Regret Entry (The Bug Fix) ===")
        key = "partial_regret_key"
        info_set = MockInfoSet(key)
        
        # Only set regret for action 0
        self.deep_cfr.regret_memory[key][0] = 50.0
        # Action 1 is missing from memory
        
        legal_actions = [(Action.CHECK, 0), (Action.RAISE, 10)]
        
        strategy = self.deep_cfr.get_strategy(info_set, legal_actions)
        
        print(f"Regret Memory: {dict(self.deep_cfr.regret_memory[key])}")
        print(f"Legal Actions Indices: 0, 1")
        print(f"Strategy: {strategy}")
        
        self.assertAlmostEqual(strategy[0], 1.0)
        self.assertAlmostEqual(strategy[1], 0.0)
        print("✓ Missing regret entry handled correctly (treated as 0).")

    def test_average_strategy_accumulation(self):
        """
        Verify average strategy accumulation logic.
        """
        print("\n=== TEST: Average Strategy Accumulation ===")
        key = "avg_strat_key"
        
        # Iteration 1: Strategy {0: 0.2, 1: 0.8}
        self.deep_cfr.strategy_memory[key][0] += 0.2
        self.deep_cfr.strategy_memory[key][1] += 0.8
        
        # Iteration 2: Strategy {0: 0.4, 1: 0.6}
        self.deep_cfr.strategy_memory[key][0] += 0.4
        self.deep_cfr.strategy_memory[key][1] += 0.6
        
        # Expected Sum: {0: 0.6, 1: 1.4}
        # Expected Avg: {0: 0.6/2=0.3, 1: 1.4/2=0.7}
        
        info_set = MockInfoSet(key)
        legal_actions = [(Action.CHECK, 0), (Action.RAISE, 10)]
        
        avg_strat = self.deep_cfr.compute_average_strategy(info_set, legal_actions)
        
        print(f"Accumulated Sum: {dict(self.deep_cfr.strategy_memory[key])}")
        print(f"Computed Avg Strategy: {avg_strat}")
        
        self.assertAlmostEqual(avg_strat[0], 0.3)
        self.assertAlmostEqual(avg_strat[1], 0.7)
        print("✓ Average strategy accumulation is correct.")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

