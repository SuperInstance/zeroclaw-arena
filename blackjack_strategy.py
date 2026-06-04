"""
Blackjack Breakthrough — Can ZeroClaws discover basic strategy?

Current state: 26.2% overall win rate, 47% best script.
Target: >40% overall by injecting strategic patterns.

Strategy:
1. Create "seeded" scripts that implement basic strategy rules
2. Run the arena with these seeds mixed in
3. Measure if strategic scripts outcompete random ones
"""

import random
import json
import os
import numpy as np
from collections import defaultdict

# Basic blackjack strategy rules (simplified, no doubling/splitting)
def basic_strategy_action(hand_value, dealer_up, is_soft):
    """Basic strategy for hit/stand decisions."""
    if is_soft:
        # Soft hand (has an ace counted as 11)
        if hand_value >= 19:
            return 'stand'
        elif hand_value == 18:
            return 'stand' if dealer_up in [7, 8] else 'hit'
        else:
            return 'hit'
    else:
        # Hard hand
        if hand_value >= 17:
            return 'stand'
        elif hand_value >= 13:
            return 'stand' if dealer_up in [2, 3, 4, 5, 6] else 'hit'
        elif hand_value == 12:
            return 'stand' if dealer_up in [4, 5, 6] else 'hit'
        else:
            return 'hit'


def generate_strategy_scripts(n=50):
    """Generate scripts that encode basic strategy as state→action rules."""
    scripts = []
    for i in range(n):
        rules = []
        for hand in range(4, 22):
            for dealer in range(1, 11):
                for soft in [True, False]:
                    action = basic_strategy_action(hand, dealer, soft)
                    # Encode as: "hand={hand},dealer={dealer},soft={soft}" -> action
                    state_key = f"h{hand}d{dealer}{'s' if soft else 'h'}"
                    rules.append((state_key, action))
        
        # Add some noise to make each script slightly different
        noise_rate = 0.05 * (i / n)  # 0-5% deviation from basic strategy
        noisy_rules = []
        for state, action in rules:
            if random.random() < noise_rate:
                noisy_rules.append((state, 'hit' if action == 'stand' else 'stand'))
            else:
                noisy_rules.append((state, action))
        
        # Encode as a script string
        script_lines = [f"{s}:{a}" for s, a in noisy_rules]
        script_str = "\n".join(script_lines)
        scripts.append(script_str)
    
    return scripts


def evaluate_script(script_str, n_games=500):
    """Evaluate a blackjack script by playing games."""
    # Parse rules
    rules = {}
    for line in script_str.strip().split('\n'):
        if ':' in line:
            state, action = line.split(':')
            rules[state.strip()] = action.strip()
    
    wins = 0
    losses = 0
    pushes = 0
    
    for _ in range(n_games):
        # Simple blackjack simulation
        deck = list(range(1, 11)) * 4  # Simplified deck (1-10, 4 suits)
        random.shuffle(deck)
        
        # Deal
        idx = 0
        player_hand = [deck[idx], deck[idx+1]]
        dealer_hand = [deck[idx+2], deck[idx+3]]
        idx += 4
        
        dealer_up = dealer_hand[0]
        
        # Player turn
        while True:
            # Calculate hand value
            value = sum(player_hand)
            aces = player_hand.count(1)
            soft = False
            if aces > 0 and value + 10 <= 21:
                value += 10
                soft = True
            
            if value >= 21:
                break
            
            # Look up strategy
            state_key = f"h{value}d{dealer_up}{'s' if soft else 'h'}"
            action = rules.get(state_key, 'hit')  # Default to hit
            
            if action == 'stand':
                break
            else:
                player_hand.append(deck[idx])
                idx += 1
        
        player_value = sum(player_hand)
        aces = player_hand.count(1)
        if aces > 0 and player_value + 10 <= 21:
            player_value += 10
        
        if player_value > 21:
            losses += 1
            continue
        
        # Dealer turn (stands on 17+)
        while True:
            dealer_value = sum(dealer_hand)
            aces = dealer_hand.count(1)
            if aces > 0 and dealer_value + 10 <= 21:
                dealer_value += 10
            
            if dealer_value >= 17:
                break
            dealer_hand.append(deck[idx])
            idx += 1
        
        dealer_value = sum(dealer_hand)
        aces = dealer_hand.count(1)
        if aces > 0 and dealer_value + 10 <= 21:
            dealer_value += 10
        
        if dealer_value > 21:
            wins += 1
        elif player_value > dealer_value:
            wins += 1
        elif player_value == dealer_value:
            pushes += 1
        else:
            losses += 1
    
    win_rate = wins / n_games
    return win_rate, wins, losses, pushes


def run_breakthrough():
    print("=" * 60)
    print("BLACKJACK BREAKTHROUGH EXPERIMENT")
    print("=" * 60)
    
    # Generate strategic scripts
    print("\n1. Generating 50 basic-strategy scripts...")
    scripts = generate_strategy_scripts(50)
    
    # Evaluate them
    print("\n2. Evaluating scripts (500 games each)...")
    results = []
    for i, script in enumerate(scripts):
        wr, w, l, p = evaluate_script(script, 500)
        results.append({"id": i, "win_rate": wr, "wins": w, "losses": l, "pushes": p})
        if i % 10 == 0:
            print(f"   Script {i}: {wr:.1%} ({w}W/{l}L/{p}P)")
    
    # Summary
    win_rates = [r['win_rate'] for r in results]
    best = max(results, key=lambda x: x['win_rate'])
    avg = np.mean(win_rates)
    
    print(f"\n3. Results Summary:")
    print(f"   Average win rate: {avg:.1%}")
    print(f"   Best win rate: {best['win_rate']:.1%} (script {best['id']})")
    print(f"   Worst win rate: {min(win_rates):.1%}")
    print(f"   Std dev: {np.std(win_rates):.1%}")
    
    # Compare with random baseline
    print(f"\n4. Random Baseline (50 random scripts):")
    random_wrs = []
    for i in range(50):
        # Random strategy
        random_rules = {}
        for hand in range(4, 22):
            for dealer in range(1, 11):
                for soft in [True, False]:
                    state_key = f"h{hand}d{dealer}{'s' if soft else 'h'}"
                    random_rules[state_key] = random.choice(['hit', 'stand'])
        script_str = "\n".join(f"{k}:{v}" for k, v in random_rules.items())
        wr, _, _, _ = evaluate_script(script_str, 200)
        random_wrs.append(wr)
    
    avg_random = np.mean(random_wrs)
    print(f"   Random average: {avg_random:.1%}")
    print(f"   Strategy advantage: {(avg - avg_random)*100:+.1f}pp")
    
    # Save results
    output = {
        "strategy_avg": float(avg),
        "strategy_best": float(best['win_rate']),
        "random_avg": float(avg_random),
        "advantage_pp": float((avg - avg_random) * 100),
        "n_scripts": 50,
        "n_games_per_script": 500,
    }
    
    out_path = os.path.expanduser("~/repos/zeroclaw-arena/blackjack-results.json")
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
    
    # Verdict
    if avg > 0.40:
        print("\n✅ BREAKTHROUGH: Strategy scripts achieve >40% win rate!")
    elif avg > 0.35:
        print("\n⚠️ PROGRESS: Strategy improves but house edge is real")
    else:
        print("\n❌ No significant improvement")


if __name__ == "__main__":
    run_breakthrough()
