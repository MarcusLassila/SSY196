import numpy as np
from math import comb, log2

def strategy(hats):
    guesses = []
    good_differences = set(range(2, len(hats), 4))
    for i in range(len(hats)):
        weight = sum(hats[:i] + hats[i + 1:])
        if abs(weight) in good_differences:
            guesses.append(1 if weight < 0 else -1)
        else:
            guesses.append(None)
    return guesses

def chance_strategy(hats):
    return [None] * (len(hats) - 1) + [1]

def simulate_game(n):
    hats = [-1 if np.random.random() < 0.5 else 1 for _ in range(n)]  # -1 represents red hat and 1 represents blue hat
    outcome = strategy(hats)
    assert len(outcome) == len(hats)
    score = 0
    for guess, hat in zip(outcome, hats):
        if guess == hat:
            score +=1
        elif guess is not None:
            score = 0
            break
    return score > 0

num_games = 100000
win_precentage = sum(simulate_game(15) for _ in range(num_games)) / num_games
print("Simulated win percentage:", win_precentage)

# 2-majority
# r b b  =>  6 correct
# b b b  =>  2 wrong

# 2-majority
# r r r b b b b  =>  70 correct
# r r b b b b b  =>  42 wrong
# r b b b b b b  =>  14 wrong
# b b b b b b b  =>   2 wrong

# 4-majority
# r r r b b b b  =>  20 correct avg (Last player random guesses if he sees 50/50 split. 40 such outcomes.)
# r r b b b b b  =>  42 correct
# r b b b b b b  =>  14 wrong
# b b b b b b b  =>   2 wrong

# 2 and 6-majority 
# r r r b b b b  =>  70 correct
# r r b b b b b  =>  42 wrong
# r b b b b b b  =>  14 correct
# b b b b b b b  =>   2 wrong

def best_win_percentage(n):
    assert log2(n + 1) % 1 == 0
    return sum(2 * comb(n, k) for k in range(n // 2, -1, -2)) / 2 ** n

print("best win percentage:", best_win_percentage(15))
