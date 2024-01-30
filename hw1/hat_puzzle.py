import numpy as np
from math import comb, log2

def strategy(hats):
    guesses = []
    ref = len(hats) // 2
    for i in range(len(hats)):
        weight = sum(hats[:i] + hats[i + 1:])
        if abs(weight - ref) < np.log2(len(hats) + 1) - 1:
            guesses.append(None)
        elif weight > ref:
            guesses.append(0)
        else:
            guesses.append(1)
    if all(guess is None for guess in guesses):
        guesses[-1] = 1
    return guesses

def chance_strategy(hats):
    return [None] * (len(hats) - 1) + [1]

def simulate_game(n = 3):
    hats = np.random.randint(0, 2, n).tolist()
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

# num_games = 1000000
# win_precentage = sum(simulate_game(7) for _ in range(num_games)) / num_games
# print(win_precentage)

# 2-majority
# r b b  =>  6 correct
# b b b  =>  2 wrong

# 2-majority
# r r r b b b b  =>  70 correct
# r r b b b b b  =>  42 wrong
# r b b b b b b  =>  14 wrong
# b b b b b b b  =>   2 wrong

# 4-majority
# r r r b b b b  =>  35 correct avg
# r r b b b b b  =>  42 correct
# r b b b b b b  =>  14 wrong
# b b b b b b b  =>   2 wrong

# 6-majority
# r r r b b b b  =>  35 correct avg
# r r b b b b b  =>  21 correct avg
# r b b b b b b  =>  14 correct
# b b b b b b b  =>   2 wrong

def find_best(n):
    assert log2(n + 1) % 1 == 0
    scores = []
    for b in range(n // 2, 0, -1):
        score = 0
        for k in range(n // 2, -1, -1):
            c = comb(n, k)
            if k > b:
                score += c
            elif k == b:
                score += 2 * c
        score /= 2 ** n
        scores.append(score)
    return max(scores)

print("best win percentage:", find_best(15))
