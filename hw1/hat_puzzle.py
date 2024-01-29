import numpy as np

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

num_games = 1000000
win_precentage = sum(simulate_game(3) for _ in range(num_games)) / num_games
print(win_precentage)
