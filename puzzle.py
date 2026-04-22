from itertools import permutations

runners = ['Red', 'Blue', 'Green', 'Yellow', 'White']

for perm in permutations(range(1, 6)):
    pos = dict(zip(runners, perm))
    if (pos['Blue'] == pos['Red'] + 1 and
        pos['Green'] == 5 and
        pos['White'] == 1 and
        pos['Yellow'] != 5 and
        pos['Red'] != 1 and
        pos['Yellow'] != 2):
        print(f"Solution found: {pos}")
        print(f"Red's position: {pos['Red']}")
