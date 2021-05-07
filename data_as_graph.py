import random


def graph_from_data():
    pass


N_L = 8000
N_S = 63
FAC = 1.3


SOURCE = 0
SINK = N_L + N_S + 1

cost = list(range(1, 13))
s_cap = [int((N_L * FAC) / N_S)] * N_S


print(f"{N_L + N_S + 2} {N_L * 12 + N_L + N_S} {SOURCE} {SINK}")

# Source naar leerling
for i in range(1, N_L + 1):
    print(f"{SOURCE} {i} 1 0")

# Leerling naar school
for i in range(1, N_L + 1):
    n_scls_chosen = min(12, N_S)
    scls_i = random.sample(range(N_L + 1, N_L + N_S + 1), n_scls_chosen)
    for j in range(n_scls_chosen):
        print(f"{i} {scls_i[j]} 1 {cost[j]}")


# School naar sink
for i in range(1 + N_L, N_S + N_L + 1):
    print(f"{i} {SINK} {s_cap[i- N_L - 1]} 0")
