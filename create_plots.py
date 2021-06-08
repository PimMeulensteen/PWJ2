from os.path import exists
from os import remove
import multiprocessing as mp
import pickle
import numpy as np
import matplotlib.pyplot as plt

from matching import Results, StudentSchools, run_tests
n_schools = 75
n_height = 20
hor_axis = np.linspace(1, 1.5, n_height)
RUNS = 40


def run_school(n):
    res = []
    file_name = f"data/{n}_{len(hor_axis)}_{RUNS}.p"
    if exists(file_name):
        try:
            res = pickle.load(open(file_name, "rb"))
        except EOFError:
            remove(file_name)
            print(f"error reading {file_name}")

    if res:
        return res

    for f in hor_axis:
        matching = StudentSchools(s=n, l=n*42, f=f)
        _, r = run_tests(matching, RUNS)
        res.append(np.mean(r))
    pickle.dump(res, open(file_name, "wb"))
    return res


def vis_data():
    data = np.random.random(size=(n_height, n_schools))
    p = mp.Pool(10)
    p.map(run_school, range(1, n_schools+1))

    for n in range(1, n_schools+1):
        res = run_school(n)
        for pos, _ in enumerate(hor_axis):
            data[n_height - 1 - pos][n-1] = res[pos]

    fig, (ax1) = plt.subplots(figsize=(13, 8), ncols=1)
    p = ax1.imshow(data, interpolation='bicubic',
                   extent=[.5, n_schools+.5, 1, 1.5], aspect='auto')
    ax1.set_ylabel("Aantal beschikbare plekken per leerling")
    ax1.set_xlabel("Aantal scholen wat meedoet aan de matching")
    ax1.set_title(
        f"Het relatief verschil in de som van de plekken tussen DA-STB en het door ons beschreven algoritme op gegenereerde data. ({RUNS} runs per datapunt)")
    fig.colorbar(p, ax=ax1)
    plt.show()


if __name__ == "__main__":
    vis_data()
