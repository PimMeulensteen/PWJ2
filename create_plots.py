from os.path import exists
from os import remove
import multiprocessing as mp
import pickle
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

from matching import StudentSchools, run_tests

n_schools = 74
n_height = 20
hor_axis = np.linspace(1, 1.5, n_height)
FAC = 9881 / 7978
RUNS = 5_000


def run_one(x: Tuple[int, int]) -> float:
    """Run one instance of a simulation. The parmeter is a tuple of the number
    of schools 'n' and the amount of avalible spots per student 'i'. Return the
    mean of 'RUNS' executions. """
    n, i = x
    f = hor_axis[i]
    matching = StudentSchools(s=n, l=n * 42, f=f)
    _, r = run_tests(matching, RUNS)
    return float(np.mean(r))


def run_school(n: int) -> List[float]:
    """ Run the simulation with 'n' number of schools. Returns a list of median
   results for each entry in 'hor_axis' """
    res = [0.0] * n_height
    file_name = f"data/{n}_{len(hor_axis)}_{RUNS}.p"
    if exists(file_name):
        try:
            res = pickle.load(open(file_name, "rb"))
            return res
        except EOFError:
            remove(file_name)
            print(f"error reading {file_name}")

    # Call the run_one function in parallel.
    p = mp.Pool(n_height // 2)
    res = p.map(run_one, zip([n] * n_height, range(n_height)))

    # Save the result to prevent double calculation.
    pickle.dump(res, open(file_name, "wb"))

    return res


def get_measurements():
    data = np.random.random(size=(n_height, n_schools))

    for n in range(1, n_schools + 1):
        res = run_school(n)

        for pos, _ in enumerate(hor_axis):
            data[n_height - 1 - pos][n - 1] = res[pos]

    return data


def vis_data():
    """ Create 2D-plot with n-schools vs spots per student. Takes 10ths of
    hours to run, but results are cached. """
    data = get_measurements()
    fig, (ax1) = plt.subplots(figsize=(13, 8), ncols=1)

    # Use bicubic interpolation to smooth the picture.
    p = ax1.imshow(
        data,
        interpolation="bicubic",
        extent=[0.5, n_schools + 0.5, 1, 1.5],
        aspect="auto",
    )
    ax1.set_ylabel("Aantal beschikbare plekken per leerling")
    ax1.set_xlabel("Aantal scholen wat meedoet aan de matching")
    ax1.set_title(
        f"Het relatief verschil in de som van de plekken tussen DA-STB en het door ons beschreven algoritme op gegenereerde data."
    )
    fig.colorbar(p, ax=ax1)
    plt.show()


def make_data_and_save(n=63, l=7672, f=None):
    if not f:
        f = FAC
    matching = StudentSchools(s=n, l=l, f=f)
    pickle.dump(run_tests(matching, RUNS), open("dump.txt", "wb"))


def load_data_and_make_hist(data=0):
    file = open("dump.txt", "rb")
    r, data = pickle.load(file)
    plt.hist(data, bins=len(set(data)))
    plt.xlim(7755, max(data) + 1)
    plt.ylabel("Frequentie")
    plt.xlabel("Som van plekken waar leerlingen zijn geplaatst")
    plt.title(
        "Histogram van resultaten van DA-STB op 7978 leerlingen, 9881 plekken en 63 scholen."
    )
    print(np.mean(data), r)
    plt.plot([r, r], [0, 1000])
    plt.show()
