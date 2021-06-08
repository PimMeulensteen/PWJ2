import random
from data_as_graph import *
from da_stb import da_stb
import os


FILENAME = "t.txt"
if __name__ == "__main__":
    scl, st = gen_data(15, 800, 1.1)
    with open(FILENAME, "w") as fd:
        fd.writelines(graph_from_data(scl, st))
    print(da_stb(scl, st))
    os.system(f"python3 mincostmaxflow.py < {FILENAME}")
