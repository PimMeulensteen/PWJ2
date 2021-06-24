from matching import StudentSchools, run_tests, Results
FAC = 9881 / 7978
RUNS = 5_000


def print_table(n=63, l=7672, f=None):
    if not f:
        f = FAC
    matching = StudentSchools(s=n, l=l, f=f)
    res_table = [[] for _ in range(13)]
    mcmf_table = [[] for _ in range(13)]
    r, q = run_tests(matching, RUNS)


    s = 0
    for i in range(13):
        pos = r.hist[i] if i < len(r.hist) else 0
        s = pos
        mcmf_table[i] = s


    for result in q:
        s = 0
        for i in range(13):
            pos = result.hist[i] if i < len(result.hist) else 0
            s = pos
            res_table[i].append(s)

    res_table = [(sum(i) / RUNS) for i in res_table]
    print(sum(i.sum_of_places for i in q) / RUNS, res_table)
    print(r.sum_of_places, mcmf_table)



print_table()
