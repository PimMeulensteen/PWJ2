import copy
from random import shuffle, sample, seed
from typing import Callable, List, Mapping
from mincostmaxflow import min_cost_cc, min_cost
WeightFunc = Callable[[int], int]


class Results():
    sum_of_places: int
    hist: List[int]
    res: List[int]

    def __init__(self, placement) -> None:
        self.res = placement
        self.sum_of_places = sum(placement)

    def bin(self):
        """ Create the histogram of the results and store it in self.hist."""
        self.hist = [0] * (max(self.res) + 1)
        for r in self.res:
            self.hist[r] += 1

    def __repr__(self) -> str:
        return f"Results(count: {len(self.res)}, sum: {self.sum_of_places}, hist: {self.hist[1:]})"

    def __gt__(self, other):
        return self.sum_of_places > other.sum_of_places


class StudentSchools():
    len_prio: int
    schools: List[int]
    students: List[List[int]]

    def __init__(self, schools=[], students=[], s=6, l=230, f=1.05) -> None:
        if not schools or not students:
            self.gen_data(s, l, f)
        else:
            self.schools = schools
            self.students = students
            self.len_prio = min(12, len(self.schools))

    def as_graph_str(self, weight: WeightFunc = lambda x: x) -> str:
        """ Return a string format of the graph correspoding to this Matching.

        The first line is a line with four non-negative integers, n,
        m, 0≤s≤n−1 and 0≤t≤n−1, separated by single spaces, where
        n is the numbers of nodes in the graph, m is the number of edges, s
        is the source and t is the sink (s≠t).

        Then follow m lines, each line consisting of four (space-separated)
        integers u, v, c and w indicating that there is an edge from u to v in
        the graph with capacity c and cost w.  """
        s = 0
        n_std = len(self.students)
        n_scl = len(self.schools)
        t = n_scl + n_std + 1

        res = f"{n_std + n_scl + 2} {n_std * self.len_prio + n_std + n_scl} \
                {s} {t}\n"

        # Source to student
        for i in range(1, n_std + 1):
            res += f"{s} {i} 1 0\n"

        # Student to school
        for i, std in enumerate(self.students):
            for spot, j in enumerate(std):
                res += f"{i + 1} {j + n_std + 1} 1 {weight(spot + 1)}\n"

        # School to sink
        for i in range(1 + n_std, n_scl + n_std + 1):
            res += f"{i} {t} {self.schools [i- n_std - 1]} 0\n"

        return res

    def as_graph(self, weight: WeightFunc = lambda x: x):
        """ Return a source, sink, list of capacities, list of costs and an
        adjency list for the graph corresponding to this matching. """

        # Add two nodes for the source and sink
        n = len(self.schools) + len(self.students) + 2

        cap = [[0 for _ in range(n)] for _ in range(n)]
        cost = [[0 for _ in range(n)] for _ in range(n)]
        adj: List[List[int]] = [[] for _ in range(n)]

        source = 0
        sink = n - 1

        # Source to student
        for i in range(1, len(self.students) + 1):
            cap[source][i] = 1
            adj[source].append(i)
            adj[i].append(source)

        # Student to school
        for i, std in enumerate(self.students):
            for spot, j in enumerate(std):
                idx = j + len(self.students) + 1
                cap[i + 1][idx] = 1
                cost[i + 1][idx] = weight(spot + 1)
                cost[idx][i + 1] = -weight(spot + 1)
                adj[i+1].append(idx)
                adj[idx].append(i + 1)

        # School to sink
        for i in range(1 + len(self.students), n - 1):
            cap[i][sink] = self.schools[i - len(self.students) - 1]
            cost[i][sink] = 0
            adj[i].append(sink)
            adj[sink].append(i)

        return source, sink, adj, cost, cap

    def gen_data(self,
                 n_schools: int = 63,
                 n_students: int = 8000,
                 n_fac: float = 1.1) -> None:
        self.len_prio = min(12, n_schools)
        school_cap = [int((n_students * n_fac) / n_schools)] * n_schools
        delta = 25
        school_cap = [j - delta if i % 2 else j +
                      delta for i, j in enumerate(school_cap)]
        student_choice: List[List[int]] = [[] for _ in range(n_students)]

        for i in range(n_students):
            student_choice[i] = sample(
                range(n_schools), self.len_prio)

        self.schools = school_cap
        self.students = student_choice

    def da_stb(self) -> Results:
        used = [0] * len(self.schools)
        placement = []
        for student in sample(self.students, len(self.students)):
            placed = False
            for rank, school_at_rank in enumerate(student):
                if used[school_at_rank] < self.schools[school_at_rank]:
                    placement.append(rank + 1)
                    used[school_at_rank] += 1
                    placed = True
                    break
            if not placed:
                placement.append(13)
        return Results(placement)

    def mincostmaxflow(self, mode='c') -> Results:
        s, t, adj, cost, cap1 = self.as_graph()
        cap = copy.deepcopy(cap1)
        if mode == 'c':
            _, _, cap2 = min_cost_cc(s, t, adj, cost, cap1)
        else:
            _, _, cap2 = min_cost(s, t, adj, cost, cap1)

        placement = []

        for i, std in enumerate(self.students):

            placed = False
            for spot, school_at_spot in enumerate(std):
                idx = school_at_spot + len(self.students) + 1
                max_cap = cap[i + 1][idx]
                rest_cap = cap2[i + 1][idx]
                flow = max_cap - rest_cap

                if flow == 1:
                    placed = True
                    placement.append(spot + 1)
                    break

            if not placed:
                placement.append(13)

        return Results(placement)

    def run_all_methods(self) -> List[Results]:
        return [self.da_stb(), self.mincostmaxflow()]


def run_tests(mtc, n=100):
    def f(x):
        return (mtc.da_stb().sum_of_places - min_cost) / min_cost
    min_cost = mtc.mincostmaxflow().sum_of_places
    return min_cost, list(map(f, range(n)))
