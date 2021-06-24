import collections
import sys
from collections import deque, defaultdict
from typing import Tuple, DefaultDict, Dict, List

input = sys.stdin.readline
INF = 10 ** 18


def edmonds_karp(cap, adj, s, t):
    """ Edmonds-Karp max-flow algorithm for ***integer*** capacity.

    `cap` stores the capacity for pairs of vertices.
    s and t are the source and sink, respectively.
    """

    flow = 0

    def bfs():
        parents = defaultdict(lambda: -1)
        # We store a queue of (vertex, bottleneck_flow) pairs.
        Q = deque()
        Q.append((s, INF))
        while len(Q):
            cur, cur_flow = Q.popleft()
            for nbr in adj[cur]:
                if parents[nbr] == -1 and cap[(cur, nbr)] != 0:
                    parents[nbr] = cur
                    new_flow = min(cur_flow, cap[(cur, nbr)])
                    if nbr == t:
                        return new_flow, parents
                    Q.append((nbr, new_flow))
        return 0, parents

    while True:
        # BFS to find a shortest augmenting path.
        new_flow, parents = bfs()

        # If none was found, break.
        if new_flow == 0:
            break
        flow += new_flow

        # Walk back from this augmenting path to source, update capacities.
        cur = t
        while cur != s:
            prev = parents[cur]
            cap[(prev, cur)] -= new_flow
            cap[(cur, prev)] += new_flow
            cur = prev

    return flow


def shortest_paths(adj, cap, cost, v0):
    """ Bellman-Ford algorithm to find a shortest path. Returns a dictionary of
    distatnce """
    n = len(cap)

    dist = defaultdict(lambda: INF)
    parent = defaultdict(lambda: -1)
    dist[v0] = 0

    m = defaultdict(lambda: 2)
    q = collections.deque([v0])

    while q:
        u = q.popleft()
        m[u] = 0
        for v in adj[u]:
            if cap[(u, v)] > 0 and dist[v] > dist[u] + cost[(u, v)]:
                dist[v] = dist[u] + cost[(u, v)]
                parent[v] = u
                if m[v] == 2:
                    m[v] = 1
                    q.append(v)
                elif m[v] == 0:
                    m[v] = 1
                    q.appendleft(v)

    return dist, parent


def min_cost(
    s: int,
    t: int,
    desired_flow: int,
    adj: List[List[int]],
    cost_ar: List[int],
    cap: List[List[int]],
) -> int:
    """  """
    flow, cost = 0, 0
    while flow < desired_flow:
        sp_dist, sp_cost = shortest_paths(adj, cap, cost_ar, s)
        if sp_dist[t] == INF:
            break

        f = desired_flow - flow
        cur = t
        while cur != s:
            f = min(f, cap[(sp_cost[cur], cur)])
            cur = sp_cost[cur]

        flow += f
        cost += f * sp_dist[t]
        cur = t
        while cur != s:
            cap[(sp_cost[cur], cur)] -= f
            cap[(cur, sp_cost[cur])] += f
            cur = sp_cost[cur]

    return cost


def main():
    _, m, s, t = list(map(int, input().split()))
    cost = defaultdict(int)
    cap = defaultdict(int)
    cap2 = defaultdict(int)
    adj = defaultdict(list)

    for _ in range(m):
        u, v, c, w = list(map(int, input().split()))
        cap[(u, v)] = c
        cap2[(u, v)] = c
        cost[(u, v)] = w
        cost[(v, u)] = -w
        adj[u].append(v)
        adj[v].append(u)

    flow = edmonds_karp(cap, adj, s, t)
    if DEBUG:
        print("Found max flow")
    m_cost = min_cost(s, t, flow, adj, cost, cap2)
    print(flow, m_cost)


DEBUG = True

if __name__ == "__main__":
    main()
