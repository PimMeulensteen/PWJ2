import csv
import collections
import sys
from collections import deque, defaultdict
from typing import Tuple, DefaultDict, Dict, List

input = sys.stdin.readline
INF = 10 ** 18


def edmonds_karp(cap: List[List[int]], adj: List[List[int]], s: int, t: int) -> int:
    """ Edmonds-Karp max-flow algorithm for ***integer*** capacity.

    `cap` stores the capacity for pairs of vertices.
    s and t are the source and sink, respectively.
    """

    flow = 0

    def bfs():
        INF = 10 ** 9
        parents = [-1 for _ in range(len(adj))]
        # We store a queue of (vertex, bottleneck_flow) pairs.
        Q = deque()
        Q.append((s, INF))
        while len(Q):
            cur, cur_flow = Q.popleft()
            for nbr in adj[cur]:
                if parents[nbr] == -1 and cap[cur][nbr] != 0:
                    parents[nbr] = cur
                    new_flow = min(cur_flow, cap[cur][nbr])
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
            cap[prev][cur] -= new_flow
            cap[cur][prev] += new_flow
            cur = prev

    return flow


NOT_SEEN = 2
IN_QUEUE = 1
TRAVERSED = 0


def shortest_paths(adj: List[List[int]], cap: List[List[int]], cost: List[List[int]], v0: int) -> Tuple[List[int], List[int]]:
    """ Shortest-Path-Faster-Algorithm """
    n = len(cap)
    dist = [INF] * n
    dist[v0] = 0
    parent = [-1] * n
    in_queue = [False] * n
    q = collections.deque([v0])
    in_queue[v0] = True
    while q:
        cur = q.popleft()
        in_queue[cur] = False
        for nb in adj[cur]:
            if cap[cur][nb] <= 0:
                continue

            weight = cost[cur][nb]
            if dist[nb] > dist[cur] + weight:
                dist[nb] = dist[cur] + weight
                parent[nb] = cur
                if not in_queue[nb]:
                    in_queue[nb] = True
                    q.append(nb)

    return dist, parent


def min_cost(s, t, desired_flow, adj, cost_ar, cap):
    flow, cost = 0, 0
    d, p = [], []
    while flow < desired_flow:
        d, p = shortest_paths(adj, cap, cost_ar, s)
        if d[t] == INF:
            break

        f = desired_flow - flow
        cur = t
        while cur != s:
            f = min(f, cap[p[cur]][cur])
            cur = p[cur]

        flow += f
        cost += f * d[t]
        cur = t
        while cur != s:
            cap[p[cur]][cur] -= f
            cap[cur][p[cur]] += f
            cur = p[cur]

    return flow, cost


def main():
    n, m, s, t = list(map(int, input().split()))

    cap = [[0 for _ in range(n)] for _ in range(n)]
    cap2 = [[0 for _ in range(n)] for _ in range(n)]
    cost = [[0 for _ in range(n)] for _ in range(n)]

    adj: List[List[int]] = [[] for _ in range(n)]
    for _ in range(m):
        u, v, c, w = list(map(int, input().split()))
        cap[u][v] = c
        cap2[u][v] = c
        cost[u][v] = w
        adj[u].append(v)
        adj[v].append(u)

    flow = edmonds_karp(cap, adj, s, t)
    m_cost = min_cost(s, t, flow, adj, cost, cap2)
    print(*m_cost)


if __name__ == "__main__":
    main()
