import csv
import collections
import sys
from collections import deque
from typing import Tuple, DefaultDict, Dict, List

# "sys.stdin.readline" is faster than the default input.
input = sys.stdin.readline
INF = 10**18
NOT_SEEN = -1


def edmonds_karp(cap: List[List[int]],
                 adj: List[List[int]],
                 s: int,
                 t: int) -> int:
    """ Edmonds-Karp max-flow algorithm for ***integer*** capacity.

    `cap` stores the capacity for pairs of vertices.
    s and t are the source and sink, respectively.
    """
    n = len(adj)
    flow = 0

    def bfs() -> Tuple[int, List[int]]:
        """ Breadth first search to find path from "s" to "t". Return the flow
        to this node as well as a list of parents of the path to get to "t" """
        parents = [NOT_SEEN for _ in range(n)]
        # We store a queue of (vertex, bottleneck_flow) pairs. There is not
        # bottleneck flow to 's', thus we set it to infinity.
        Q = deque([(s, INF)])
        while Q:
            cur, cur_flow = Q.popleft()
            for nbr in adj[cur]:
                if parents[nbr] == NOT_SEEN and cap[cur][nbr] != 0:
                    parents[nbr] = cur
                    new_flow = min(cur_flow, cap[cur][nbr])
                    if nbr == t:
                        return new_flow, parents
                    Q.append((nbr, new_flow))

        return 0, parents

    while True:
        # BFS to find a shortest augmenting path.
        new_flow, parents = bfs()

        # If no new flow is found, break.
        if new_flow == 0:
            break
        flow += new_flow

        # Walk back from this augmenting path to source using the list of
        # parents. Update the capacities along the way.
        cur = t
        while cur != s:
            prev = parents[cur]
            cap[prev][cur] -= new_flow
            cap[cur][prev] += new_flow
            cur = prev

    return flow


def shortest_paths(adj: List[List[int]],
                   cap: List[List[int]],
                   cost: List[List[int]],
                   s: int) -> Tuple[List[int], List[int]]:
    """ Shortest-Path-Faster-Algorithm """
    n = len(cap)
    dist = [INF] * n
    dist[s] = 0
    parent = [NOT_SEEN] * n

    # List to store information if a node is in the queue. Indexing (O(1)) is
    # faster than "in Q" (which is O(n)).
    in_queue = [False] * n
    in_queue[s] = True

    Q = deque([s])
    while Q:
        cur = Q.popleft()
        in_queue[cur] = False

        for nbr in adj[cur]:
            if cap[cur][nbr] <= 0:
                continue

            weight = cost[cur][nbr]
            # If it is faster to get to nbr using cur, update the distance and
            # partent of nbr.
            if dist[nbr] > dist[cur] + weight:
                dist[nbr] = dist[cur] + weight
                parent[nbr] = cur
                if not in_queue[nbr]:
                    in_queue[nbr] = True
                    Q.append(nbr)

    return dist, parent


def min_cost(s: int,
             t: int,
             desired_flow: int,
             adj: List[List[int]],
             cost_ar: List[List[int]],
             cap: List[List[int]]) -> Tuple[int, int]:
    flow = 0
    cost = 0
    while flow < desired_flow:
        # Find shortest path
        dist, prnt = shortest_paths(adj, cap, cost_ar, s)
        # If there is not path to t, we are done.
        if dist[t] == INF:
            break

        # find max flow on that path
        f = desired_flow - flow
        cur = t
        while cur != s:
            f = min(f, cap[prnt[cur]][cur])
            cur = prnt[cur]

        # Apply the flow
        flow += f
        cost += f * dist[t]
        cur = t
        while cur != s:
            cap[prnt[cur]][cur] -= f
            cap[cur][prnt[cur]] += f
            cur = prnt[cur]
        if DEBUG:
            print(flow, cost)
    return flow, cost


def main():
    """ The first line of input contains a line with four
    non-negative integers (n, m, s, t) separated by single spaces, where 'n'
    is the numbers of nodes in the graph, 'm' is the number of edges, 's' is
    the source and 't' is the sink (s != t). Nodes are numbered from 0 to n−1.
    Then follow 'm' lines, each line consisting of four (space-separated)
    integers 'u', 'v', 'c' and 'w' indicating that there is an edge from 'u'
    to 'v' in the graph with capacity 'c' and cost 'w'. """
    n, m, s, t = list(map(int, input().split()))

    cap = [[0 for _ in range(n)] for _ in range(n)]
    cap2 = [[0 for _ in range(n)] for _ in range(n)]
    cost = [[0 for _ in range(n)] for _ in range(n)]
    adj: List[List[int]] = [[] for _ in range(n)]

    if DEBUG:
        print("Created data arrays")
    for _ in range(m):
        u, v, c, w = list(map(int, input().split()))
        cap[u][v] = c
        cap2[u][v] = c
        cost[u][v] = w
        cost[v][u] = -w
        adj[u].append(v)
        adj[v].append(u)

    if DEBUG:
        print("Done reading from input")
    flow = edmonds_karp(cap, adj, s, t)
    if DEBUG:
        print("Found max flow")
    res = min_cost(s, t, flow, adj, cost, cap2)
    if DEBUG:
        print("Found min cost")
    print(*res)


DEBUG = 1
if __name__ == "__main__":
    main()
