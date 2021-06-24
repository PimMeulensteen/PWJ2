import copy
import sys
from collections import deque
from typing import Tuple, DefaultDict, Dict, List

# "sys.stdin.readline" is faster than the default input.
input = sys.stdin.readline
INF = 10 ** 18
NOT_SEEN = -1


def edmonds_karp(cap: List[List[int]], adj: List[List[int]], s: int, t: int) -> int:
    """ Edmonds-Karp max-flow algorithm for ***integer*** capacity.

    `cap` stores the capacity for pairs of vertices.
    adj stores adjacent vertices for every vertex.
    s and t are the source and sink, respectively.
    """
    # number of vertices in the graph
    n = len(adj)
    flow = 0

    def bfs() -> Tuple[int, List[int]]:
        """ Breadth first search to find path from "s" to "t". Return the flow
        to this node as well as a list of parents of the path to get to "t" """
        parents = [NOT_SEEN for _ in range(n)]
        # We store a queue of (vertex, bottleneck_flow) pairs. There is no
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


def cheapest_path(
    adj: List[List[int]], cap: List[List[int]], cost: List[List[int]], s: int
) -> Tuple[List[int], List[int]]:
    """ Shortest-Path-Faster-Algorithm """
    # TODO redundant variable
    n = len(cap)
    # TODO ouwie
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


def bellman_ford(adj: List[List[int]], cap: List[List[int]], cost: List[List[int]], s: int) -> List[int]:
    """  Bellman-Ford for detecting negative cycles for use in Cycle-
    Cancelling alg. if no negative cycles, return None
 """
    n = len(adj)
    dist = [INF] * n
    parent = [-1] * n
    dist[s] = 0

    # Repeat |V| - 1 times.
    for _ in range(1, len(cap)):
        for node, nbs in enumerate(adj):
            for nb in nbs:
                weight = cost[node][nb]
                if dist[node] + weight < dist[nb] and cap[node][nb] > 0:
                    dist[nb] = dist[node] + weight
                    parent[nb] = node

    # Check for negative cycle.
    C = -1
    for node, nbs in enumerate(adj):
        for nb in nbs:
            weight = cost[node][nb]
            if dist[node] != INF and dist[node] + weight < dist[nb]:
                # Store one of the vertex of
                # the negative weight cycle
                C = node
                break

    if C == -1:
        return False
    else:
        for _ in range(n):
            C = parent[C]

        # To store the cycle vertex
        cycle = []
        v = C

        while True:
            cycle.append(v)
            if v == C and len(cycle) > 1:
                break
            v = parent[v]

        # Reverse cycle[]
        cycle.reverse()
        return cycle


def min_cost_cc(
    s: int,
    t: int,
    adj: List[List[int]],
    cost_ar: List[List[int]],
    cap: List[List[int]]
) -> Tuple[int, int, List[List[int]]]:
    cap2 = copy.deepcopy(cap)

    # Create a feasable flow.
    maxflow = edmonds_karp(cap2, adj, s, t)

    # Find the minimal cost flow.
    cycle = bellman_ford(adj, cost_ar, cap2, s)
    while cycle:
        if -1 in cycle:
            break

        # smallest capacity of the cycle.
        flow = min(cap2[u][v] for u, v in zip(cycle, cycle[1:]))

        for u, v in zip(cycle, cycle[1:]):
            cap2[u][v] -= flow
            cap2[v][u] += flow

        cycle = bellman_ford(adj, cost_ar, cap2, s)

    # Calculate the actual cost of the flow.
    tot_cost = 0
    for source, targets in enumerate(zip(cap, cap2)):
        for target, (old_cap, new_cap) in enumerate(zip(*targets)):
            flw = old_cap - new_cap
            tot_cost += flw * cost_ar[source][target] if flw > 0 else 0

    return maxflow, tot_cost, cap2


def min_cost(
    s: int,
    t: int,
    adj: List[List[int]],
    cost_ar: List[List[int]],
    capp: List[List[int]],
    desired_flow: int = 0,
) -> Tuple[int, int, List[List[int]]]:
    if not desired_flow:
        cap2 = copy.deepcopy(capp)
        desired_flow = edmonds_karp(cap2, adj, s, t)
    flow = 0
    cost = 0
    while flow < desired_flow:
        # Find shortest path
        dist, prnt = cheapest_path(adj, capp, cost_ar, s)
        # If there is not path to t, we are done.
        if dist[t] == INF:
            break

        # find max flow on that path
        f = desired_flow - flow
        cur = t
        while cur != s:
            f = min(f, capp[prnt[cur]][cur])
            cur = prnt[cur]

        # Apply the flow
        flow += f
        cost += f * dist[t]
        cur = t
        while cur != s:
            capp[prnt[cur]][cur] -= f
            capp[cur][prnt[cur]] += f
            cur = prnt[cur]

    return flow, cost, capp
