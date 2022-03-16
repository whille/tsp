#!/usr/bin/env python
# Solve a traveling salesman problem on a randomly generated set of
# points using lazy constraints.   The base MIP model only includes
# 'degree-2' constraints, requiring each node to have exactly
# two incident edges.  Solutions to this model may contain subtours -
# tours that don't visit every city.  The lazy constraint callback
# adds new constraints to cut them off.


import math
import random
import tsp_cb
import tsp_mtz
import tabu_search


def rand_distance(n):
    random.seed(1)
    # 2D position of n nodes
    points = [(random.randint(0, 100), random.randint(0, 100)) for i in range(n)]
    # Dictionary of Euclidean distance between each pair of points
    # triangle matrix, avoid i == j
    dist = {(i, j):
            math.sqrt(sum((points[i][k]-points[j][k])**2 for k in range(2)))
            for i in range(n) for j in range(i)}
    return dist


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1])
    dist = rand_distance(n)
    tsp_cb.calc(n, dist)
    # mtz is 10X slower than callback, tested: n = 40
    tsp_mtz.calc(n, dist)
    tabu_search.calc_tsp(n, dist)
