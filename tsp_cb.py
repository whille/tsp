#!/usr/bin/env python3
from itertools import combinations
import gurobipy as gp
from gurobipy import GRB


# Callback - use lazy constraints to eliminate sub-tours
def subtourelim(model, where):
    if where == GRB.Callback.MIPSOL:
        vals = model.cbGetSolution(model._vars)
        # find the shortest cycle in the selected edge list
        tour = subtour(vals, model._n)
        if len(tour) < model._n:
            # add subtour elimination constr. for every pair of cities in tour
            model.cbLazy(gp.quicksum(model._vars[i, j]
                         for i, j in combinations(tour, 2)) <= len(tour)-1)


# Given a tuplelist of edges, find the shortest subtour
# find under degree-2 constrain
def subtour(vals, n):
    # make a list of edges selected in the solution
    edges = gp.tuplelist((i, j) for i, j in vals.keys()
                         if vals[i, j] > 0.5)
    unvisited = list(range(n))
    cycle = range(n+1)  # initial length has 1 more city
    while unvisited:  # true if list is non-empty
        thiscycle = []
        neighbors = unvisited
        while neighbors:
            current = neighbors[0]
            thiscycle.append(current)
            unvisited.remove(current)
            neighbors = [j for i, j in edges.select(current, '*')
                         if j in unvisited]
        if len(cycle) > len(thiscycle):
            cycle = thiscycle
    print(f"cycle: {cycle}")
    return cycle


def calc(n, dist):
    m = gp.Model()
    vars = m.addVars(dist.keys(), obj=dist, vtype=GRB.BINARY, name='e')
    for i, j in vars.keys():
        # restrict symmetric tsp
        vars[j, i] = vars[i, j]  # edge in opposite direction
    # Add degree-2 constraint, for each node, 2 edges exists
    m.addConstrs(vars.sum(i, '*') == 2 for i in range(n))
    m._vars = vars
    m._n = n
    m.Params.LazyConstraints = 1
    m.optimize(subtourelim)
    show(m, n)


def show(m, n):
    vals = m.getAttr('X', m._vars)
    tour = subtour(vals, n)
    assert len(tour) == n

    print('')
    print('Optimal tour: %s' % str(tour))
    print('Optimal cost: %g' % m.ObjVal)
    print('')
