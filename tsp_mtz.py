#!/usr/bin/env python
from tsp_cb import show, GRB, gp


def mtz(model, n, x):
    """mtz: Miller-Tucker-Zemlin's model for the (asymmetric) traveling salesman problem
    (potential formulation)
    Parameters:
        - n: number of nodes
        - x[i,j]: edge(i,j)
    Returns a model, ready to be solved.
    """
    u = model.addVars(range(n), lb=0, ub=n-1, vtype=GRB.INTEGER, name='u')
    for i in range(n):
        for j in range(1, n):
            if i != j:
                model.addConstr(u[i] - u[j] + n * x[i, j] <= n - 1, "MTZ(%s,%s)" % (i, j))


def show_iip(m, fname):
    if m.Status == GRB.INFEASIBLE:
        m.computeIIS()
        m.write(fname)


def calc(n, dist):
    for i, j in list(dist.keys()):
        dist[j, i] = dist[i, j]  # edge in opposite direction
    m = gp.Model()
    # Callback - use lazy constraints to eliminate sub-tours
    vars = m.addVars(dist.keys(), obj=dist, vtype=GRB.BINARY, name="e")
    # asymmetric tsp
    m.addConstrs(vars.sum(i, '*') == 1 for i in range(n))
    m.addConstrs(vars.sum('*', i) == 1 for i in range(n))
    m._vars = vars
    m._n = n
    mtz(m, m._n, m._vars)
    m.optimize()
    show(m, n)
