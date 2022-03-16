# ref: https://towardsdatascience.com/optimization-techniques-tabu-search-36f197ef8e25
import math
import random
import time
from itertools import combinations


class TabuSearch:
    def __init__(self, evaluate_fn, neighbor_op, n):
        self.evaluate = evaluate_fn
        self.neighbor_op = neighbor_op
        self.n = n
        self.init_params()

    def init_params(self):
        # page 103, Knox[212]
        # tabu size:
        self.tenure = 3 * self.n
        self.max_tries = 4
        self.max_iter = int(0.0003 * self.n ** 4)
        print(f"max_iter: {self.max_iter}")

    def run(self, init_fn):
        # TODO multi-process
        best = (math.inf, None)
        for _ in range(self.max_tries):
            best_v, best_sol = self._run(init_fn)
            if best_v < best[0]:
                best = (best_v, best_sol)
        return best

    def _run(self, init_fn):
        self.dic_tabu = {}
        self.dic_longM = {}
        # (v, act, solution)
        cur_sol = init_fn()
        best = (self.evaluate(cur_sol), None, cur_sol)
        for _ in range(self.max_iter):
            # print(f"cur_sol: {cur_sol}")
            iter_best = (math.inf, None, None)
            tabu_best = (math.inf, None, None)
            for act, sol in self.neighbor_op(cur_sol):
                v = self.evaluate(sol)
                if v < iter_best[0]:
                    iter_best = (v, act, sol)
                if act not in self.dic_tabu and v < tabu_best[0]:
                    tabu_best = (v, act, sol)
            if iter_best[0] < best[0]:
                # aspiration criterion
                print(f"-> {iter_best[-1]}, v: {iter_best[0]}")
                best = iter_best
                _best = iter_best
            elif tabu_best[-1] is not None:
                _best = tabu_best
            else:
                _best = iter_best
            cur_sol = _best[-1]
            self.update_tabu(_best[1])
        self.show_tabu()
        return best[0], best[-1]

    def update_tabu(self, new_key):
        # print(f"update_tabu: {new_key}")
        # print(f"dic_tabu: {self.dic_tabu}")
        self.dic_longM[new_key] = self.dic_longM.get(new_key, 0) + 1
        for key in list(self.dic_tabu.keys()):
            # decrement Tenure
            self.dic_tabu[key] -= 1
            if self.dic_tabu[key] <= 0:
                del self.dic_tabu[key]
        self.dic_tabu[new_key] = self.tenure

    def show_tabu(self):
        print(f"dic_tabu: {self.dic_tabu}")
        print(f"dic_longM: {self.dic_longM}")


class TSP():
    def __init__(self, dist, N):
        self.dist = dist
        for i, j in list(dist.keys()):
            dist[j, i] = dist[i, j]  # edge in opposite direction
        self.N = N

    def total_distance(self, route):
        total = 0
        n = len(route)
        for i in range(n):
            total += self.dist[(route[i], route[(i+1) % n])]
        return total

    def swap2(self, route):
        for p1, p2 in combinations(range(len(route)), 2):
            if p1 >= p2:
                continue
            r = list((route))
            r[p1], r[p2] = r[p2], r[p1]
            yield (p1, p2), tuple(r)

    def opt2(self, route):
        """ab->cd-> -> ac<-bd->, where bc are not adjacent"""
        for a in range(self.N):
            for c in range((a + 3), a - 3 + self.N):
                r = list(route)
                i, j = a + 1, c
                while i < j:
                    r[i % self.N], r[j % self.N] = r[j % self.N], r[i % self.N]
                    i += 1
                    j -= 1
                yield (a, c), tuple(r)

    def initial_solution(self, mode='greedy'):
        """
        mode :
            "greedy" : advance step by choosing optimal one
            "random" : randomly generate a series number
        """
        if mode == 'greedy':
            route = [random.randint(0, self.N - 1)]
            i = 0
            while len(route) < self.N:
                next_n = min([j for j in range(self.N) if j != i and j not in route],
                             key=lambda j: self.dist[i, j])
                route.append(next_n)
                i = next_n
        elif mode == 'random':
            route = list(range(self.N))
            random.shuffle(route)
        print(f"init route: {route}, v: {self.total_distance(route)}")
        return route


def calc_tsp(n, dist):
    tsp = TSP(dist, n)

    start = time.time()
    tabu = TabuSearch(tsp.total_distance, tsp.opt2, n)
    best_v, best_route = tabu.run(lambda: tsp.initial_solution('greedy'))
    end = time.time()

    print(f"best v: {best_v}, route: {best_route}")
    print(f"the time cost: {end - start}")
