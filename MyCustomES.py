import math
import random
from typing import List, Dict, Any, TypedDict, Literal, Tuple, Optional, Callable

import numpy as np

from QOPS_test import get_Z_family_values

VarType = Literal["int", "float"]


class VarSpec(TypedDict):
    lo: float
    hi: float
    type: VarType  # "int" or "float"


Vector = List[float]
SearchSpace = List[VarSpec]
Individual = Dict[str, Any]
ObjectiveFn = Callable[[Vector], float]


class customES:
    def __init__(
        self,
        search_space: SearchSpace,
        pop_size: int = 30,
        num_children: int = 20,
        max_gens: int = 100,
        target_fitness: float = 1e-5,
        seed: Optional[int] = None,
    ) -> None:
        """
        Mixed (int/float) Evolution Strategies with self-adaptation and early stopping.

        search_space: list of {"lo": float, "hi": float, "type": "int"|"float"}
        """
        self.search_space = search_space
        self.pop_size = pop_size
        self.num_children = num_children
        self.max_gens = max_gens
        self.target_fitness = target_fitness
        self.history: List[Tuple[int, float]] = []  # (gen, best_fitness)
        if seed is not None:
            random.seed(seed)
        self.tester = None
        self.fam_idx = 0

    # ---------- Public API ----------

    def objective_function(self, sols):
        id = 0
        nan_id = []
        testcase_to_execute = []
        testcase_to_execute_exp = []
        for solution in sols:
            indexes, weights = solution[0:len(solution)//2], solution[len(solution)//2::]
            fam = self.tester.applicable_families[self.fam_idx]
            paulies = get_Z_family_values(self.tester.CUT.num_qubits, indexes)
            pauli_dict = {}
            for pauli, prob in zip(paulies, weights):
                pauli_dict[pauli] = prob
            testcase = {"test_case": pauli_dict, "family_index": fam[0], "M": fam[-1]}
            if testcase["test_case"] == {}:
                nan_id.append(id)
            else:
                test_case = self.tester.get_test_case_theoretics_Z(testcase)
                exp = self.tester.get_theoretical_exp_from_test_case_M3(test_case)
                testcase_to_execute.append(testcase["test_case"])
                testcase_to_execute_exp.append(exp)
            id+=1

        results = self.tester.execute_test_cases(testcase_to_execute)
        RESULT = []
        for i in range(len(sols)):
            if i in nan_id:
                RESULT.append(np.inf)
            else:
                RESULT.append(abs(testcase_to_execute_exp[i]-results[i]))

        return RESULT

    def run(self) -> Individual:
        """Run the (μ + λ)-ES and return the best individual."""
        population = self._init_population()
        population.sort(key=lambda c: c["fitness"])
        best = population[0]
        self.history = [(0, best["fitness"])]

        for gen in range(1, self.max_gens + 1):
            if best["fitness"] >= self.target_fitness:
                print(f"Stopping early at gen {gen-1}, fitness={best['fitness']}")
                break

            parents = [population[i % len(population)] for i in range(self.num_children)]
            children = [self._mutate(par) for par in parents]

            CHILDVEC = [self.decode_vector(c["vector"]) for c in children]
            CHILDRESULT = self.objective_function(CHILDVEC)
            for c,cr in zip(children, CHILDRESULT):
                c["fitness"] = cr

            union = population + children
            union.sort(key=lambda c: c["fitness"],reverse=True)

            if union[0]["fitness"] > best["fitness"]:
                best = union[0]
            population = union[: self.pop_size]

            self.history.append((gen, best["fitness"]))
            print(f" > gen {gen}, fitness={best['fitness']}")
        return best, self.history

    def decode_vector(self, vec: Vector) -> List[float | int]:
        """Return a list with ints cast to int and floats kept as float."""
        out: List[float | int] = []
        for spec, val in zip(self.search_space, vec):
            out.append(int(val) if spec["type"] == "int" else float(val))
        return out

    # ---------- Internals ----------
    def _is_int(self, spec: VarSpec) -> bool:
        return spec["type"] == "int"

    def _random_component(self, spec: VarSpec) -> float:
        lo, hi = spec["lo"], spec["hi"]
        if self._is_int(spec):
            return float(random.randint(int(math.ceil(lo)), int(math.floor(hi))))
        return lo + (hi - lo) * random.random()

    def _random_vector(self) -> Vector:
        return [self._random_component(spec) for spec in self.search_space]

    def _random_gaussian(self, mean: float = 0.0, stdev: float = 1.0) -> float:
        return random.gauss(mean, stdev)

    def _mutate_problem(self, vector: Vector, stdevs: Vector) -> Vector:
        child: Vector = []
        for i, v in enumerate(vector):
            val = v + stdevs[i] * self._random_gaussian()
            lo, hi = self.search_space[i]["lo"], self.search_space[i]["hi"]
            if self._is_int(self.search_space[i]):
                val = round(val)
            # clamp
            if val < lo:
                val = lo
            if val > hi:
                val = hi
            child.append(float(val))
        return child

    def _mutate_strategy(self, stdevs: Vector) -> Vector:
        n = len(stdevs)
        tau = 1.0 / math.sqrt(2.0 * n)
        tau_p = 1.0 / math.sqrt(2.0 * math.sqrt(n))
        g_global = self._random_gaussian()
        return [sd * math.exp(tau_p * g_global + tau * self._random_gaussian()) for sd in stdevs]

    def _mutate(self, parent: Individual) -> Individual:
        vec = self._mutate_problem(parent["vector"], parent["strategy"])
        strat = self._mutate_strategy(parent["strategy"])
        return {"vector": vec, "strategy": strat}

    def _init_population(self) -> List[Individual]:
        # Strategy init: 5% of range; ensure >=1.0 for integer dims to avoid early freeze after rounding
        strat_hi = []
        for spec in self.search_space:
            rng = spec["hi"] - spec["lo"]
            hi = 0.05 * rng
            if self._is_int(spec):
                hi = max(hi, 1.0)
            strat_hi.append(hi)

        pop: List[Individual] = []
        FitVec = []
        Vec = []
        Strat = []
        for _ in range(self.pop_size):
            vec = self._random_vector()
            strat = [random.uniform(0.0, h) for h in strat_hi]
            Vec.append(vec)
            Strat.append(strat)
            FitVec.append(self.decode_vector(vec))

        FitrRsult = self.objective_function(FitVec)

        for vec,strat,fit in zip(Vec,Strat,FitrRsult):
            pop.append({"vector": vec, "strategy": strat, "fitness": fit})

        return pop


# ---------- Example usage ----------
if __name__ == "__main__":
    # Mixed int/float search space
    space = []
    for _ in range(16):
        space.append({"lo": 1.0, "hi": 7.0, "type": "int"})

    for _ in range(16):
        space.append({"lo": -5.0, "hi": 5.0, "type": "float"})

    search_space: SearchSpace = space

    es = customES(
        search_space=search_space,
        pop_size=5,
        num_children=3,
        max_gens=100,
        target_fitness=700,
        seed=None,
        # objective_function can be injected; defaults to Sphere
        # objective_function=lambda v: sum(abs(x) for x in v),  # example
    )

    best = es.run()
    decoded = es.decode_vector(best["vector"])
    print(f"done! Solution: f={best['fitness']}, s={decoded}")