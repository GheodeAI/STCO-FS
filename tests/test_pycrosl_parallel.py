import os
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/stco-matplotlib-tests")
sys.path.insert(0, str(ROOT / "Experiments" / "Paper"))

from PyCROSL.AbsObjectiveFunc import AbsObjectiveFunc
from PyCROSL.CoralPopulation import Coral, CoralPopulation


class ToyObjective(AbsObjectiveFunc):
    def __init__(self):
        super().__init__(1, "max", sup_lim=100, inf_lim=0)

    def objective(self, solution):
        time.sleep(0.001)
        return float(solution[0])

    def random_solution(self):
        return np.zeros(1)

    def repair_solution(self, solution):
        return solution


def test_parallel_coral_evaluation_preserves_order_and_counter():
    objective = ToyObjective()
    corals = [Coral(np.array([index]), objective) for index in range(100)]
    population = CoralPopulation.__new__(CoralPopulation)

    result = population.evaluate_fitnesses(corals, n_jobs=4)

    assert result == corals
    assert objective.counter == len(corals)
    assert all(coral.fitness_calculated for coral in corals)
    assert [coral.fitness for coral in corals] == [float(index) for index in range(100)]
