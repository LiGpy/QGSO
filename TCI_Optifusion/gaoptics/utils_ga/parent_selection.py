"""
The pygad.utils_ga.parent_selection module has all the built-in parent selection operators.
"""

import numpy
import torch
import time


class ParentSelection:
    def steady_state_selection(self, fitness, num_parents):

        """
        Selects the parents using the steady-state selection technique. Later, these parents will mate to produce the offspring.
        It accepts 2 parameters:
            -fitness: The fitness values of the solutions in the current population.
            -num_parents: The number of parents to be selected.
        It returns an Tensor of the selected parents.
        """
        fitness_sorted = fitness.sort(descending=True)[1]
        # Selecting the best individuals in the current generation as parents
        # for producing the offspring of the next generation.
        parents = self.population[fitness_sorted[:num_parents]]
        return parents, fitness_sorted[:num_parents]

    def divide_selection(self, fitness, num_parents, div_rate=0.2):
        """
        从高到底排序，但保持父本的多样性
        """
        device = fitness.device
        fitness_sorted_idx = fitness.sort()[1]
        parents_pre = self.population[fitness_sorted_idx]
        parents = torch.zeros(num_parents, self.population.size(1)).to(device)
        i_select = 0
        i_now = 0
        while i_now < self.population.size(0) and i_select < num_parents:
            if i_select == 0:
                parents[0] = parents_pre[i_now]
                i_select += 1
            elif ((parents_pre[i_now] - parents[:i_select]).square().sum(dim=1).sqrt() / (self.population.size(1) ** 0.5)).min() > div_rate:
                parents[i_select] = parents_pre[i_now]
                i_select += 1

            i_now += 1
        parents[i_select:] = torch.rand(parents[i_select:].size(0), self.population.size(1)).to(device)
        return parents


