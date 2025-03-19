"""
The pygad.utils_ga.mutation module has all the built-in mutation operators.
"""

import numpy
import random
import torch


class Mutation:
    def adaptive_mutation(self, offspring):

        """
        基于“mutation_num_genes”参数应用自适应变异。随机选择数量相等的基因进行突变。这个数字取决于解决方案的适应度。
        The random values are selected based on the 2 parameters 'random_mutation_min_val' and 'random_mutation_max_val'.
        It accepts a single parameter:
            -offspring: The offspring to mutate.
        It returns an array of the mutated offspring.
        """
        num_genes = round(self.len_pop.basics.piece_num * 0.4 + 1)

        self.mutation_num_genes = (num_genes, num_genes, num_genes, num_genes)
        phi_inf = self.len_pop.basics.phi_inf
        n = len(offspring)
        device = offspring.device
        range_min = 0
        range_max = 1

        mute_idx = torch.zeros(n, 4).to(device)
        item_mute = (torch.randint(0, num_genes, (1, n)).to(device))[0]
        mute_idx[torch.where(item_mute >= 0)[0], item_mute] = 1
        mute_idx = torch.clamp(mute_idx + (torch.randint(0, 3, (n, 4)).to(device)), max=1)

        # 曲率

        num_c = self.mutation_num_genes[0]
        sel_idx = mute_idx[:, 0].unsqueeze(1).expand(n, num_c).flatten()
        mutation_indices_c = torch.randint(self.len_pop.basics.index_c[0], self.len_pop.basics.index_c[1], (n, num_c)).to(device)
        random_value_c = (range_min + torch.rand(n, num_c).to(device) * (range_max - range_min)).flatten()
        item_indices_c = torch.where(mutation_indices_c >= 0)[0].to(device).long()
        mutation_indices_c = mutation_indices_c.flatten()
        offspring[item_indices_c[sel_idx == 1], mutation_indices_c[sel_idx == 1]] = random_value_c[sel_idx == 1]

        # 间距
        num_d = self.mutation_num_genes[1]
        sel_idx = mute_idx[:, 1].unsqueeze(1).expand(n, num_d).flatten()
        mutation_indices_d = torch.randint(self.len_pop.basics.index_d[0], self.len_pop.basics.index_d[1], (n, num_d)).to(device)
        item_indices_d = torch.where(mutation_indices_d >= 0)[0].to(device).long()
        mutation_indices_d = mutation_indices_d.flatten()
        phi_d = (phi_inf[mutation_indices_d][:, 0] + offspring[item_indices_d, mutation_indices_d] * phi_inf[mutation_indices_d][:, 1]).reshape(n, num_d)
        item = torch.rand(n, num_d).to(device).T
        weight = (item / item.sum(dim=0)).T
        item_d = (weight.T * phi_d.sum(dim=1)).T
        random_value_d = torch.clamp(
            (item_d.flatten() - phi_inf[mutation_indices_d][:, 0]) / phi_inf[mutation_indices_d][:, 1], 0, 1)
        offspring[item_indices_d[sel_idx == 1], mutation_indices_d[sel_idx == 1]] = random_value_d[sel_idx == 1]

        # 材料折射率
        num_n = self.mutation_num_genes[2]
        sel_idx = mute_idx[:, 2].unsqueeze(1).expand(n, num_n).flatten()
        mutation_indices_n = torch.randint(self.len_pop.basics.index_n[0], self.len_pop.basics.index_n[1], (n, num_n)).to(device)
        random_value_n = (range_min + torch.rand(n, num_n).to(device) * (range_max - range_min)).flatten()
        item_indices_n = torch.where(mutation_indices_n >= 0)[0].to(device).long()
        mutation_indices_n = mutation_indices_n.flatten()
        offspring[item_indices_n[sel_idx == 1], mutation_indices_n[sel_idx == 1]] = random_value_n[sel_idx == 1]

        # 材料阿贝数
        num_V = self.mutation_num_genes[3]
        sel_idx = mute_idx[:, 3].unsqueeze(1).expand(n, num_V).flatten()
        mutation_indices_V = torch.randint(self.len_pop.basics.index_V[0], self.len_pop.basics.index_V[1], (n, num_V)).to(device)
        random_value_V = (range_min + torch.rand(n, num_V).to(device) * (range_max - range_min)).flatten()
        item_indices_V = torch.where(mutation_indices_V >= 0)[0].to(device).long()
        mutation_indices_V = mutation_indices_V.flatten()
        if not self.op_n:
            offspring[item_indices_V[sel_idx == 1], mutation_indices_V[sel_idx == 1]] = 1 - offspring[
                item_indices_V[sel_idx == 1], mutation_indices_V[sel_idx == 1]]

        offspring[item_indices_V[sel_idx == 1], mutation_indices_V[sel_idx == 1]] = random_value_V[sel_idx == 1]

        return torch.clamp(offspring, 0, 1)
