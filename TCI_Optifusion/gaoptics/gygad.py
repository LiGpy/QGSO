import numpy
import random
import cloudpickle
import time
import warnings
import concurrent.futures
import inspect
import logging
from gaoptics import utils_ga
from gaoptics import helper
from gaoptics import visualize
import sys
import torch
import math


class GA:
    def __init__(self, len_pop):
        self.start_glass_num = None
        self.local_batch_size = None
        self.big_iter = None
        self.last_parent_fitness = None
        self.num_elitism = None
        self.mutation_num_genes = None
        self.doe_num = None
        self.aspheric_num = None
        self.parent_b_size = None
        self.pop_size = None
        self.fitness_batch_size = None
        self.num_parents_mating = None
        self.sol_per_pop = None
        self.num_genes = None
        self.population = None
        self.len_pop = len_pop
        self.device = len_pop.basics.device
        self.vari_flag_all = None

        # Round initial_population and population
        # Holds the fitness of the solutions in each generation.
        self.solutions_fitness = []

        # 上一代适应度
        self.last_generation_fitness = None
        # 上一代父本
        self.last_generation_parents = None
        # 上一代杂交后代
        self.last_generation_offspring_crossover = None
        # 上一代变异后代
        self.last_generation_offspring_mutation = None
        # 保存在last_generation_formation属性中的适应度值之前的一代适应度值。添加在PyGAD 2.16.2中。
        self.previous_generation_fitness = None
        # 在PyGAD 2.18.0中添加。NumPy数组，根据'keep_elitism'参数中传递的值保存当前一代的精英。只有当“keep_elitism”参数具有非零值时，它才有效。
        self.last_generation_elitism = None
        # 添加在PyGAD 2.19.0中。NumPy数组，保存当前一代精英主义的索引。只有当“keep_elitism”参数具有非零值时，它才有效。
        self.last_generation_elitism_indices = None

        self.num_generations = None
        self.num_sphere = self.len_pop.basics.sphere_gener_num
        self.num_asphere = self.len_pop.basics.asphere_gener_num

    def cal_pop_fitness(self, pop_use=None):
        if pop_use is None:
            pop_now = self.population
        else:
            pop_now = pop_use

        fitness_func = self.len_pop.fitness_func

        device = self.device
        pop_fitness = torch.zeros(len(pop_now)).to(device)

        solutions_indices = torch.linspace(0, len(pop_now) - 1, len(pop_now)).to(device).long()
        # Number of batches.
        num_batches = int(len(solutions_indices) / self.fitness_batch_size)
        # For each batch, get its indices and call the fitness function.
        for batch_idx in range(num_batches + 1):
            if batch_idx < num_batches:
                batch_first_index = batch_idx * self.fitness_batch_size
                batch_last_index = (batch_idx + 1) * self.fitness_batch_size
            else:
                batch_first_index = num_batches * self.fitness_batch_size
                batch_last_index = len(pop_now)
            if batch_last_index > batch_first_index:
                batch_indices = solutions_indices[batch_first_index:batch_last_index]
                batch_solutions = pop_now[batch_indices, :]

                item = self.len_pop.basics.max_pop
                self.len_pop.basics.max_pop = len(batch_solutions)
                batch_fitness = fitness_func(batch_solutions)
                self.len_pop.basics.max_pop = item

                pop_fitness[batch_indices] = batch_fitness
        return pop_fitness

    def sa(self, rand_range=0.1, iterMax=300, min_delta_fit=0.0025, end_mean_iter=50, opti_flag=None):

        i = 0
        device = self.population.device
        mean_fit_all = []
        fit_last = self.cal_pop_fitness()
        pop_final = self.population.clone()
        fit_final = fit_last.clone()
        mean_fit_all.append(fit_final.mean())
        while i <= iterMax:
            t1 = time.time()
            # 随机变化部分
            rand_delta = ((torch.rand(self.population.size()).to(
                device)).T * rand_range - rand_range / 2).T * torch.Tensor(self.len_pop.order1.lr_rate).to(device)

            # if opti_flag is not None:
            #     rand_delta[:, ~opti_flag] = 0

            rand_delta[:, ~self.len_pop.order1.vari_flag] = 0
            # 实验
            pop_last = self.population.clone()
            self.population = torch.clamp((pop_last.clone() + rand_delta), 0, 1)

            # if i == 10:
            #     self.len_pop.order1.final_select_flag = True
            #     self.len_pop.basics.acc_fit = 10
            #     self.cal_pop_fitness()
            #     self.len_pop.order1.final_select_flag = False

            fit_new = self.cal_pop_fitness()
            # 最佳种群和适应度转换
            final_replace_idx = fit_final - fit_new > 0
            pop_final[final_replace_idx] = self.population[final_replace_idx].clone()
            fit_final[final_replace_idx] = fit_new[final_replace_idx].clone()
            mean_fit_all.append(fit_final.mean())

            # 当前种群和适应度转换
            diff_fit = fit_last - fit_new

            sa_T = torch.clamp(fit_last * 0.1, max=1)
            p_accept = torch.exp(diff_fit / sa_T)
            rand_accept = torch.rand(self.population.size(0)).to(device)
            index_not_replace = rand_accept > p_accept
            self.population[index_not_replace] = pop_last[index_not_replace].clone()
            fit_new[index_not_replace] = fit_last[index_not_replace].clone()
            fit_last = fit_new.clone()
            best_solution_fitness = fit_final.min()
            new_mean_fit = fit_final.mean()
            t2 = time.time()
            print('sa_iter:{generation}   time:{time}   bestfit:{bestfit}  mean_fit:{item}'.format(generation=i,
                                                                                                   bestfit=best_solution_fitness
                                                                                                   ,
                                                                                                   item=new_mean_fit.item(),
                                                                                                   time=t2 - t1))

            if i >= end_mean_iter and (mean_fit_all[-end_mean_iter] - mean_fit_all[-1]) / (
                    end_mean_iter - 1) <= min_delta_fit:
                break
            i = i + 1

        self.population = pop_final

    def adam(self, input_data, iter_num=80, beta1=0.9, beta2=0.9, delta_x=1e-6, lr_base=1e-3, minima_iter=20, cos_T=10,
             sa_num=20, minima_diff_flag=0.1):
        print('adam:')
        # 计算梯度
        pop_origin = input_data.clone()
        elitism = input_data[:, self.vari_flag_all]

        device = elitism.device
        lr_now = torch.Tensor(self.len_pop.order1.lr_rate).to(device)[self.vari_flag_all]
        n, l = elitism.size()
        m = torch.zeros(n, l).to(self.device)
        v = torch.zeros(n, l).to(self.device)
        t = torch.zeros(n, l).to(self.device)
        elitism = elitism.unsqueeze(1)
        diff = (torch.eye(l) * delta_x).unsqueeze(0).to(device)

        i = 0

        pop_use = torch.ones_like(pop_origin)
        pop_use[:, self.vari_flag_all] = elitism.squeeze(1)
        pop_use[:, ~self.vari_flag_all] = pop_origin[:, ~self.vari_flag_all]

        fit_start = self.cal_pop_fitness(pop_use=pop_use)
        fit_all = []
        pop_final = elitism.squeeze(1).clone()
        fit_final = fit_start.clone()
        fit_all.append(fit_final.clone())
        while i <= iter_num:
            t += 1
            lr = lr_base * torch.ones(1, n).to(self.device)[0] * ((1 + math.cos(i * torch.pi / cos_T)) + 0.2) / 2.2
            item_1 = (elitism + diff).reshape(-1, l)
            item_0 = (elitism - diff).reshape(-1, l)
            item = torch.cat((item_1, item_0), dim=0)

            #
            item_pop_1 = pop_origin.unsqueeze(1).expand(pop_origin.size(0), l, pop_origin.size(1)).reshape(-1,
                                                                                                           pop_origin.size(
                                                                                                               1))
            item_pop = torch.cat((item_pop_1, item_pop_1), dim=0)
            pop_use = torch.ones_like(item_pop)
            pop_use[:, self.vari_flag_all] = item
            pop_use[:, ~self.vari_flag_all] = item_pop[:, ~self.vari_flag_all]

            fit_now = (self.cal_pop_fitness(pop_use=pop_use))
            y1 = fit_now[:len(fit_now) // 2].reshape(n, l)
            y0 = fit_now[len(fit_now) // 2:].reshape(n, l)
            grad = ((y1 - y0) / (2 * delta_x))
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            m_t_hat = m / (1 - beta1 ** t)
            v_t_hat = v / (1 - beta2 ** t)
            delta_x_final = (lr * (m_t_hat / (1e-8 + v_t_hat ** 0.5)).transpose(0, 1)).transpose(0, 1) * lr_now
            elitism = torch.clamp((elitism.squeeze(1) - delta_x_final).unsqueeze(1), 0, 1)

            pop_use = torch.ones_like(pop_origin)
            pop_use[:, self.vari_flag_all] = elitism.squeeze(1)
            pop_use[:, ~self.vari_flag_all] = pop_origin[:, ~self.vari_flag_all]
            fit_now = self.cal_pop_fitness(pop_use=pop_use)
            replace_idx = fit_final - fit_now > 0
            pop_final[replace_idx] = (elitism.squeeze(1))[replace_idx].clone()
            fit_final[replace_idx] = fit_now[replace_idx].clone()
            print('adam_now_mean:{x}'.format(x=fit_final.mean().item()))
            fit_all.append(fit_final.clone())

            if len(fit_all) >= minima_iter + 1:
                diff_rate = (fit_all[-(minima_iter + 1)] - fit_all[-1]) / fit_all[-(minima_iter + 1)]
                if diff_rate.mean() < 0.001 and diff_rate.max() < 0.005:
                    break

            # exp
            i = i + 1
        print('adam_good:{x}'.format(x=(fit_start - fit_final).mean().item()))
        pop_origin[:, self.vari_flag_all] = pop_final
        return pop_origin

    def run(self):
        """
        运行遗传算法。这是遗传算法经过多代进化的主要方法。
        """
        self.design_sphere()
        # if self.len_pop.basics.use_aspheric:
        #     self.design_asphere(sphere_structure)
        # self.len_pop.basics.final_select_flag = True
        # self.cal_pop_fitness()

    # def design_asphere(self, sphere_structure):
    #
    #     self.len_pop.init_pop('asphere', sphere_structure=sphere_structure)
    #     self.len_pop.order1.stage = 'asphere'
    #     self.population = self.len_pop.order1.init_population
    #     # Number of genes in the solution.
    #     self.num_genes = self.population.shape[1]
    #     # Number of solutions in the population.
    #     self.sol_per_pop = self.population.shape[0]
    #     self.num_parents_mating = max(round(self.sol_per_pop * 0.06), 1)
    #     self.num_elitism = max(round(self.sol_per_pop * 0.02), 1)
    #     self.fitness_batch_size = self.sol_per_pop
    #     # The population size.
    #     self.pop_size = (self.sol_per_pop, self.num_genes)
    #     self.num_generations = self.num_asphere
    #     self.big_iter = 10
    #     self.start_glass_num = 0
    #     self.local_batch_size = 3
    #     for generation in range(0, self.num_generations):
    #         print('generation_asphere:{generation}: '.format(generation=generation))
    #         self.len_pop.order1.now_gener = generation
    #         if (generation + 1) % (self.big_iter // 2) == 0:
    #             self.len_pop.order1.opti_stage = 'all_wave'
    #         else:
    #             self.len_pop.order1.opti_stage = 'single_wave'
    #         batch_size = self.local_batch_size
    #         self.parent_b_size = self.num_parents_mating // batch_size
    #         local_vari_num = self.sol_per_pop // (self.parent_b_size * 2) * 2
    #
    #         if generation == self.start_glass_num:
    #             self.len_pop.order1.use_real_glass = True
    #
    #         if (generation + 1) % self.big_iter == 0:
    #             self.vari_flag_all = self.len_pop.order1.vari_flag.clone()
    #             if generation >= self.start_glass_num:
    #                 self.vari_flag_all[self.len_pop.order1.n_list + self.len_pop.order1.v_list] = False
    #         else:
    #             self.vari_flag_all = torch.zeros_like(self.len_pop.order1.vari_flag).bool()
    #             item_flag = self.len_pop.order1.vari_flag.clone()
    #             if generation >= self.start_glass_num:
    #                 item_flag[self.len_pop.order1.n_list + self.len_pop.order1.v_list] = False
    #             valid_list = torch.where(item_flag)[0]
    #             self.vari_flag_all[valid_list[torch.randperm(valid_list.numel())][:local_vari_num]] = True
    #
    #         sa_rate = 0.1
    #         adam_rate = 1e-3
    #         div_rate = 0.15
    #         self.sa(rand_range=sa_rate, opti_flag=self.vari_flag_all)
    #         self.last_generation_fitness = self.cal_pop_fitness()
    #         # 筛选母本，每个母本之间有一定的差异性
    #         self.last_generation_parents = self.select_parents(self.last_generation_fitness,
    #                                                            num_parents=self.num_parents_mating, div_rate=div_rate)
    #
    #         # ADAM进一步优化母本
    #         if generation >= 0:
    #             for local_epoch in range(batch_size):
    #                 if local_epoch * self.parent_b_size < self.num_parents_mating:
    #                     if (local_epoch + 1) * self.parent_b_size >= self.num_parents_mating:
    #                         select_parent_index = torch.linspace(local_epoch * self.parent_b_size,
    #                                                              self.num_parents_mating - 1,
    #                                                              self.num_parents_mating - local_epoch * self.parent_b_size).to(
    #                             self.last_generation_parents.device).long()
    #                     else:
    #                         select_parent_index = torch.linspace(local_epoch * self.parent_b_size,
    #                                                              (local_epoch + 1) * self.parent_b_size - 1,
    #                                                              self.parent_b_size).to(
    #                             self.last_generation_parents.device).long()
    #                     select_parent = self.last_generation_parents[select_parent_index]
    #                     select_parent2 = self.adam(select_parent, lr_base=adam_rate)
    #
    #                     self.last_generation_parents[select_parent_index] = select_parent2
    #
    #         if (generation + 1) % self.big_iter == 0:
    #             self.len_pop.order1.final_select_flag = True
    #         else:
    #             self.len_pop.order1.final_select_flag = False
    #
    #         # self.len_pop.order1.final_select_flag = True
    #         self.last_parent_fitness = self.cal_pop_fitness(pop_use=self.last_generation_parents)
    #         self.len_pop.order1.final_select_flag = False
    #         self.last_generation_elitism = self.select_elitism(self.last_parent_fitness, num_elitism=self.num_elitism,
    #                                                            div_rate=div_rate)
    #
    #         if generation % 2 == 0:
    #             mutation_asphere = True
    #         else:
    #             mutation_asphere = False
    #
    #         self.last_generation_offspring_mutation = self.mutation(self.last_generation_parents, offspring_size=(
    #             self.sol_per_pop - self.num_elitism, self.num_genes), mutation_asphere=mutation_asphere)
    #         self.population[0:self.num_elitism, :] = self.last_generation_elitism
    #         self.population[self.num_elitism:, :] = self.last_generation_offspring_mutation

    def design_sphere(self):
        # self.len_pop.order1.final_select_flag = True
        self.len_pop.init_pop('sphere')
        self.len_pop.order1.stage = 'sphere'
        self.population = self.len_pop.order1.init_population
        # Number of genes in the solution.
        self.num_genes = self.population.shape[1]
        # Number of solutions in the population.
        self.sol_per_pop = self.population.shape[0]
        self.num_parents_mating = max(round(self.sol_per_pop * 0.06), 1)
        self.num_elitism = max(round(self.sol_per_pop * 0.02), 1)
        self.fitness_batch_size = self.sol_per_pop
        # The population size.
        self.pop_size = (self.sol_per_pop, self.num_genes)
        self.num_generations = self.num_sphere
        self.big_iter = 0.1
        div_rate_all = 0.2
        self.start_glass_num = 100

        for generation in range(0, self.num_generations):
            div_rate = div_rate_all
            print('generation_sphere:{generation}: '.format(generation=generation))
            self.len_pop.order1.now_gener = generation
            if (generation + 1) > (self.big_iter // 2):
                self.len_pop.order1.opti_stage = 'all_wave'
            else:
                self.len_pop.order1.opti_stage = 'single_wave'
            # self.len_pop.order1.opti_stage = 'all_wave'
            if generation == self.start_glass_num:
                self.len_pop.order1.use_real_glass = True
            batch_size = 3
            self.parent_b_size = max(self.num_parents_mating // batch_size, 2)
            local_vari_num = self.sol_per_pop // (self.parent_b_size * 2)

            if (generation + 1) % self.big_iter == 0:
                self.vari_flag_all = self.len_pop.order1.vari_flag.clone()
                if generation >= self.start_glass_num:
                    self.vari_flag_all[self.len_pop.order1.n_list + self.len_pop.order1.v_list] = False
            else:
                self.vari_flag_all = torch.zeros_like(self.len_pop.order1.vari_flag).bool()
                item_flag = self.len_pop.order1.vari_flag.clone()
                if generation >= self.start_glass_num:
                    item_flag[self.len_pop.order1.n_list + self.len_pop.order1.v_list] = False
                valid_list = torch.where(item_flag)[0]
                self.vari_flag_all[valid_list[torch.randperm(valid_list.numel())][:local_vari_num]] = True

            self.sa()

            # self.len_pop.order1.final_select_flag = True
            # self.len_pop.basics.acc_fit = 1
            # self.cal_pop_fitness()
            # self.len_pop.order1.final_select_flag = False

            self.last_generation_fitness = self.cal_pop_fitness()
            # 筛选母本，每个母本之间有一定的差异性
            self.last_generation_parents = self.select_parents(self.last_generation_fitness,
                                                               num_parents=self.num_parents_mating, div_rate=div_rate)
            # ADAM进一步优化母本
            for local_epoch in range(batch_size):
                if local_epoch * self.parent_b_size < self.num_parents_mating:
                    if (local_epoch + 1) * self.parent_b_size >= self.num_parents_mating:
                        select_parent_index = torch.linspace(local_epoch * self.parent_b_size,
                                                             self.num_parents_mating - 1,
                                                             self.num_parents_mating - local_epoch * self.parent_b_size).to(
                            self.last_generation_parents.device).long()
                    else:
                        select_parent_index = torch.linspace(local_epoch * self.parent_b_size,
                                                             (local_epoch + 1) * self.parent_b_size - 1,
                                                             self.parent_b_size).to(
                            self.last_generation_parents.device).long()
                    select_parent = self.last_generation_parents[select_parent_index]
                    select_parent2 = self.adam(select_parent)
                    self.last_generation_parents[select_parent_index] = select_parent2

            # self.len_pop.order1.final_select_flag = True
            # self.len_pop.basics.acc_fit = 1
            # self.cal_pop_fitness()
            # self.len_pop.order1.final_select_flag = False

            if (generation + 1) % self.big_iter == 0:
                self.len_pop.order1.final_select_flag = True
            else:
                self.len_pop.order1.final_select_flag = False
            # self.len_pop.order1.final_select_flag = True
            self.last_parent_fitness = self.cal_pop_fitness(pop_use=self.last_generation_parents)

            if generation == self.num_generations - 1:
                break
            self.len_pop.order1.final_select_flag = False
            self.last_generation_elitism = self.select_elitism(self.last_parent_fitness, num_elitism=self.num_elitism,
                                                               div_rate=div_rate)

            self.last_generation_offspring_mutation = self.mutation(self.last_generation_parents, offspring_size=(
                self.sol_per_pop - self.num_elitism, self.num_genes))
            self.population[0:self.num_elitism, :] = self.last_generation_elitism
            self.population[self.num_elitism:, :] = self.last_generation_offspring_mutation

        return self.last_generation_parents

    def select_parents(self, fitness, num_parents, div_rate=0.1):
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
            elif ((parents_pre[i_now] - parents[:i_select]).square().sum(dim=1).sqrt() / (
                    self.population.size(1) ** 0.5)).min() > div_rate:
                parents[i_select] = parents_pre[i_now]
                i_select += 1

            i_now += 1
        parents[i_select:] = torch.rand(parents[i_select:].size(0), self.population.size(1)).to(device)
        return parents

    def select_elitism(self, fitness, num_elitism, div_rate=0.1):
        """
        从高到底排序，但保持父本的多样性
        """
        device = fitness.device
        fitness_sorted_idx = fitness.sort()[1]
        parents_pre = self.last_generation_parents[fitness_sorted_idx]
        elitism = torch.zeros(num_elitism, self.last_generation_parents.size(1)).to(device)
        i_select = 0
        i_now = 0
        while i_now < self.last_generation_parents.size(0) and i_select < num_elitism:
            if i_select == 0:
                elitism[0] = parents_pre[i_now]
                i_select += 1
            elif ((parents_pre[i_now] - elitism[:i_select]).square().sum(dim=1).sqrt() / (
                    self.last_generation_parents.size(1) ** 0.5)).min() > div_rate:
                elitism[i_select] = parents_pre[i_now]
                i_select += 1

            i_now += 1
        elitism[i_select:] = torch.rand(elitism[i_select:].size(0), self.last_generation_parents.size(1)).to(device)
        return elitism

    def mutation(self, parents, offspring_size, mutation_asphere=False):

        """
        基于“mutation_num_genes”参数应用自适应变异。随机选择数量相等的基因进行突变。这个数字取决于解决方案的适应度。
        The random values are selected based on the 2 parameters 'random_mutation_min_val' and 'random_mutation_max_val'.
        It accepts a single parameter:
            -offspring: The offspring to mutate.
        It returns an array of the mutated offspring.
        """
        device = parents.device
        indices = torch.randint(0, parents.shape[0], (1, offspring_size[0])).to(device)
        parent_idx = indices[0]
        offspring = parents[parent_idx]
        device = offspring.device
        offspring_ori = offspring.clone()
        num_genes = 2
        if mutation_asphere:
            offspring = self.len_pop.find_asphere(offspring, mutated_surface=num_genes)
        else:
            phi_inf = self.len_pop.order1.phi_inf
            n = len(offspring)
            # 材料折射率
            item = torch.randint(0, len(self.len_pop.order1.n_list), (n, num_genes)).to(device)
            mutation_indices_n = torch.Tensor(self.len_pop.order1.n_list).to(device)[item].long()
            mutation_indices_v = mutation_indices_n + 1
            item_indices_nv = torch.where(mutation_indices_n >= 0)[0].to(device).long()
            mater_lab = self.len_pop.order1.material_lab
            mater_n = torch.Tensor(mater_lab.nd).to(device)[1:]
            mater_v = torch.Tensor(mater_lab.vd).to(device)[1:]
            mater_index = torch.randint(0, len(mater_n), (n, num_genes)).to(device)
            random_value_n = mater_n[mater_index].flatten()
            random_value_v = mater_v[mater_index].flatten()
            mutation_indices_n = mutation_indices_n.flatten()
            mutation_indices_v = mutation_indices_v.flatten()
            offspring[item_indices_nv, mutation_indices_n] = (random_value_n - phi_inf[mutation_indices_n][:, 0]) / \
                                                             phi_inf[mutation_indices_n][:, 1]
            offspring[item_indices_nv, mutation_indices_v] = (random_value_v - phi_inf[mutation_indices_v][:, 0]) / \
                                                             phi_inf[mutation_indices_v][:, 1]

            # 曲率
            c_list = self.len_pop.order1.c_list.copy()
            for idx_c in self.len_pop.order1.c_list:
                if ~self.len_pop.order1.vari_flag[idx_c]:
                    c_list.remove(idx_c)
            item = torch.randint(0, len(c_list), (n, num_genes)).to(device)
            mutation_indices_c = torch.Tensor(c_list).to(device)[item].long()
            item_indices_c = torch.where(mutation_indices_c >= 0)[0].to(device).long()
            mutation_indices_c = mutation_indices_c.flatten()
            # phi_c = (phi_inf[mutation_indices_c][:, 0] + offspring[item_indices_c, mutation_indices_c] * phi_inf[
            #                                                                                                  mutation_indices_c][
            #                                                                                              :, 1]).reshape( n,
            #     num_genes)
            #
            #
            # item_idx = torch.Tensor(c_list).to(device).unsqueeze(1) - mutation_indices_c.unsqueeze(0)
            # item_idx[item_idx <= 0] = 999
            # back_idx = item_idx.min(dim=0).values
            # back_idx[back_idx == 999] = 2
            # item_idx = torch.Tensor(c_list).to(device).unsqueeze(1) - mutation_indices_c.unsqueeze(0)
            # item_idx[item_idx >= 0] = -999
            # fore_idx = item_idx.max(dim=0).values
            # fore_idx[fore_idx == -999] = -2
            #
            #
            # n2 = torch.ones_like(back_idx)
            # n2[back_idx == 4] = offspring[item_indices_c, torch.clamp(mutation_indices_c + 2, max=offspring.size(1) - 1)][back_idx == 4] * (
            #         self.len_pop.basics.INDX[1] - self.len_pop.basics.INDX[0]) + self.len_pop.basics.INDX[0]
            # n2 = n2.reshape(n, num_genes)
            #
            # n1 = torch.ones_like(fore_idx)
            # n1[fore_idx == -4] = offspring[item_indices_c, mutation_indices_c - 2][fore_idx == -4] * (
            #         self.len_pop.basics.INDX[1] - self.len_pop.basics.INDX[0]) + self.len_pop.basics.INDX[0]
            # n1 = n1.reshape(n, num_genes)
            #
            # item = torch.rand(n, num_genes).to(device).T
            # weight = (item / item.sum(dim=0)).T
            #
            # item_focal = (weight.T * (phi_c * (n2 - n1)).sum(dim=1)).T
            #
            # random_value_c = torch.clamp(
            #     ((item_focal / (n2 - n1)).flatten() - phi_inf[mutation_indices_c][:, 0]) / phi_inf[mutation_indices_c][
            #                                                                                :,
            #                                                                                1], 0, 1)
            random_value_c = torch.rand(mutation_indices_c.size()).to(device)
            offspring[item_indices_c, mutation_indices_c] = random_value_c

            # 间距
            d_list = self.len_pop.order1.d_list.copy()
            for idx_d in self.len_pop.order1.d_list:
                if ~self.len_pop.order1.vari_flag[idx_d]:
                    d_list.remove(idx_d)
            item = torch.randint(0, len(d_list), (n, num_genes)).to(device)
            mutation_indices_d = torch.Tensor(d_list).to(device)[item].long()
            item_indices_d = torch.where(mutation_indices_d >= 0)[0].to(device).long()
            mutation_indices_d = mutation_indices_d.flatten()
            phi_d = (phi_inf[mutation_indices_d][:, 0] + offspring[item_indices_d, mutation_indices_d] * phi_inf[
                                                                                                             mutation_indices_d][
                                                                                                         :, 1]).reshape(n, num_genes)
            item = torch.rand(n, num_genes).to(device).T
            weight = (item / item.sum(dim=0)).T
            item_d = (weight.T * phi_d.sum(dim=1)).T
            random_value_d = torch.clamp(
                (item_d.flatten() - phi_inf[mutation_indices_d][:, 0]) / phi_inf[mutation_indices_d][:, 1], 0, 1)
            offspring[item_indices_d, mutation_indices_d] = random_value_d

        offspring[:, ~self.len_pop.order1.vari_flag] = offspring_ori[:, ~self.len_pop.order1.vari_flag]

        return torch.clamp(offspring, 0, 1)
