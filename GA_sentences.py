import numpy as np
import string
import random
import argparse


parser = argparse.ArgumentParser(description='Genetic Algorithm')
parser.add_argument('--input_str', type=str,
                    required=True, help='Input string')
parser.add_argument('--num_pop', type=int, required=True,
                    help='Number of population')
parser.add_argument('--mul_rate', type=float,
                    required=True, help='Multation rate')

args = parser.parse_args()

str_total = string.ascii_letters + " " + "0123456789" + "_"


def initialize_pop(str_len):
    letters = str_total
    return ''.join(random.choice(letters) for i in range(str_len))


def evaluate_fitness(sentence, ind):
    count = 0
    sentence = list(sentence)
    ind = list(ind)
    for i in range(len(sentence)):
        if sentence[i] == ind[i]:
            count += 1
    return count


def tournament_selection(pop, pop_fitness, selection_size, tournament_size=4):
    num_individuals = len(pop)
    indices = np.array(range(num_individuals))
    selected_indices = []

    while len(selected_indices) < selection_size:
        np.random.shuffle(indices)
        for i in range(0, num_individuals, tournament_size):
            _max = pop_fitness[indices[i]]
            max_i = indices[i]
            tournament_range = indices[i:i + tournament_size]
            for j in tournament_range:
                if pop_fitness[j] > _max:
                    _max = pop_fitness[j]
                    max_i = j
            selected_indices.append(max_i)
    return selected_indices


def variation(pop):
    num_individuals = len(pop)
    num_para = len(pop[0])
    indices = np.array(range(num_individuals))
    np.random.shuffle(indices)
    offspring = []
    for i in range(0, num_individuals, 2):
        idx1 = indices[i]
        idx2 = indices[i+1]
        offspring1 = list(pop[idx1])
        offspring2 = list(pop[idx2])

        random_arr = np.random.rand(1, num_para)

        for j in range(num_para):
            if random_arr[0][j] < 0.5:
                temp = offspring1[j]
                offspring1[j] = offspring2[j]
                offspring2[j] = temp

        offspring.append(offspring1)
        offspring.append(offspring2)

    offspring = np.array(offspring)
    return offspring


def mutation(mul_rate, ind):
    for i in range(len(ind)):
        r = random.random()
        if r <= mul_rate:
            letters = list(str_total)
            r_s = random.choice(letters)
            ind[i] = r_s
    return ind


def compare_str(sentence, pop):
    arr_sen = list(sentence)
    for i in range(len(pop)):
        if list(pop[i]) == arr_sen:
            return True
    return False


def GA(sentence, num_pop, mul_rate):
    str_len = len(sentence)
    pop = []
    fitness_pop = []
    gen = 0

    # Initate random letters to pop
    for i in range(num_pop):
        pop.append(list(initialize_pop(str_len)))

    # Evaluate fitness
    for i in range(num_pop):
        fitness_pop.append(evaluate_fitness(sentence, pop[i]))

    # Sử dụng tournament_size 4 và selection_size là bằng kích thước quần thể
    selection_size = len(pop)
    tournament_size = 4

    while not compare_str(sentence, pop):
        # Tạo ra các cá thể con và đánh giá chúng
        offspring = variation(pop)

        for i in range(len(pop)):
            offspring[i] = mutation(mul_rate, offspring[i])

        offspring_fitness = []
        for i in range(len(offspring)):
            offspring_fitness.append(evaluate_fitness(sentence, offspring[i]))

        # Tạo ra quần thể pool gồm quần thể hiện tại pop và offspring
        pool = np.vstack((pop, offspring))
        pool_fitness = np.hstack((fitness_pop, offspring_fitness))

        # Thực hiện tournament selection trên quần thể pool
        pool_indices = tournament_selection(
            pool, pool_fitness, selection_size, tournament_size)

        # Thay thế quần thể hiện tại bằng những cá thể được chọn ra từ pool.
        pop = pool[pool_indices, :]
        pop_fitness = pool_fitness[pool_indices]
        gen += 1
        print("\nGen: ", str(gen))
        print(pop)
    return gen


GA(args.input_str, args.num_pop, args.mul_rate)
