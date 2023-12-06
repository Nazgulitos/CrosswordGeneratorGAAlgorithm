import random
import time

def select_parents(population):
    """
    Selects two individuals according the best fitness scores
    :param population: the list of all individuals
    :return: two individuals with the highest fitness score
    """
    population_copy = population.copy()

    # Sorting the population by fitness score descending
    population_copy = sorted(population_copy, key=lambda individual: individual[-1], reverse=True)

    # Selecting individuals with the highest fitness score
    parent1 = population_copy[0]
    parent2 = population_copy[1]

    return parent1, parent2


def select_worst_individual(population):
    """
    Select one individual with the highest fitness score
    :param population: the list of all individuals
    :return: the individual with the highest fitness score
    """

    # List of each individual fitness score
    fitness_scores = [individual[-1] for individual in population]

    # Selects the index of individual with the highest fitness score
    worst_index = fitness_scores.index(min(fitness_scores))

    # Return the best individual
    return worst_index


def select_best_individual(population):
    """
    Select one individual with the highest fitness score
    :param population: the list of all individuals
    :return: the individual with the highest fitness score
    """
    # List of each individual fitness score
    fitness_scores = [individual[-1] for individual in population]

    # Selects the index of individual with the highest fitness score
    best_index = fitness_scores.index(max(fitness_scores))

    # Return the best individual
    return population[best_index]


def _is_grid_connected(grid):
    """
    Checks wheter current crossword is connected using DFS method
    :param grid: current crossword - individual
    :return: True, if it's connected, False, otherwise
    """

    adjacency_list = {i: [] for i in range(len(grid))}

    for i, (word, start, orientation) in enumerate(grid):
        for j, (other_word, other_start, other_orientation) in enumerate(grid[i + 1:]):
            if orientation == 'h' and other_orientation == 'v':
                intersect = (start[0], other_start[1])
                if other_start[0] <= intersect[0] < other_start[0] + len(other_word) and start[1] <= intersect[1] < \
                        start[1] + len(word):
                    adjacency_list[i].append(i + 1 + j)
                    adjacency_list[i + 1 + j].append(i)

            elif orientation == 'v' and other_orientation == 'h':
                intersect = (other_start[0], start[1])
                if other_start[1] <= intersect[1] < other_start[1] + len(other_word) and start[0] <= intersect[0] < \
                        start[0] + len(word):
                    adjacency_list[i].append(i + 1 + j)
                    adjacency_list[i + 1 + j].append(i)

    # Perform DFS to check connectivity
    visited = set()

    def dfs(node):
        visited.add(node)
        for neighbor in adjacency_list[node]:
            if neighbor not in visited:
                dfs(neighbor)

    # Start DFS from the first node
    dfs(0)

    # Check if all nodes are visited
    return len(visited) == len(grid)


class CrosswordGenerator:
    """
    This class represents all functions related to generating crossword using Genetic Algorithm
    """

    def __init__(self, words_list: list, gr_size: int, mut_rate: float):
        """
        Initializes the list of words, grid size, and mutation rate
        :param words_list: list with words should be used in crossword
        :param gr_size: shows size of crossword grid
        :param mut_rate: rate that show the probability of mutation occurrence
        """
        self.words = words_list
        self.grid_size = gr_size
        self.mutation_rate = mut_rate
        self.row = list(range(self.grid_size))
        self.col = list(range(self.grid_size))
        # It equals to the length of word that has maximal length
        self.MaxLen = len(max(words_list, key=len))


    def generate_crossword(self):
        """
        The main function that call all necessary function and try to get the proper crossword
        :return: the final proper crossword that may be improper
        """
        # Record the start time
        start_time = time.time()

        # Counter for iterations of generating
        cnt = 0

        # Initial population
        population = self._initialize_population()
        population2 = population.copy()

        # Initial best crossword
        best_individual = select_best_individual(population)
        cnt += 1

        # Loop that controls individuals generation
        while best_individual[-1] != 0 and cnt != 100000:

            temp_population = []

            while len(population) > 0:

                # Select the two individuals with the highest fitness score
                parent1, parent2 = select_parents(population)

                population.remove(parent1)
                population.remove(parent2)
                # Create an offspring through crossover
                offspring = self._crossover(parent1, parent2)

                # Mutate the offspring
                mutated_offspring = self._mutate(offspring)

                # Update the population with the new individual
                temp_population.append(self._initialize_fitness_score(mutated_offspring[:-1]))

            # Sorting the population by fitness score descending
            population = sorted(population2, key=lambda individual: individual[-1], reverse=True)

            i = 0
            # Filling the new population with old individuals
            while len(temp_population) != max_population_size:
                temp_population.append(population[i])
                i += 1

            # Searching the individual with the best fitness score
            best_individual = select_best_individual(population)

            cnt += 1

            # In case, when the population stucks in local maximum
            if cnt == 50000:
                temp_population = self._initialize_population()

            print("Current crossword Puzzle:")
            print(best_individual[-1])
            print(best_individual)
            bestie = best_individual.copy()
            self._print(bestie)
            population = temp_population

        # Record the end time
        end_time = time.time()

        # Calculate and print the elapsed time
        elapsed_time = end_time - start_time
        print(f"Elapsed Time: {elapsed_time} seconds")

        print(best_individual[-1])

        print("Crossword Puzzle:")
        self._print(best_individual.copy())
        return best_individual

    def _initialize_population(self):
        """
        Generates the concrete number of individuals for initial generation
        :return: population with individuals
        """
        population = []
        for i in range(max_population_size):
            # Generate the one individual(crossword grid)
            crossword_grid = self.generate_grid()
            # for row in crossword_grid:
            #     print(row)
            # self._print(crossword_grid)
            # print("---------------------------------")
            # self.row = list(range(self.grid_size))
            # self.col = list(range(self.grid_size))

            # Initializes fitness score for the individual and add to population
            population.append(self._initialize_fitness_score(crossword_grid))
            # for item in population[i]:
            #     print(item)
            # print("---------------------------------")

        return population

    def generate_grid(self):
        # field = [[' ' for _ in range(grid_size)] for _ in range(grid_size)]
        grid = [[] for _ in range(len(self.words))]
        for i in range(len(self.words)):
            if random.choice([True, False]):
                self._place_word_horizontally(grid[i], self.words[i])
            else:
                self._place_word_vertically(grid[i], self.words[i])

        return grid

    def _place_word_horizontally(self, grid, word):
        # if len(self.row) != 0:
        #     row = random.choice(self.row)
        # else:
        #     self._place_word_vertically(grid, words)
        #     return
        col = random.randint(0, self.MaxLen - len(word))
        row = random.randint(0, self.MaxLen - 1)

        grid.append([row, col])
        grid.append('h')

        # self.row.remove(row)
        # if row + 1 in self.row:
        #     self.row.remove(row + 1)
        # if row - 1 in self.row:
        #     self.row.remove(row - 1)

    def _place_word_vertically(self, grid, word):
        set1 = set([i for i in range(0, self.grid_size)])
        set2 = set([i for i in range(0, self.grid_size - len(word))])
        row = random.randint(0, self.MaxLen - len(word))
        col = random.randint(0, self.MaxLen - 1)
        # if len(self.col) != 0:
        #     col = random.choice(self.col)
        # else:
        #     self._place_word_horizontally(grid, words)
        #     return

        grid.append([row, col])
        grid.append('v')

        # self.col.remove(col)
        # if col + 1 in self.col:
        #     self.col.remove(col + 1)
        # if col - 1 in self.col:
        #     self.col.remove(col - 1)

    def _crossover(self, parent1, parent2):
        """
        Uses parents to create new offspring through
        :param parent1:
        :param parent2:
        :return:
        """
        offspring = []
        for i in range(len(parent1)):
            if random.choice([True, False]):
                offspring.append(parent1[i])
            else:
                offspring.append(parent2[i])

        # crossover_point = random.randint(1, len(self.words))
        #
        # # Perform crossover by swapping the rows at the crossover point
        # offspring = parent1[:crossover_point] + parent2[crossover_point:]
        #
        # return offspring
        # Randomly select two crossover points

        # crossover_point1 = random.randint(1, len(self.words) - 1)
        # crossover_point2 = random.randint(crossover_point1 + 1, len(self.words))
        #
        # # Perform two-point crossover by swapping the rows between the two points
        # offspring = parent1[:crossover_point1] + parent2[crossover_point1:crossover_point2] + parent1[crossover_point2:]

        return offspring

    def _mutate(self, individual):
        """

        :param individual:
        :return:
        """
        temp_words = self.words

        if random.random() < self.mutation_rate:
            mutation_type = random.choice(['change_word', 'change_coordinates', 'change_orientation'])

            if mutation_type == 'change_word':
                # Randomly change the word at this position
                word1 = random.choice(temp_words)
                word2 = random.choice(temp_words)
                while word1 == word2:
                    word2 = random.choice(temp_words)
                index1 = self.words.index(word1)
                index2 = self.words.index(word2)

                if individual[index1][2] == 'h':
                    new_start_col = random.randint(0, self.MaxLen - len(word2))
                    individual[index1][1] = (individual[index1][1][0], new_start_col)
                elif individual[index1][2] == 'v':
                    new_start_row = random.randint(0, self.MaxLen - len(word2))
                    individual[index1][1] = (new_start_row, individual[index1][1][1])

                if individual[index2][2] == 'h':
                    new_start_col = random.randint(0, self.MaxLen - len(word1))
                    individual[index2][1] = (individual[index2][1][0], new_start_col)
                elif individual[index2][2] == 'v':
                    new_start_row = random.randint(0, self.MaxLen - len(word1))
                    individual[index2][1] = (new_start_row, individual[index2][1][1])

            elif mutation_type == 'change_coordinates':
                # Randomly change coordinates of a word
                word = random.choice(temp_words)
                index = temp_words.index(word)

                if individual[index][2] == 'h':
                    new_start_col = random.randint(0, self.MaxLen - len(word))
                    new_start_row = random.randint(0, self.MaxLen)
                    individual[index][1] = (new_start_row, new_start_col)
                elif individual[index][2] == 'v':
                    new_start_col = random.randint(0, self.MaxLen)
                    new_start_row = random.randint(0, self.MaxLen - len(word))
                    individual[index][1] = (new_start_row, new_start_col)

            elif mutation_type == 'change_orientation':
                # Randomly change the orientation of the word at this position
                word = random.choice(temp_words)
                index = temp_words.index(word)

                individual[index][2] = 'v' if individual[index][2] == 'h' else 'h'

        return individual

    def _initialize_fitness_score(self, grid):
        """

        :param grid:
        :return:
        """
        points = 0
        temp_grid = grid
        if temp_grid[0][0] not in self.words:
            for i in range(len(self.words)):
                temp_grid[i].insert(0, self.words[i])

        # Check if the grid is connected
        if not(_is_grid_connected(temp_grid)):
            # Penalize the grid for being disconnected
            # points += len(self.words)
            points -= 1
        # else:
        #     points -= len(self.words)

        # for row1 in grid:
        #     for row2 in grid:
        #         if row1 != row2:
        #             if (abs(row1[1][1] - row2[1][1]) <= 1 and row1[2] == row2[2] == "v") or (
        #                     abs(row1[1][0] - row2[1][0]) <= 1 and row1[2] == row2[2] == "h"):
        #                 points -= 1
        #             else:
        #                 points += 1
        #             if (row1[2] == "h" and row2[2] == "v"):
        #                 set = [i for i in range(row1[1][1], row1[1][1] + len(row1[0]))]
        #                 set2 = [i for i in range(row2[1][0], row2[1][0] + len(row2[0]))]
        #                 if (abs(row1[1][0] - row2[1][0]) <= 1 and row2[1][1] in set) or ((abs(
        #                         row2[1][1] - row1[1][1]) <= 1 or abs(
        #                     row2[1][1] + len(row2[0]) - row1[1][1] + len(row1[0]))) and row1[1][0] in set2):
        #                     points -= 1
        #             else:
        #                 set = [i for i in range(row2[1][1], row2[1][1] + len(row2[0]))]
        #                 set2 = [i for i in range(row1[1][0], row1[1][0] + len(row1[0]))]
        #                 if (abs(row2[1][0] - row1[1][0]) <= 1 and row1[1][1] in set) or ((abs(
        #                         row1[1][1] - row2[1][1]) <= 1 or abs(
        #                     row2[1][1] + len(row2[0]) - row1[1][1] + len(row1[0]))) and row2[1][0] in set2):
        #                     points -= 1

        intersections = 0
        temp_set = [i for i in range(0, len(self.words))]
        for i, (word, start, orientation) in enumerate(temp_grid):
            for j, (other_word, other_start, other_orientation) in enumerate(temp_grid[i + 1:]):
                if orientation == 'h' and other_orientation == 'v':
                    set_rows = [i for i in range(other_start[0], other_start[0] + len(other_word))]
                    set_col = [i for i in range(start[1], start[1] + len(word))]
                    if (abs(start[1] + len(word) - 1 - other_start[1]) == 1 and start[0] in set_rows) or (
                            other_start[1] in set_col and start[0] - other_start[0] == - 1):
                        points -= 1
                elif orientation == 'v' and other_orientation == 'h':
                    set_rows = [i for i in range(start[0], start[0] + len(word))]
                    set_col = [i for i in range(other_start[1], other_start[1] + len(word))]
                    if (abs(start[1] - other_start[1]) == 1 and other_start[0] in set_rows and not(start[1] in set_col)) or (
                            start[1] in set_col and abs(start[0] + len(word) - 1 - other_start[0]) == 1):
                        points -= 1
                elif orientation == 'v' and other_orientation == 'v':
                    set_rows = set([i for i in range(other_start[0], other_start[0] + len(other_word))])
                    set_rows2 = set([i for i in range(start[0], start[0] + len(word))])
                    if (abs(start[1] - other_start[1]) <= 1 and len(set_rows.intersection(set_rows2)) != 0) or (
                            start[1] == other_start[1] and abs(start[0] + len(word) - 1 - other_start[0]) == 1):
                        points -= 1
                elif orientation == 'h' and other_orientation == 'h':
                    set_rows = set([i for i in range(other_start[1], other_start[1] + len(other_word))])
                    set_rows2 = set([i for i in range(start[1], start[1] + len(word))])
                    if (abs(start[0] - other_start[0]) <= 1 and len(set_rows.intersection(set_rows2)) != 0) or (
                            start[0] == other_start[0] and abs(start[1] + len(word) - 1 - other_start[1]) == 1):
                        points -= 1

                if orientation == 'h' and other_orientation == 'v':
                    intersect = (start[0], other_start[1])
                    if other_start[0] <= intersect[0] < other_start[0] + len(other_word) and start[1] <= intersect[1] < \
                            start[1] + len(word):
                        if word[intersect[1] - start[1]] == other_word[intersect[0] - other_start[0]]:
                            # points += 1
                            intersections += 1
                        else:
                            points -= 1
                elif orientation == 'v' and other_orientation == 'h':
                    intersect = (other_start[0], start[1])
                    if other_start[1] <= intersect[1] < other_start[1] + len(other_word) and start[0] <= intersect[0] < \
                            start[0] + len(word):
                        if word[intersect[0] - start[0]] == other_word[intersect[1] - other_start[1]]:
                            # points += 1
                            intersections += 1
                        else:
                            points -= 1
        # if intersections == len(self.words) - 1:
        #     points += 10
        # else:
        #     points -= 5
        # if penalty == aim_penalty:
        #     points += 5
        # elif penalty >= aim_penalty/2:
        #     points += 3
        # elif penalty <= aim_penalty/2 and penalty != 0:
        #     points += 1
        #
        # points += aim_penalty - penalty
        temp_grid.append(points)
        return temp_grid

    def _print(self, crossie):
        """

        :param crossword_grid:
        :return:
        """
        crossword_grid = []
        for thing in crossie:
            crossword_grid.append(thing)
        grid = [[' ' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        index = 0
        # for row in crossword_grid:
        #     if isinstance(row, int):
        #         crossword_grid.remove(row)
        #     elif len(row) == 3:
        #         row.remove(row[0])
        #     else:
        #         break
        for row in crossword_grid:
            if isinstance(row, int):
                continue
            if row[-1] == "h":
                for i in range(len(self.words[index])):
                    grid[row[1][0]][i + row[1][1]] = self.words[index][i]
            else:
                for i in range(len(self.words[index])):
                    grid[i + row[1][0]][row[1][1]] = self.words[index][i]
            index += 1
        for row in grid:
            print(' '.join(row))


words = ['apple', 'banana', 'cherry']
# words = ['zoo', 'goal', 'owl', 'as']
grid_size: int = 20
max_population_size = 1000
mutation_rate = 0.6
crossword_generator = CrosswordGenerator(words, grid_size, mutation_rate)
crossword = crossword_generator.generate_crossword()
