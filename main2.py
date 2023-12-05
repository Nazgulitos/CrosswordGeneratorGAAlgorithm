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

    :param grid:
    :return:
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

        # Initial best crossword
        best_individual = select_best_individual(population)
        cnt += 1

        # Loop that controls individuals generation
        while best_individual[-1] < 15 and cnt <= 1000000:

            # Select the two individuals with the highest fitness score
            parent1, parent2 = select_parents(population)

            # Create an offspring through crossover
            offspring = self._crossover(parent1, parent2)

            # Mutate the offspring
            mutated_offspring = self._mutate(offspring)

            # Replacing the existed random individual with the new offspring with its fitness score
            population[random.randint(0, max_population_size - 1)] = \
                self._initialize_fitness_score(mutated_offspring[:-1])


            # population[select_worst_individual(population)] = \
            #     self._initialize_fitness_score(mutated_offspring[:-1])

            # Searching the individual with the best fitness score
            best_individual = select_best_individual(population)

            cnt += 1

            #
            if cnt >= 10000:
                self.mutation_rate += 0.01

            print("Current crossword Puzzle:")
            print(best_individual[-1])
            print(best_individual)

        # Record the end time
        end_time = time.time()

        # Calculate and print the elapsed time
        elapsed_time = end_time - start_time
        print(f"Elapsed Time: {elapsed_time} seconds")

        print(best_individual[-1])

        print("Crossword Puzzle:")
        self._print(best_individual)
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
            self._print(crossword_grid)
            print("---------------------------------")

            # self.row = list(range(self.grid_size))
            # self.col = list(range(self.grid_size))

            # Initializes fitness score for the individual and add to population
            population.append(self._initialize_fitness_score(crossword_grid))
            # for item in population[i]:
            #     print(item)
            # print("---------------------------------")

        return population

    def generate_grid(self):
        """
        Generates the crossword grid - individual
        :return: the individual
        """

        # Grid for controlling improper crossings, location etc
        crossword_grid = [[' ' for _ in range(grid_size)] for _ in range(grid_size)]

        # List for the words, their coordinates, etc
        individual = [[] for _ in range(len(self.words))]

        # Loop for location word on crossword grid
        for i in range(len(self.words)):
            # Randomly locate the word
            if random.choice([True, False]):
                self._place_word_horizontally(individual[i], words[i], crossword_grid, individual, self.words)
            else:
                self._place_word_vertically(individual[i], words[i], crossword_grid, individual, self.words)

        return individual

    def _place_word_horizontally(self, word_data, word, crossword_grid, individual, words):
        """
        Locates the word horizontally on random place
        :param word_data: list for current word's items: the word,
            coordinates of initial letter, and orientation of word
        :param word: current word to be located
        :param crossword_grid: grid with located word in current crossword(individual)
        """
        flag = False

        while not flag:
            cnt = 0
            # Randomly choose the row and col for a word
            row = random.randint(0, self.grid_size - 1)
            col = random.randint(0, self.grid_size - len(word))

            set1 = set([l for l in range(col, col + len(word))])
            flag2 = True
            indix = 0
            for rowi in individual:
                if len(rowi) != 0:
                    if rowi[1] == "h":
                        set2 = set([i for i in range(rowi[0][1], rowi[0][1] + len(words[indix]))])
                        if abs(rowi[0][0] - row) == 1 and len(set1.intersection(set2)) != 0:
                            flag2 = False
                            break
                indix += 1
            if not (flag2):
                continue

            # [[[1, 2], 'v'], [[9, 0], 'h'], [], []]

            # Check whether the potential place for word is free and satisfies the conditions
            for j in range(col, col + len(word)):
                if crossword_grid[row][j] != '0' and crossword_grid[row][j] != '*' and crossword_grid[row][j] != 'h':
                    cnt += 1
            if cnt == len(word):
                flag = True

        # Continue if a word can be located
        if flag:
            for j in range(col, col + len(word)):
                # h for horizontal words to avoid words rewriting
                crossword_grid[row][j] = "h"

                # 0 for horizontal words to avoid together stucked words
                if row - 1 >= 0:
                    crossword_grid[row - 1][j] = "0"
                if row + 1 < self.grid_size:
                    crossword_grid[row + 1][j] = "0"

            # * for horizontal and vertical words to avoid stucked words
            if col - 1 >= 0:
                crossword_grid[row][col - 1] = "*"
            if col + len(word) + 1 < self.grid_size:
                crossword_grid[row][col + len(word)] = "*"

        # save the data about word
        word_data.append([row, col])
        word_data.append('h')

    def _place_word_vertically(self, word_data, word, crossword_grid, individual, words):
        """
        Locates the word vertically on random place
        :param word_data: list for current word's items: the word,
            coordinates of initial letter, and orientation of word
        :param word: current word to be located
        :param crossword_grid: grid with located word in current crossword(individual)
        :return:
        """

        flag = False

        while not flag:
            cnt = 0
            # Randomly choose the row and col for a word
            row = random.randint(0, self.grid_size - len(word))
            col = random.randint(0, self.grid_size - 1)

            indix = 0
            set1 = set([l for l in range(col, col + len(word))])
            flag2 = True
            for rowi in individual:
                if len(rowi) != 0:
                    if rowi[1] == "h":
                        set2 = set([i for i in range(rowi[0][1], rowi[0][1] + len(words[indix]))])
                        if abs(rowi[0][1] - col) == 1 and len(set1.intersection(set2)) != 0:
                            flag2 = False
                            break
                indix += 1
            if not (flag2):
                continue

            # Check whether the potential place for word is free and satisfies the conditions
            for i in range(row, row + len(word)):
                if crossword_grid[i][col] != '1' and crossword_grid[i][col] != '*' and crossword_grid[i][col] != 'v':
                    cnt += 1
            if cnt == len(word):
                flag = True

        # Continue if a word can be located
        if flag:
            for i in range(row, row + len(word)):
                # h for horizontal words to avoid words rewriting
                crossword_grid[i][col] = "v"

                # 1 for vertical words to avoid together stucked words
                if col - 1 >= 0:
                    crossword_grid[i][col - 1] = "1"
                if col + 1 < self.grid_size:
                    crossword_grid[i][col + 1] = "1"

            # * for horizontal and vertical words to avoid stucked words
            if row - 1 >= 0:
                crossword_grid[row - 1][col] = "*"
            if row + len(word) + 1 < self.grid_size:
                crossword_grid[row + len(word)][col] = "*"

        # save the data about word
        word_data.append([row, col])
        word_data.append('v')

    def _crossover(self, parent1, parent2):
        """
        Uses parents to create new offspring through
        :param parent1:
        :param parent2:
        :return:
        """
        # offspring = []
        # for i in range(len(parent1)):
        #     if random.choice([True, False]):
        #         offspring.append(parent1[i])
        #     else:
        #         offspring.append(parent2[i])

        crossover_point = random.randint(1, len(self.words))

        # Perform crossover by swapping the rows at the crossover point
        offspring = parent1[:crossover_point] + parent2[crossover_point:]
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
                    new_start_col = random.randint(0, self.grid_size - len(word2))
                    individual[index1][1] = (individual[index1][1][0], new_start_col)
                elif individual[index1][2] == 'v':
                    new_start_row = random.randint(0, self.grid_size - len(word2))
                    individual[index1][1] = (new_start_row, individual[index1][1][1])

                if individual[index2][2] == 'h':
                    new_start_col = random.randint(0, self.grid_size - len(word1))
                    individual[index2][1] = (individual[index2][1][0], new_start_col)
                elif individual[index2][2] == 'v':
                    new_start_row = random.randint(0, self.grid_size - len(word1))
                    individual[index2][1] = (new_start_row, individual[index2][1][1])

            elif mutation_type == 'change_coordinates':
                # Randomly change coordinates of a word
                word = random.choice(temp_words)
                index = temp_words.index(word)

                if individual[index][2] == 'h':
                    new_start_col = random.randint(0, self.grid_size - len(word))
                    individual[index][1] = (individual[index][1][0], new_start_col)
                elif individual[index][2] == 'v':
                    new_start_row = random.randint(0, self.grid_size - len(word))
                    individual[index][1] = (new_start_row, individual[index][1][1])

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
                temp_grid[i].insert(0, words[i])

        # Check if the grid is connected
        if _is_grid_connected(temp_grid):
            # Penalize the grid for being disconnected
            points += 10
        # else:
        #     points -= 5

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
        for i, (word, start, orientation) in enumerate(temp_grid):
            for j, (other_word, other_start, other_orientation) in enumerate(temp_grid[i + 1:]):
                if orientation == 'h' and other_orientation == 'v':
                    intersect = (start[0], other_start[1])
                    if other_start[0] <= intersect[0] < other_start[0] + len(other_word) and start[1] <= intersect[1] < \
                            start[1] + len(word):
                        if word[intersect[1] - start[1]] == other_word[intersect[0] - other_start[0]]:
                            points += 1
                            intersections += 1
                        # else:
                        #     points -= 1
                elif orientation == 'v' and other_orientation == 'h':
                    intersect = (other_start[0], start[1])
                    if other_start[1] <= intersect[1] < other_start[1] + len(other_word) and start[0] <= intersect[0] < \
                            start[0] + len(word):
                        if word[intersect[0] - start[0]] == other_word[intersect[1] - other_start[1]]:
                            points += 1
                            intersections += 1
                        # else:
                        #     points -= 1
        # if intersections == len(self.words) - 1:
        #     points += 10
        # else:
        #     points -= 5
        temp_grid.append(points)
        return temp_grid



    def _print(self, crossword_grid):
        """

        :param crossword_grid:
        :return:
        """
        grid = [[' ' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        index = 0
        for row in crossword_grid:
            if isinstance(row, int):
                crossword_grid.remove(row)
            elif len(row) == 3:
                row.remove(row[0])
            else:
                break
        for row in crossword_grid:
            if row[1] == "h":
                for i in range(len(words[index])):
                    grid[row[0][0]][i + row[0][1]] = words[index][i]
            else:
                for i in range(len(words[index])):
                    grid[i + row[0][0]][row[0][1]] = words[index][i]
            index += 1
        for row in grid:
            print(' '.join(row))


words = ['apple', 'banana', 'cherry', 'ape']
# words = ['zoo', 'goal', 'owl', 'as']
grid_size: int = 20
max_population_size = 1000
mutation_rate = 0.6
crossword_generator = CrosswordGenerator(words, grid_size, mutation_rate)
crossword = crossword_generator.generate_crossword()
