import random
import time
# convergence_threshold =10
class CrosswordGenerator:

    def __init__(self, words: list, grid_size: int, mutation_rate: float):
        self.words = words
        self.grid_size = grid_size
        self.mutation_rate = mutation_rate
        self.row = list(range(self.grid_size))
        self.col = list(range(self.grid_size))

    def generate_crossword(self):
        # Record the start time
        start_time = time.time()

        population = self._initialize_population()
        for _ in range(max_population_size):
            parent1, parent2 = self._select_parents(population)
            offspring = self._crossover(parent1, parent2)
            mutated_offspring = self._mutate(offspring)
            population[random.randint(0, max_population_size - 1)] = self._initialize_fitness_score(
                mutated_offspring[:-1])
        best_individual = self._select_best_individual(population)

        cnt = 0
        while best_individual[-1] != len(self.words) - 1:
            self.mutation_rate += 0.001
            if cnt == 1000000:
                break
            parent1, parent2 = self._select_parents(population)
            offspring = self._crossover(parent1, parent2)
            mutated_offspring = self._mutate(offspring)
            population[random.randint(0, max_population_size - 1)] = self._initialize_fitness_score(
                mutated_offspring[:-1])
            best_individual = self._select_best_individual(population)
            cnt += 1

        # Record the end time
        end_time = time.time()

        # Calculate and print the elapsed time
        elapsed_time = end_time - start_time
        print(f"Elapsed Time: {elapsed_time} seconds")
        print("Crossword Puzzle:")
        self._print(best_individual)
        return best_individual
        return 0

    def _select_best_individual(self, population):

        # Calculate the fitness score for each individual in the population
        fitness_scores = [individual[-1] for individual in population]

        # Find the index of the individual with the highest fitness score
        best_index = fitness_scores.index(max(fitness_scores))

        # Return the best individual
        return population[best_index]

    def _mutate(self, individual):
        temp_words = self.words

        # for i in range(self.grid_size):
        if random.random() < self.mutation_rate:
            # With probability self.mutation_rate, perform mutation
            if random.choice([True, False]):
                # if random.choice([True, False]):
                # Randomly change the word at this position
                word1 = random.choice(temp_words)
                word2 = random.choice(temp_words)
                while word1 == word2:
                    word2 = random.choice(temp_words)
                index1 = self.words.index(word1)
                index2 = self.words.index(word2)
                individual[index1][0] = word2
                individual[index2][0] = word1
            # else:
            # change coordinates of a word

            else:
                word = random.choice(temp_words)
                index = temp_words.index(word)
                # Randomly change the orientation of the word at this position
                if individual[index][2] == 'h':
                    individual[index][2] = 'v'
                else:
                    individual[index][2] = 'h'

        return individual

    def _crossover(self, parent1, parent2):

        # Randomly select a crossover point
        crossover_point = random.randint(1, len(self.words))

        # Perform crossover by swapping the rows at the crossover point
        offspring = parent1[:crossover_point] + parent2[crossover_point:]

        return offspring

    def _select_parents(self, population):

        # Calculate the fitness score for each individual in the population
        fitness_scores = [individual[-1] for individual in population]

        fitness_scores.sort(reverse=True)

        parent1 = population[fitness_scores[0]]
        parent2 = population[fitness_scores[1]]

        # parent1 = random.choice(population)
        # parent2 = random.choice(population)

        return parent1, parent2

    def _initialize_population(self):
        population = []

        # Generate random crossword grids for the population
        for i in range(max_population_size):
            crossword_grid = self._generate_random_grid()
            # for row in crossword_grid:
            #     print(row)
            # self._print(crossword_grid)

            self.row = list(range(self.grid_size))
            self.col = list(range(self.grid_size))

            population.append(self._initialize_fitness_score(crossword_grid))
            # for item in population[i]:
            #     print(item)
            # print("---------------------------------")

        return population

    def _initialize_fitness_score(self, grid):
        temp_grid = grid
        if temp_grid[0][0] not in self.words:
            for i in range(len(self.words)):
                temp_grid[i].insert(0, words[i])

        intersections = 0
        for i, (word, start, orientation) in enumerate(temp_grid):
            for j, (other_word, other_start, other_orientation) in enumerate(temp_grid[i + 1:]):
                if orientation == 'h' and other_orientation == 'v':
                    intersect = (start[0], other_start[1])
                    if intersect[0] >= other_start[0] and intersect[0] < other_start[0] + len(other_word) and intersect[
                        1] >= start[1] and intersect[1] < start[1] + len(word):
                        # intersections.append((intersect, word[intersect[1] - start[1]], other_word[intersect[0] - other_start[0]]))
                        if word[intersect[1] - start[1]] == other_word[intersect[0] - other_start[0]]:
                            intersections += 1
                        # else:
                        #     intersections -= 1
                elif orientation == 'v' and other_orientation == 'h':
                    intersect = (other_start[0], start[1])
                    if intersect[1] >= other_start[1] and intersect[1] < other_start[1] + len(other_word) and intersect[
                        0] >= start[0] and intersect[0] < start[0] + len(word):
                        # intersections.append((intersect, word[intersect[0] - start[0]], other_word[intersect[1] - other_start[1]]))
                        if word[intersect[0] - start[0]] == other_word[intersect[1] - other_start[1]]:
                            intersections += 1
                        # else:
                        #     intersections -= 1
        temp_grid.append(intersections)
        return temp_grid

    def _print(self, crossword_grid):
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

    def _generate_random_grid(self):
        # field = [[' ' for _ in range(grid_size)] for _ in range(grid_size)]
        grid = [[] for _ in range(len(self.words))]
        for i in range(len(self.words)):
            if random.choice([True, False]):
                self._place_word_horizontally(grid[i], words[i])
            else:
                self._place_word_vertically(grid[i], words[i])

        return grid

    def _place_word_horizontally(self, grid, word):
        if len(self.row) != 0:
            row = random.choice(self.row)
        else:
            self._place_word_vertically(grid, words)
            return
        col = random.randint(0, self.grid_size - len(word))

        grid.append([row, col])
        grid.append('h')

        self.row.remove(row)
        if row + 1 in self.row:
            self.row.remove(row + 1)
        if row - 1 in self.row:
            self.row.remove(row - 1)

    def _place_word_vertically(self, grid, word):
        row = random.randint(0, self.grid_size - len(word))
        if len(self.col) != 0:
            col = random.choice(self.col)
        else:
            self._place_word_horizontally(grid, words)
            return

        grid.append([row, col])
        grid.append('v')

        self.col.remove(col)
        if col + 1 in self.col:
            self.col.remove(col + 1)
        if col - 1 in self.col:
            self.col.remove(col - 1)


# words = ['apple', 'banana', 'cherry', 'date', 'elderberry']
words = ['zoo', 'goal', 'owl', 'as']
grid_size = 5
mutation_rate = 0.5
max_population_size = 10000

crossword_generator = CrosswordGenerator(words, grid_size, mutation_rate)
crossword = crossword_generator.generate_crossword()
