import random
import time

grid_size: int = 20
max_population_size = 1000
mutation_rate = 0.01


def _is_connected(crossword_grid):
    visited = set()

    def dfs(node):
        if node in visited:
            return
        visited.add(node)

        # Check neighbors in horizontal and vertical directions
        for i, (word, start, orientation) in enumerate(crossword_grid):
            if orientation == 'h':
                if start[0] <= node[0] < start[0] + len(word) and start[1] == node[1]:
                    dfs((start[0] - 1, start[1]))
                    dfs((start[0] + len(word), start[1]))
            elif orientation == 'v':
                if start[0] == node[0] and start[1] <= node[1] < start[1] + len(word):
                    dfs((start[0], start[1] - 1))
                    dfs((start[0], start[1] + len(word)))

    # Start DFS from the first letter of the first word
    dfs((crossword_grid[0][1][0], crossword_grid[0][1][1]))

    # Check if all letters are visited
    return len(visited) == sum(len(word) for word, _, _ in crossword_grid)


def _select_parents(population):

    # Calculate the fitness score for each individual in the population
    fitness_scores = [individual[-1] for individual in population]

    fitness_scores.sort(reverse=True)

    parent1 = population[fitness_scores[0]]
    parent2 = population[fitness_scores[1]]

    # parent1 = random.choice(population)
    # parent2 = random.choice(population)

    return parent1, parent2


def _select_best_individual(population):

    # Calculate the fitness score for each individual in the population
    fitness_scores = [individual[-1] for individual in population]

    # Find the index of the individual with the highest fitness score
    best_index = fitness_scores.index(max(fitness_scores))

    # Return the best individual
    return population[best_index]


class CrosswordGenerator:
    def __init__(self, lexicon: list):
        self.words = lexicon
        self.grid_size = grid_size
        self.mutation_rate = mutation_rate
        self.row = list(range(self.grid_size))
        self.col = list(range(self.grid_size))

    def generate_crossword(self):
        # Record the start time
        start_time = time.time()
        cnt = 0
        population = self._initialize_population()

        parent1, parent2 = _select_parents(population)
        offspring = self._crossover(parent1, parent2)
        mutated_offspring = self._mutate(offspring)
        population[random.randint(0, max_population_size - 1)] = self._initialize_fitness_score(
            mutated_offspring[:-1])
        best_individual = _select_best_individual(population)
        cnt += 1

        while best_individual[-1] != len(self.words) - 1:
            parent1, parent2 = _select_parents(population)
            offspring = self._crossover(parent1, parent2)
            mutated_offspring = self._mutate(offspring)
            population[random.randint(0, max_population_size - 1)] = self._initialize_fitness_score(
                mutated_offspring[:-1])
            best_individual = _select_best_individual(population)
            cnt += 1
            if cnt == 1000:
                break

        # Record the end time
        end_time = time.time()

        # Calculate and print the elapsed time
        elapsed_time = end_time - start_time
        print(f"Elapsed Time: {elapsed_time} seconds")

        print("Crossword Puzzle:")
        self._print(best_individual)
        return best_individual

    def _initialize_population(self):
        population = []
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

    def _generate_random_grid(self):
        field = [[' ' for _ in range(grid_size)] for _ in range(grid_size)]
        grid = [[] for _ in range(len(self.words))]
        for i in range(len(self.words)):
            if random.choice([True, False]):
                self._place_word_horizontally(grid[i], self.words[i], field)
            else:
                self._place_word_vertically(grid[i], self.words[i], field)

        return grid

    def _place_word_horizontally(self, grid, word, field):
        cnt = 0
        while cnt != 400:
            row = random.randint(0, self.grid_size - len(word))
            col = random.randint(0, self.grid_size - len(word))
            flag = True
            for j in range(col, col + len(word)):
                if field[row][j] != '1' and field[row][j] != ' ':
                    flag = False
                    break
            if flag:
                index = 0
                for j in range(col, col + len(word)):
                    field[row][j] = word[index]
                    if row - 1 >= 0:
                        field[row - 1][j] = "0"
                    if row + 1 <= self.grid_size:
                        field[row + 1][j] = "0"
                    index += 1
                if col - 1 >= 0:
                    field[row][col - 1] = "*"
                if col + len(word) + 1 <= self.grid_size:
                    field[row][col + len(word)] = "*"
                break
            cnt += 1
        else:
            self._place_word_vertically(grid, words, field)
            return
        grid.append([row, col])
        grid.append('h')

    def _place_word_vertically(self, grid, word, field):
        cnt = 0
        while cnt != 400:
            row = random.randint(0, self.grid_size - len(word))
            col = random.randint(0, self.grid_size - len(word))
            flag = True
            for i in range(row, row + len(word)):
                if field[i][col] != '0' and field[i][col] != ' ':
                    flag = False
                    break
            if flag:
                index = 0
                for i in range(row, row + len(word)):
                    field[i][col] = word[index]
                    if col - 1 >= 0:
                        field[i][col - 1] = "1"
                    if col + 1 <= self.grid_size:
                        field[i][col + 1] = "1"
                    index += 1
                if row - 1 >= 0:
                    field[row - 1][col] = "*"
                if row + len(word) + 1 <= self.grid_size:
                      field[row + len(word)][col] = "*"
                break
            cnt += 1
        else:
            self._place_word_horizontally(grid, words, field)
            return
        grid.append([row, col])
        grid.append('h')

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
                        else:
                            intersections -= 4
                elif orientation == 'v' and other_orientation == 'h':
                    intersect = (other_start[0], start[1])
                    if intersect[1] >= other_start[1] and intersect[1] < other_start[1] + len(other_word) and intersect[
                        0] >= start[0] and intersect[0] < start[0] + len(word):
                        # intersections.append((intersect, word[intersect[0] - start[0]], other_word[intersect[1] - other_start[1]]))
                        if word[intersect[0] - start[0]] == other_word[intersect[1] - other_start[1]]:
                            intersections += 1
                        else:
                            intersections -= 4
        temp_grid.append(intersections)
        return temp_grid

    def _crossover(self, parent1, parent2):

        # Randomly select a crossover point
        crossover_point = random.randint(1, len(self.words))

        # Perform crossover by swapping the rows at the crossover point
        offspring = parent1[:crossover_point] + parent2[crossover_point:]

        return offspring

    def _mutate(self, individual):
        temp_words = self.words

        if random.random() < self.mutation_rate:
            # With probability self.mutation_rate, perform mutation
            if random.choice([True, False]):
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

            else:
                # Randomly change coordinates of a word
                word = random.choice(temp_words)
                index = temp_words.index(word)

                if individual[index][2] == 'h':
                    new_start_col = random.randint(0, self.grid_size - len(word))
                    individual[index][1] = (individual[index][1][0], new_start_col)
                elif individual[index][2] == 'v':
                    new_start_row = random.randint(0, self.grid_size - len(word))
                    individual[index][1] = (new_start_row, individual[index][1][1])

                # Randomly change the orientation of the word at this position
                individual[index][2] = 'v' if individual[index][2] == 'h' else 'h'

        return individual

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


words = ['apple', 'banana', 'cherry', 'date', 'elderberry']
# words = ['zoo', 'goal', 'owl', 'as']

crossword_generator = CrosswordGenerator(words)
crossword = crossword_generator.generate_crossword()
