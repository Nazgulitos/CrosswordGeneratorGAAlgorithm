# Crossword Generation using Genetic Algorithm(GA)

### Genetic Algorithm Flow:

1. **Initialization:** The algorithm begins by randomly generating a population, where each individual represents a list of words with their starting coordinates and orientations on the crossword grid.

2. **Fitness Evaluation:** Fitness of each crossword is evaluated using a fitness function. This function checks for proper word placement, grid connectivity, and penalizes incorrect placements.

3. **Parent Selection:** Two parents are selected from the population based on their fitness scores. Individuals with the highest fitness scores are chosen as parents.

4. **Crossover:** New offspring is created by randomly selecting words and their parameters (coordinates, orientation) from one of the parents.

5. **Mutation:** The offspring undergoes mutation with a certain probability (0.6 in this case), which may involve changing words, coordinates, or orientations.

6. **Replacement:** The new population includes the offspring and individuals from the old population with the highest fitness scores until the population size is reached.

7. **Termination:** The process continues until a stopping criterion is met, such as finding a solution with fitness 0 or reaching a maximum number of iterations.

### Fitness Function:

The fitness function evaluates the quality of a crossword puzzle. It penalizes disconnected grids, improper placements, and crossings of words, guiding the evolutionary process toward better solutions.

### Variation Operators:

1. **Crossover:** Offspring is created by randomly selecting words from two parents while maintaining word order.
2. **Mutation:** With a certain probability, mutation occurs, introducing changes to the crossword grid, including altering words, coordinates, or orientations.

### EA Parameters:

- **Population Size (max_population_size):** 1000
- **Mutation Rate (mutation_rate):** 0.6
- **Grid Size (grid_size):** 20

### Program Behavior:
My rpogram can solve the task for a maximum of 4 words. For 5 or more words, it may get stuck in a local maximum, which I couldn't resolve.



