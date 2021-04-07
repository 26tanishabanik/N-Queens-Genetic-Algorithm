from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def main():
    import random
    import logging
    import numpy as np
    from tqdm.notebook import trange
    class NQueens():
        def __init__(self, n_queens):
            self.n_queens = n_queens   

        def print_board(self, candidate):
            board = np.zeros((self.n_queens, self.n_queens))

            for q_col, q_row in enumerate(candidate):
                board[q_row, q_col] = 1

            return board

        def check_attack(self, row, col, candidate):
            # Check if this queen attacks other queens in the candidate
            for q_col, q_row in enumerate(candidate):
                if q_col == col and q_row == row:
                    # skip if its the same queen we are checking
                    continue
                else:
                    # Check if this queen shares a row or column
                    if q_col == col or q_row == row:
                        return True
                    # Check if this queen shares a diagnol
                    elif abs(row - q_row) == abs(col - q_col):
                        return True
            else:
                return False

        def check_single_attack(self, row, col, q_row, q_col):
            # Check if this queen shares a row or column
            if q_col == col or q_row == row:
                return True
            # Check if this queen shares a diagnol
            elif abs(row - q_row) == abs(col - q_col):
                return True
            else:
                return False

        def check_conflicts(self, row, col, candidate):
            conflicts = 0

            for q_col, q_row in enumerate(candidate):
                if q_col == col and q_row == row:
                    # skip if its the same queen we are checking
                    continues
                if self.check_single_attack(row, col, q_row, q_col):
                    conflicts += 1

            return conflicts
    from time import time
    from copy import deepcopy        
    class HybridGA(NQueens):
        def __init__(self, n_queens=4, pop_size=10):
            super().__init__(n_queens)
            self.pop_size = pop_size
            self.population = []

        def greedy_init(self):
            self.population = []
            for c in range(self.pop_size):

                # Find a greedy candidate
                candidate = []
                for col in range(self.n_queens):
                    if col == 0:
                        candidate.append(0)
                    else:
                        conflicts = []
                        # Get the conflicts of each queen with previously placed queens
                        for row in range(self.n_queens):
                            conflict = self.check_conflicts(row, col, candidate)
                            conflicts.append(conflict)

                        # Get the least conflicting rows
                        least_conflicts = [i for i, x in enumerate(conflicts) if x == min(conflicts)]
                        # Randomly select and append
                        candidate.append(random.choice(least_conflicts))

                # Append to population
                self.population.append(candidate)

        def decimal_to_binary(self, num):
            return bin(num).replace("0b", "")

        def binary_to_decimal(self, num):
            return int(num, 2)

        def fitness(self, candidate):
            fitness = 0

            for q_col, q_row in enumerate(candidate):
                if not self.check_attack(q_row, q_col, candidate):
                    fitness += 1

            return fitness

        def crossover(self, parentA, parentB, not_crossover_prob=0.3):
            if random.random() < not_crossover_prob:
                return random.choice([parentA, parentB])

            children = []
            fitnesses = []

            for split in range(1, len(parentA) - 1):
                # Get 2 children candidates
                childA = parentA[:split] + parentB[split:]
                childB = parentB[:split] + parentA[split:]
                # Append children candidates
                children.append(childA)
                children.append(childB)
                # Get fitness
                fitnesses.append(self.fitness(childA))
                fitnesses.append(self.fitness(childB))

            # Return the best child
            return children[np.array(fitnesses).argmax()]

        def mutate(self, child, mutation_prob=0.2):
            if random.random() < mutation_prob:
                # Randomly select and mutate a queen
                idx = random.randint(0, self.n_queens - 1)
                child[idx] = random.randint(0, self.n_queens - 1)

                return child
            else:
                return child

        def min_conflict(self, candidate):
            col = random.randint(0, self.n_queens - 1)

            potentials = []
            fitnesses = []
            for row in range(self.n_queens):
                temp = deepcopy(candidate)
                temp[col] = row
                potentials.append(temp)
                fitnesses.append(self.fitness(temp))

            return potentials[np.array(fitnesses).argmax()]


        def evolve(self, timeout=2000):
            # Init population
            self.greedy_init()

            best = {'fitness': 0, 'solution': list(range(self.n_queens))}
            gen = 0

            while(1):
                # Evaluate fitness of population
                fitnesses = np.array(list(map(self.fitness, self.population)))
                weights = [fitness / sum(fitnesses) for fitness in fitnesses]

                # Get best candidate
                best_idx = fitnesses.argsort()[::-1][0]
                best_candidate = self.population[best_idx]
                best_fitness = fitnesses[best_idx]

                # Check the best candidate
                if best_fitness > best['fitness']:
                    best['fitness'] = best_fitness
                    best['solution'] = best_candidate
                    print('Current Best:', best)

                # Check if solution is found
                if best['fitness'] == self.n_queens:
                    print('Found the Solution')
                    print(self.print_board(best['solution']))
                    slt=self.print_board(best['solution'])
                    return best,slt

                new_population = []
                for p in range(self.pop_size):
                    # Select parents
                    parentA, parentB = random.choices(self.population, weights=weights, k=2)
                    # Crossover, mutation and min-conflict
                    child = self.crossover(parentA, parentB)
                    child = self.mutate(child)
                    final_child = self.min_conflict(child)

                    new_population.append(final_child)

                # Append best to next gen population
                new_population.append(best_candidate)

                self.population = new_population

                gen += 1

                if gen > timeout:
                    print('Current Best:', best)
                    print('Solution Not Found, Please Retry')
                    return best  
    ob1=HybridGA()
    slt=ob1.evolve()
    print(slt[1])
    # n_queens=6
    # solver = HybridGA(n_queens)
    # solver.evolve(timeout=500)  

    return render_template('index.html',slt=slt[1])

@app.route('/custom', methods=['POST'])
def custom():
 if request.method == 'POST':
    import random
    import logging
    import numpy as np
    from tqdm.notebook import trange
    class NQueens():
        def __init__(self, n_queens):
            self.n_queens = n_queens   

        def print_board(self, candidate):
            board = np.zeros((self.n_queens, self.n_queens))

            for q_col, q_row in enumerate(candidate):
                board[q_row, q_col] = 1

            return board

        def check_attack(self, row, col, candidate):
            # Check if this queen attacks other queens in the candidate
            for q_col, q_row in enumerate(candidate):
                if q_col == col and q_row == row:
                    # skip if its the same queen we are checking
                    continue
                else:
                    # Check if this queen shares a row or column
                    if q_col == col or q_row == row:
                        return True
                    # Check if this queen shares a diagnol
                    elif abs(row - q_row) == abs(col - q_col):
                        return True
            else:
                return False

        def check_single_attack(self, row, col, q_row, q_col):
            # Check if this queen shares a row or column
            if q_col == col or q_row == row:
                return True
            # Check if this queen shares a diagnol
            elif abs(row - q_row) == abs(col - q_col):
                return True
            else:
                return False

        def check_conflicts(self, row, col, candidate):
            conflicts = 0

            for q_col, q_row in enumerate(candidate):
                if q_col == col and q_row == row:
                    # skip if its the same queen we are checking
                    continues
                if self.check_single_attack(row, col, q_row, q_col):
                    conflicts += 1

            return conflicts
    from time import time
    from copy import deepcopy        
    class HybridGA(NQueens):
        def __init__(self, n_queens=int(request.form['n_queen']), pop_size=10):
            super().__init__(n_queens)
            self.pop_size = pop_size
            self.population = []

        def greedy_init(self):
            self.population = []
            for c in range(self.pop_size):

                # Find a greedy candidate
                candidate = []
                for col in range(self.n_queens):
                    if col == 0:
                        candidate.append(0)
                    else:
                        conflicts = []
                        # Get the conflicts of each queen with previously placed queens
                        for row in range(self.n_queens):
                            conflict = self.check_conflicts(row, col, candidate)
                            conflicts.append(conflict)

                        # Get the least conflicting rows
                        least_conflicts = [i for i, x in enumerate(conflicts) if x == min(conflicts)]
                        # Randomly select and append
                        candidate.append(random.choice(least_conflicts))

                # Append to population
                self.population.append(candidate)

        def decimal_to_binary(self, num):
            return bin(num).replace("0b", "")

        def binary_to_decimal(self, num):
            return int(num, 2)

        def fitness(self, candidate):
            fitness = 0

            for q_col, q_row in enumerate(candidate):
                if not self.check_attack(q_row, q_col, candidate):
                    fitness += 1

            return fitness

        def crossover(self, parentA, parentB, not_crossover_prob=0.3):
            if random.random() < not_crossover_prob:
                return random.choice([parentA, parentB])

            children = []
            fitnesses = []

            for split in range(1, len(parentA) - 1):
                # Get 2 children candidates
                childA = parentA[:split] + parentB[split:]
                childB = parentB[:split] + parentA[split:]
                # Append children candidates
                children.append(childA)
                children.append(childB)
                # Get fitness
                fitnesses.append(self.fitness(childA))
                fitnesses.append(self.fitness(childB))

            # Return the best child
            return children[np.array(fitnesses).argmax()]

        def mutate(self, child, mutation_prob=0.2):
            if random.random() < mutation_prob:
                # Randomly select and mutate a queen
                idx = random.randint(0, self.n_queens - 1)
                child[idx] = random.randint(0, self.n_queens - 1)

                return child
            else:
                return child

        def min_conflict(self, candidate):
            col = random.randint(0, self.n_queens - 1)

            potentials = []
            fitnesses = []
            for row in range(self.n_queens):
                temp = deepcopy(candidate)
                temp[col] = row
                potentials.append(temp)
                fitnesses.append(self.fitness(temp))

            return potentials[np.array(fitnesses).argmax()]


        def evolve(self, timeout=2000):
            # Init population
            self.greedy_init()

            best = {'fitness': 0, 'solution': list(range(self.n_queens))}
            gen = 0

            while(1):
                # Evaluate fitness of population
                fitnesses = np.array(list(map(self.fitness, self.population)))
                weights = [fitness / sum(fitnesses) for fitness in fitnesses]

                # Get best candidate
                best_idx = fitnesses.argsort()[::-1][0]
                best_candidate = self.population[best_idx]
                best_fitness = fitnesses[best_idx]

                # Check the best candidate
                if best_fitness > best['fitness']:
                    best['fitness'] = best_fitness
                    best['solution'] = best_candidate
                    print('Current Best:', best)

                # Check if solution is found
                if best['fitness'] == self.n_queens:
                    print('Found the Solution')
                    print(self.print_board(best['solution']))
                    slt=self.print_board(best['solution'])
                    return best,slt

                new_population = []
                for p in range(self.pop_size):
                    # Select parents
                    parentA, parentB = random.choices(self.population, weights=weights, k=2)
                    # Crossover, mutation and min-conflict
                    child = self.crossover(parentA, parentB)
                    child = self.mutate(child)
                    final_child = self.min_conflict(child)

                    new_population.append(final_child)

                # Append best to next gen population
                new_population.append(best_candidate)

                self.population = new_population

                gen += 1

                if gen > timeout:
                    print('Current Best:', best)
                    print('Solution Not Found, Please Retry')
                    return best  
    ob1=HybridGA()
    slt=ob1.evolve()
    print(slt[1])
    

    return render_template('index.html',slt=slt[1])



if __name__ == "__main__":
    app.run()
                          
    


