#    Generates data for the VAE
#    Uses DEAP https://deap.readthedocs.io/en/master/index.html

import random
import argparse
import importlib
import pickle
import time, os
import numpy as np

from deap import base
from deap import creator
from deap import tools

# global
CONSTRAINT_ID: int
DATA_DIRECTORY = 'data/' # directory storing the output data

#----------
# Get command line arguments
#----------
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--equation', help='equation name should correspond to the equation file, e.g., enter eq01 to import the equation file eq01.py')
parser.add_argument('-v', '--num_variables', help='enter the number of variables')
parser.add_argument('-s', '--seed', help='enter -1 for a random seed, or enter the seed number. if the argument is not provided, it defaults to -1.')
parser.add_argument('-d', '--debug', action='store_true', help='if argument is used, generate debug info.')
parser.add_argument('-t', '--time', action='store_true', help='if argument is used, calculate run time.')

args = parser.parse_args()

#----------
# Import equation module
#----------
if args.equation:
    equation_name = args.equation
    eq = importlib.__import__(equation_name) # import equation module
else:
    exit("Error. Please specify equation name in the command line. Use --help for more information.")

#----------
# Get number of variables
#----------
if args.num_variables:
    NUM_VARIABLES = int(args.num_variables)
    if NUM_VARIABLES < eq.MIN_VARIABLES:
        exit("Error. Minimum number of variables for this function is %d. Use --help for more information." % eq.MIN_VARIABLES)
else:
    NUM_VARIABLES = eq.MIN_VARIABLES

#----------
# Set seed
#----------
if not args.seed or int(args.seed) < 0: # if args.seed is not provided or is negative
    seed = int(time.time()) # use current time as random seed
else:
    seed = int(args.seed)
print('Seed', seed)
random.seed(seed)

#----------
# Set flags
#----------
DEBUG = args.debug
CALCULATE_RUNTIME = args.time

#----------
# Start set up DEAP
#----------
# CXPB  is the probability with which two individuals are crossed
# MUTPB is the probability for mutating an individual
CXPB, MUTPB = 0.5, 0.2

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Attribute generator 
#                      define 'attr_bool' to be an attribute ('gene')
#                      which corresponds to real numbers sampled uniformly
#                      from the specified range
toolbox.register("attr_float", random.uniform, eq.DATAGEN_MIN_RANGE, eq.DATAGEN_MAX_RANGE)

# Structure initializers
#                         define 'individual' to be an individual
#                         consisting of floats
toolbox.register("individual", tools.initRepeat, creator.Individual, 
    toolbox.attr_float, NUM_VARIABLES)

# define the population to be a list of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)  

#----------
# Operator registration
#----------
def checkBounds(min, max):
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                for i in range(len(child)):
                    if child[i] > max:
                        child[i] = max
                    elif child[i] < min:
                        child[i] = min
            return offspring
        return wrapper
    return decorator

# register the crossover operator
toolbox.register("mate", tools.cxUniform, indpb=0.05)
toolbox.decorate("mate", checkBounds(eq.DATAGEN_MIN_RANGE, eq.DATAGEN_MAX_RANGE))

# register a mutation operator
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.decorate("mutate", checkBounds(eq.DATAGEN_MIN_RANGE, eq.DATAGEN_MAX_RANGE))

# operator for selecting individuals for breeding the next
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of three individuals
# drawn randomly from the current generation.
toolbox.register("select", tools.selTournament, tournsize=3)

#----------
# register the goal / fitness function
#----------
# unnormalise each variable in the individual into their original range
def unnormalise_to_range(individual):
    result = []
    for i in range(NUM_VARIABLES):
        xi_std = (individual[i] - eq.DATAGEN_MIN_RANGE)/(eq.DATAGEN_MAX_RANGE - eq.DATAGEN_MIN_RANGE)
        # xi_scaled = xi_std * (eq.VARIABLE_RANGE[i]['max'] - eq.VARIABLE_RANGE[i]['min']) + eq.VARIABLE_RANGE[i]['min']
        xi_scaled = xi_std * (eq.X_MAX_RANGE - eq.X_MIN_RANGE) + eq.X_MIN_RANGE
        result.append(xi_scaled)
    return result


def evaluate_constraint(individual):
    unnormalised_individual = unnormalise_to_range(individual)
    constraint_value = eq.CONSTRAINTS[CONSTRAINT_ID](unnormalised_individual, NUM_VARIABLES)
    return constraint_value,


toolbox.register("evaluate", evaluate_constraint)

#----------
# End set up DEAP
#----------

def data_generator_ga():

    min_list = []
    max_list = []
    avg_list = []
    std_list = []

    # create an initial population of individuals
    pop = toolbox.population(n=eq.DATAGEN_POP)
    
    if DEBUG:
        print("Start of evolution")
    
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    if DEBUG:
        print("  Evaluated %i individuals" % len(pop))

    # Extracting all the fitnesses of 
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0
    
    # Begin the evolution
    while g < eq.DATAGEN_GEN:
        # A new generation
        g = g + 1

        if DEBUG:
            if (g % 100) == 0:
                print("-- Generation %i --" % g)
        
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
    
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
    
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # print("  Evaluated %i individuals" % len(invalid_ind))
        
        # The population is entirely replaced by the offspring
        pop[:] = offspring
        
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        
        # print("  Min %s" % min(fits))
        # print("  Max %s" % max(fits))
        # print("  Avg %s" % mean)
        # print("  Std %s" % std)
        min_list.append(min(fits))
        max_list.append(max(fits))
        avg_list.append(mean)
        std_list.append(std)

        best_ind_for_gen = tools.selBest(pop, 1)[0]
        fitness_in_gen = best_ind_for_gen.fitness.values[0]
        # print(fitness_in_gen)

        if fitness_in_gen == 0.0:
            break
    
    if DEBUG:
        print("-- End of (successful) evolution --")
    
    best_ind = tools.selBest(pop, 1)[0]

    if DEBUG:
        print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))


    unnormalised_item = unnormalise_to_range(best_ind)
    # print(unnormalised_item)
    # print("Converted to actual values %s", unnormalised_item)

    # plt.plot(min_list, label='min')
    # plt.plot(max_list, label='max')
    # plt.plot(avg_list, label='avg')
    # plt.plot(std_list, label='std')
    # plt.xlabel("Generation")
    # plt.ylabel("Fitness")
    # plt.legend()
    # plt.savefig('output_ga/' + str(seed) + '_ga.png', dpi=72, bbox_inches='tight', pad_inches=0)

    # only return an individual if it meets the constraints

    if (eq.CONSTRAINTS[CONSTRAINT_ID](unnormalised_item, NUM_VARIABLES) == 0.0): # if constraint is met
        return best_ind
    else:
        return None


def generate_data():
    print("Generating data for constraint", CONSTRAINT_ID)

    total = 0
    data = []
    while total < eq.NUM_DATA_POINTS:
        valid_data = data_generator_ga()
        if valid_data:
            data.append(list(valid_data)) # VAE takes in a 2D array
            total = total + 1
            if total % 1000 == 0:
                print('Data points generated: %d out of %d' % (total, eq.NUM_DATA_POINTS))

    current_file = DATA_DIRECTORY + equation_name + '_v' + str(NUM_VARIABLES) + '_constraint' + str(CONSTRAINT_ID) + '.pkl'
    pickle.dump([seed, data], open(current_file, 'wb'))
    print("Seed and data saved to", current_file)


def main():
    global CONSTRAINT_ID

    # if the directory to save data does not exist, create it
    if not os.path.exists(DATA_DIRECTORY):
        os.makedirs(DATA_DIRECTORY)

    #----------
    # Run optimiser
    #----------
    for i in range(len(eq.CONSTRAINTS)):
        CONSTRAINT_ID = i
        
        if CALCULATE_RUNTIME:
            start = time.time()

        generate_data()
        
        if CALCULATE_RUNTIME:
            end = time.time()
            total_time = end - start
            if total_time < 60.0:
                unit = "seconds"
            elif total_time < 3600.0:
                total_time = total_time/60.0
                unit = "minutes"
            else:
                total_time = total_time/3600.0
                unit = "hours"
            print("Run time %.2lf " % total_time + unit)
    

if __name__ == "__main__":
    main()