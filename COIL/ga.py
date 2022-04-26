#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

import importlib
import argparse
import random
import pickle
import numpy as np
import time, os

from deap import base
from deap import creator
from deap import tools

from matplotlib import pyplot as plt

# directories
RESULTS_DIRECTORY = 'results/'
IMAGE_DIRECTORY = 'image/'

# image settings
IMAGE_TYPE = 'pdf'
TITLE_FONT_SIZE = 18
LEGEND_FONT_SIZE = 14
AXIS_FONT_SIZE = 14
#----------
# Get command line arguments
#----------
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--equation', help='equation name should correspond to the equation file, e.g., enter eq01 to import the equation file eq01.py')
parser.add_argument('-v', '--num_variables', help='enter the number of variables')
parser.add_argument('-r', '--runs', help='enter the number of runs. if the argument is not provided, it defaults to 1 run')
parser.add_argument('-s', '--seed', help='enter -1 for a random seed, or enter the seed number. if the argument is not provided, it defaults to -1.')
parser.add_argument('-d', '--debug', action='store_true', help='if argument is used, generate debug info.')
parser.add_argument('-i', '--image', action='store_true', help='if argument is used, a GA image is generated for each run.')
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
GA_NUM_INDIVIDUALS = NUM_VARIABLES

ANSWER = eq.calculate_optimal(NUM_VARIABLES)

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
# Set number of runs
#----------
if args.runs:
    NUM_RUNS = int(args.runs)
else:
    NUM_RUNS = 1

DEBUG = args.debug
GENERATE_IMAGE = args.image
CALCULATE_RUNTIME = args.time

#----------
# Start set up DEAP
#----------
# CXPB  is the probability with which two individuals are crossed
# MUTPB is the probability for mutating an individual
CXPB, MUTPB = 0.5, 0.2

GA_MIN_RANGE = -1.0
GA_MAX_RANGE = 1.0

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute generator 
#                      define 'attr_bool' to be an attribute ('gene')
#                      which corresponds to real numbers sampled uniformly
#                      from the specified range
toolbox.register("attr_float", random.uniform, GA_MIN_RANGE, GA_MAX_RANGE)

# Structure initializers
#                         define 'individual' to be an individual
#                         consisting of floats
toolbox.register("individual", tools.initRepeat, creator.Individual, 
    toolbox.attr_float, GA_NUM_INDIVIDUALS)

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
toolbox.decorate("mate", checkBounds(GA_MIN_RANGE, GA_MAX_RANGE))

# register a mutation operator
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.decorate("mutate", checkBounds(GA_MIN_RANGE, GA_MAX_RANGE))

# operator for selecting individuals for breeding the next
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of three individuals
# drawn randomly from the current generation.
toolbox.register("select", tools.selTournament, tournsize=3)

#----------
# register the goal / fitness function
#----------
# Here we use our own custom developed fitness function

# unnormalise each variable in the individual into their original range
def unnormalise_to_range(individual):
    result = []
    for i in range(NUM_VARIABLES):
        xi_std = (individual[i] - GA_MIN_RANGE)/(GA_MAX_RANGE - GA_MIN_RANGE)
        # xi_scaled = xi_std * (eq.VARIABLE_RANGE[i]['max'] - eq.VARIABLE_RANGE[i]['min']) + eq.VARIABLE_RANGE[i]['min']
        xi_scaled = xi_std * (eq.X_MAX_RANGE - eq.X_MIN_RANGE) + eq.X_MIN_RANGE
        result.append(xi_scaled)
    return result


def calculate_objective_and_constraints(individual):
    unnormalised_item = unnormalise_to_range(individual)
    result = {}
    result['obj'] = eq.func(unnormalised_item, NUM_VARIABLES)
    
    for i, constraint in enumerate(eq.CONSTRAINTS):
        result[i] = constraint(unnormalised_item, NUM_VARIABLES)

    return result


def eval_winner_for_func(duel0, duel1):
    if (eq.MIN_OR_MAX_FLAG == 0): # minimise
        if (duel0.calculated_values['obj'] < duel1.calculated_values['obj']):
            return True
        else:
            return False
    else: # maximise
        if (duel0.calculated_values['obj'] > duel1.calculated_values['obj']):
            return True
        else:
            return False


def play(duel):
    if duel[0].calculated_values['obj'] == duel[1].calculated_values['obj']: 
        duel[0].gathered_score += 1
        duel[1].gathered_score += 1
    elif eval_winner_for_func(duel[0], duel[1]):
        duel[0].gathered_score += 1
    else:
        duel[1].gathered_score += 1

    for i in range(len(eq.CONSTRAINTS)):
        duel0_constraint_performance = duel[0].calculated_values[i]
        duel1_constraint_performance = duel[1].calculated_values[i]

        if (duel0_constraint_performance != 0.0) and (duel1_constraint_performance != 0.0): # both wrong
            if duel0_constraint_performance < duel1_constraint_performance:
                duel[0].gathered_score += 1
            else:
                duel[1].gathered_score += 1
        else:
            if (duel0_constraint_performance == 0.0):
                duel[0].gathered_score += 1

            if (duel1_constraint_performance == 0.0):
                duel[1].gathered_score += 1

    duel[0].num_matches += 1
    duel[1].num_matches += 1


def eval_objective_function(item):
    if (item.num_matches == 0):
        return 0
    else:
        return item.gathered_score/item.num_matches

#----------
# End set up DEAP
#----------

def pure_ga(run):
    if GENERATE_IMAGE:
        min_list = []
        max_list = []
        avg_list = []
        std_list = []
        avg_obj_list = []
        avg_dist_from_constraint = []

    # create an initial population of 200 individuals (where
    # each individual is a list of integers)
    pop = toolbox.population(n=eq.GA_POP)

    if DEBUG:
        print("Start of evolution")

    for ind in pop:
        # calculate objective to produce a list
        ind.calculated_values = calculate_objective_and_constraints(ind)
        ind.gathered_score = 0
        ind.num_matches = 0

    for ind in pop:
        participants = random.sample(pop, 5)
        for t in range(10): # 10 tournaments
            duel = random.sample(participants, 2)
            play(duel)

    for ind in pop:
        ind.fitness.values = (eval_objective_function(ind),)

    if DEBUG:
        print("  Evaluated %i individuals" % len(pop))

    # Extracting all the fitnesses of 
    fits = [ind.fitness.values[0] for ind in pop]

    # Variable keeping track of the number of generations
    g = 0
    
    # Begin the evolution
    while g < eq.GA_GEN:
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

                # fitness values of the children must be recalculated later
                del child1.fitness.values
                del child2.fitness.values


        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
            del mutant.fitness.values # make all invalid

        # Evaluate the individuals with an invalid fitness
        # invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        # calculate objective value and constraints values for each individual
        for ind in offspring:
            ind.calculated_values = calculate_objective_and_constraints(ind)
            ind.gathered_score = 0
            ind.num_matches = 0

        for x in offspring:
            participants = random.sample(offspring, 5)

            for t in range(10): # 10 tournaments
                duel = random.sample(participants, 2)
                play(duel)


        for ind in offspring:
            ind.fitness.values = (eval_objective_function(ind),)


        # print("  Evaluated %i individuals" % len(invalid_ind))
        
        # The population is entirely replaced by the offspring
        pop[:] = offspring
        
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        

        if GENERATE_IMAGE:
            distance_from_answer = []
            distance_from_constraint = []

            for ind in pop:
                unnormalised_item = unnormalise_to_range(ind)
                performance = eq.func(unnormalised_item, NUM_VARIABLES)
                distance_from_answer.append(np.abs(ANSWER - performance))
                for constraint in eq.CONSTRAINTS:
                    distance_from_constraint.append(constraint(unnormalised_item, NUM_VARIABLES)) # add all the distances from all constraints
            
            length = len(pop)
            mean = sum(fits) / length
            sum2 = sum(x*x for x in fits)
            std = abs(sum2 / length - mean**2)**0.5
            
            # print("  Min %s" % min(fits))
            # print("  Max %s" % max(fits))
            # print("  Avg %s" % mean)
            # print("  Std %s" % std)
            # print(a)
            min_list.append(min(fits))
            max_list.append(max(fits))
            avg_list.append(mean)
            avg_obj_list.append(sum(distance_from_answer) / length)
            avg_dist_from_constraint.append(sum(distance_from_constraint) / length)
            std_list.append(std)
        
    if DEBUG:
        print("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]
    if DEBUG:
        print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

    unnormalised_item = unnormalise_to_range(best_ind)

    result = eq.func(unnormalised_item, NUM_VARIABLES)
    
    run_result = {
    'variables': [],
    'distance_from_each_constraint': [],
    'distance_from_constraints': None,
    'objective': result,
    'optimal': ANSWER,
    'distance_from_optimal': np.abs(ANSWER - result),
    }
    
    total_distance_from_constraints = 0.0

    for i, constraint in enumerate(eq.CONSTRAINTS):
        constraint_value = constraint(unnormalised_item, NUM_VARIABLES)
        total_distance_from_constraints += constraint_value
        if DEBUG:
            print('Constraint', i, constraint_value)

        run_result['distance_from_each_constraint'].append(constraint_value)

    run_result['distance_from_constraints'] = total_distance_from_constraints

    if GENERATE_IMAGE:
        axes = plt.gca()
        # axes.set_ylim([-1000000,0])
        # plt.plot(min_list, label='min')
        # plt.plot(max_list, label='max')
        # plt.plot(avg_list, label='avg')
        plt.plot(avg_obj_list, linewidth=3, label='GA', color="#4c72b0")
        plt.title('Average distance from objective', fontsize=TITLE_FONT_SIZE)
        # plt.plot(std_list, label='std')
        plt.xlabel("Generation", fontsize=AXIS_FONT_SIZE)
        plt.ylabel("Distance", fontsize=AXIS_FONT_SIZE)
        plt.legend(fontsize=LEGEND_FONT_SIZE)
        plt.tick_params(labelsize=AXIS_FONT_SIZE)
        if not os.path.exists(IMAGE_DIRECTORY):
            os.makedirs(IMAGE_DIRECTORY)
        plt.savefig(IMAGE_DIRECTORY + equation_name + '_' + str(seed) + '_run' + str(run) + '_ga_obj.' + IMAGE_TYPE, bbox_inches='tight', pad_inches=0)

        plt.clf()
        plt.plot(avg_dist_from_constraint, linewidth=3, label='GA', color="#4c72b0")
        plt.title('Average distance from constraint', fontsize=TITLE_FONT_SIZE)
        # plt.plot(std_list, label='std')
        plt.xlabel("Generation", fontsize=AXIS_FONT_SIZE)
        plt.ylabel("Distance", fontsize=AXIS_FONT_SIZE)
        plt.legend(fontsize=LEGEND_FONT_SIZE)
        plt.tick_params(labelsize=AXIS_FONT_SIZE)
        if not os.path.exists(IMAGE_DIRECTORY):
            os.makedirs(IMAGE_DIRECTORY)
        plt.savefig(IMAGE_DIRECTORY + equation_name + '_' + str(seed) + '_run' + str(run) + '_ga_constraint.' + IMAGE_TYPE, bbox_inches='tight', pad_inches=0)




    if DEBUG:
        print('result \t(' + str(ANSWER) + ') \t', result)

    for i in range(GA_NUM_INDIVIDUALS):
        if DEBUG:
            variable_name = 'x' + str(i)
            print(variable_name + '\t (' + str(eq.X_MIN_RANGE) + ',' + str(eq.X_MAX_RANGE) + ') \t', unnormalised_item[i])

        run_result['variables'].append(unnormalised_item[i])

    return run_result


def main():
    #----------
    # Run optimiser
    #----------
    run_results = []

    if CALCULATE_RUNTIME:
        start = time.time()
    
    for run in range(NUM_RUNS):
        run_result = pure_ga(run)
        run_results.append(run_result)
    
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


    #----------
    # Save results
    #----------
    if not os.path.exists(RESULTS_DIRECTORY):
      os.makedirs(RESULTS_DIRECTORY)
    results_file = RESULTS_DIRECTORY + equation_name + '_v' + str(NUM_VARIABLES) + '_ga' + '.pkl'    
    pickle.dump([seed, run_results], open(results_file, 'wb'))
    print("Seed and results saved to", results_file)


    #----------
    # Print stats
    #----------
    run_results_objective_error = []
    run_results_constraint_error = []
    for run in range(NUM_RUNS):
        run_results_objective_error.append(run_results[run]['distance_from_optimal'])
        run_results_constraint_error.append(run_results[run]['distance_from_constraints'])
    print("mean objective error", np.mean(np.array(run_results_objective_error), axis=0))
    print("std objective error", np.std(np.array(run_results_objective_error), axis=0))
    print("mean constraints error", np.mean(np.array(run_results_constraint_error), axis=0))
    print("std constraints error", np.std(np.array(run_results_constraint_error), axis=0))




if __name__ == "__main__":
    main()