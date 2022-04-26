#    Generates data manually for the VAE for Constraint 2

import random
import argparse
import importlib
import pickle
import time, os
import numpy as np

# globals
CONSTRAINT_ID: str
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


def unnormalise_to_range(individual):
    result = []
    for i in range(NUM_VARIABLES):
        # xi_std = (individual[i] - eq.VARIABLE_RANGE[i]['min'])/(eq.VARIABLE_RANGE[i]['max'] - eq.VARIABLE_RANGE[i]['min'])
        xi_std = (individual[i] - eq.X_MIN_RANGE)/(eq.X_MAX_RANGE - eq.X_MIN_RANGE)
        xi_scaled = xi_std * (eq.DATAGEN_MAX_RANGE  - eq.DATAGEN_MIN_RANGE) + eq.DATAGEN_MIN_RANGE
        result.append(xi_scaled)
    return result


def data_point_generator():
    data_point = []
    random_value = random.uniform(eq.X_MIN_RANGE, eq.X_MAX_RANGE)

    if (NUM_VARIABLES % 2) == 0: # even
        random_value_1 = random_value + random.uniform(eq.LOWER_BOUND, eq.UPPER_BOUND)
        data_point = [random_value, random_value_1]
        count = 2
    else: # odd
        random_value_0 = random_value - random.uniform(eq.LOWER_BOUND, eq.UPPER_BOUND)
        random_value_1 = random_value + random.uniform(eq.LOWER_BOUND, eq.UPPER_BOUND)
        data_point = [random_value_0, random_value, random_value_1]
        count = 3

    while count < NUM_VARIABLES:
        data_point = [data_point[0] - random.uniform(eq.LOWER_BOUND, eq.UPPER_BOUND)] + data_point # prepend val - rand() to the list
        data_point.append(data_point[-1] + random.uniform(eq.LOWER_BOUND, eq.UPPER_BOUND)) # append val + rand() to the list
        count += 2

    if (data_point[0] >= eq.X_MIN_RANGE) and (data_point[-1] <= eq.X_MAX_RANGE):
        if DEBUG:
            print(data_point)
        return unnormalise_to_range(data_point)
    else:
        return None


def generate_data():
    print("Generating data for constraint", CONSTRAINT_ID)

    total = 0
    data = []
    while total < eq.NUM_DATA_POINTS:
        valid_data = data_point_generator()
        if DEBUG:
            print(valid_data)
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