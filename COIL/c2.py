# generate data: specify number of data points generated
NUM_DATA_POINTS = 5000

# the minimum number of variables required for the equation to work
MIN_VARIABLES = 2

# specify objective function
def func(x, num_var):
    result = 0.0
    for i in range(num_var):
        result += x[i]**2
    return result

# specify whether to minimise or maximise function, 0 for min 1 for max
MIN_OR_MAX_FLAG = 0

# set the min and max range for the variables
X_MIN_RANGE = -50.0
X_MAX_RANGE = 50.0

# specify constraints (return 0 if constraint is met, otherwise return absolute distance)
LOWER_BOUND = 8.0
UPPER_BOUND = 10.0
def c0_calculator(x1, x2):
    result = x2 - x1
    if (result <= UPPER_BOUND) and (result >= LOWER_BOUND): # inside bounds
        return 0.0
    else:
        if result < LOWER_BOUND:
            distance = result - LOWER_BOUND
        else:
            distance = result - UPPER_BOUND
        
        if distance >= 0: # always return a positive distance
            return distance
        else:
            return (-distance)

def c0(x, num_var):
    result = 0.0
    less_one_variable = num_var - 1
    for i in range(less_one_variable):
        result += c0_calculator(x[i], x[i+1])
    return result

# list of constraints: add specified constraints to this list in order for them to be considered
CONSTRAINTS = [
c0,
]

# calculate the optimal result for the function for the constraint(s) to be met
def calculate_optimal(num_var):
    total_list = []
    negative_list = []
    positive_list = []
    if (num_var % 2) == 0: # even
        total_list = [-4.0, 4.0]
        count = 2
    else: # odd (the smallest odd number of variables for this function is 3)
        total_list = [-8.0, 0.0, 8.0]
        count = 3

    while count < num_var:
        total_list = [total_list[0] - LOWER_BOUND] + total_list # prepend val - 8 to the list
        total_list.append(total_list[-1] + LOWER_BOUND) # append val + 8 to the list
        count += 2
        
    result = 0.0
    for item in total_list:
        result += (item)**2
    return result

# Note for c2, we use generate_data_c2.py instead of GA data generator
# generate data: specify num gen and num pop for the data generator GA
DATAGEN_GEN = 500
DATAGEN_POP = 200

# generate data: specify min and max range for data
DATAGEN_MIN_RANGE = -1.0
DATAGEN_MAX_RANGE = 1.0

# learn representation: specify the number of latent variables and epochs for the vae
# NUM_LATENT = NUM_VARIABLES
NUM_EPOCHS = 200

# optimise: specify num gen and num pop for the optimiser GA
VAEGA_GEN = 50
VAEGA_POP = 20

# optimse: the range for the GA to generate random numbers for the latent variable
VAEGA_MIN_RANGE = -2.0
VAEGA_MAX_RANGE = 2.0

# comparison GA: specify num gen and num pop for the GA
# GA_NUM_INDIVIDUALS = NUM_VARIABLES # the number of individuals for the GA is the number of variables
GA_GEN = 50
GA_POP = 20
