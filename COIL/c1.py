# generate data: specify number of data points generated
NUM_DATA_POINTS = 5000

# the minimum number of variables required for the equation to work
MIN_VARIABLES = 1

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
def c0(x, num_var):
    # the constraint is: (45 - x[0]) + (45 - x[1]) + ... + (45 - x[n]) <= 0
    result = 45.0 * num_var
    for i in range(num_var):
        result -= x[i]

    if result <= 0.0:
        return 0.0
    else:
        return result

# calculate the optimal result for the function for the constraint(s) to be met
def calculate_optimal(num_var):
    X_VAL_AT_OPTIMAL = 45.0
    return X_VAL_AT_OPTIMAL * X_VAL_AT_OPTIMAL * num_var

# list of constraints: add specified constraints to this list in order for them to be considered
CONSTRAINTS = [
c0,
]

# generate data: specify num gen and num pop for the data generator GA
DATAGEN_GEN = 200 #500
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


