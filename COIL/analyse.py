import numpy as np
import argparse
import importlib
import os, time
import pickle
import matplotlib.pyplot as plt

RESULTS_DIRECTORY = 'results/'
IMAGE_DIRECTORY = 'image/'

image_format = 'pdf' # png

# if the directory to save image does not exist, create it
if not os.path.exists(IMAGE_DIRECTORY):
    os.makedirs(IMAGE_DIRECTORY)

#----------
# Get command line arguments
#----------
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--equation', help='equation name should correspond to the equation file, e.g., enter eq01 to import the equation file eq01.py')

args = parser.parse_args()

#----------
# Import equation module
#----------
if args.equation:
    equation_name = args.equation
    eq = importlib.__import__(equation_name) # import equation module
else:
    exit("Error. Please specify equation name in the command line. Use --help for more information.")


MIN_VARIABLES = eq.MIN_VARIABLES

if equation_name == 'c1':
    MAX_VARIABLES = 10
if equation_name == 'c2':
    MAX_VARIABLES = 10

NUM_RUNS = 100
variable_list = []


for i in range(MIN_VARIABLES, MAX_VARIABLES+1):
    variable_list.append(str(i))


def get_results(experiment):
    mean_objective_error = []
    mean_constraint_error = []
    stderr_objective_error = []
    stderr_constraint_error = []
    # if experiment == 'vaega':
        # experiment = 'quickvaega'
    for variable in variable_list:
        data_file = RESULTS_DIRECTORY + equation_name + '_v' + variable + '_' + experiment + '.pkl'

        optimise_seed, run_results = pickle.load(open(data_file, 'rb'))

        run_results_objective_error = []
        run_results_constraint_error = []

        for run in range(NUM_RUNS):
            # print('variables', run_results[run]['variables'])
            # print(run_results[run]['distance_from_optimal'])
            run_results_objective_error.append(run_results[run]['distance_from_optimal']/run_results[run]['optimal']*100.0)
            run_results_constraint_error.append(run_results[run]['distance_from_constraints']/float(variable))

        mean_objective_error.append(np.mean(np.array(run_results_objective_error), axis=0))
        stderr_objective_error.append(np.std(np.array(run_results_objective_error), axis=0)/np.sqrt(NUM_RUNS))

        mean_constraint_error.append(np.mean(np.array(run_results_constraint_error), axis=0))
        stderr_constraint_error.append(np.std(np.array(run_results_constraint_error), axis=0)/np.sqrt(NUM_RUNS))
    return mean_objective_error, stderr_objective_error, mean_constraint_error, stderr_constraint_error


def draw_image(dataset1, dataset1_stderr, dataset1_name, dataset2, dataset2_stderr, dataset2_name, xlabel, ylabel, title, imagename):
    error_kw = dict(lw=0.8, capsize=3, capthick=0.8)
    x = np.arange(len(variable_list))  # the label locations
    width = 0.4  # the width of the bars
    fig, ax = plt.subplots()
    plt.tick_params(labelsize=14)
    rects1 = ax.bar(x - width/2, dataset1, width, yerr=dataset1_stderr, error_kw=error_kw, label=dataset1_name, color="#4c72b0")
    rects2 = ax.bar(x + width/2, dataset2, width, yerr=dataset2_stderr, error_kw=error_kw, label=dataset2_name, color="#dd8452")

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_title(title, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xticks(list(range(len(variable_list))))
    ax.set_xticklabels(variable_list)
    # ax.set_xticks(np.arange(min(x),max(x),1))
    ax.legend(fontsize=14)

    # ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    # plt.show()

    plt.savefig(IMAGE_DIRECTORY + equation_name + '_' + imagename, dpi=72, bbox_inches='tight', pad_inches=0)



vaega_mean_objective_error, vaega_stderr_objective_error, vaega_mean_constraint_error, vaega_stderr_constraint_error = get_results('coil')
ga_mean_objective_error, ga_stderr_objective_error, ga_mean_constraint_error, ga_stderr_constraint_error = get_results('ga')

coil = vaega_mean_objective_error
coil_stderr = vaega_stderr_objective_error
coil_name = 'COIL'
ga = ga_mean_objective_error
ga_stderr = ga_stderr_objective_error
ga_name = 'GA'
xlabel = 'Number of variables'
ylabel = 'Average percentage error'
title = equation_name.upper() + ': Average percentage objective error'
imagename = 'objective_error.' + image_format
draw_image(ga, ga_stderr, ga_name, coil, coil_stderr, coil_name, xlabel, ylabel, title, imagename)

coil = vaega_mean_constraint_error
coil_stderr = vaega_stderr_constraint_error
coil_name = 'COIL'
ga = ga_mean_constraint_error
ga_stderr = ga_stderr_constraint_error
ga_name = 'GA'
xlabel = 'Number of variables'
ylabel = 'Average constraint error'
title = equation_name.upper() + ': Average constraint error (per variable)'
imagename = 'constraint_error.' + image_format
draw_image(ga, ga_stderr, ga_name, coil, coil_stderr, coil_name, xlabel, ylabel, title, imagename)

