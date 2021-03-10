import sys
import numpy as np
import os.path

# ----- Set the paths -------------------------------- #
if os.path.exists("/home/guillaume/Work/2018-07-NILM/nilm"):
    #path2load = '/home/guillaume/Work/2018-07-NILM/nilm/UK_Dale/house_1/'
    path2store = '/home/guillaume/Work/2018-07-NILM/nilm/_store_eco/'

else:
    #path2load = '/home/guillaume/Work/2018-07-NILM/nilm/UK_Dale/house_1/'
    path2store = '/home/guillaume/Work/2018-07-NILM/nilm/_store_eco/'

# ----- Data preprocess ------------------------------ #

ll_select = ['fridge', 'washing_machine', 'microwave',
             'dishwasher', 'kettle']

sampling_rate = '5T'
samples_per_min = 1/5

# ----- OMP ------------------------------------------ #
omp_tolerance = .03

# ----- GMM ------------------------------------------ #
number_of_components = 50
components_to_extract = 25

#------ Community -------------------------------------#
resolution=1

# ----- NAMES for the savings ------------------------ #

# name_resultDic = "result_ukdale_{}_omp{}_clust{}.pickle".format(
#     sampling_rate, omp_tolerance, number_of_components)

name_resultDic = "result_ukdale_{}_omp{}_clust{}_extract{}_resolution{}.pickle".format(
    sampling_rate, omp_tolerance, number_of_components,components_to_extract,resolution)

name_store_dataset  = "Eco_5min" # test_ukdale_preprocess.py

# import experiments.test_ukdale.test_ukdale_omp
# del sys.modules['experiments.test_ukdale.test_ukdale_omp']
