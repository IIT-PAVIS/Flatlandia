"""
Name: demo_coarse_localization.py
Description: Script to train a model for the coarse camera localization task.
-----
Author: Matteo Taiana.
Licence: MIT. Copyright 2023, PAVIS, Istituto Italiano di Tecnologia.
Acknowledgment:
This project has received funding from the European Union's Horizon 2020
research and innovation programme under grant agreement No 870743.
"""


# Reads all the similarities from the binary .npz files (each one contains information on one Query Graph/Reference Graph pair).
# Saves them as one JSON file indexed by the query token, with a 'valid' flag.
import json
import numpy as np
import sys
from os import listdir, getcwd, path
from glob import glob

# Internal imports
sys.path.append('code_for_matching/')
from mapo_utils import GraphLoader    # Dataset for real data.


########
# MAIN #
########
input_path = getcwd() + '/data/similarities/Final/'
output_path = getcwd() + '/data/similarities/'

my_dict = {} 

# Get all the tokens returned by the data loader.
# WARNING: the data loader has to be modified so that it returns the token, for this to work.
tokens = []
for dataset_fold in ['training', 'validation', 'testing']:
    loader   = GraphLoader(n_neighbors=10, data_mode=dataset_fold, query_mode='GT', simple=False)
    for graph_id, (token, reference, query, miscellaneous, n_classes) in enumerate(loader):
        tokens.append(token)

print('There are {:d} tokens in the dataset.'.format(len(tokens)))

# list file and directories
input_file_names = glob(input_path + '/*.npz')

for token in tokens:
    try:
        input_file_name = input_path + token + '.npz'
        data = np.load(input_file_name)  # Todo: this will fail when the file a file is not available (which can happen when we cannot compute the similarity, for some reason.)

        data_has_problems = np.isnan(data['similarities_v2']).any() or  np.isnan(data['similarities']).any() or np.isnan(data['residuals']).any() 

 
        if data_has_problems:
            residuals = [-1.0]
            similarities = [-1.0]
            similarities_v2 = [-1.0]
            valid = False
            print('Marking example {:s} as invalid because of NaNs.'.format(path.basename(input_file_name)))
        else:
            residuals       = data['residuals'].tolist()
            similarities    = data['similarities'].tolist()
            similarities_v2 = data['similarities_v2'].tolist()
            valid           = True
    except:
        residuals       = [-1.0]
        similarities    = [-1.0]
        similarities_v2 = [-1.0]
        valid = False
        print('Marking example {:s} as invalid because of missing file.'.format(path.basename(input_file_name)))

    my_dict[token] = {'residuals':residuals,
                   'similarities':similarities,
                   'similarities_v2':similarities_v2,
                   'valid':valid}

with open(output_path + '/similarities_final.json', 'w') as outfile:
         json.dump(my_dict, outfile, indent=4)

