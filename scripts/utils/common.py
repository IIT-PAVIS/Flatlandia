"""
Name: common.py
Description: global functions and parameters, used by several functions.
-----
Author: Matteo Toso.
Licence: MIT. Copyright 2023, PAVIS, Istituto Italiano di Tecnologia.
Acknowledgment:
This project has received funding from the European Union's Horizon 2020
research and innovation programme under grant agreement No 870743.
"""
import pathlib

# Automatically find the root path of this git repository
rootpath = '/'.join(str(pathlib.Path(__file__).parent.resolve()).split('/')[:-2]) + '/'
dataset_path = rootpath + 'data/flatlandia.json'
local_maps_path = rootpath + 'data/local_maps.json'
out_path = rootpath + '/models/'
region_proposals_path = rootpath + 'data/region_proposal.json'

# To be filled by the user as part of set up
mapillary_access_token = 'MLY|XXXXXXXXXXXXXXX|XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
wandb_user = 'XXXXXXXXXX'
