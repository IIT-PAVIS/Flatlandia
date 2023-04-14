import matplotlib.pyplot as plt
import torch
import json
from torch.utils.data import DataLoader
from os import getcwd
import numpy as np
from sys import path
import dgl
from os import makedirs

# TODO: this is missing some import files, and needs careful cleaning; not fundamental for the initial release!
# Internal imports
path.append('code_for_matching/')



from mapo_utils import GraphLoader    # Dataset for real data.
from dataset import RegionRetrievalData    # Dataset for synthetic data.
import plotting as plotting
from gcns import GCN, forward 
from loss_functions import compute_loss_v1, compute_loss_v2, compute_loss_v3, compute_loss_v4
from metrics import compute_metrics
from similarity import compute_similarity_to_query, find_subgraphs_similar_to_query, compute_similarity_to_query_procrustes, compute_similarity_to_query_procrustes_preselection
from new_region_retrieval import prepare_input_data


def topy(x):
    """ Simple function to turn torch tensors into numpy arrays, to lighten notation """
    return x.clone().detach().numpy()



object_classes = [
    'driveway', 'marking--discrete--arrow--left', 'marking--discrete--arrow--right',
    'marking--discrete--arrow--split-left-or-straight', 'marking--discrete--arrow--split-right-or-straight',
    'marking--discrete--arrow--straight', 'crosswalk-plain', 'crosswalk-zebra', 'give-way-row', 'give-way-single',
    'stop-line', 'marking--discrete--symbol--bicycle', 'marking--discrete--text', 'other-marking', 'banner',
    'object--sign--advertisement', 'object--sign--information', 'object--sign--store',
    'object--support--traffic-sign-frame', 'object--traffic-sign', 'object--traffic-sign--direction',
    'object--traffic-sign--information-parking', 'object--trash-can', 'object--bench', 'object--street-light',
    'object--support--pole', 'object--traffic-light--general-upright', 'object--traffic-light--general-horizontal',
    'object--traffic-light--general-single', 'object--traffic-light--general-other',
    'object--traffic-light--pedestrians', 'object--traffic-light--cyclists', 'object--manhole', 'object--junction-box',
    'object--water-valve', 'object--bike-rack', 'object--catch-basin', 'object--cctv-camera', 'object--fire-hydrant',
    'object--mailbox', 'object--parking-meter', 'object--phone-booth']



########
# MAIN #
########
n_classes = 42

n_bins    = 12

output_path = 'data/similarities/Final/'
makedirs(output_path, exist_ok=True)


######################################
# Compute similarities and save them #
######################################
for dataset_fold in ['training', 'validation', 'testing']:
    print('\nWorking on {:s}'.format(dataset_fold))
    loader = GraphLoader(n_neighbors=10, data_mode=dataset_fold, query_mode='GT', simple=False)
    for graph_id, (token, reference, query, miscellaneous, n_classes) in enumerate(loader):
        print('\nWorking on graph #{:d}'.format(graph_id))
        q_graph, r_graph, repeated_query_info, node_locations, edge_connectivity = prepare_input_data(reference, query, miscellaneous, n_classes, include_location_in_emb=False)

        # HACK: Add inverse edges so that nodes with just an incoming edge to a query node will be considered its neighbour.
        source = np.concatenate((reference['nn_edges'][0], reference['nn_edges'][1]))
        dest   = np.concatenate((reference['nn_edges'][1], reference['nn_edges'][0]))
        r_graph_with_doubled_edges = dgl.graph((source, dest))  # Connectivity.
        r_graph_with_doubled_edges.ndata['h'] = torch.from_numpy(np.array(reference['nodes_class_encoding'])).type(torch.float32) # Node embeddings.

        radius = 0.2  # Radius for selecting the neighbourhood around each node in the Ref. Graph.
        similarities, similarities_v2, residuals, success = compute_similarity_to_query_procrustes_preselection(reference, query, radius)

        if success:
            output_file_name = output_path + '/{:s}.npz'.format(token)
            np.savez(output_file_name, similarities=similarities, similarities_v2=similarities_v2, residuals=residuals)
        else:
            print('Computation on graph {:d} (token = {:s}) did not work for some reason.'.format(graph_id, token))
