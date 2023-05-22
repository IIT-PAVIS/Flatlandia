"""
Name: dataloader.py
Description: Functions to access the Flatlandia dataset
-----
Author: Matteo Toso.
Licence: MIT. Copyright 2023, PAVIS, Istituto Italiano di Tecnologia.
Acknowledgment:
This project has received funding from the European Union's Horizon 2020
research and innovation programme under grant agreement No 870743.
"""


import json
from torch.utils.data import Dataset
import numpy as np
from Flatlandia.scripts.utils.common import dataset_path, local_maps_path, region_proposals_path
import mapillary.interface as mly
import requests
import io
from PIL import Image


id_to_scene = {'0': 'BarcelonaCentre_Calle_de_Pere_IV_2_0', '1': 'BarcelonaCentre_Calle_de_Pere_IV_1',
               '2': 'BarcelonaCentre_Gran_de_Gracia_0', '3': 'BerlinCentre_Rosenthaler_Platz_0',
               '4': 'BerlinCentre_Torstrasse_0', '5': 'LisbonCentre_Largo_Duque_do_cadaval_0',
               '6': 'LisbonCentre_R_de_Angora_0', '7': 'LisbonCentre_Encarnado',
               '8': 'LisbonCentre_Square_Pedro_IV_0', '9': 'LisbonCentre_Praca_martim_moniz_0',
               '10': 'LisbonCentre_Largo_de_Sao_Domingos_0', '11': 'LisbonCentre_Largo_do_Regedor_0',
               '12': 'ParisRosaPark_R_Jean_Oberle_2_0', '13': 'ParisRosaPark_R_Jean_Oberle_0',
               '14': 'ParisRosaPark_Bd_Macdonald_3', '15': 'ParisRosaPark_R_Gaston_Tessier_0',
               '16': 'ParisCentre_R_de_Rivoli_0', '17': 'ParisRosaPark_Prv_Rosa_Parks_0',
               '18': 'ParisRosaPark_Linear_Forest_0', '19': 'ViennaCentre_Unter_Vieduktgasse_0'
               }


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

simplified_object_classes = [
    'driveway', 'arrow--marking', 'arrow--marking',
    'arrow--marking', 'arrow--marking',
    'arrow--marking', 'crosswalk', 'crosswalk', 'give-way', 'give-way',
    'stop-line', 'bicycle--symbol', 'text', 'other-marking', 'banner',
    'sign', 'sign', 'sign',
    'traffic-sign', 'traffic-sign', 'traffic-sign',
    'traffic-sign', 'trash-can', 'bench', 'street-light',
    'support-pole', 'traffic-light', 'traffic-light',
    'traffic-light', 'traffic-light',
    'traffic-light', 'traffic-light', 'manhole', 'junction-box',
    'water-valve', 'bike-rack', 'catch-basin', 'cctv-camera', 'fire-hydrant',
    'mailbox', 'parking-meter', 'phone-booth']


def object_classes_dict(x, simple=True):
    """
    This function returns the full label of the
    :param x: Int, internal class representation
    :param simple: if True, returns macro-classes (e.g., 'object--traffic-light--general-single', 'object--traffic-light--general-other',
    'object--traffic-light--pedestrians' are classified as 'traffic-light')
    :return: a string describing the Panoptic class
    """
    if simple:
        return simplified_object_classes[x]
    else:
        return object_classes[x]


def get_distance_matrix(xy):
    if len(xy.shape) < 3:
        xy = xy.reshape(1, len(xy), -1)

    inner = np.matmul(xy, xy.transpose(0, 2, 1))
    xx = np.sum(xy ** 2, axis=2, keepdims=True)
    pairwise_distance = np.abs(xx - 2 * inner + xx.transpose(0, 2, 1))
    return np.sqrt(pairwise_distance)


class FlatlandiaLoader(Dataset):
    def __init__(self, data_mode='training', data_subset=False):
        """
        :param data_mode: select between 'training', 'testing' and 'validation'
        :param data_subset: if True only use visual queries with more than 4 object detections
        """
        with open(dataset_path, 'r') as infile:
            data = json.load(infile)
        self.all = not data_subset
        self.maps = data['maps']
        self.token_to_scene = data['miscellanea']['split']['img_to_scene']
        self.tokens = data['miscellanea']['split']['simple'][data_mode]
        if self.all:
            self.tokens = self.tokens + data['miscellanea']['split']['hard'][data_mode]

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        token = self.tokens[idx]
        scene = self.maps[str(self.token_to_scene[token])]
        try:
            out = {
            'reference_map': str(self.token_to_scene[token]),
            'reference_xy': [scene['objects'][x]['xy'] for x in scene['object_list']],
            'reference_class': [scene['objects'][x]['class'] for x in scene['object_list']],
            'query_token': token,
            'query_xy': scene['queries'][token]['xy'],
            'query_theta': scene['queries'][token]['theta'],
            'query_matches': [scene['object_list'].index(x) for x in
                              scene['queries'][token]['detected_objects']],
            'query_detections': [scene['queries'][token]['detections'][x] for x in
                                 scene['queries'][token]['detected_objects']],
            'intrinsics': scene['queries'][token]['intrinsics']
        }
        except:
            print('Error, the desired problem is not available')
            out = None
        return out

    def get_region_proposal(self, token, mode='GT', select='CoarseBB'):
        with open(region_proposals_path, 'r') as infile:
            rp = json.load(infile)
        if self.all:
            return rp[mode]['all'][select][token][0]
        else:
            return rp[mode]['easy'][select][token][0]


def get_image(token, mapillary_access_token, resolution=2048):
    mly.set_access_token(mapillary_access_token)
    image_url = mly.image_thumbnail(image_id=token, resolution=resolution)
    image_data = requests.get(image_url, stream=True).content
    image = Image.open(io.BytesIO(image_data))
    return image


def nodes_to_graph(nodes_xy, classes, knn):
    """
    :param nodes_xy:
    :param classes:
    :param knn:
    :return: idx: [NE,2] source and destination node id for each graph edge
             node_embedding: [N, 44] concatenated one-hot class encoding [42] and long/lat [2] of each object
             edge_embedding: [NE, 2] concatenated distance and relative orientation of each graph edge
    """
    # 1) compute the distance between pairs of nodes

    distances = get_distance_matrix(xy)

    # 2) define connectivity
    if knn is None:
        idx = fully_connected_idx(len(xy))
    else:
        idx_i = np.tile(range(len(xy)), (min([knn, len(xy)]), 1)).T.reshape(-1)
        idx_f = np.array(np.argsort(distances, 1)[:, :min([knn, len(xy)])]).reshape(-1)
        idx = [idx_i, idx_f]
    idx = np.array(idx).T
    # 3) generate class embedding
    temp = np.eye(42, dtype=int)
    class_embedding = [temp[i] for i in classes]
    edge_distance = [distances[i, j] for [i, j] in idx]
    edge_orientation = [np.arctan2((xy[j] - xy[i])[1], (xy[j] - xy[i])[0]) for i, j in idx]
    node_embedding = np.hstack([np.array(class_embedding), xy])
    edge_embedding = np.array(edge_distance + edge_orientation).reshape(2, -1).T
    return idx, node_embedding, edge_embedding


def fully_connected_idx(size):
    """
    :param size: number of nodes
    :return: indexes for a fully-connected graph
    """
    temp = np.tile(np.array(range(size)).reshape(1, -1), (size, 1))
    idx_source = temp.T.reshape(-1)
    idx_destination = temp.reshape(-1)
    return idx_source, idx_destination


def assemble_graphs(problem, region_proposal=None, query='GT', knn_q=None, knn_r=3, augmentation=True):
    """
    :param problem: dataset entry containing information about the visual query and the reference map
    :param region_proposal: [N] list of reference map objects to be used in the graph. If None, use all
    :param query: 'GT'/'depth' select how a local map of the object seen in the visual query is generated
    :param knn_q: number of nearest neighbors connected to each query node; if None use a fully connected graph
    :param knn_r: number of nearest neighbors connected to each reference node; if None use a fully connected graph
    :param augmentation: if False, query map has camera in 0,0 with orientation 0; if True apply random roto-translation
    :return: out: a dictionary of graph obtained using only the reference or the query nodes ('r', 'q'), using only
    edges connecting same-class query and reference objects ('r_xor_q'), or using all edges above (r_and_q).
    """

    # 1) if a subset of the reference object is provided, down-sample the reference data
    if region_proposal is None:
        ref_xy = np.array(problem['reference_xy'])
        ref_c = np.array(problem['reference_class'])
        mask = np.zeros([len(problem['query_matches']), len(ref_xy)], dtype=int)
        for i, j in enumerate(problem['query_matches']):
            mask[i, j] = 1
    else:
        ref_xy = np.array(problem['reference_xy'])[region_proposal]
        ref_c = np.array(problem['reference_class'])[region_proposal]
        matches = []
        for x in problem['query_matches']:
            try:
                matches.append(region_proposal.index(x))
            except:
                matches.append(None)
        mask = np.zeros([len(problem['query_matches']), len(ref_xy)], dtype=int)
        for i, j in enumerate(matches):
            if j is not None:
                mask[i, j] = 1

    # 2) compute the local query map
    with open(local_maps_path, 'r') as infile:
        precomputed_local_maps = json.load(infile)
    q_xy = precomputed_local_maps[problem['reference_map']][problem['query_token']][query]
    q_c = np.array(problem['reference_class'])[problem['query_matches']]

    if augmentation:
        delta_t = np.random.uniform(0, 1, [1, 2])
        delta_o = np.random.uniform(0, 2 * np.pi)
        delta_rot = np.array([np.cos(delta_o), -np.sin(delta_o), np.sin(delta_o), np.cos(delta_o)]).reshape([2, 2])
        delta_s = np.random.uniform(0.5, 1.5)
        q_xy_old = q_xy.copy()
        q_xy = ((q_xy - np.mean(q_xy, 0)) @ delta_rot.T) * delta_s + np.mean(q_xy, 0) + delta_t
        q_pose = (-np.mean(q_xy_old, 0) @ delta_rot.T) * delta_s + np.mean(q_xy_old, 0) + delta_t
        q_pose = np.hstack([q_pose, np.array(delta_o).reshape(1, 1)]).reshape(3)
    else:
        q_pose = np.array([0, 0, 0])

    # 3) remove reference nodes with no class representation in the query

    filter = [x in q_c for x in ref_c]
    ref_xy = ref_xy[filter]
    ref_c = ref_c[filter]
    mask = mask.T[filter]

    # 4) create the graphs

    q_idx, q_n, q_e = nodes_to_graph(q_xy, q_c, knn_q)
    r_idx, r_n, r_e = nodes_to_graph(ref_xy, ref_c, knn_r)

    rq_idx = np.argwhere(r_n[:, :-2] @ q_n[:, :-2].T)
    rq_idx[:, 1] += len(ref_xy)
    rq_idx = np.hstack([rq_idx, rq_idx[:, ::-1]]).reshape(-1, 2)

    rq_n = np.vstack([r_n, q_n])
    edge_distance = [np.linalg.norm(rq_n[j, -2:]-rq_n[i, -2:]) for [i, j] in rq_idx]
    edge_orientation = [np.arctan2((rq_n[j, -1]-rq_n[i, -1]), (rq_n[j, -2]-rq_n[i, -2])) for i, j in rq_idx]
    rq_e = np.array(edge_distance + edge_orientation).reshape(2, -1).T

    out = {
        'q_pose': q_pose,
        'q': {'idx': q_idx, 'e_n': q_n, 'e_e': q_e},
        'r': {'idx': r_idx, 'e_n': r_n, 'e_e': r_e},
        'r_and_q': {'idx': rq_idx, 'e_n': rq_n, 'e_e': rq_e},
        'r_or_q': {'idx': np.vstack([r_idx, q_idx+len(ref_xy), rq_idx]),
                   'e_n': rq_n, 'e_e': np.vstack([r_e, q_e, rq_e])},
        'r_xor_q': {'idx': np.vstack([r_idx, q_idx + len(ref_xy)]),
                   'e_n': rq_n, 'e_e': np.vstack([r_e, q_e])},
        'ref_to_q_match': mask
    }

    return out


def main():

    dataset = FlatlandiaLoader()
    temp = dataset[0]
    graphs = assemble_graphs(temp)

    return


if __name__ == "__main__":
    main()
