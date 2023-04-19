"""
Name: plotting.py
Description: Functions to visualize the dataset queries and the reference maps
-----
Author: Matteo Toso.
Licence: MIT. Copyright 2023, PAVIS, Istituto Italiano di Tecnologia.
Acknowledgment:
This project has received funding from the European Union's Horizon 2020
research and innovation programme under grant agreement No 870743.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from Flatlandia.scripts.utils.common import mapillary_access_token, local_maps_path
from Flatlandia.scripts.utils.dataloader import object_classes_dict
import mapillary.interface as mly
import numpy as np
import requests
import io
import json
from PIL import Image


colormap = {'arrow--marking': '#9BBFBB',
            'bicycle--symbol': '#FFA023',
            'sign': '#EF6B5A',
            'traffic-sign': '#9BC73E',
            'trash-can': '#F9E94B',
            'street-light': '#59C2EE',
            'support-pole': '#9B89BA',
            'traffic-light': '#999594',
            'manhole': '#30BDB4',
            'junction-box': '#F2A5C9',
            'bench': '#B8BEDE',
            'bike-rack': '#E7EEE1',
            'catch-basin': '#2F528F',
            'cctv-camera': '#E2F0D9',
            'fire-hydrant': '#F2E8F1',
            'text': '#44546A'}


def visualize_problem(data, save_path=None):
    """
    :param data: one of the Flatlandia dataset instances
    :param save_path: full path where to save the plot; if None, the plot is only shown
    :return: Plot of the reference map, query image with detections, and the local map from projecting the detections
    """

    # 1) plot all objects on a map
    map = {}
    for x in range(len(data['reference_xy'])):
        object_class = object_classes_dict(data['reference_class'][x])
        if map.get(object_class) is None:
            map[object_class] = []
        map[object_class].append(data['reference_xy'][x])

    fig, ax = plt.subplot_mosaic([['Left', 'TopRight'], ['Left', 'BottomRight']], gridspec_kw={'width_ratios': [2, 1]})
    for cs in list(map.keys()):
        points = np.array(map[cs]).T
        ax['Left'].scatter(points[0], points[1], c=colormap[cs], label=cs, s=35)

    ax['Left'].scatter(data['query_xy'][0], data['query_xy'][1], s=70, label='Camera', c='#003865', alpha=0.8)

    # Coordinates of our triangle
    theta = (np.pi * data['query_theta']/180)
    p0 = data['query_xy']
    p1 = 0.075 * np.array([np.cos(theta + 1.), np.sin(theta + 1.)]) + p0
    p2 = 0.075 * np.array([np.cos(theta - 1.), np.sin(theta - 1.)]) + p0
    pts = np.array([p0, p1, p2])
    p = patches.Polygon(pts, alpha=0.4)
    ax['Left'].add_patch(p)

    xy = (np.array(data['reference_xy'])[data['query_matches']]).T
    ax['Left'].scatter(xy[0], xy[1], s=70, facecolors='none', edgecolors='r', label='Detected')

    # 2) Plot the image with anotatated detections
    try:
        mly.set_access_token(mapillary_access_token)
        image_url = mly.image_thumbnail(image_id=data['query_token'], resolution=2048)
        image_data = requests.get(image_url, stream=True).content
        image = Image.open(io.BytesIO(image_data))
        ax['TopRight'].imshow(image)
    except:
        print('Unable to load image; check internet connection is enabled and Mapillary security token is working')
    for ii, bb in enumerate(data['query_detections']):
        label = object_classes_dict(data['reference_class'][data['query_matches'][ii]])
        rect = patches.Rectangle((bb[0], bb[1]), bb[2] - bb[0], bb[3] - bb[1], linewidth=1, edgecolor=colormap[label], facecolor='none')
        ax['TopRight'].add_patch(rect)

    # 3) Plot the local map based on the detected objects

    with open(local_maps_path, 'r') as infile:
        localmaps = json.load(infile)
    q_xy = localmaps[data['reference_map']][data['query_token']]['GT']
    q_c = np.array(data['reference_class'])[data['query_matches']]
    for xy, c in zip(q_xy, q_c):
        ax['BottomRight'].scatter(-xy[1], xy[0], s=50, c=colormap[object_classes_dict(c)])
    ax['BottomRight'].scatter(0, 0, s=70, label='Camera', c='#003865', alpha=0.8)

    # Coordinates of our triangle
    theta = 0.5 * np.pi
    p0 = [0, 0]
    p1 = 0.075 * np.array([np.cos(theta + 1.), np.sin(theta + 1.)]) + p0
    p2 = 0.075 * np.array([np.cos(theta - 1.), np.sin(theta - 1.)]) + p0
    pts = np.array([p0, p1, p2])
    p = patches.Polygon(pts, alpha=0.4)
    ax['BottomRight'].add_patch(p)

    ax['TopRight'].axis('off')
    ax['TopRight'].get_xaxis().set_visible(False)
    ax['TopRight'].get_yaxis().set_visible(False)

    ax['Left'].axis('equal')
    plt.tight_layout()
    ax['TopRight'].set_title("Query: {}".format(data['query_token']))
    ax['BottomRight'].set_title("Local map")
    ax['Left'].set_title("Scene: {}".format(data['reference_map']))
    ax['Left'].legend(fancybox=True, shadow=True, ncol=5)

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

    return


def main():
    from dataloader import FlatlandiaLoader

    data = FlatlandiaLoader()
    for x in data:
        visualize_problem(x)
    return


if __name__ == '__main__':
    main()
