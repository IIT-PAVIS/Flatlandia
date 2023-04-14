"""
Name: models.py
Description: Script to train a model for the fine camera localization task.
-----
Author: Matteo Toso.
Licence: MIT. Copyright 2023, PAVIS, Istituto Italiano di Tecnologia.
Acknowledgment:
This project has received funding from the European Union's Horizon 2020
research and innovation programme under grant agreement No 870743.
"""
import torch

from scripts.utils.dataloader import FlatlandiaLoader, assemble_graphs
from scripts.utils.models import select_module
from scripts.utils.common import wandb_user, out_path
import dgl
import argparse
import numpy as np
import random
import torch
import wandb


parser = argparse.ArgumentParser()
parser.add_argument("--lr", help="- Power of the learning rate", type=int, default=4)
parser.add_argument("--epochs", help="Number of training epochs", type=int, default=1000)
parser.add_argument("--seed", help="Random seed, to ensure repeatability", type=int, default=2)
parser.add_argument("--experiment_name", help="Name of the experiment", type=str, default='Flatlandia-FineBaseline')
parser.add_argument("--noise", help="Apply rigid transformation to the query", type=int, default=1, choices=[0, 1])
parser.add_argument("--easy_dataset", help="remove queries with less than 5 nodes", type=int, default=0, choices=[0, 1])
parser.add_argument("--batch", help="Batch size", type=int, default=5)
parser.add_argument("--model_type", help="Type of model to apply", type=str, default='MLP+ATT+MLP',
                    choices=['MLP', 'GAT+MLP', 'MLP+ATT+MLP', 'GAT+ATT+MLP'])
parser.add_argument("--data_type", help="Type of query data to use", type=str, default='GT',
                    choices=['GT', 'depth'])
parser.add_argument("--rp", help="Type of precomputed region proposal", type=str, default='CoarseBB',
                    choices=['simil_nq','simil_2nq', 'simil_circle', 'triplet_nq', 'triplet_2nq', 'triplet_circle',
                             'mix_nq', 'mix_2nq', 'mix_circle', 'NoisyGT', 'CoarseBB'])
args = parser.parse_args()

config = {
    'lr': 1 / np.power(10, args.lr),
    'epochs': args.epochs,
    'seed': args.seed,
    'exp_name': args.experiment_name,
    'noise': bool(args.noise),
    'simplified': bool(args.easy_dataset),
    'batch_size': args.batch,
    'model_type': args.model_type,
    'data_type': args.data_type,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'rp': args.rp
}


def solve_regression(data, model, rp, settings, augment):
    """
    :param data: instance of the Flatlandia dataloader
    :param model: a network to regress pose from Flatlandia problems, initialized in the main function
    :param rp: a list ids (integers) of the reference map objects in a region likely to contain the query; if None, the
        full reference map is used
    :param settings: the json dictionary with the experiment's options
    :param augment: if True, the query local map is randomly roto-translated
    :return:
    """

    # 0) Given a Flatlandia dataset item, we aggregate the information in graphs
    p = assemble_graphs(data, region_proposal=rp, query=settings['data_type'], knn_q=None, knn_r=3,
                        augmentation=augment)

    q_xor_r = p['r_xor_q']['idx'].T
    q_and_r = p['r_xor_q']['idx'].T

    q_xor_r_graph = dgl.graph((q_xor_r[0], q_xor_r[1])).to(model.device)
    q_and_r_graph = dgl.graph((q_and_r[0], q_and_r[1])).to(model.device)
    nv = torch.from_numpy(p['r_and_q']['e_n']).to(model.device).type(torch.float32)
    ev = torch.from_numpy(p['r_and_q']['e_e']).to(model.device).type(torch.float32)
    gt_pose = torch.from_numpy(np.hstack([data.get('query_xy'), data.get('query_theta') * np.pi / 180]))

    # 1) The graphs, the edge and node embeddings, and the number of reference map nodes, are passed to the model
    _, _, pose = model([q_xor_r_graph, q_and_r_graph], nv, ev, len(p['r']['e_n']))

    def distance_3dof(pose1, pose2):
        # We compute the similarity of two poses in Flatlandia as their euclidan distance and the angular distance of
        # their orientation
        distance = (pose2 - pose1)
        spatial_d = torch.linalg.norm(distance[..., :2])
        angular_d = torch.arctan2(torch.sin(distance[..., 2]), torch.cos(distance[..., 2])) % (2 * torch.pi)
        if angular_d > torch.pi:
            angular_d = 2 * torch.pi - angular_d
        errors = torch.hstack([spatial_d, angular_d])
        return errors

    error = distance_3dof(pose, gt_pose.to(model.device))

    return error


def run_epoch(dataset, model, settings, optimizer=None, shuffle=False, augment=True):
    """
    :param dataset: instance of the Flatlandia dataloader
    :param model: a network to regress pose from Flatlandia problems, initialized in the main function
    :param settings: the json dictionary with the experiment's options
    :param optimizer: the pytorch optimizer used to train the model
    :param shuffle: if True, the dataset examples are shuffled at the beginning of the epoch
    :param augment: if True, the query local map is randomly roto-translated
    :return:
        metrics_backup: [pose_error, rotation_error] for each dataset sample
        metrics: [pose_error, rotation_error] median metrics of the reconstructed poses over the epoch
    """
    metrics = []
    success = 0
    problem_ids = list(range(len(dataset)))
    ids = []
    if shuffle:
        random.shuffle(problem_ids)

    for s, p in enumerate(problem_ids):

        data = dataset[p]
        rp = dataset.get_region_proposal(data['query_token'], settings['data_type'], settings['rp'])
        error = solve_regression(data, model, rp, settings, augment)

        if (error != error).any():
            print('skipping {}'.format(p))
            continue
        success += 1
        ids.append(p)

        loss = torch.sum(error)

        metrics.append(error.cpu().clone().detach().numpy())
        if optimizer is not None:
            loss.backward()
            if (s + 1) % settings['batch_size'] == 0:
                optimizer.step()
                optimizer.zero_grad()

    metrics_backup = [np.array(metrics).copy(), ids]
    metrics = np.mean(np.array(metrics).reshape(-1, 2), axis=0)
    print(
        '({}/{}):: loss_ {:.3f}, distance_error {:.3f}, angular_error {:.3f}'.format(
            success, len(dataset), metrics.mean(), metrics[0], metrics[1]))
    return metrics_backup, metrics


def main():

    # 0) Fix the seed for reproducibility, initialize the W&B logging, initialize models and optimizers
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    set_train = FlatlandiaLoader(data_mode='training')
    set_test = FlatlandiaLoader(data_mode='testing')
    set_validation = FlatlandiaLoader(data_mode='validation')

    model = select_module(config['model_type'])
    model = model.to(config['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    optimizer.zero_grad()

    wandb.init(project=config['exp_name'], entity=wandb_user, config=config)
    save_path = "{}/{}/{}".format(out_path, config['exp_name'], wandb.run.name)
    os.mkdirs(save_path)

    current_best = 90000

    # 1) Start the training
    for epoch in range(config['epochs']):
        print('Epoch {}'.format(epoch))
        _, metrics_t = run_epoch(set_train, model, config, shuffle=True, optimizer=optimizer,
                                 augment=config['noise'])
        with torch.no_grad():
            _, metrics_v = run_epoch(set_validation, model, config, augment=False)
        wandb.log({
            'T - Loss': metrics_t.mean(),
            'T - GPS Error': metrics_t[0] * 215,
            'T - Orientation Error': metrics_t[1] * (180/np.pi),
            'V - Loss': metrics_v.mean(),
            'V - GPS Error': metrics_v[0] * 215,
            'V - Orientation Error': metrics_v[1] * (180/np.pi),
        })

        torch.save(model.state_dict(), save_path + '\model_epoch_{}'.format(epoch))
        if metrics_v.mean() < current_best:
            current_best = metrics_v.mean()
            torch.save(model.state_dict(), save_path + '\model_best')

    # 2) Evaluate the trained model

    print('Evaluating best scoring model on the test set.')
    model.load_state_dict(torch.load(save_path + '\model_best.pth'))
    with torch.no_grad():
        _, metrics_v = run_epoch(set_test, model, config, augment=False)

    return


if __name__ == "__main__":
    main()
