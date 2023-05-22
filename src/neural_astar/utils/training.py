"""Helper functions for training
Author: Ryo Yonetani
Affiliation: OSX
"""

from __future__ import annotations

import random
import re
from glob import glob

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim
from neural_astar.planner.astar import VanillaAstar


def load_from_ptl_checkpoint(checkpoint_path: str) -> dict:
    """
    Load model weights from PyTorch Lightning checkpoint.

    Args:
        checkpoint_path (str): (parent) directory where .ckpt is stored.

    Returns:
        dict: model state dict
    """

    ckpt_file = sorted(glob(f"{checkpoint_path}/**/*.ckpt", recursive=True))[-1]
    print(f"load {ckpt_file}")
    state_dict = torch.load(ckpt_file)["state_dict"]
    state_dict_extracted = dict()
    for key in state_dict:
        if "planner" in key:
            state_dict_extracted[re.split("planner.", key)[-1]] = state_dict[key]

    return state_dict_extracted


class PlannerModule(pl.LightningModule):
    def __init__(self, planner, config):
        super().__init__()
        self.planner = planner
        self.vanilla_astar = VanillaAstar()
        self.config = config

    def forward(self, map_designs, start_maps, goal_maps):
        return self.planner(map_designs, start_maps, goal_maps)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.RMSprop(self.planner.parameters(), self.config.params.lr)

    def training_step(self, train_batch, batch_idx):
        map_designs, start_maps, goal_maps, opt_trajs, _ = train_batch
        outputs = self.forward(map_designs, start_maps, goal_maps)
        loss = nn.L1Loss()(outputs.histories, opt_trajs)
        self.log("metrics/train_loss", loss)

        return loss

    def all_accuracies(self, true_labels, suggested_labels, true_costs, num_thresholds, minimize=True):
        num_examples = len(true_labels)
        valid = 0
        meets_threshold = [0] * num_thresholds
        for true_label, suggested_label, true_cost in zip(true_labels, suggested_labels, true_costs):
            valid += 1
            cost_ratio = np.sum(suggested_label * true_cost) / np.sum(true_label * true_cost)
            if not minimize:
                cost_ratio = 1.0 / cost_ratio

            assert cost_ratio > 0.99  # cost is not better than optimal...

            for i in range(len(meets_threshold)):
                if cost_ratio - 1.0 < 10.0 ** (-i-1):
                    meets_threshold[i] += 1

        threshold_dict = {f"below_{10. ** (1-i)}_percent_acc": val / num_examples for i, val in enumerate(meets_threshold)}
        threshold_dict['valid_acc'] = valid / num_examples
        for key, value in threshold_dict.items():
            self.log(key, value)

    def validation_step(self, val_batch, batch_idx):
        map_designs, start_maps, goal_maps, opt_trajs, opt_vertex = val_batch
        outputs = self.forward(map_designs, start_maps, goal_maps)
        loss = nn.L1Loss()(outputs.histories, opt_trajs)

        path_optimal = opt_trajs.detach().cpu().numpy()
        path_model = outputs.paths.detach().cpu().numpy()
        vertex_optimal = opt_vertex.detach().cpu().numpy()

        # perfect_match_accuracy
        matching_correct = np.sum(np.abs(path_optimal - path_model), axis=-1)
        avg_matching_correct = (matching_correct < 0.5).mean()

        # cost_ratio
        #cost_ratio = (np.sum(path_model * vertex_optimal, axis=1) / np.sum(path_optimal * vertex_optimal, axis=1)).mean()

        self.all_accuracies(path_optimal, path_model, vertex_optimal, 6)
        self.log("metrics/val_perfect_match_accuracy", avg_matching_correct)
        self.log("metrics/val_loss", loss)

        # For shortest path problems:
        if map_designs.shape[1] == 1:
            va_outputs = self.vanilla_astar(map_designs, start_maps, goal_maps)
            pathlen_astar = va_outputs.paths.sum((1, 2, 3)).detach().cpu().numpy()
            pathlen_model = outputs.paths.sum((1, 2, 3)).detach().cpu().numpy()
            p_opt = (pathlen_astar == pathlen_model).mean()

            exp_astar = va_outputs.histories.sum((1, 2, 3)).detach().cpu().numpy()
            exp_na = outputs.histories.sum((1, 2, 3)).detach().cpu().numpy()
            p_exp = np.maximum((exp_astar - exp_na) / exp_astar, 0.0).mean()

            h_mean = 2.0 / (1.0 / (p_opt + 1e-10) + 1.0 / (p_exp + 1e-10))

            self.log("metrics/p_opt", p_opt)
            self.log("metrics/p_exp", p_exp)
            self.log("metrics/h_mean", h_mean)

        return loss


def set_global_seeds(seed: int) -> None:
    """
    Set random seeds

    Args:
        seed (int): random seed
    """

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
    random.seed(seed)
