import torch
import torch.nn as nn
import numpy as np
import os
import json
import time
from torch.autograd import grad
from torch.autograd import Function
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

EPS = 1e-4
class DistanceGradientNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, point, hyperplane_A, hyperplane_b, v1, v2):
        num_points = point.shape[0]
        distances = torch.zeros(num_points, device=point.device, dtype=point.dtype)
        gradients = torch.zeros(num_points, 3, device=point.device, dtype=point.dtype)
        
        num_vertices = v1.shape[1]
        num_hyperplanes = hyperplane_A.shape[1]
        
        ### Get the perpendicular distance from the point to faces
        # map point c to c1, its projection onto the hyperplane
        points = point.repeat(1, num_hyperplanes).view(num_points, num_hyperplanes, -1)
        Ap_minus_b = torch.sum(hyperplane_A * points, dim=-1, keepdim=True) - hyperplane_b
        
        # check sign; if there is negative distance, directly assign distance
        Ap_minus_b = Ap_minus_b.squeeze(-1)
        Ap_minus_b[torch.isinf(Ap_minus_b)] = -torch.inf
        Ap_minus_b[torch.isnan(Ap_minus_b)] = -torch.inf
        is_negative_distance = torch.all(Ap_minus_b <= 0, dim=-1)
        max_negative_pairs = torch.max(Ap_minus_b, dim=-1)

        distances[is_negative_distance] = max_negative_pairs.values[is_negative_distance]    
        gradients[is_negative_distance] = hyperplane_A[is_negative_distance, max_negative_pairs.indices[is_negative_distance]]
        
        # we only need to consider the positive distances in the rest of the discussion
        non_negative_distance = torch.logical_not(is_negative_distance)
        num_nonnegative_points = non_negative_distance.sum().item()
        points = points[non_negative_distance]
        Ap_minus_b = Ap_minus_b.unsqueeze(-1)[non_negative_distance]
        hyperplane_A = hyperplane_A[non_negative_distance]
        hyperplane_b = hyperplane_b[non_negative_distance]
        v1 = v1[non_negative_distance]
        v2 = v2[non_negative_distance]
        
        # if the points projected onto hyperplanes are on the faces, then it is the perpendicular distance we want
        hyperplane_projected_points = (points - Ap_minus_b * hyperplane_A).view(num_nonnegative_points * num_hyperplanes, 3, 1)
        batched_hyperplane_A = hyperplane_A.repeat(1, num_hyperplanes, 1).view(num_nonnegative_points * num_hyperplanes, num_hyperplanes, 3)
        batched_hyperplane_b = hyperplane_b.repeat(1, num_hyperplanes, 1).view(num_nonnegative_points * num_hyperplanes, num_hyperplanes, 1)
        Apprime_minus_b = torch.bmm(batched_hyperplane_A, hyperplane_projected_points) - batched_hyperplane_b
        Apprime_minus_b = Apprime_minus_b.view(num_nonnegative_points, num_hyperplanes, num_hyperplanes)
        Apprime_minus_b[torch.isinf(Apprime_minus_b)] = -torch.inf
        Apprime_minus_b[torch.isnan(Apprime_minus_b)] = -torch.inf
        on_zonotope = torch.all(Apprime_minus_b <= EPS, dim=-1, keepdim=True).view(num_nonnegative_points, num_hyperplanes)

        hyperplane_projected_points = hyperplane_projected_points.view(num_nonnegative_points, num_hyperplanes, 3)
        perpendicular_distances = torch.linalg.norm(points - hyperplane_projected_points, dim=-1).view(num_nonnegative_points, num_hyperplanes)
        perpendicular_distances[torch.logical_not(on_zonotope)] = torch.inf
        perpendicular_distances[torch.isinf(perpendicular_distances)] = torch.inf
        perpendicular_distances[torch.isnan(perpendicular_distances)] = torch.inf
        min_perpendicular_pair = perpendicular_distances.min(dim=1)
        distances[non_negative_distance] = min_perpendicular_pair.values
        gradients[non_negative_distance] = hyperplane_A[torch.arange(hyperplane_A.shape[0], device=point.device), min_perpendicular_pair.indices] ### TODO
        ### Get the distance from the point to edges
        # NOTE: should probably pad vertices with fake edges such that every zonotope have the same number of verticess
        edge_distance_points = point[non_negative_distance].repeat(1, num_vertices).view(num_nonnegative_points, num_vertices, 3)
        v2_minus_v1_square = torch.square(v2 - v1)
        v2_minus_v1_square_sum = torch.sum(v2_minus_v1_square, dim=-1, keepdim=True) # + EPS
        
        t_hat = torch.sum((edge_distance_points - v1) * (v2 - v1), dim=-1, keepdim=True) / v2_minus_v1_square_sum
        t_star = t_hat.clamp(0,1)
        v = v1 + t_star * (v2 - v1)
        edge_distances = torch.linalg.norm(edge_distance_points - v, dim=-1, keepdim=True).view(num_nonnegative_points, num_vertices)
        edge_distances[torch.isnan(edge_distances)] = torch.inf
        
        edge_distance_pair = edge_distances.min(dim=1)
        edge_distances = edge_distance_pair.values
        use_edge_distance = edge_distances < distances[non_negative_distance] ### TODO
        edge_indices = torch.arange(edge_distances.shape[0], device=point.device)[use_edge_distance]
        distance_edge_indices = torch.arange(distances.shape[0], device=point.device)[non_negative_distance][use_edge_distance]
        distances[distance_edge_indices] = edge_distances[use_edge_distance]
        # print(f"distance_edge_indices = {distance_edge_indices}")
        #import pdb; pdb.set_trace()
        
        # coeff = torch.logical_and((t_hat < 1.), (t_hat > 0.)).float()
        # dsdp = (coeff * (v2_minus_v1_square / v2_minus_v1_square_sum))[edge_indices, edge_distance_pair.indices[edge_indices]]
        # edge_distance_points = edge_distance_points[edge_indices, edge_distance_pair.indices[edge_indices]]
        # v = v[edge_indices, edge_distance_pair.indices[edge_indices]]
        # gradients[distance_edge_indices] = (edge_distance_points - v + (v - edge_distance_points) * dsdp) / edge_distances[use_edge_distance].view(-1,1)
        
        edge_distance_points = edge_distance_points[edge_indices, edge_distance_pair.indices[edge_indices]]
        v = v[edge_indices, edge_distance_pair.indices[edge_indices]]
        gradients[distance_edge_indices] = (edge_distance_points - v) / edge_distances[use_edge_distance].view(-1,1)

        return distances.view(-1,1), gradients