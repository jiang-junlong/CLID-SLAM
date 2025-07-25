#!/usr/bin/env python3
# @file      data_sampler.py
# @author    Junlong Jiang     [jiangjunlong@mail.dlut.edu.cn]
# Copyright (c) 2025 Junlong Jiang, all rights reserved

import torch

from model.local_point_cloud_map import LocalPointCloudMap
from utils.config import Config
from utils.tools import transform_torch


class DataSampler:
    def __init__(self, config: Config):
        self.config = config
        self.dev = config.device

    def sample_pin(self, points_torch, normal_torch, sem_label_torch, color_torch):
        """
        Sample training sample points for current scan, get the labels for online training
        input and output are all torch tensors
        points_torch is in the sensor's local coordinate system, not yet transformed to the global system
        """

        # T0 = get_time()

        dev = self.dev
        surface_sample_range = self.config.surface_sample_range_m
        surface_sample_n = self.config.surface_sample_n
        freespace_behind_sample_n = self.config.free_behind_n
        freespace_front_sample_n = self.config.free_front_n
        all_sample_n = (
            surface_sample_n + freespace_behind_sample_n + freespace_front_sample_n + 1
        )  # 1 as the exact measurement
        free_front_min_ratio = self.config.free_sample_begin_ratio
        free_sample_end_dist = self.config.free_sample_end_dist_m

        # get sample points
        point_num = points_torch.shape[0]
        distances = torch.linalg.norm(
            points_torch, dim=1, keepdim=True
        )  # ray distances (scaled)

        # Part 0. the exact measured point
        measured_sample_displacement = torch.zeros_like(distances)
        measured_sample_dist_ratio = torch.ones_like(distances)

        # Part 1. close-to-surface uniform sampling
        # uniform sample in the close-to-surface range (+- range)
        surface_sample_displacement = (
            torch.randn(point_num * surface_sample_n, 1, device=dev)
            * surface_sample_range
        )

        repeated_dist = distances.repeat(surface_sample_n, 1)
        surface_sample_dist_ratio = (
            surface_sample_displacement / repeated_dist + 1.0
        )  # 1.0 means on the surface
        if sem_label_torch is not None:
            surface_sem_label_tensor = sem_label_torch.repeat(
                1, surface_sample_n
            ).transpose(0, 1)
        if color_torch is not None:
            color_channel = color_torch.shape[1]
            surface_color_tensor = color_torch.repeat(surface_sample_n, 1)

        # Part 2. free space (in front of surface) uniform sampling
        # if you want to reconstruct the thin objects (like poles, tree branches) well, you need more freespace samples to have
        # a space carving effect

        sigma_ratio = 2.0
        repeated_dist = distances.repeat(freespace_front_sample_n, 1)
        free_max_ratio = 1.0 - sigma_ratio * surface_sample_range / repeated_dist
        free_diff_ratio = free_max_ratio - free_front_min_ratio
        free_sample_front_dist_ratio = (
            torch.rand(point_num * freespace_front_sample_n, 1, device=dev)
            * free_diff_ratio
            + free_front_min_ratio
        )
        free_sample_front_displacement = (
            free_sample_front_dist_ratio - 1.0
        ) * repeated_dist
        if sem_label_torch is not None:
            free_sem_label_front = torch.zeros_like(repeated_dist)
        if color_torch is not None:
            free_color_front = torch.zeros(
                point_num * freespace_front_sample_n, color_channel, device=dev
            )

        # Part 3. free space (behind surface) uniform sampling
        repeated_dist = distances.repeat(freespace_behind_sample_n, 1)
        free_max_ratio = free_sample_end_dist / repeated_dist + 1.0
        free_behind_min_ratio = 1.0 + sigma_ratio * surface_sample_range / repeated_dist
        free_diff_ratio = free_max_ratio - free_behind_min_ratio

        free_sample_behind_dist_ratio = (
            torch.rand(point_num * freespace_behind_sample_n, 1, device=dev)
            * free_diff_ratio
            + free_behind_min_ratio
        )

        free_sample_behind_displacement = (
            free_sample_behind_dist_ratio - 1.0
        ) * repeated_dist
        if sem_label_torch is not None:
            free_sem_label_behind = torch.zeros_like(repeated_dist)
        if color_torch is not None:
            free_color_behind = torch.zeros(
                point_num * freespace_behind_sample_n, color_channel, device=dev
            )

        # T1 = get_time()

        # all together
        all_sample_displacement = torch.cat(
            (
                measured_sample_displacement,
                surface_sample_displacement,
                free_sample_front_displacement,
                free_sample_behind_displacement,
            ),
            0,
        )
        all_sample_dist_ratio = torch.cat(
            (
                measured_sample_dist_ratio,
                surface_sample_dist_ratio,
                free_sample_front_dist_ratio,
                free_sample_behind_dist_ratio,
            ),
            0,
        )

        repeated_points = points_torch.repeat(all_sample_n, 1)
        repeated_dist = distances.repeat(all_sample_n, 1)
        all_sample_points = repeated_points * all_sample_dist_ratio

        # depth tensor of all the samples
        depths_tensor = repeated_dist * all_sample_dist_ratio

        # get the weight vector as the inverse of sigma
        weight_tensor = torch.ones_like(depths_tensor)

        surface_sample_count = point_num * (surface_sample_n + 1)
        if (
            self.config.dist_weight_on
        ):  # far away surface samples would have lower weight
            weight_tensor[:surface_sample_count] = (
                1
                + self.config.dist_weight_scale * 0.5
                - (repeated_dist[:surface_sample_count] / self.config.max_range)
                * self.config.dist_weight_scale
            )  # [0.6, 1.4]
        # TODO: also try add lower weight for surface samples with large incidence angle

        # behind surface weight drop-off because we have less uncertainty behind the surface
        if self.config.behind_dropoff_on:
            dropoff_min = 0.2 * free_sample_end_dist
            dropoff_max = free_sample_end_dist
            dropoff_diff = dropoff_max - dropoff_min
            behind_displacement = all_sample_displacement
            dropoff_weight = (dropoff_max - behind_displacement) / dropoff_diff
            dropoff_weight = torch.clamp(dropoff_weight, min=0.0, max=1.0)
            dropoff_weight = dropoff_weight * 0.8 + 0.2
            weight_tensor = weight_tensor * dropoff_weight

        # give a flag indicating the type of the sample [negative: freespace, positive: surface]
        weight_tensor[surface_sample_count:] *= -1.0

        # ray-wise depth
        distances = distances.squeeze(1)

        # assign sdf labels to the samples
        # projective distance as the label: behind +, in-front -
        sdf_label_tensor = all_sample_displacement.squeeze(
            1
        )  # scaled [-1, 1] # as distance (before sigmoid)

        # assign the normal label to the samples
        normal_label_tensor = None
        if normal_torch is not None:
            normal_label_tensor = normal_torch.repeat(all_sample_n, 1)

        # assign the semantic label to the samples (including free space as the 0 label)
        sem_label_tensor = None
        if sem_label_torch is not None:
            sem_label_tensor = torch.cat(
                (
                    sem_label_torch.unsqueeze(-1),
                    surface_sem_label_tensor,
                    free_sem_label_front,
                    free_sem_label_behind,
                ),
                0,
            ).int()

        # assign the color label to the close-to-surface samples
        color_tensor = None
        if color_torch is not None:
            color_tensor = torch.cat(
                (
                    color_torch,
                    surface_color_tensor,
                    free_color_front,
                    free_color_behind,
                ),
                0,
            )

        # T2 = get_time()
        # Convert from the all ray surface + all ray free order to the ray-wise (surface + free) order
        all_sample_points = (
            all_sample_points.reshape(all_sample_n, -1, 3)
            .transpose(0, 1)
            .reshape(-1, 3)
        )
        sdf_label_tensor = (
            sdf_label_tensor.reshape(all_sample_n, -1).transpose(0, 1).reshape(-1)
        )
        sdf_label_tensor *= -1  # convert to the same sign as

        weight_tensor = (
            weight_tensor.reshape(all_sample_n, -1).transpose(0, 1).reshape(-1)
        )
        # depths_tensor = depths_tensor.reshape(all_sample_n, -1).transpose(0, 1).reshape(-1)

        if normal_torch is not None:
            normal_label_tensor = (
                normal_label_tensor.reshape(all_sample_n, -1, 3)
                .transpose(0, 1)
                .reshape(-1, 3)
            )
        if sem_label_torch is not None:
            sem_label_tensor = (
                sem_label_tensor.reshape(all_sample_n, -1).transpose(0, 1).reshape(-1)
            )
        if color_torch is not None:
            color_tensor = (
                color_tensor.reshape(all_sample_n, -1, color_channel)
                .transpose(0, 1)
                .reshape(-1, color_channel)
            )

        # ray distance (distances) is not repeated

        # T3 = get_time()

        # print("time for sampling I:", T1-T0)
        # print("time for sampling II:", T2-T1)
        # print("time for sampling III:", T3-T2)
        # all super fast, all together in 0.5 ms

        return (
            all_sample_points,
            sdf_label_tensor,
            normal_label_tensor,
            sem_label_tensor,
            color_tensor,
            weight_tensor,
        )

    def sample(
        self, points_torch, local_point_cloud_map: LocalPointCloudMap, cur_pose_torch
    ):
        dev = self.dev
        surface_sample_range = self.config.surface_sample_range_m
        surface_sample_n = self.config.surface_sample_n
        freespace_behind_sample_n = self.config.free_behind_n
        freespace_front_sample_n = self.config.free_front_n
        all_sample_n = (
            surface_sample_n + freespace_behind_sample_n + freespace_front_sample_n + 1
        )  # 1 as the exact measurement
        free_front_min_ratio = self.config.free_sample_begin_ratio
        free_sample_end_dist = self.config.free_sample_end_dist_m

        # get sample points
        point_num = points_torch.shape[0]
        distances = torch.linalg.norm(
            points_torch, dim=1, keepdim=True
        )  # ray distances (scaled)

        # Part 0. the exact measured point
        measured_sample_displacement = torch.zeros_like(distances)
        measured_sample_dist_ratio = torch.ones_like(distances)

        # Part 1. close-to-surface uniform sampling
        # uniform sample in the close-to-surface range (+- range)
        surface_sample_displacement = (
            torch.randn(point_num * surface_sample_n, 1, device=dev)
            * surface_sample_range
        )

        repeated_dist = distances.repeat(surface_sample_n, 1)
        surface_sample_dist_ratio = (
            surface_sample_displacement / repeated_dist + 1.0
        )  # 1.0 means on the surface

        # Part 2. free space (in front of surface) uniform sampling
        # if you want to reconstruct the thin objects (like poles, tree branches) well, you need more freespace samples to have
        # a space carving effect

        sigma_ratio = 2.0
        repeated_dist = distances.repeat(freespace_front_sample_n, 1)
        free_max_ratio = 1.0 - sigma_ratio * surface_sample_range / repeated_dist
        free_diff_ratio = free_max_ratio - free_front_min_ratio
        free_sample_front_dist_ratio = (
            torch.rand(point_num * freespace_front_sample_n, 1, device=dev)
            * free_diff_ratio
            + free_front_min_ratio
        )
        free_sample_front_displacement = (
            free_sample_front_dist_ratio - 1.0
        ) * repeated_dist

        # Part 3. free space (behind surface) uniform sampling
        repeated_dist = distances.repeat(freespace_behind_sample_n, 1)
        free_max_ratio = free_sample_end_dist / repeated_dist + 1.0
        free_behind_min_ratio = 1.0 + sigma_ratio * surface_sample_range / repeated_dist
        free_diff_ratio = free_max_ratio - free_behind_min_ratio

        free_sample_behind_dist_ratio = (
            torch.rand(point_num * freespace_behind_sample_n, 1, device=dev)
            * free_diff_ratio
            + free_behind_min_ratio
        )

        free_sample_behind_displacement = (
            free_sample_behind_dist_ratio - 1.0
        ) * repeated_dist

        # all together
        all_sample_displacement = torch.cat(
            (
                measured_sample_displacement,
                surface_sample_displacement,
                free_sample_front_displacement,
                free_sample_behind_displacement,
            ),
            0,
        )
        all_sample_dist_ratio = torch.cat(
            (
                measured_sample_dist_ratio,
                surface_sample_dist_ratio,
                free_sample_front_dist_ratio,
                free_sample_behind_dist_ratio,
            ),
            0,
        )

        repeated_points = points_torch.repeat(all_sample_n, 1)
        repeated_dist = distances.repeat(all_sample_n, 1)
        all_sample_points = repeated_points * all_sample_dist_ratio
        ####################################### Added By Jiang Junlong #################################################
        # 根据表面采样平移量计算符号
        sdf_sign = torch.where(surface_sample_displacement.squeeze(1) < 0, 1, -1)
        mask = torch.ones(
            point_num * all_sample_n, dtype=torch.bool, device=self.config.device
        )

        sdf_label_tensor = -1 * all_sample_displacement.squeeze(1)
        surface_sample_count = point_num * (surface_sample_n + 1)
        surface_sample_points = all_sample_points[point_num:surface_sample_count]
        surface_sample_points_G = transform_torch(surface_sample_points, cur_pose_torch)
        dist, valid_mask = local_point_cloud_map.region_specific_sdf_estimation(
            surface_sample_points_G
        )
        mask[point_num:surface_sample_count] = valid_mask
        sdf_label_tensor[point_num:surface_sample_count] = sdf_sign * dist
        # sdf_label = torch.clamp(sdf_label, -0.4, 0.4)

        # depth tensor of all the samples
        depths_tensor = repeated_dist * all_sample_dist_ratio
        # get the weight vector as the inverse of sigma
        weight_tensor = torch.ones_like(depths_tensor)
        if (
            self.config.dist_weight_on
        ):  # far away surface samples would have lower weight
            weight_tensor[:surface_sample_count] = (
                1
                + self.config.dist_weight_scale * 0.5
                - (repeated_dist[:surface_sample_count] / self.config.max_range)
                * self.config.dist_weight_scale
            )  # [0.6, 1.4]

        weight_tensor[surface_sample_count:] *= -1.0

        # Convert from the all ray surface + all ray free order to the ray-wise (surface + free) order

        all_sample_points = (
            all_sample_points.reshape(all_sample_n, -1, 3)
            .transpose(0, 1)
            .reshape(-1, 3)
        )
        sdf_label_tensor = (
            sdf_label_tensor.reshape(all_sample_n, -1).transpose(0, 1).reshape(-1)
        )

        weight_tensor = (
            weight_tensor.reshape(all_sample_n, -1).transpose(0, 1).reshape(-1)
        )

        mask = mask.reshape(all_sample_n, -1).transpose(0, 1).reshape(-1)
        return all_sample_points[mask], sdf_label_tensor[mask], weight_tensor[mask]
