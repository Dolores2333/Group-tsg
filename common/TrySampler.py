# -*- coding: utf-8 _*_
# @Time : 24/6/2021 10:37 am
# @Author: ZHA Mengyue
# @FileName: TrySampler.py
# @Software: N-BEATScopy
# @Blog: https://github.com/Dolores2333

"""
Timeseries Sampler
"""

import numpy as np
import gin


"""
Temporarily, the out put is of size 
(group_type, batch_size, num_blocks_in_group, vector_size)
"""


def stack_for_blocks(element, num):
    return np.stack([element for _ in range(num)], axis=1)


@gin.configurable
class TimeseriesSampler:
    def __init__(self,
                 timeseries: np.ndarray,
                 insample_size: int,
                 outsample_size: int,
                 window_sampling_limit: int,
                 num_trend_blocks: int,
                 num_seasonality_blocks: int,
                 num_generic_blocks: int,
                 batch_size: int = 1024):
        """
        Timeseries sampler.

        :param timeseries: Timeseries data to sample from. Shape: timeseries, timesteps
        :param insample_size: Insample window size. If timeseries is shorter then it will be 0-padded and masked.
        :param outsample_size: Outsample window size. If timeseries is shorter then it will be 0-padded and masked.
        :param window_sampling_limit: Size of history the sampler should use.
        :param batch_size: Number of sampled windows.
        """
        self.timeseries = [ts for ts in timeseries]
        self.window_sampling_limit = window_sampling_limit
        self.batch_size = batch_size
        self.insample_size = insample_size
        self.outsample_size = outsample_size
        self.num_trend_blocks = num_trend_blocks
        self.num_seasonality_blocks = num_seasonality_blocks
        self.num_generic_blocks = num_generic_blocks

    def __iter__(self):
        """
        Batches of sampled windows.

        :return: Batches of:
         Insample: "batch size, insample size"
         Insample mask: "batch size, insample size"
         Outsample: "batch size, outsample size"
         Outsample mask: "batch size, outsample size"
        """
        while True:
            insample = np.zeros((self.batch_size, self.insample_size))
            insample_mask = np.zeros((self.batch_size, self.insample_size))
            outsample = np.zeros((self.batch_size, self.outsample_size))
            outsample_mask = np.zeros((self.batch_size, self.outsample_size))
            """
            len(2d array) returns the number of rows of this array
            np.random.randint(random range, random repeats)
            """
            sampled_ts_indices = np.random.randint(len(self.timeseries),
                                                   size=self.batch_size)

            for i, sample_index in enumerate(sampled_ts_indices):
                sampled_timeseries = self.timeseries[sample_index]
                cut_point = np.random.randint(low=max(1, len(sampled_timeseries)-self.window_sampling_limit),
                                              high=len(sampled_timeseries),
                                              size=1)[0]

                lower_bound = max(0, cut_point-self.insample_size)
                higher_bound = min(len(sampled_timeseries), cut_point+self.outsample_size)

                insample_window = sampled_timeseries[lower_bound:cut_point]
                insample[i, -len(insample_window):] = insample_window
                insample_mask[i, -len(insample_window):] = 1.0

                outsample_window = sampled_timeseries[cut_point:higher_bound]
                outsample[i, :len(outsample_window)] = outsample_window
                outsample_mask[i, :len(outsample_window)] = 1.0

            trend_group_insample = stack_for_blocks(insample, self.num_trend_blocks)
            trend_group_insample_mask = stack_for_blocks(insample_mask, self.num_trend_blocks)
            seasonality_group_insample = stack_for_blocks(insample, self.num_seasonality_blocks)
            seasonality_group_insample_mask = stack_for_blocks(insample_mask, self.num_seasonality_blocks)
            generic_group_insample = stack_for_blocks(insample, self.num_generic_blocks)
            generic_group_insample_mask = stack_for_blocks(insample_mask, self.num_generic_blocks)

            layer_insample = np.stack((trend_group_insample,
                                       seasonality_group_insample,
                                       generic_group_insample),
                                      axis=0)
            layer_insample_mask = np.stack((trend_group_insample_mask,
                                            seasonality_group_insample_mask,
                                            generic_group_insample_mask),
                                           axis=0)
            """
            trend_group_outsample = stack_for_blocks(outsample, self.num_trend_blocks)
            trend_group_outsample_mask = stack_for_blocks(outsample_mask, self.num_trend_blocks)
            seasonality_group_outsample = stack_for_blocks(outsample, self.num_seasonality_blocks)
            seasonality_group_outsample_mask = stack_for_blocks(outsample_mask, self.num_seasonality_blocks)
            generic_group_outsample = stack_for_blocks(outsample, self.num_generic_blocks)
            generic_group_outsample_mask = stack_for_blocks(outsample_mask, self.num_generic_blocks)

            layer_outsample = np.stack((trend_group_outsample,
                                        seasonality_group_outsample,
                                        generic_group_outsample),
                                       axis=0)
            layer_outsample_mask = np.stack((trend_group_outsample_mask,
                                             seasonality_group_outsample_mask,
                                             generic_group_outsample_mask),
                                            axis=0)
            """
            layer_outsample = outsample
            layer_outsample_mask = outsample_mask
            yield layer_insample, layer_insample_mask, layer_outsample, layer_outsample_mask

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.insample_size))
        insample_mask = np.zeros((len(self.timeseries), self.insample_size))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.insample_size:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0

        trend_group_insample = stack_for_blocks(insample, self.num_trend_blocks)
        trend_group_insample_mask = stack_for_blocks(insample_mask, self.num_trend_blocks)
        seasonality_group_insample = stack_for_blocks(insample, self.num_seasonality_blocks)
        seasonality_group_insample_mask = stack_for_blocks(insample_mask, self.num_seasonality_blocks)
        generic_group_insample = stack_for_blocks(insample, self.num_generic_blocks)
        generic_group_insample_mask = stack_for_blocks(insample_mask, self.num_generic_blocks)

        layer_insample = np.stack((trend_group_insample,
                                   seasonality_group_insample,
                                   generic_group_insample),
                                  axis=0)
        layer_insample_mask = np.stack((trend_group_insample_mask,
                                        seasonality_group_insample_mask,
                                        generic_group_insample_mask),
                                       axis=0)

        return layer_insample, layer_insample_mask
