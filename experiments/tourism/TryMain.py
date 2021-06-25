# -*- coding: utf-8 _*_
# @Time : 20/6/2021 3:30 pm
# @Author: ZHA Mengyue
# @FileName: TryMain.py
# @Software: N-BEATScopy
# @Blog: https://github.com/Dolores2333


"""
Tourism Experiment
"""
import logging
import os
from typing import Dict

import gin
import numpy as np
import pandas as pd
import torch as t
from fire import Fire

from common.experiment import Experiment
from common.TrySampler import TimeseriesSampler
from common.torch.ops import to_tensor
from common.torch.snapshots import SnapshotManager
from datasets.tourism import TourismDataset, TourismMeta
from experiments.trainer import trainer
from summary.utils import group_values

from models.TryNBeats import GroupNBeats


"""
TourismExperiment inherits from Experiment
applies on Group N-BEATS
"""


class TourismExperiment(Experiment):
    @gin.configurable()
    def instance(self,
                 repeat: int,
                 lookback: int,
                 loss: str,
                 history_size: Dict[str, float],
                 iterations: Dict[str, int],
                 model_type: str):
        dataset = TourismDataset.load(training=True)
        """
        return TourismDataset(ids=np.array(ids),
                              groups=np.array(groups),
                              horizons=np.array(horizons),
                              values=np.array(values))
        id[i] is the id of i-th ts
        groups[i] indicates the group i-th ts belongs to, like Monthly, Yearly...
        horizon[i] is the horizon aka outsample_size aka forecast_size of i-th ts
        values[i, :] is the values of i-th ts
        """

        forecasts = []
        for seasonal_pattern in TourismMeta.seasonal_patterns:
            history_size_in_horizon = history_size[seasonal_pattern]
            horizon = TourismMeta.horizons_map[seasonal_pattern]
            input_size = lookback * horizon

            # Training Set
            """
            Filter values array by group indices and clean it from NaNs.
            """
            training_values = group_values(dataset.values, dataset.groups, seasonal_pattern)

            '''
            Obtain the training_set, model, snapshot_manager
            '''
            # now traning_set is an instance
            """
            Return: batch * insample
                    batch * insample_mask
                    batch * outsample
                    batch * outsample_mask
            """
            training_set = TimeseriesSampler(timeseries=training_values,
                                             insample_size=input_size,
                                             outsample_size=horizon,
                                             window_sampling_limit=int(horizon*history_size_in_horizon))
            """
            Only the last positions of length window_sampling_limit+input_size can be utilized
            if len(ts)-wimdow_sampling <1 the whole ts will be used
            """
            # Following initializes the model architecture
            model = GroupNBeats(input_size=input_size, output_size=horizon)
            # Training Part
            snapshot_manager = SnapshotManager(snapshot_dir=os.path.join(self.root,
                                                                         'snapshots',
                                                                         seasonal_pattern),
                                               total_iterations=iterations[seasonal_pattern])
            # Train the model
            model = trainer(snapshot_manager=snapshot_manager,
                            model=model,
                            training_set=iter(training_set),
                            timeseries_frequency=TourismMeta.frequency_map[seasonal_pattern],
                            loss_name=loss,
                            iterations=iterations[seasonal_pattern])

            # Evaluate the model
            # Build forecasts for each ts
            x, x_mask = map(to_tensor, training_set.last_insample_window())
            model.eval()
            with t.no_grad():
                forecasts.extend(model(x, x_mask).cpu().detach().numpy())

        forecasts_df = pd.DataFrame(forecasts,
                                    columns=[f'V{i+1}'
                                             for i in range(np.max(TourismMeta.horizons))])
        forecasts_df.index = dataset.ids
        forecasts_df.index.name = 'id'
        forecasts_df.to_csv(os.path.join(self.root,
                                         'forecast.csv'))


if __name__ == '__main__':
    logging.root.setLevel(logging.INFO)
    Fire(TourismExperiment)
