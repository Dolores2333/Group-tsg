"""
N-BEATS Layers
Each layer composes of Generic and Seasonality blocks of different lengths
"""
from typing import Tuple
import numpy as np
import torch as t
import gin


def stack_for_blocks(element, num):
    return t.stack([element.clone() for _ in range(num)], dim=1)


class GroupBlock(t.nn.Module):
    """
    Group blocks for basis function to be assigned
    """
    def __init__(self,
                 num_blocks,
                 input_size,
                 layer_size: int,
                 num_layers: int,
                 theta_size: int,
                 basis_function: t.nn.Module
                 ):
        super(GroupBlock, self).__init__()
        self.layers = t.nn.ModuleList([GroupLinearLayer(din=input_size,
                                                        dout=layer_size,
                                                        num_blocks=num_blocks)] +
                                      [GroupLinearLayer(din=layer_size,
                                                        dout=layer_size,
                                                        num_blocks=num_blocks)
                                       for _ in range(num_layers-1)])
        self.basis_parameters = GroupLinearLayer(din=layer_size,
                                                 dout=theta_size,
                                                 num_blocks=num_blocks)
        self.basis_function = basis_function

    def forward(self, x: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        """
        x is of size (batch_size, num_units, din)
        """
        group_input = x
        for layer in self.layers:
            group_input = t.relu(layer(group_input))
        thetas = self.basis_parameters(group_input)
        # now thetas is of Size (batch_size, num_blocks, theta_size)
        backcast, forecast = self.basis_function(thetas)
        """(bs, num_blocks, backcast_size), (bs, num_blocks, forecast_size)"""
        return backcast, forecast


class GroupBasisFunction(t.nn.Module):
    """
    Realize Group Basis for basis type to be assigned
    """
    def __init__(self,
                 backcast_size,
                 forecast_size,
                 num_blocks,
                 degree,
                 basis_type):
        super(GroupBasisFunction, self).__init__()
        self.num_blocks = num_blocks
        if basis_type == 'Trend':
            self.group_basis_function = TrendBasis(degree_of_polynomial=degree,
                                                   backcast_size=backcast_size,
                                                   forecast_size=forecast_size)
        elif basis_type == 'Seasonality':
            self.group_basis_function = SeasonalityBasis(harmonics=degree,
                                                         backcast_size=backcast_size,
                                                         forecast_size=forecast_size)
        else:
            self.group_basis_function = GenericBasis(backcast_size=backcast_size,
                                                     forecast_size=forecast_size)

    def forward(self, thetas):
        """
        Output should be in the size of (batch_size, num_blocks, output_size)
        """
        """
        Input theta is like (batch_size, num_blocks, theta_size)
        """
        backcasts = []
        forecasts = []
        # change thetas to (num_blocks, batch_size, din)
        thetas = thetas.permute(1, 0, 2)
        # (num_blocks, batch_size, theta_size)

        for i, block_thetas in enumerate(thetas):
            # block_theta is of size (batch_size, theta_size) is the thetas for i-th block
            block_backcasts, block_forecasts = self.group_basis_function(block_thetas)

            # block_backcasts = block_backcasts.numpy().tolist()
            # block_forecasts = block_forecasts.numpy().tolist()
            backcasts.append(block_backcasts)
            forecasts.append(block_forecasts)

        backcasts = t.stack(backcasts, dim=0)
        forecasts = t.stack(forecasts, dim=0)

        backcasts = backcasts.permute(1, 0, 2)
        forecasts = forecasts.permute(1, 0, 2)

        return backcasts, forecasts


class GroupTrend(t.nn.Module):
    def __init__(self,
                 num_blocks,
                 input_size,
                 layer_size: int,
                 num_layers: int,
                 degree_of_polynomial: int,
                 output_size):
        super(GroupTrend, self).__init__()
        self.group_trend_basis = GroupBasisFunction(backcast_size=input_size,
                                                    forecast_size=output_size,
                                                    num_blocks=num_blocks,
                                                    degree=degree_of_polynomial,
                                                    basis_type='Trend')
        self.group_trend_blocks = GroupBlock(num_blocks=num_blocks,
                                             input_size=input_size,
                                             layer_size=layer_size,
                                             num_layers=num_layers,
                                             theta_size=2 * (degree_of_polynomial + 1),
                                             basis_function=self.group_trend_basis)

    def forward(self, x):
        trend_backcasts, trend_forecasts = self.group_trend_blocks(x)
        return trend_backcasts, trend_forecasts


class GroupSeasonality(t.nn.Module):
    def __init__(self,
                 num_blocks,
                 input_size,
                 layer_size: int,
                 num_layers: int,
                 num_of_harmonics: int,
                 output_size):
        super(GroupSeasonality, self).__init__()
        self.group_seasonality_basis = GroupBasisFunction(backcast_size=input_size,
                                                          forecast_size=output_size,
                                                          num_blocks=num_blocks,
                                                          degree=num_of_harmonics,
                                                          basis_type='Seasonality')
        self.group_seasonality_blocks = GroupBlock(num_blocks=num_blocks,
                                                   input_size=input_size,
                                                   layer_size=layer_size,
                                                   num_layers=num_layers,
                                                   theta_size=4 * int(np.ceil(num_of_harmonics
                                                                              / 2
                                                                              * output_size)
                                                                      - (num_of_harmonics - 1)),
                                                   basis_function=self.group_seasonality_basis)

    def forward(self, x):
        seasonality_backcasts, seasonality_forecasts = self.group_seasonality_blocks(x)
        return seasonality_backcasts, seasonality_forecasts


class GroupGeneric(t.nn.Module):
    def __init__(self,
                 num_blocks,
                 input_size,
                 layer_size: int,
                 num_layers: int,
                 output_size):
        super(GroupGeneric, self).__init__()
        self.group_generic_basis = GroupBasisFunction(backcast_size=input_size,
                                                      forecast_size=output_size,
                                                      num_blocks=num_blocks,
                                                      basis_type='Generic',
                                                      degree=0)
        self.group_generic_blocks = GroupBlock(num_blocks=num_blocks,
                                               input_size=input_size,
                                               layer_size=layer_size,
                                               num_layers=num_layers,
                                               theta_size=input_size + output_size,
                                               basis_function=self.group_generic_basis)

    def forward(self, x):
        generic_backcasts, generic_forecasts = self.group_generic_blocks(x)
        return generic_backcasts, generic_forecasts


class GroupNBeatsLayer(t.nn.Module):
    def __init__(self,
                 num_trend_blocks,
                 num_seasonality_blocks,
                 num_generic_blocks,
                 input_size,
                 trend_layer_size,
                 seasonality_layer_size,
                 generic_layer_size,
                 num_trend_layers,
                 num_seasonality_layers,
                 num_generic_layers,
                 degree_of_polynomial,
                 num_of_harmonics,
                 output_size):
        super(GroupNBeatsLayer, self).__init__()
        self.group_trend = GroupTrend(num_blocks=num_trend_blocks,
                                      input_size=input_size,
                                      layer_size=trend_layer_size,
                                      num_layers=num_trend_layers,
                                      degree_of_polynomial=degree_of_polynomial,
                                      output_size=output_size)
        self.group_seasonality = GroupSeasonality(num_blocks=num_seasonality_blocks,
                                                  input_size=input_size,
                                                  layer_size=seasonality_layer_size,
                                                  num_layers=num_seasonality_layers,
                                                  num_of_harmonics=num_of_harmonics,
                                                  output_size=output_size)
        self.group_generic = GroupGeneric(num_blocks=num_generic_blocks,
                                          input_size=input_size,
                                          layer_size=generic_layer_size,
                                          num_layers=num_generic_layers,
                                          output_size=output_size)
        self.num_trend_blocks = num_trend_blocks
        self.num_seasonality_blocks = num_seasonality_blocks
        self.num_generic_blocks = num_generic_blocks

    def forward(self, x):
        x_trend = x[0]
        x_seasonality = x[1]
        x_generic = x[2]
        # (bs, num_blocks, input_size)
        trend_backcasts, trend_forecasts = self.group_trend(x_trend)
        seasonality_backcasts, seasonality_forecasts = self.group_seasonality(x_seasonality)
        generic_backcasts, generic_forecasts = self.group_generic(x_generic)
        """
        Fuse Answers only simple average temporarily
        Actually there are a lot of ways to try
        """
        avg_backcast, avg_forecast = fuse_answers(trend_backcasts,
                                                  seasonality_backcasts,
                                                  generic_backcasts,
                                                  trend_forecasts,
                                                  seasonality_forecasts,
                                                  generic_forecasts)

        backcast, forecast = bloom_answers(avg_backcast,
                                           avg_forecast,
                                           self.num_trend_blocks,
                                           self.num_seasonality_blocks,
                                           self.num_generic_blocks)
        return backcast, forecast


def fuse_answers(trend_backcasts, seasonality_backcasts, generic_backcasts,
                 trend_forecasts, seasonality_forecasts, generic_forecasts):
    # Fuse answers from different types of blocks
    # (bs, num_blocks, back/fore_cast_size) ————> (bs, num_blocks, back/fore_cast_size)
    avg_group_backcasts = group_average(trend_backcasts, seasonality_backcasts, generic_backcasts)
    avg_group_forecasts = group_average(trend_forecasts, seasonality_forecasts, generic_forecasts)

    # Fuse answers from different blocks
    # (bs, num_blocks, back/fore_cast_size) ————> (bs, back/fore_cast_size)
    avg_backcast = block_average(avg_group_backcasts)
    avg_forecast = block_average(avg_group_forecasts)
    return avg_backcast, avg_forecast


def bloom_answers(avg_backcast, avg_forecast, num_trend_blocks, num_seasonality_blocks, num_generic_blocks):
    # Reconstruct the size to (bs, num_blocks, back/fore_cast_size)
    trend_group_backcasts = stack_for_blocks(avg_backcast, num_trend_blocks)
    seasonality_group_backcasts = stack_for_blocks(avg_backcast, num_seasonality_blocks)
    generic_group_backcasts = stack_for_blocks(avg_backcast, num_generic_blocks)

    trend_group_forecasts = stack_for_blocks(avg_forecast, num_trend_blocks)
    seasonality_group_forecasts = stack_for_blocks(avg_forecast, num_seasonality_blocks)
    generic_group_forecasts = stack_for_blocks(avg_forecast, num_generic_blocks)

    # Reconstruct the size to (group_type, bs, num_blocks, back/fore_cast_size)
    backcast = t.stack((trend_group_backcasts,
                        seasonality_group_backcasts,
                        generic_group_backcasts), dim=0)
    forecast = t.stack((trend_group_forecasts,
                        seasonality_group_forecasts,
                        generic_group_forecasts), dim=0)

    return backcast, forecast


""" 
Fuse backcast, forecast from trend, seasonality and generic blocks
t_, s_, g_ is of the size (batch_size, num_blocks, back/fore_cast_size)
the output should be of the size (batch_size, num_blocks, back/fore_cast_size)
"""


def group_average(t_, s_, g_):
    total = t.stack((t_, s_, g_), dim=0)
    avg = (total * 1.0).mean(dim=0, keepdim=False)
    return avg


"""
Average backcast, forecast among blocks
x_ is of the size (batch_size, num_blocks, back/fore_cast_size)
the output should be of the size(batch_size, back/fore_cast_size)
"""


def block_average(x_):
    return (x_ * 1.0).mean(dim=1, keepdim=False)


@gin.configurable()
class GroupNBeats(t.nn.Module):
    def __init__(self,
                 num_trend_blocks,
                 num_seasonality_blocks,
                 num_generic_blocks,
                 input_size,
                 trend_layer_size,
                 seasonality_layer_size,
                 generic_layer_size,
                 num_trend_layers,
                 num_seasonality_layers,
                 num_generic_layers,
                 output_size,
                 num_model_layers,
                 degree_of_polynomial,
                 num_of_harmonics):
        super(GroupNBeats, self).__init__()
        self.layer = GroupNBeatsLayer(num_trend_blocks=num_trend_blocks,
                                      num_seasonality_blocks=num_seasonality_blocks,
                                      num_generic_blocks=num_generic_blocks,
                                      input_size=input_size,
                                      trend_layer_size=trend_layer_size,
                                      seasonality_layer_size=seasonality_layer_size,
                                      generic_layer_size=generic_layer_size,
                                      num_trend_layers=num_trend_layers,
                                      num_seasonality_layers=num_seasonality_layers,
                                      num_generic_layers=num_generic_layers,
                                      output_size=output_size,
                                      degree_of_polynomial=degree_of_polynomial,
                                      num_of_harmonics=num_of_harmonics)
        """
        num_groups is the number of Layers composes of 
        trend, seasonality and generic groups
        """
        self.layers = t.nn.ModuleList([self.layer for _ in range(num_model_layers)])

    """
    Temporarily, the out put is of size 
    (group_type, batch_size, num_blocks_in_group, vector_size)
    what is the size of forecast?
    """

    def forward(self, x: t.Tensor, input_mask: t.Tensor) -> t.Tensor:
        residuals = x.flip(dims=(3,))
        input_mask = input_mask.flip(dims=(3,))
        forecast = x[:, :, :, -1:]
        for i, layer in enumerate(self.layers):
            layer_backcast, layer_forecast = layer(residuals)
            residuals = (residuals - layer_backcast) * input_mask
            forecast = layer_forecast + forecast
        # (group_types, bs, num_blocks, size)-->(bs, size)
        trend_forecasts = forecast[0]
        seasonality_forecasts = forecast[1]
        generic_forecasts = forecast[2]
        avg_group_forecasts = group_average(trend_forecasts,
                                            seasonality_forecasts,
                                            generic_forecasts)
        forecast = block_average(avg_group_forecasts)
        return forecast


"""
The output is of the size (batch_size, backcast_size), (batch_size, forecast_size)
"""


class GenericBasis(t.nn.Module):
    """
    Generic basis function.
    """
    def __init__(self, backcast_size: int, forecast_size: int):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, theta: t.Tensor):
        return theta[:, :self.backcast_size], theta[:, -self.forecast_size:]


class TrendBasis(t.nn.Module):
    """
    Polynomial function to model trend.
    """
    def __init__(self, degree_of_polynomial: int, backcast_size: int, forecast_size: int):
        super().__init__()
        self.polynomial_size = degree_of_polynomial + 1  # degree of polynomial with constant term
        self.backcast_time = t.nn.Parameter(
            t.tensor(np.concatenate([np.power(np.arange(backcast_size, dtype=np.float) / backcast_size, i)[None, :]
                                     for i in range(self.polynomial_size)]), dtype=t.float32),
            requires_grad=False)
        self.forecast_time = t.nn.Parameter(
            t.tensor(np.concatenate([np.power(np.arange(forecast_size, dtype=np.float) / forecast_size, i)[None, :]
                                     for i in range(self.polynomial_size)]), dtype=t.float32), requires_grad=False)

    def forward(self, theta: t.Tensor):
        backcast = t.einsum('bp,pt->bt', theta[:, self.polynomial_size:], self.backcast_time)
        forecast = t.einsum('bp,pt->bt', theta[:, :self.polynomial_size], self.forecast_time)
        return backcast, forecast


class SeasonalityBasis(t.nn.Module):
    """
    Harmonic functions to model seasonality.
    """
    def __init__(self, harmonics: int, backcast_size: int, forecast_size: int):
        super().__init__()
        self.frequency = np.append(np.zeros(1, dtype=np.float32),
                                   np.arange(harmonics, harmonics / 2 * forecast_size,
                                             dtype=np.float32) / harmonics)[None, :]
        backcast_grid = -2 * np.pi * (
                np.arange(backcast_size, dtype=np.float32)[:, None] / forecast_size) * self.frequency
        forecast_grid = 2 * np.pi * (
                np.arange(forecast_size, dtype=np.float32)[:, None] / forecast_size) * self.frequency
        self.backcast_cos_template = t.nn.Parameter(t.tensor(np.transpose(np.cos(backcast_grid)), dtype=t.float32),
                                                    requires_grad=False)
        self.backcast_sin_template = t.nn.Parameter(t.tensor(np.transpose(np.sin(backcast_grid)), dtype=t.float32),
                                                    requires_grad=False)
        self.forecast_cos_template = t.nn.Parameter(t.tensor(np.transpose(np.cos(forecast_grid)), dtype=t.float32),
                                                    requires_grad=False)
        self.forecast_sin_template = t.nn.Parameter(t.tensor(np.transpose(np.sin(forecast_grid)), dtype=t.float32),
                                                    requires_grad=False)

    def forward(self, theta: t.Tensor):
        params_per_harmonic = theta.shape[1] // 4
        backcast_harmonics_cos = t.einsum('bp,pt->bt', theta[:, 2 * params_per_harmonic:3 * params_per_harmonic],
                                          self.backcast_cos_template)
        backcast_harmonics_sin = t.einsum('bp,pt->bt', theta[:, 3 * params_per_harmonic:], self.backcast_sin_template)
        backcast = backcast_harmonics_sin + backcast_harmonics_cos
        forecast_harmonics_cos = t.einsum('bp,pt->bt',
                                          theta[:, :params_per_harmonic], self.forecast_cos_template)
        forecast_harmonics_sin = t.einsum('bp,pt->bt', theta[:, params_per_harmonic:2 * params_per_harmonic],
                                          self.forecast_sin_template)
        forecast = forecast_harmonics_sin + forecast_harmonics_cos

        return backcast, forecast


class GroupLinearLayer(t.nn.Module):
    def __init__(self, din, dout, num_blocks):
        super(GroupLinearLayer, self).__init__()

        self.w = t.nn.Parameter(0.01 * t.randn(num_blocks, din, dout))
    """
    x = (batch_size, num_blocks, din)
    w = (num_blocks, din, dout)
    x <—— x.permute(1, 0, 2) = (num_blocks, batch_size, din)
    torch.bmm(x, w) = (num_blocks, batch_size, dout)
    x <—— x.permute(1, 0, 2) = (batch_size, num_units, dout)
    """
    def forward(self, x):
        # print(x.detach().numpy().shape)
        x = x.permute(1, 0, 2)
        x = t.bmm(x, self.w)
        return x.permute(1, 0, 2)
