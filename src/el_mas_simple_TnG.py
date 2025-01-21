#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 01:16:09 2025

@author: lucas
"""

import torch
import torch.nn.functional as F
from math import sqrt

class GnT(object):
    """
    Generate-and-Test algorithm for feed forward neural networks with one hidden layer,
    using maturity-threshold based replacement.
    """
    def __init__(
            self,
            net,
            hidden_activation,
            opt,
            decay_rate=0.99,
            replacement_rate=1e-4,
            init='kaiming',
            device="cpu",
            maturity_threshold=20,
            loss_func=F.mse_loss,
            accumulate=False,
    ):
        super(GnT, self).__init__()
        self.device = device
        self.net = net
        self.loss_func = loss_func
        self.accumulate = accumulate

        # Store optimizer
        self.opt = opt

        # Algorithm hyperparameters
        self.replacement_rate = replacement_rate
        self.decay_rate = decay_rate
        self.maturity_threshold = maturity_threshold

        # Utility tracking for hidden layer
        self.util = torch.zeros(self.net.hidden.out_features).to(self.device)
        self.bias_corrected_util = torch.zeros(self.net.hidden.out_features).to(self.device)
        self.ages = torch.zeros(self.net.hidden.out_features).to(self.device)
        self.mean_feature_act = torch.zeros(self.net.hidden.out_features).to(self.device)
        self.accumulated_num_features_to_replace = 0

        # Calculate initialization bounds
        if hidden_activation == 'selu':
            init = 'lecun'
        self.bound = self.compute_bound(hidden_activation=hidden_activation, init=init)

    def compute_bound(self, hidden_activation, init='kaiming'):
        if hidden_activation in ['swish', 'elu']:
            hidden_activation = 'relu'
            
        if init == 'default':
            bound = sqrt(1 / self.net.hidden.in_features)
        elif init == 'xavier':
            bound = (torch.nn.init.calculate_gain(nonlinearity=hidden_activation) * 
                    sqrt(6 / (self.net.hidden.in_features + self.net.hidden.out_features)))
        elif init == 'lecun':
            bound = sqrt(3 / self.net.hidden.in_features)
        else:
            bound = (torch.nn.init.calculate_gain(nonlinearity=hidden_activation) * 
                    sqrt(3 / self.net.hidden.in_features))
        return bound

    def update_utility(self, features):
        with torch.no_grad():
            self.util *= self.decay_rate
            
            # Bias correction
            bias_correction = 1 - self.decay_rate ** self.ages
            
            self.mean_feature_act *= self.decay_rate
            self.mean_feature_act += (1 - self.decay_rate) * features.mean(dim=0)
            bias_corrected_act = self.mean_feature_act / bias_correction

            output_weight_mag = self.net.output.weight.data.abs().mean(dim=0)
            new_util = output_weight_mag * features.abs().mean(dim=0)

            self.util += (1 - self.decay_rate) * new_util
            self.bias_corrected_util = self.util / bias_correction

    def test_features(self, features):
        if self.replacement_rate == 0:
            return torch.empty(0, dtype=torch.long).to(self.device), 0

        self.ages += 1
        self.update_utility(features)

        # Find eligible features
        eligible_feature_indices = torch.where(self.ages > self.maturity_threshold)[0]
        if eligible_feature_indices.shape[0] == 0:
            return torch.empty(0, dtype=torch.long).to(self.device), 0

        num_new_features_to_replace = self.replacement_rate * eligible_feature_indices.shape[0]
        
        if self.accumulate:
            self.accumulated_num_features_to_replace += num_new_features_to_replace
            num_new_features_to_replace = int(self.accumulated_num_features_to_replace)
            self.accumulated_num_features_to_replace -= num_new_features_to_replace
        else:
            if num_new_features_to_replace < 1:
                if torch.rand(1) <= num_new_features_to_replace:
                    num_new_features_to_replace = 1
            num_new_features_to_replace = int(num_new_features_to_replace)

        if num_new_features_to_replace == 0:
            return torch.empty(0, dtype=torch.long).to(self.device), 0

        # Find features to replace
        features_to_replace = torch.topk(-self.bias_corrected_util[eligible_feature_indices],
                                       num_new_features_to_replace)[1]
        features_to_replace = eligible_feature_indices[features_to_replace]

        # Reset utility for new features
        self.util[features_to_replace] = 0
        self.mean_feature_act[features_to_replace] = 0.

        return features_to_replace, num_new_features_to_replace

    def gen_new_features(self, features_to_replace, num_features_to_replace):
        if num_features_to_replace == 0:
            return

        with torch.no_grad():
            # Reset and reinitialize input weights
            self.net.hidden.weight.data[features_to_replace, :] = 0.0
            self.net.hidden.weight.data[features_to_replace, :] += torch.empty(
                num_features_to_replace, self.net.hidden.in_features
            ).uniform_(-self.bound, self.bound).to(self.device)
            
            self.net.hidden.bias.data[features_to_replace] = 0

            # Update output layer bias and reset output weights
            self.net.output.bias.data += (self.net.output.weight.data[:, features_to_replace] * 
                                        self.mean_feature_act[features_to_replace] / 
                                        (1 - self.decay_rate ** self.ages[features_to_replace])).sum(dim=1)
            self.net.output.weight.data[:, features_to_replace] = 0
            self.ages[features_to_replace] = 0

    def update_optim_params(self, features_to_replace, num_features_to_replace):
        if hasattr(self.opt, 'state') and num_features_to_replace > 0:
            # Reset optimizer state for hidden layer weights
            self.opt.state[self.net.hidden.weight]['exp_avg'][features_to_replace, :] = 0.0
            self.opt.state[self.net.hidden.bias]['exp_avg'][features_to_replace] = 0.0
            self.opt.state[self.net.hidden.weight]['exp_avg_sq'][features_to_replace, :] = 0.0
            self.opt.state[self.net.hidden.bias]['exp_avg_sq'][features_to_replace] = 0.0
            
            # Reset optimizer state for output weights
            self.opt.state[self.net.output.weight]['exp_avg'][:, features_to_replace] = 0.0
            self.opt.state[self.net.output.weight]['exp_avg_sq'][:, features_to_replace] = 0.0

    def gen_and_test(self, features):
        features_to_replace, num_features_to_replace = self.test_features(features)
        self.gen_new_features(features_to_replace, num_features_to_replace)
        self.update_optim_params(features_to_replace, num_features_to_replace)