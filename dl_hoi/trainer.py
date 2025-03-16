# # fix horrible
# from pathlib import Path
# import sys
# path_root = Path(__file__).parents[2]
# sys.path.append(str(path_root))

import sys
import json
import pickle
import argparse
from .nets.ffnn import FFNN
from .nets.linear import MyLinear
from dl_hoi.algos.bp import Backprop
from dl_hoi.algos.cbp import ContinualBackprop
from dl_hoi.utils.miscellaneous import *

def expr(params: dict):
    agent_type = params['agent']
    env_file = params['env_file']
    num_data_points = int(params['num_data_points'])
    beta_1 = params['beta_1']
    beta_2 = params['beta_2']
    weight_decay = params['weight_decay']
    accumulate = params['accumulate']
    perturb_scale = params['perturb_scale']

    num_inputs = params['num_inputs']
    num_features = params['num_features']
    hidden_activation = params['hidden_activation']
    step_size = params['step_size']
    opt = params['opt']
    replacement_rate = params["replacement_rate"]
    decay_rate = params["decay_rate"]
    util_type='adaptable_contribution'
    init = 'kaiming'
    mt = params["mt"]
    util_type = params["util_type"]
    init = params["init"]

    device = params["device"]

    if agent_type == 'linear':
        net = MyLinear(
            input_size=num_inputs,
        )
    else:
        net = FFNN(
            input_size=num_inputs,
            num_features=num_features,
            hidden_activation=hidden_activation,
        )

    if agent_type == 'bp' or agent_type == 'linear' or agent_type == 'l2':
        learner = Backprop(
            net=net,
            step_size=step_size,
            opt=opt,
            beta_1=beta_1,
            beta_2=beta_2,
            weight_decay=weight_decay,
            to_perturb=(perturb_scale > 0),
            perturb_scale=perturb_scale,
            device=device,
        )
    elif agent_type == 'cbp':
        learner = ContinualBackprop(
            net=net,
            step_size=step_size,
            opt=opt,
            beta_1=beta_1,
            beta_2=beta_2,
            replacement_rate=replacement_rate,
            decay_rate=decay_rate,
            device=device,
            maturity_threshold=mt,
            util_type=util_type,
            init=init,
            accumulate=accumulate,
        )

    with open(env_file, 'rb+') as f:
        inputs, outputs, _ = pickle.load(f)

    errs = torch.zeros((num_data_points), dtype=torch.float)

    for i in tqdm(range(num_data_points)):
        x, y = inputs[i: i+1], outputs[i: i+1]
        err = learner.learn(x=x, target=y)
        errs[i] = err

    data_to_save = {
        'errs': errs.numpy()
    }
    return data_to_save


def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', help="Path to the file containing the parameters for the experiment",
                        type=str, default='temp_cfg/0.json')
    args = parser.parse_args(arguments)
    cfg_file = args.c

    with open(cfg_file, 'r') as f:
        params = json.load(f)

    data = expr(params)

    with open(params['data_file'], 'wb+') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
