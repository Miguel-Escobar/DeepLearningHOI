import sys
import json
import copy
import argparse
import subprocess
from lop.utils.miscellaneous import *

debug = False

if debug:
    import pprint

def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', help="Path of the file containing the parameters of the experiment",
                        type=str, default='cfg/a.json')
    args = parser.parse_args(arguments)
    cfg_file = args.c

    with open(cfg_file, 'r') as f:
        params = json.load(f)

    list_params, hyper_param_settings = get_configurations(params=params)

    # make a directory for temp cfg files
    bash_command = "mkdir -p temp_cfg/"
    subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)

    # generate_env_data data for all the runs
    if params['gen_prob_data']:
        if debug:
            print("Generando datos del problema")
            # print("Directorio de la data del problema: " + params['env_data_dir']) # debug
            # print("Número de ejecuciones: " + str(params['num_runs'])) # debug
            pprint.pprint(params) # debug
        assert params['env_data_dir'] != '' and params['env_data_dir'] != "/", "No se ha especificado un directorio para los datos del problema o estas a punto de borrar todo el disco. No borres todo el disco." # Esto en verdad debiera ser más seguro, pero no sé cómo.

        bash_command = "rm --force " + params['env_data_dir'] + '*'
        subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
        bash_command = "mkdir -p " + params['env_data_dir']
        subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
        # make a directory for env temp cfg files and ensure it is empty
        bash_command = "mkdir -p env_temp_cfg/"
        subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
        bash_command = "rm -rf env_temp_cfg/*"
        subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)

        for idx in tqdm(range(params['num_runs'])):
            if debug: print("Generando config para la teacher data de la run " + str(idx))
            new_cfg_file = 'env_temp_cfg/'+str(idx)+'.json'
            new_params = copy.deepcopy(params)
            new_params['env_file'] = new_params['env_data_dir'] + str(idx)
            if 'target_net_dir' in new_params.keys():
                if new_params['target_net_dir'] == '':
                    pass
                else:
                    new_params['target_net_file'] = new_params['target_net_dir'] + str(idx)
            try:    f = open(new_cfg_file, 'w+')
            except: f = open(new_cfg_file, 'w+')
            with open(new_cfg_file, 'w+') as f:
                json.dump(new_params, f)
        return
    if debug: print("No generando datos del problema")
    bash_command = "rm -r --force " + params['data_dir']
    subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
    bash_command = "mkdir " + params['data_dir']
    subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)

    """
        Set and write all the parameters for the individual config files
    """
    for setting_index, param_setting in enumerate(hyper_param_settings):
        new_params = copy.deepcopy(params)
        for idx, param in enumerate(list_params):
            new_params[param] = param_setting[idx]
        new_params['index'] = setting_index
        new_params['data_dir'] = params['data_dir'] + str(setting_index) + '/'

        """
            Make the data directory
        """
        print("Directorio de la data: " + new_params['data_dir']) # debug
        bash_command = "mkdir -p " + new_params['data_dir']
        subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)

        for idx in tqdm(range(params['num_runs'])):
            new_params['data_file'] = new_params['data_dir'] + str(idx)
            new_params['env_file'] = new_params['env_data_dir'] + str(idx)

            """
                write data in config files
            """
            new_cfg_file = 'temp_cfg/'+str(setting_index*params['num_runs']+idx)+'.json'
            try:    f = open(new_cfg_file, 'w+')
            except: f = open(new_cfg_file, 'w+')
            with open(new_cfg_file, 'w+') as f:
                json.dump(new_params, f, indent=4)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
