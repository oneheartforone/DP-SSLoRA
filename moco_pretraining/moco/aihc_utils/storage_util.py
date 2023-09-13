import os
import datetime

from pathlib import Path

import getpass

if str(getpass.getuser()) == 'jby':
    STORAGE_ROOT = Path('../../home/jby/chexpert_experiments')
else:
    STORAGE_ROOT = Path('../../results/Linear_probe_directly_epsilon5_paper')


def get_storage_folder(exp_name, exp_type):

    try:
        jobid = os.environ["SLURM_JOB_ID"]
    except:
        jobid = None

    datestr = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    username = str(getpass.getuser())

    fname = f'{exp_name}_{exp_type}_{datestr}_SLURM{jobid}' if jobid is not None else f'{exp_name}_{exp_type}_{datestr}'

    path_name = STORAGE_ROOT / fname
    os.makedirs(path_name)

    print(f'Experiment storage is at {path_name}')
    return path_name