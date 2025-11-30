from MMSA import MMSA_run
from MMSA.config import get_config_regression
config_file_path = './config/config_regression.json'
# MMSA_run('tetfn', 'mosi', seeds=[1111], gpu_ids=[0],model_save_dir='./save_models',log_dir='./logs',config_file=config_file_path,res_save_dir='./results')
# MMSA_run('tetfn_crt', 'mosi', seeds=[1111], gpu_ids=[0],model_save_dir='./save_models',log_dir='./logs',config_file=config_file_path,res_save_dir='./results')
MMSA_run('tetfn_crt2', 'mosi', seeds=[1111], gpu_ids=[0],model_save_dir='./save_models',log_dir='./logs',config_file=config_file_path,res_save_dir='./results')