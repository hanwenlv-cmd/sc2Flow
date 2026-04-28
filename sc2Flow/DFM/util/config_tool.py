
data_config = {
        'mcfarland':{
            'k':5,
            'split_key':'split',
            'SMILES_key':'SMILES',
            'perturbation_key':'condition',
            'DataPath':'/home/usr/sc2Flow_lab/datasets/mcfarland_2020_pre.h5ad',
            'dosage_key':'dose_val',
            'cell_type_key':'cell_type',
            'control_key':'control',
            'test_key':'ood',
            'cell_drug_dose_key':'cov_drug_dose_name'
        },
        'zhaoSims': {
            'k': 4,
            'split_key': 'split',
            'SMILES_key': 'SMILES',
            'perturbation_key': 'condition',
            'DataPath': '/home/usr/sc2Flow_lab/datasets/zhaoSims2021_pre.h5ad',
            'dosage_key': 'dose_val',
            'cell_type_key': 'cell_type',
            'control_key': 'control',
            'test_key': 'ood',
            'cell_drug_dose_key':'cov_drug_dose_name'
        },
        'sciplex':{
            'k': 1,
            'split_key':'mode',
            'SMILES_key':'SMILES',
            'perturbation_key':'product_name',
            'DataPath':'/home/usr/sc2Flow_lab/datasets/sciplex3_pre_2000.h5ad',
            'dosage_key':'dose_val',
            'cell_type_key': 'cell_type',
            'control_key':'is_control',
            'test_key':'test',
            'cell_drug_dose_key':'cov_drug_dose_name',

        }
    }

class Config:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def keys(self):
        return [attr for attr in dir(self) if not attr.startswith('_')]

def get_config(config_name):
    return Config(data_config[config_name])