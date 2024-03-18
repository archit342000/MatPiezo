from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
import json
from mendeleev import element

def get_encoding(value, class_dict, num_clases):
    # print(value)
    # print(class_dict)
    encoding = [0] * num_clases
    # print(encoding)
    encoding[class_dict[value]] = 1
    return encoding

def get_discrete_class_dict(values):

    # Remove all the duplicate values
    values = list(np.unique(values))

    # Remove all the None values
    values = [value for value in values if value is not None]

    # Get the class dictionary
    class_dict = {}
    for i in range(len(values)):
        class_dict[values[i]] = i+1

    class_dict[None] = 0

    return class_dict

def get_continuous_class_dict(values, num_classes):

    # Remove all the None values
    values = [value for value in values if value is not None]

    # Get the labels for the classes
    labels = list(np.arange(1, num_classes))

    intervals = list(pd.qcut(values, num_classes-1, labels=labels))

    class_dict = {}
    for i in range(len(values)):
        class_dict[values[i]] = intervals[i]

    class_dict[None] = 0

    return class_dict

def get_atomic_numbers_encoding():

    atomic_numbers_encoding = {}
    class_dict = get_discrete_class_dict(list(range(1, 119)))

    for i in range(1, 119):
        atomic_numbers_encoding[i] = get_encoding(i, class_dict, 119)

    atomic_numbers_encoding[None] = get_encoding(None, class_dict, 119)

    return atomic_numbers_encoding

def get_atomic_volume_encoding():

    atomic_volume_encoding = {}
    atomic_volumnes = [element(i).atomic_volume for i in range(1, 119)]
    class_dict = get_continuous_class_dict(atomic_volumnes, 11)
    for i in range(1, 119):
        atomic_volume_encoding[i] = get_encoding(atomic_volumnes[i-1] ,class_dict, 11)

    atomic_volume_encoding[None] = get_encoding(None, class_dict, 11)
    return atomic_volume_encoding

def get_atomic_weights_encoding():

    atomic_weights_encoding = {}
    atomic_weights = [element(i).atomic_weight for i in range(1, 119)]
    class_dict = get_continuous_class_dict(atomic_weights, 11)
    for i in range(1, 119):
        atomic_weights_encoding[i] = get_encoding(atomic_weights[i-1] ,class_dict, 11)

    atomic_weights_encoding[None] = get_encoding(None, class_dict, 11)
    return atomic_weights_encoding

def get_atomic_radius_encoding():

    atomic_radius_encoding = {}
    atomic_radius = [element(i).atomic_radius for i in range(1, 119)]
    class_dict = get_continuous_class_dict(atomic_radius, 11)
    for i in range(1, 119):
        atomic_radius_encoding[i] = get_encoding(atomic_radius[i-1] ,class_dict, 11)

    atomic_radius_encoding[None] = get_encoding(None, class_dict, 11)
    return atomic_radius_encoding

def get_block_encoding():

    block_encoding = {}
    block = ['s', 'p', 'd', 'f']
    class_dict = get_discrete_class_dict(block)
    for i in range(1, 119):
        block_encoding[i] = get_encoding(element(i).block ,class_dict, 5)

    block_encoding[None] = get_encoding(None, class_dict, 5)
    return block_encoding

def get_group_id_encoding():

    group_id_encoding = {}
    group_id = [i for i in range(1, 19)]
    class_dict = get_discrete_class_dict(group_id)
    for i in range(1, 119):
        group_id_encoding[i] = get_encoding(element(i).group_id ,class_dict, 19)

    group_id_encoding[None] = get_encoding(None, class_dict, 19)
    return group_id_encoding

def get_period_encoding():

    period_encoding = {}
    period = [i for i in range(1, 8)]
    class_dict = get_discrete_class_dict(period)
    for i in range(1, 119):
        period_encoding[i] = get_encoding(element(i).period ,class_dict, 8)

    period_encoding[None] = get_encoding(None, class_dict, 8)
    return period_encoding

def get_covalent_radius_encoding():

    covalent_radius_encoding={}
    covalent_radius=[element(i).covalent_radius_cordero for i in range(1,119)]
    class_dict = get_continuous_class_dict(covalent_radius, 11)
    for i in range(1, 119):
        covalent_radius_encoding[i] = get_encoding(covalent_radius[i-1] ,class_dict, 11)

    covalent_radius_encoding[None] = get_encoding(None, class_dict, 11)
    return covalent_radius_encoding

def get_vdw_radius_encoding():

    vdw_radius_encoding = {}
    vdw_radius = [element(i).vdw_radius for i in range(1, 119)]
    class_dict = get_continuous_class_dict(vdw_radius, 11)
    for i in range(1, 119):
        vdw_radius_encoding[i] = get_encoding(vdw_radius[i-1] ,class_dict, 11)

    vdw_radius_encoding[None] = get_encoding(None, class_dict, 11)
    return vdw_radius_encoding

def get_en_pauling_encoding():

    en_pauling_encoding = {}
    en_pauling = [element(i).en_pauling for i in range(1, 119)]
    class_dict = get_continuous_class_dict(en_pauling, 11)
    for i in range(1, 119):
        en_pauling_encoding[i] = get_encoding(en_pauling[i-1] ,class_dict, 11)

    en_pauling_encoding[None] = get_encoding(None, class_dict, 11)
    return en_pauling_encoding

def get_c6_encoding():

    c6_encoding = {}
    c6 = [element(i).c6 for i in range(1, 119)]
    class_dict = get_continuous_class_dict(c6, 11)
    for i in range(1, 119):
        c6_encoding[i] = get_encoding(c6[i-1] ,class_dict, 11)

    c6_encoding[None] = get_encoding(None, class_dict, 11)
    return c6_encoding

def get_dipole_polarizability_encoding():

    dipole_polarizability_encoding = {}
    dipole_polarizability = [element(i).dipole_polarizability for i in range(1, 119)]
    class_dict = get_continuous_class_dict(dipole_polarizability, 11)
    for i in range(1, 119):
        dipole_polarizability_encoding[i] = get_encoding(dipole_polarizability[i-1] ,class_dict, 11)

    dipole_polarizability_encoding[None] = get_encoding(None, class_dict, 11)
    return dipole_polarizability_encoding

def get_electron_affinity_encoding():

    electron_affinity_encoding = {}
    electron_affinity = [element(i).electron_affinity for i in range(1, 119)]
    class_dict = get_continuous_class_dict(electron_affinity, 11)
    for i in range(1, 119):
        electron_affinity_encoding[i] = get_encoding(electron_affinity[i-1] ,class_dict, 11)

    electron_affinity_encoding[None] = get_encoding(None, class_dict, 11)
    return electron_affinity_encoding

def get_nvalence_encoding():

    nvalence_encoding = {}
    nvalence = [element(i).nvalence() for i in range(1, 119)]
    class_dict = get_discrete_class_dict(nvalence)
    for i in range(1, 119):
        nvalence_encoding[i] = get_encoding(nvalence[i-1] ,class_dict, 18)

    nvalence_encoding[None] = get_encoding(None, class_dict, 18)
    return nvalence_encoding

def get_first_ionization_energy_encoding():

    first_ionization_energy_encoding = {}
    ionenergies = [element(i).ionenergies for i in range(1, 119)]
    first_ionization_energy = []
    for i in range(118):
        if 1 in ionenergies[i]:
            first_ionization_energy.append(ionenergies[i][1])
        else:
            first_ionization_energy.append(None)

    class_dict = get_continuous_class_dict(first_ionization_energy, 11)
    for i in range(1, 119):
        first_ionization_energy_encoding[i] = get_encoding(first_ionization_energy[i-1] ,class_dict, 11)

    first_ionization_energy_encoding[None] = get_encoding(None, class_dict, 11)
    return first_ionization_energy_encoding

def get_atom_features(features_params):

    atomic_features_encoding = {}
    
    if features_params['nvalence']:
        print('Getting nvalence encoding...')
        atomic_features_encoding['nvalence'] = get_nvalence_encoding()

    if features_params['atomic_numbers']:
        print('Getting atomic numbers encoding...')
        atomic_features_encoding['atomic_numbers'] = get_atomic_numbers_encoding()

    if features_params['atomic_volume']:
        print('Getting atomic volume encoding...')
        atomic_features_encoding['atomic_volume'] = get_atomic_volume_encoding()
    
    if features_params['atomic_weights']:
        print('Getting atomic weights encoding...')
        atomic_features_encoding['atomic_weights'] = get_atomic_weights_encoding()

    if features_params['atomic_radius']:
        print('Getting atomic radius encoding...')
        atomic_features_encoding['atomic_radius'] = get_atomic_radius_encoding()

    if features_params['block']:
        print('Getting block encoding...')
        atomic_features_encoding['block'] = get_block_encoding()

    if features_params['group_id']:
        print('Getting group id encoding...')
        atomic_features_encoding['group_id'] = get_group_id_encoding()

    if features_params['period']:
        print('Getting period encoding...')
        atomic_features_encoding['period'] = get_period_encoding()

    if features_params['covalent_radius']:
        print('Getting covalent radius encoding...')
        atomic_features_encoding['covalent_radius'] = get_covalent_radius_encoding()

    if features_params['vdw_radius']:
        print('Getting vdw radius encoding...')
        atomic_features_encoding['vdw_radius'] = get_vdw_radius_encoding()

    if features_params['en_pauling']:
        print('Getting en pauling encoding...')
        atomic_features_encoding['en_pauling'] = get_en_pauling_encoding()

    if features_params['c6']:
        print('Getting c6 encoding...')
        atomic_features_encoding['c6'] = get_c6_encoding()

    if features_params['dipole_polarizability']:
        print('Getting dipole polarizability encoding...')
        atomic_features_encoding['dipole_polarizability'] = get_dipole_polarizability_encoding()

    if features_params['electron_affinity']:
        print('Getting electron affinity encoding...')
        atomic_features_encoding['electron_affinity'] = get_electron_affinity_encoding()

    if features_params['first_ionization_energy']:
        print('Getting first ionization energy encoding...')
        atomic_features_encoding['first_ionization_energy'] = get_first_ionization_energy_encoding()

    atom_features = []
    atom_features.append([])
    for key in atomic_features_encoding:
        atom_features[0].extend(atomic_features_encoding[key][None])

    for i in range(1, 119):
        atom_features.append([])
        for key in atomic_features_encoding:
            atom_features[i].extend(atomic_features_encoding[key][i])

    return atom_features

def build_config(config_path, features_params):

    atomic_features = get_atom_features(features_params)
    config = {}
    for i in range(119):
        config[i] = atomic_features[i]

    with open(config_path, 'w') as f:
        json.dump(config, f)

    return config

config_path = 'atom_dict_cgcnn.json'
features_params = {'atomic_numbers': True, 
                   'atomic_volume': True,
                   'atomic_weights': False, 
                   'atomic_radius': False, 
                   'block': True, 
                   'group_id': True, 
                   'period': False, 
                   'covalent_radius': True, 
                   'vdw_radius': False, 
                   'en_pauling': True, 
                   'c6': False, 
                   'dipole_polarizability': False, 
                   'electron_affinity': True, 
                   'nvalence': True, 
                   'first_ionization_energy': True}

build_config(config_path, features_params)