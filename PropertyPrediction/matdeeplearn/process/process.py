import os
import glob
import sys
import time
import csv
import json
import warnings
import numpy as np
import ase
import glob
from ase import io
from scipy.stats import rankdata
from scipy import interpolate
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor
from mendeleev import element

##torch imports
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Dataset, Data, InMemoryDataset
from torch_geometric.utils import dense_to_sparse, degree, add_self_loops
import torch_geometric.transforms as T
from torch_geometric.utils import degree
from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd

################################################################################
# Data splitting
################################################################################

def split_data(
    dataset,
    train_ratio,
    val_ratio,
    test_ratio,
    seed=np.random.randint(1, 1e6),
    save=False,
):

    dataset_size = len(dataset)
    if (train_ratio + val_ratio + test_ratio) <=1 :
        labels = dataset.data.y
        labels = labels.squeeze()
        print(labels.shape)

        cats = pd.qcut(labels, q=330, labels=False)

        train_idx, valid_test_idx = train_test_split(np.arange(dataset_size),
                                                        test_size=1-train_ratio,
                                                        random_state=seed,
                                                        shuffle=True,
                                                        stratify=cats)
        
        valid_idx, test_idx = train_test_split(valid_test_idx,
                                                test_size=test_ratio/(test_ratio+val_ratio),
                                                random_state=seed,
                                                shuffle=True,
                                                stratify=cats[valid_test_idx])
        
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, valid_idx)
        test_dataset = torch.utils.data.Subset(dataset, test_idx)
        return train_dataset, val_dataset, test_dataset
    else:
        print("invalid ratios")

def split_data_CV(dataset, num_folds=5, seed=np.random.randint(1, 1e6), save=False):
    print("Seed: ", seed)
    dataset_size = len(dataset)

    labels = dataset.data.y
    labels = labels.squeeze()
    print(labels.shape)

    cats = pd.qcut(labels, q=330, labels=False)

    cv_splits = []
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    splits = skf.split(np.arange(dataset_size), cats)

    file = open('split_indices.txt', 'a')

    for train_index, test_index in splits:
        file.write(f'{test_index}\n')
        cv_splits.append(torch.utils.data.Subset(dataset, test_index))

    file.close()
    return cv_splits

################################################################################
# Pytorch datasets
################################################################################

##Fetch dataset; processes the raw data if specified
def get_dataset(data_path, target_index, reprocess="False", processing_args=None):
    if processing_args == None:
        processed_path = "processed"
    else:
        processed_path = processing_args.get("processed_path", "processed")

    if processing_args==None or processing_args['targets'] == "True" :
        transforms = GetY(index=target_index)
    else:
        transforms = None

    if os.path.exists(data_path) == False:
        print("Data not found in:", data_path)
        sys.exit()

    if reprocess == "True":
        os.system("rm -rf " + os.path.join(data_path, processed_path))
        process_data(data_path, processed_path, processing_args)

    if os.path.exists(os.path.join(data_path, processed_path, "data.pt")) == True:
        dataset = StructureDataset(
            data_path,
            processed_path,
            transforms,
        )
        print(dataset.data)
    elif os.path.exists(os.path.join(data_path, processed_path, "data0.pt")) == True:
        dataset = StructureDataset_large(
            data_path,
            processed_path,
            transforms,
        )
    else:
        process_data(data_path, processed_path, processing_args)
        if os.path.exists(os.path.join(data_path, processed_path, "data.pt")) == True:
            dataset = StructureDataset(
                data_path,
                processed_path,
                transforms,
            )
        elif os.path.exists(os.path.join(data_path, processed_path, "data0.pt")) == True:
            dataset = StructureDataset_large(
                data_path,
                processed_path,
                transforms,
            )        
    return dataset


##Dataset class from pytorch/pytorch geometric; inmemory case
class StructureDataset(InMemoryDataset):
    def __init__(
        self, data_path, processed_path="processed", transform=None, pre_transform=None
    ):
        self.data_path = data_path
        self.processed_path = processed_path
        super(StructureDataset, self).__init__(data_path, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_dir(self):
        return os.path.join(self.data_path, self.processed_path)

    @property
    def processed_file_names(self):
        file_names = ["data.pt"]
        return file_names


##Dataset class from pytorch/pytorch geometric
class StructureDataset_large(Dataset):
    def __init__(
        self, data_path, processed_path="processed", transform=None, pre_transform=None
    ):
        self.data_path = data_path
        self.processed_path = processed_path
        super(StructureDataset_large, self).__init__(
            data_path, transform, pre_transform
        )

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_dir(self):
        return os.path.join(self.data_path, self.processed_path)

    @property
    def processed_file_names(self):
        # file_names = ["data.pt"]
        file_names = []
        for file_name in glob.glob(self.processed_dir + "/data*.pt"):
            file_names.append(os.path.basename(file_name))
        # print(file_names)
        return file_names

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, "data_{}.pt".format(idx)))
        return data


################################################################################
#  Processing
################################################################################
def create_global_feat(atoms_index_arr):
    comp    = np.zeros(108)
    temp    = np.unique(atoms_index_arr,return_counts=True)
    for i in range(len(temp[0])):
            comp[temp[0][i]]=temp[1][i]/temp[1].sum()
    return comp.reshape(1,-1)

def process_data(data_path, processed_path, processing_args):

    # Create 32-sized one-hot encoding of the 32 point groups
    point_groups = np.eye(32)
    point_groups_dict = {
        "1": 0,
        "-1": 1,
        "2": 2,
        "m": 3,
        "2/m": 4,
        "222": 5,
        "mm2": 6,
        "mmm": 7,
        "4": 8,
        "-4": 9,
        "4/m": 10,
        "422": 11,
        "4mm": 12,
        "-42m": 13,
        "4/mmm": 14,
        "3": 15,
        "-3": 16,
        "32": 17,
        "3m": 18,
        "-3m": 19,
        "6": 20,
        "-6": 21,
        "6/m": 22,
        "622": 23,
        "6mm": 24,
        "-6m2": 25,
        "6/mmm": 26,
        "23": 27,
        "m-3": 28,
        "432": 29,
        "-43m": 30,
        "m-3m": 31,
    }

    crystal_systems = np.eye(7)
    crystal_systems_dict = {
        "triclinic": 0,
        "monoclinic": 1,
        "orthorhombic": 2,
        "tetragonal": 3,
        "trigonal": 4,
        "hexagonal": 5,
        "cubic": 6,
    }

    ##Begin processing data
    print("Processing data to: " + os.path.join(data_path, processed_path))
    assert os.path.exists(data_path), "Data path not found in " + data_path

    # split_size = processing_args['split_size']

    ##Load dictionary
    if processing_args["dictionary_source"] != "generated":
        if processing_args["dictionary_source"] == "default":
            print("Using default dictionary.")
            atom_dictionary = get_dictionary(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "dictionary_default.json",
                )
            )
        elif processing_args["dictionary_source"] == "blank":
            print(
                "Using blank dictionary. Warning: only do this if you know what you are doing"
            )
            atom_dictionary = get_dictionary(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), "dictionary_blank.json"
                )
            )
        else:
            dictionary_file_path = os.path.join(
                data_path, processing_args["dictionary_path"]
            )
            if os.path.exists(dictionary_file_path) == False:
                print("Atom dictionary not found, exiting program...")
                sys.exit()
            else:
                print("Loading atom dictionary from file.")
                atom_dictionary = get_dictionary(dictionary_file_path)

    ##Load targets
    if processing_args['targets'] == "True":
        target_property_file = os.path.join(data_path, processing_args["target_path"])
        assert os.path.exists(target_property_file), (
            "targets not found in " + target_property_file
        )
        with open(target_property_file) as f:
            reader = csv.reader(f)
            target_data = [row for row in reader]
    else:
        files = os.listdir(data_path)
        target_data = [[file.split('.')[0]] for file in files if file.endswith(".cif")]

    ##Read db file if specified
    ase_crystal_list = []
    if processing_args["data_format"] == "db":
        db = ase.db.connect(os.path.join(data_path, "data.db"))
        row_count = 0
        # target_data=[]
        for row in db.select():
            # target_data.append([str(row_count), row.get('target')])
            ase_temp = row.toatoms()
            ase_crystal_list.append(ase_temp)
            row_count = row_count + 1
            if row_count % 500 == 0:
                print("db processed: ", row_count)

    ##Process structure files and create structure graphs
    data_list = []
    for index in range(0, len(target_data)):
        structure_id = target_data[index][0]
        data = Data()

        ##Read in structure file using ase
        if processing_args["data_format"] != "db":
            try:
                ase_crystal = ase.io.read(
                    os.path.join(
                        data_path, structure_id + "." + processing_args["data_format"]
                    )
                )
            except:
                print("Error reading file: " + structure_id)
                continue
            data.ase = ase_crystal
        else:
            ase_crystal = ase_crystal_list[index]
            data.ase = ase_crystal

        # # Read in structure file using pymatgen
        # try:
        #     structure = Structure.from_file(
        #         os.path.join(
        #             data_path, structure_id + "." + processing_args["data_format"]
        #         )
        #     )
        # except:
        #     print("Error reading file: " + structure_id)
        #     continue

        structure = AseAtomsAdaptor.get_structure(ase_crystal)

        ##Get min, max, average, geometric mean and standard deviation of atomic radii
        symbols = list(structure.symbol_set)
        atomic_radii = []
        for symbol in symbols:
            try:
                ele = element(symbol)
                at_rad = ele.atomic_radius
                if at_rad == None:
                    continue
                atomic_radii.append(at_rad)
            except:
                continue
        
        # min_atomic_radius = min(atomic_radii)
        # max_atomic_radius = max(atomic_radii)
        # avg_atomic_radius = np.mean(atomic_radii)
        # geo_atomic_radius = np.exp(np.mean(np.log(atomic_radii)))
        # std_atomic_radius = np.std(atomic_radii)
        # radius_feats = torch.Tensor([min_atomic_radius, max_atomic_radius, avg_atomic_radius, geo_atomic_radius, std_atomic_radius])
        # radius_feats.unsqueeze_(0)
        # data.radius_feats = radius_feats
        
        ##Get point group
        try:
            sga = SpacegroupAnalyzer(structure)
            point_group = sga.get_point_group_symbol()
            data.pge = point_groups[point_groups_dict[point_group]]
            data.pge = np.array([data.pge])
            data.pge = torch.Tensor(data.pge)
        except:
            continue
        # data.pge = data.pge.reshape(1, -1)

        ##Get crystal system
        try:
            crystal_system = sga.get_crystal_system()
            data.cse = crystal_systems[crystal_systems_dict[crystal_system]]
            data.cse = np.array([data.cse])
            data.cse = torch.Tensor(data.cse)
        except:
            continue

        ##Compile structure sizes (# of atoms) and elemental compositions
        if index == 0:
            length = [len(ase_crystal)]
            elements = [list(set(ase_crystal.get_chemical_symbols()))]
        else:
            length.append(len(ase_crystal))
            elements.append(list(set(ase_crystal.get_chemical_symbols())))

        ##Obtain distance matrix with ase
        distance_matrix = ase_crystal.get_all_distances(mic=True)

        ##Create sparse graph from distance matrix
        distance_matrix_trimmed = threshold_sort(
            distance_matrix,
            processing_args["graph_max_radius"],
            processing_args["graph_max_neighbors"],
            adj=False,
        )

        distance_matrix_trimmed = torch.Tensor(distance_matrix_trimmed)
        out = dense_to_sparse(distance_matrix_trimmed)
        edge_index = out[0]
        edge_weight = out[1]

        self_loops = True
        if self_loops == True:
            edge_index, edge_weight = add_self_loops(
                edge_index, edge_weight, num_nodes=len(ase_crystal), fill_value=0
            )
            data.edge_index = edge_index
            data.edge_weight = edge_weight

            distance_matrix_mask = (
                distance_matrix_trimmed.fill_diagonal_(1) != 0
            ).int()
        elif self_loops == False:
            data.edge_index = edge_index
            data.edge_weight = edge_weight

            distance_matrix_mask = (distance_matrix_trimmed != 0).int()

        data.edge_descriptor = {}
        data.edge_descriptor["distance"] = edge_weight
        data.edge_descriptor["mask"] = distance_matrix_mask

        if processing_args['targets'] == "True":
            target = target_data[index][1:]
            y = torch.Tensor(np.array([target], dtype=np.float32))
            data.y = y

        _atoms_index     = ase_crystal.get_atomic_numbers()
        gatgnn_glob_feat = create_global_feat(_atoms_index)
        gatgnn_glob_feat = np.repeat(gatgnn_glob_feat,len(_atoms_index),axis=0)
        data.glob_feat   = torch.Tensor(gatgnn_glob_feat).float()

        # pos = torch.Tensor(ase_crystal.get_positions())
        # data.pos = pos
        z = torch.LongTensor(ase_crystal.get_atomic_numbers())
        data.z = z

        ###placeholder for state feature
        u = np.zeros((3))
        u = torch.Tensor(u[np.newaxis, ...])
        data.u = u

        if processing_args["targets"] == "True":
            data.structure_id = [[structure_id] * len(data.y)]
        else:
            data.structure_id = [[structure_id]]

        if processing_args["verbose"] == "True" and (
            (index + 1) % 500 == 0 or (index + 1) == len(target_data)
        ):
            print("Data processed: ", index + 1, "out of", len(target_data))
            # if index == 0:
            # print(data)
            # print(data.edge_weight, data.edge_attr[0])

        data_list.append(data)

    ##
    n_atoms_max = max(length)
    species = list(set(sum(elements, [])))
    species.sort()
    num_species = len(species)
    if processing_args["verbose"] == "True":
        print(
            "Max structure size: ",
            n_atoms_max,
            "Max number of elements: ",
            num_species,
        )
        print("Unique species:", species)
    crystal_length = len(ase_crystal)
    data.length = torch.LongTensor([crystal_length])

    ##Generate node features
    if processing_args["dictionary_source"] != "generated":
        ##Atom features(node features) from atom dictionary file
        for index in range(0, len(data_list)):
            atom_fea = np.vstack(
                [
                    atom_dictionary[str(data_list[index].ase.get_atomic_numbers()[i])]
                    for i in range(len(data_list[index].ase))
                ]
            ).astype(float)
            data_list[index].x = torch.Tensor(atom_fea)
    elif processing_args["dictionary_source"] == "generated":
        ##Generates one-hot node features rather than using dict file
        from sklearn.preprocessing import LabelBinarizer

        lb = LabelBinarizer()
        lb.fit(species)
        for index in range(0, len(data_list)):
            data_list[index].x = torch.Tensor(
                lb.transform(data_list[index].ase.get_chemical_symbols())
            )

    ##Adds node degree to node features (appears to improve performance)
    for index in range(0, len(data_list)):
        data_list[index] = OneHotDegree(
            data_list[index], processing_args["graph_max_neighbors"] + 1
        )

    ##Get graphs based on voronoi connectivity; todo: also get voronoi features
    ##avoid use for the time being until a good approach is found
    processing_args["voronoi"] = "False"
    if processing_args["voronoi"] == "True":
        from pymatgen.core.structure import Structure
        from pymatgen.analysis.structure_analyzer import VoronoiConnectivity
        # from pymatgen.io.ase import AseAtomsAdaptor

        Converter = AseAtomsAdaptor()

        for index in range(0, len(data_list)):
            pymatgen_crystal = Converter.get_structure(data_list[index].ase)
            # double check if cutoff distance does anything
            Voronoi = VoronoiConnectivity(
                pymatgen_crystal, cutoff=processing_args["graph_max_radius"]
            )
            connections = Voronoi.max_connectivity

            distance_matrix_voronoi = threshold_sort(
                connections,
                9999,
                processing_args["graph_max_neighbors"],
                reverse=True,
                adj=False,
            )
            distance_matrix_voronoi = torch.Tensor(distance_matrix_voronoi)

            out = dense_to_sparse(distance_matrix_voronoi)
            edge_index_voronoi = out[0]
            edge_weight_voronoi = out[1]

            edge_attr_voronoi = distance_gaussian(edge_weight_voronoi)
            edge_attr_voronoi = edge_attr_voronoi.float()

            data_list[index].edge_index_voronoi = edge_index_voronoi
            data_list[index].edge_weight_voronoi = edge_weight_voronoi
            data_list[index].edge_attr_voronoi = edge_attr_voronoi
            if index % 500 == 0:
                print("Voronoi data processed: ", index)

    ##makes SOAP and SM features from dscribe
    if processing_args["SOAP_descriptor"] == "True":
        if True in data_list[0].ase.pbc:
            periodicity = True
        else:
            periodicity = False

        from dscribe.descriptors import SOAP
            
        make_feature_SOAP = SOAP(
            species=species,
            rcut=processing_args["SOAP_rcut"],
            nmax=processing_args["SOAP_nmax"],
            lmax=processing_args["SOAP_lmax"],
            sigma=processing_args["SOAP_sigma"],
            periodic=periodicity,
            sparse=False,
            average="inner",
            rbf="gto",
            crossover=False,
        )
        for index in range(0, len(data_list)):
            features_SOAP = make_feature_SOAP.create(data_list[index].ase)
            data_list[index].extra_features_SOAP = torch.Tensor(features_SOAP)
            if processing_args["verbose"] == "True" and index % 500 == 0:
                if index == 0:
                    print(
                        "SOAP length: ",
                        features_SOAP.shape,
                    )
                print("SOAP descriptor processed: ", index)

    elif processing_args["SM_descriptor"] == "True":
        if True in data_list[0].ase.pbc:
            periodicity = True
        else:
            periodicity = False

        from dscribe.descriptors import SineMatrix, CoulombMatrix
        
        if periodicity == True:
            make_feature_SM = SineMatrix(
                n_atoms_max=n_atoms_max,
                permutation="eigenspectrum",
                sparse=False,
                flatten=True,
            )
        else:
            make_feature_SM = CoulombMatrix(
                n_atoms_max=n_atoms_max,
                permutation="eigenspectrum",
                sparse=False,
                flatten=True,
            )
            
        for index in range(0, len(data_list)):
            features_SM = make_feature_SM.create(data_list[index].ase)
            data_list[index].extra_features_SM = torch.Tensor(features_SM)
            if processing_args["verbose"] == "True" and index % 500 == 0:
                if index == 0:
                    print(
                        "SM length: ",
                        features_SM.shape,
                    )
                print("SM descriptor processed: ", index)

    ##Generate edge features
    if processing_args["edge_features"] == "True":

        ##Distance descriptor using a Gaussian basis
        distance_gaussian = GaussianSmearing(
            0, 1, processing_args["graph_edge_length"], 0.2
        )
        # print(GetRanges(data_list, 'distance'))
        NormalizeEdge(data_list, "distance")
        # print(GetRanges(data_list, 'distance'))
        for index in range(0, len(data_list)):
            data_list[index].edge_attr = distance_gaussian(
                data_list[index].edge_descriptor["distance"]
            )
            if processing_args["verbose"] == "True" and (
                (index + 1) % 500 == 0 or (index + 1) == len(target_data)
            ):
                print("Edge processed: ", index + 1, "out of", len(target_data))

    Cleanup(data_list, ["ase", "edge_descriptor"])

    if os.path.isdir(os.path.join(data_path, processed_path)) == False:
        os.mkdir(os.path.join(data_path, processed_path))

    ##Save processed dataset to file
    if processing_args["dataset_type"] == "inmemory":
        data, slices = InMemoryDataset.collate(data_list)
        torch.save((data, slices), os.path.join(data_path, processed_path, "data.pt"))

    elif processing_args["dataset_type"] == "large":
        for i in range(0, len(data_list)):
            torch.save(
                data_list[i],
                os.path.join(
                    os.path.join(data_path, processed_path), "data_{}.pt".format(i)
                ),
            )


################################################################################
#  Processing sub-functions
################################################################################

##Selects edges with distance threshold and limited number of neighbors
def threshold_sort(matrix, threshold, neighbors, reverse=False, adj=False):
    mask = matrix > threshold
    distance_matrix_trimmed = np.ma.array(matrix, mask=mask)
    if reverse == False:
        distance_matrix_trimmed = rankdata(
            distance_matrix_trimmed, method="ordinal", axis=1
        )
    elif reverse == True:
        distance_matrix_trimmed = rankdata(
            distance_matrix_trimmed * -1, method="ordinal", axis=1
        )
    distance_matrix_trimmed = np.nan_to_num(
        np.where(mask, np.nan, distance_matrix_trimmed)
    )
    distance_matrix_trimmed[distance_matrix_trimmed > neighbors + 1] = 0

    if adj == False:
        distance_matrix_trimmed = np.where(
            distance_matrix_trimmed == 0, distance_matrix_trimmed, matrix
        )
        return distance_matrix_trimmed
    elif adj == True:
        adj_list = np.zeros((matrix.shape[0], neighbors + 1))
        adj_attr = np.zeros((matrix.shape[0], neighbors + 1))
        for i in range(0, matrix.shape[0]):
            temp = np.where(distance_matrix_trimmed[i] != 0)[0]
            adj_list[i, :] = np.pad(
                temp,
                pad_width=(0, neighbors + 1 - len(temp)),
                mode="constant",
                constant_values=0,
            )
            adj_attr[i, :] = matrix[i, adj_list[i, :].astype(int)]
        distance_matrix_trimmed = np.where(
            distance_matrix_trimmed == 0, distance_matrix_trimmed, matrix
        )
        return distance_matrix_trimmed, adj_list, adj_attr


##Slightly edited version from pytorch geometric to create edge from gaussian basis
class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, resolution=50, width=0.05, **kwargs):
        super(GaussianSmearing, self).__init__()
        offset = torch.linspace(start, stop, resolution)
        # self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.coeff = -0.5 / ((stop - start) * width) ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.unsqueeze(-1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


##Obtain node degree in one-hot representation
def OneHotDegree(data, max_degree, in_degree=False, cat=True):
    idx, x = data.edge_index[1 if in_degree else 0], data.x
    deg = degree(idx, data.num_nodes, dtype=torch.long)
    deg = F.one_hot(deg, num_classes=max_degree + 1).to(torch.float)

    if x is not None and cat:
        x = x.view(-1, 1) if x.dim() == 1 else x
        data.x = torch.cat([x, deg.to(x.dtype)], dim=-1)
    else:
        data.x = deg

    return data


##Obtain dictionary file for elemental features
def get_dictionary(dictionary_file):
    with open(dictionary_file) as f:
        atom_dictionary = json.load(f)
    return atom_dictionary


##Deletes unnecessary data due to slow dataloader
def Cleanup(data_list, entries):
    for data in data_list:
        for entry in entries:
            try:
                delattr(data, entry)
            except Exception:
                pass


##Get min/max ranges for normalized edges
def GetRanges(dataset, descriptor_label):
    mean = 0.0
    std = 0.0
    for index in range(0, len(dataset)):
        if len(dataset[index].edge_descriptor[descriptor_label]) > 0:
            if index == 0:
                feature_max = dataset[index].edge_descriptor[descriptor_label].max()
                feature_min = dataset[index].edge_descriptor[descriptor_label].min()
            mean += dataset[index].edge_descriptor[descriptor_label].mean()
            std += dataset[index].edge_descriptor[descriptor_label].std()
            if dataset[index].edge_descriptor[descriptor_label].max() > feature_max:
                feature_max = dataset[index].edge_descriptor[descriptor_label].max()
            if dataset[index].edge_descriptor[descriptor_label].min() < feature_min:
                feature_min = dataset[index].edge_descriptor[descriptor_label].min()

    mean = mean / len(dataset)
    std = std / len(dataset)
    return mean, std, feature_min, feature_max


##Normalizes edges
def NormalizeEdge(dataset, descriptor_label):
    mean, std, feature_min, feature_max = GetRanges(dataset, descriptor_label)

    for data in dataset:
        data.edge_descriptor[descriptor_label] = (
            data.edge_descriptor[descriptor_label] - feature_min
        ) / (feature_max - feature_min)


# WIP
def SM_Edge(dataset):
    from dscribe.descriptors import (
        CoulombMatrix,
        SOAP,
        MBTR,
        EwaldSumMatrix,
        SineMatrix,
    )

    count = 0
    for data in dataset:
        n_atoms_max = len(data.ase)
        make_feature_SM = SineMatrix(
            n_atoms_max=n_atoms_max,
            permutation="none",
            sparse=False,
            flatten=False,
        )
        features_SM = make_feature_SM.create(data.ase)
        features_SM_trimmed = np.where(data.mask == 0, data.mask, features_SM)
        features_SM_trimmed = torch.Tensor(features_SM_trimmed)
        out = dense_to_sparse(features_SM_trimmed)
        edge_index = out[0]
        edge_weight = out[1]
        data.edge_descriptor["SM"] = edge_weight

        if count % 500 == 0:
            print("SM data processed: ", count)
        count = count + 1

    return dataset


################################################################################
#  Transforms
################################################################################

##Get specified y index from data.y
class GetY(object):
    def __init__(self, index=0):
        self.index = index

    def __call__(self, data):
        # Specify target.
        if self.index != -1:
            # data.y = data.y[0][self.index]
            data.y=data.y[0]
        return data
