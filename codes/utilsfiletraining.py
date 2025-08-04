import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Sampler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler, LabelEncoder, PowerTransformer, QuantileTransformer
from sklearn.model_selection import KFold, train_test_split, GroupKFold, LeaveOneGroupOut
from sklearn.metrics import pairwise_distances
from scipy.stats import linregress
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import random
from torch_geometric import seed_everything
from torch_geometric.utils import softmax
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GCNConv, GATConv, GraphConv, MFConv, GATv2Conv, TransformerConv, GINConv
from torch_geometric.nn import Linear
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool as gsp
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PYGDataLoader
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from tabulate import tabulate
import pandas as pd
import matplotlib.pyplot as plt
from permetrics.regression import RegressionMetric
from rdkit import Chem
from rdkit.Chem import MolFromSmiles, SDWriter, Draw
from rdkit.Chem import AllChem, AddHs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import DataStructs
from rdkit import RDLogger
from tqdm import tqdm
import seaborn as sns
import learn2learn as l2l
from functools import lru_cache
from collections import defaultdict
import os

def get_key_by_value(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None

def get_df_for_act(act='activity', target_column='lngamma'):
    if act == 'activity':
        df_train = pd.read_csv('training.csv', sep=';')
        df_test = pd.read_csv('test.csv', sep=';')
        
        df_train = df_train[['T/K', target_column, 'SMILES_IL', 'SMILES_solute', 'Chemical_family', 'Name_Solute']]
        df_train.rename(columns={
            'T/K': 'temps', target_column: 'y', 'SMILES_IL': 'smiles',
            'SMILES_solute': 'smiles_solutes', 'Name_Solute': 'solutes',
        }, inplace=True)
        
        df_test = df_test[['T/K', target_column, 'SMILES_IL', 'SMILES_solute', 'Chemical_family', 'Name_Solute']]
        df_test.rename(columns={
            'T/K': 'temps', target_column: 'y', 'SMILES_IL': 'smiles',
            'SMILES_solute': 'smiles_solutes', 'Name_Solute': 'solutes',
        }, inplace=True)

        return df_train, df_test

def get_joined_df(df_train, df_test, flag_scale_target, flag_transformation, flag_two_transforms):
    if flag_scale_target:
        if flag_transformation == 'quantile_normal':
          scaler = QuantileTransformer(output_distribution='normal')
        elif flag_transformation == 'quantile':
          scaler = QuantileTransformer()
        elif flag_transformation == 'power':
          scaler = PowerTransformer(standardize=True)
        elif flag_transformation == 'minmax':
          scaler = MinMaxScaler()
        elif flag_transformation == 'roboust':
          scaler = RobustScaler()
        elif flag_transformation == 'standard':
          scaler = StandardScaler()
        
        df_train['y'] = scaler.fit_transform(df_train[['y']].values)
        df_test['y'] = scaler.transform(df_test[['y']].values)
        
        if flag_two_transforms:
          scaler2 = MinMaxScaler()
          df_train['y'] = scaler2.fit_transform(df_train[['y']].values)
          df_test['y'] = scaler2.transform(df_test[['y']].values)

    temp_scaler = MinMaxScaler()
    df_train['temps'] = temp_scaler.fit_transform(df_train[['temps']].values)
    df_test['temps'] = temp_scaler.transform(df_test[['temps']].values)
    
    df_train['split'] = 0
    df_test['split'] = 1
    df = pd.concat([df_train, df_test], ignore_index=True)
    df.reset_index(inplace=True, drop=True)
    if flag_two_transforms:
        return df, [scaler, scaler2]
    else: 
        return df, [scaler]


@lru_cache(maxsize=None)
def expand_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    frags = Chem.GetMolFrags(mol, asMols=True)
    cation = anion = None

    for frag in frags:
        charge = Chem.GetFormalCharge(frag)
        frag_smiles = Chem.MolToSmiles(frag)
        if charge > 0:
            cation = frag_smiles
        elif charge < 0:
            anion = frag_smiles

    return cation, anion

def expansion_smiles_cat_ani(smiles_vals):
    smiles_cations = []
    smiles_anions = []
    for smiles in smiles_vals:
      smiles_cation, smiles_anion = expand_smiles(smiles)
      smiles_cations.append(smiles_cation)
      smiles_anions.append(smiles_anion)
    return smiles_cations, smiles_anions


def normalize_smiles(smile, nrm):
  cosmo_flag = False
  if '_cosmo' in smile:
    smile = smile.replace('_cosmo', '')
    cosmo_flag = True
  mol = Chem.MolFromSmiles(smile)
  mol_norm = nrm.normalize(mol)
  smile_norm = Chem.MolToSmiles(mol_norm, True)
  if cosmo_flag:
    smile_norm = smile_norm + '_cosmo'
  return smile_norm


def get_df(act, target_column, flag_verbose, flag_cleaning, flag_scale_target, flag_transformation, flag_two_transforms):
    df_train, df_test = get_df_for_act(act, target_column)
    df, scalers = get_joined_df(df_train, df_test, flag_scale_target, flag_transformation, flag_two_transforms)
    if flag_verbose: df['y'].hist()
    smiles_cations, smiles_anions = expansion_smiles_cat_ani(df['smiles'].values)
    df['smiles_cation'] = smiles_cations
    df['smiles_anion'] = smiles_anions
    RDLogger.DisableLog('rdApp.*')
    nrm = rdMolStandardize.Normalizer()

    smiles_cation_unique = df['smiles_cation'].unique()
    smiles_anion_unique = df['smiles_anion'].unique()
    smiles_solutes_unique = df['smiles_solutes'].unique()
    
    smiles_cation_norm_dict = {}
    smiles_anion_norm_dict = {}
    smiles_solutes_norm_dict = {}
    for smile in smiles_cation_unique:
      smiles_cation_norm_dict[smile] = normalize_smiles(smile, nrm)
    for smile in smiles_anion_unique:
      smiles_anion_norm_dict[smile] = normalize_smiles(smile, nrm)
    for smile in smiles_solutes_unique:
      smiles_solutes_norm_dict[smile] = normalize_smiles(smile, nrm)
    
    df['smiles_cation'] = df['smiles_cation'].map(smiles_cation_norm_dict)
    df['smiles_anion'] = df['smiles_anion'].map(smiles_anion_norm_dict)
    df['smiles'] = df['smiles_cation'] + '.' + df['smiles_anion']
    df['smiles_solutes'] = df['smiles_solutes'].map(smiles_solutes_norm_dict)
    RDLogger.EnableLog('rdApp.*')
    return df, scalers


def molecular_similarity(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    gen = rdFingerprintGenerator.GetMorganGenerator(2, 2048)
    fp1 = gen.GetFingerprint(mol1)
    fp2 = gen.GetFingerprint(mol2)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def get_molecules_similarity(df, flag_verbose, savedir):
    solute_name_dict = df[['solutes', 'smiles_solutes']].drop_duplicates().set_index('solutes').to_dict()['smiles_solutes']
    substances_iupac_list = df['solutes'].unique()

    substances_iupac_list = sorted(substances_iupac_list, key=lambda x: df[df['solutes'] == x].shape[0], reverse=True)

    solute_names = list(solute_name_dict.keys())
    similarity_matrix = np.zeros((len(solute_names), len(solute_names)))

    for i, smile1 in enumerate(solute_name_dict.values()):
        for j, smile2 in enumerate(solute_name_dict.values()):
            similarity_matrix[i, j] = molecular_similarity(smile1, smile2)
    similarity_matrix = similarity_matrix - np.eye(len(similarity_matrix))

    similarity_matrix_df = pd.DataFrame(similarity_matrix, index=solute_names, columns=solute_names)
    similarity_matrix_max_vals = pd.DataFrame(index=solute_names)
    similarity_matrix_max_vals['max'] = similarity_matrix_df.max(axis=1)
    similarity_matrix_max_vals['max_col_name'] = similarity_matrix_df.idxmax(axis=1)

    similarity_matrix_max_vals['count'] = similarity_matrix_max_vals.index.map(df[df['split'] == 0]['solutes'].value_counts())
    similarity_matrix_max_vals['count'] = similarity_matrix_max_vals['count'].fillna(0).astype(int)

    topn_similar = similarity_matrix_max_vals[similarity_matrix_max_vals['count'] > 128].sort_values(by="max", ascending=False).head(20)
    topn_dissimilar = similarity_matrix_max_vals[similarity_matrix_max_vals['count'] > 128].sort_values(by="max", ascending=True).head(20)

    sampls = topn_similar.sample(n=6, random_state=42).index.to_list()
    sampln = topn_dissimilar.sample(n=6, random_state=42).index.to_list()
    sampl = [*sampls, *sampln]

    for comp1 in sampl:
      for comp2 in sampl:
        if comp1 != comp2:
          similarity_matrix_df.loc[comp1, comp2] = 0.0
          similarity_matrix_df.loc[comp2, comp1] = 0.0

    if flag_verbose:
        sns.heatmap(similarity_matrix_df, cmap='Blues')
        if flag_verbose >= 2: plt.show()
        else: plt.savefig(f"{savedir}/similarity_map.svg", format="svg", dpi=600.0)

        print('Top similar')
        print(topn_similar)
        print('Top dissimilar')
        print(topn_dissimilar)

    return solute_name_dict, substances_iupac_list, similarity_matrix_max_vals, sampl


def smiles_to_fingerprints(smiles_list_cation, smiles_list_anion, radius=2, n_bits=1024):
    fps_smi_dict = dict()
    smiles_list_nest = [smiles_list_cation.tolist(), smiles_list_anion.tolist()]
    smiles_list = [x for xs in smiles_list_nest for x in xs]
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius,fpSize=n_bits)
        fp = mfpgen.GetFingerprint(mol)
        
        fps_smi_dict[smi] = np.array(fp)
    return fps_smi_dict

def get_fingerprints_for_molecules(smiles_list_cation, smiles_list_anion, df, mask_train):
    fps_smi_dict = smiles_to_fingerprints(smiles_list_cation, smiles_list_anion, radius=2, n_bits=1024)

    unique_smiles_df = df[mask_train][['smiles', 'smiles_cation', 'smiles_anion']].drop_duplicates()
    valid_smiles = unique_smiles_df['smiles'].values
    unique_smiles_df['cation_fp'] = unique_smiles_df['smiles_cation'].map(fps_smi_dict)
    unique_smiles_df['anion_fp'] = unique_smiles_df['smiles_anion'].map(fps_smi_dict)

    fps = [
        np.concatenate([cation, anion])
        for cation, anion in zip(unique_smiles_df['cation_fp'], unique_smiles_df['anion_fp'])
    ]
    fps = np.array(fps)

    return fps, valid_smiles

def cluster_smiles(smiles_list_cation, smiles_list_anion, df, mask_train, n_clusters=4):
    fps, valid_smiles = get_fingerprints_for_molecules(smiles_list_cation, smiles_list_anion, df, mask_train)
    
    dist_matrix = pdist(fps, metric='hamming')
    linkage_matrix = linkage(dist_matrix, method='average')
    cluster_ids = fcluster(linkage_matrix, t=n_clusters, criterion='maxclust')
    clustered_df = pd.DataFrame({'smiles': valid_smiles, 'cluster': cluster_ids})

    return clustered_df


def get_adaptation_training_indicies(df, mask_train, shot_size, method, random_state_adjust, n_clusters=8):
    if method == 'random':
        return df[mask_train].sample(n=min(mask_train.sum(), shot_size), random_state=random_state_adjust).index
    elif method == 'centroids':
        smiles_list_task_cation = df[mask_train].smiles_cation.unique()
        smiles_list_task_anion = df[mask_train].smiles_anion.unique()

        clustered_task_df = cluster_smiles(smiles_list_task_cation, smiles_list_task_anion, df, mask_train, n_clusters=n_clusters)

        df_clustered = df[mask_train].merge(
            clustered_task_df[['smiles', 'cluster']],
            on='smiles',
            how='inner'
        )
        
        samples_per_cluster = shot_size // n_clusters
        
        sampled_indices = (
            df_clustered.groupby('cluster', group_keys=False)
            .apply(lambda x: x.sample(n=min(len(x), samples_per_cluster), random_state=random_state_adjust))
            .index
        )

        return sampled_indices
        


def prepare_splits_pre_training(df_orig, compounds_names_to_test, solute_name_dict):
    
    df = df_orig.copy()
    df['split'] = df['split'].apply(lambda x: 1 if x == -1 else (0 if x not in (0, 1) else x))

    tasks_names_to_test = [solute_name_dict[comp_name] for comp_name in compounds_names_to_test] 

    for task in tasks_names_to_test:
        df.loc[(df['smiles_solutes'] == task) & (df['split'] == 0), 'split'] = -3

    return df

def prepare_splits_fine_tuning(df_orig, comp_name, validation_schema, solute_name_dict, 
                               use_cross_val, use_adjust_val_subset, shot_size, fold_to_use, 
                               adapt_selection_method='random', random_state_adjust=42, n_clusters=8):
    
    df = df_orig.copy()
    df['split'] = df['split'].apply(lambda x: 1 if x == -1 else (0 if x not in (0, 1) else x))

    task = solute_name_dict[comp_name]

    if validation_schema == 'random':
        kf = KFold(n_splits=5, shuffle=True, random_state=random_state_adjust)
    elif validation_schema in ('smiles', 'cations', 'anions'):
        kf = GroupKFold(n_splits=5)
    
        
    df.loc[(df['smiles_solutes'] == task) & (df['split'] == 1), 'split'] = -1

    mask_train = (df['smiles_solutes'] == task) & (df['split'] == 0)
    df.loc[mask_train, 'split'] = 9

    mask_train_9 = (df['smiles_solutes'] == task) & (df['split'] == 9)
    df.loc[df[mask_train_9].sample(frac=0.2, random_state=random_state_adjust).index, 'split'] = 3

    if not use_cross_val:
        mask_train_9 = (df['smiles_solutes'] == task) & (df['split'] == 9)
        adaptation_training_indicies = get_adaptation_training_indicies(df, mask_train_9, shot_size, adapt_selection_method, random_state_adjust, n_clusters)
        df.loc[adaptation_training_indicies, 'split'] = 2
    else:
        use_adjust_val_subset_flag = False

        mask_train_9 = (df['smiles_solutes'] == task) & (df['split'] == 9)
        if use_adjust_val_subset and mask_train_9.sum() > shot_size:
            extra_samples = df[mask_train_9].sample(n=mask_train_9.sum() - shot_size, random_state=random_state_adjust)
            df.loc[extra_samples.index, 'split'] = 4
            use_adjust_val_subset_flag = True

        if validation_schema == 'random':
            splits_to_enum = kf.split(df[mask_train_9])
        else:
            group_col = {
                'cations': 'smiles_cation',
                'anions': 'smiles_anion',
                'smiles': 'smiles'
            }[validation_schema]
            splits_to_enum = kf.split(df[mask_train_9], groups=df.loc[mask_train_9, group_col])

        for fold_index, (train_idx, valid_idx) in enumerate(splits_to_enum):
            if fold_index == fold_to_use:
                df.loc[df.iloc[valid_idx].index, 'split'] = 2 
                break

    return df


def extract_tasks_and_targets(df):
    tasks = df['smiles_solutes'].values
    tasks_dict = {task: i for i, task in enumerate(np.unique(tasks))}
    tasks = np.array([tasks_dict[task] for task in tasks])
    df['ids'] = tasks
    
    y = df['y'].values
    return df, tasks, tasks_dict, y


def get_atom_features(smi):
    HYBRIDIZATION_MAP = {
      HybridizationType.SP: 0,
      HybridizationType.SP2: 1,
      HybridizationType.SP3: 2,
      HybridizationType.SP3D: 3,
      HybridizationType.SP3D2: 4,
      HybridizationType.UNSPECIFIED: 5,
      HybridizationType.S: 6,
      HybridizationType.OTHER: 7
    }

    mol = Chem.MolFromSmiles(smi)
    mol = Chem.RemoveHs(mol)
    atomic_number = []
    num_hs = []
    hybr = []
    charges = []
    aromacity = []
    degrees = []

    for atom in mol.GetAtoms():
        atomic_number.append(atom.GetAtomicNum())
        num_hs.append(atom.GetTotalNumHs(includeNeighbors=True))
        hybr.append(HYBRIDIZATION_MAP.get(atom.GetHybridization(), HYBRIDIZATION_MAP[HybridizationType.UNSPECIFIED]))
        charge = atom.GetFormalCharge()
        charges.append(charge if not np.isnan(charge) else 0)
        aromacity.append(atom.GetIsAromatic())
        degrees.append(atom.GetDegree())

    result = np.array([atomic_number, num_hs, hybr, charges, aromacity, degrees])
    return np.transpose(result)

def get_edges_info(smi):
  mol = Chem.MolFromSmiles(smi)
  mol = Chem.RemoveHs(mol)
  row, col, bonds_types = [], [], []

  for i, bond in enumerate(mol.GetBonds()):
      start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
      row += [start, end]
      col += [end, start]
      bonds_types += [bond.GetBondTypeAsDouble(), bond.GetBondTypeAsDouble()]
  return np.array([row, col], dtype=np.int_), np.array(bonds_types, dtype=np.float32)

memo = dict()

def dual_mol_dataframe_row_into_pytorch_geometric_molecular_graph(row, smile_col, memo):
  if row[smile_col] in memo:
    x, edge_index, edge_attr = memo[row[smile_col]]
  else:
    c_x = get_atom_features(row['smiles_cation'])
    a_x = get_atom_features(row['smiles_anion'])

    temp_up = np.concatenate([c_x, np.zeros([c_x.shape[0],a_x.shape[1]])], axis=1)
    temp_down = np.concatenate([np.zeros([a_x.shape[0],c_x.shape[1]]), a_x], axis=1)
    b = np.concatenate([temp_up, temp_down])
    x = torch.tensor(b, dtype=torch.float)

    c_edge_index, c_edge_weights = get_edges_info(row['smiles_cation'])
    a_edge_index, a_edge_weights = get_edges_info(row['smiles_anion'])

    b = np.concatenate([c_edge_index, a_edge_index], axis=1)
    edge_index = torch.tensor(b, dtype=torch.long)

    b = np.concatenate([c_edge_weights, a_edge_weights])
    edge_attr = torch.tensor(b, dtype=torch.float).view(-1,1)

    memo[row[smile_col]] = (x, edge_index, edge_attr)
  y = torch.tensor(row['y'], dtype=torch.float)
  id = row['ids']
  temps = torch.tensor(row['temps'], dtype=torch.float)
  data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
              y=y, id=id, smiles=row[smile_col], temps=temps,
              )
  return data

def single_mol_dataframe_row_into_pytorch_geometric_molecular_graph(row, smile_col, memo):
  if row[smile_col] in memo:
    x, edge_index, edge_attr = memo[row[smile_col]]
  else:
    s_x = get_atom_features(row[smile_col])
    x = torch.tensor(s_x, dtype=torch.float)

    s_edge_index, s_edge_weights = get_edges_info(row[smile_col])
    edge_index = torch.tensor(s_edge_index, dtype=torch.long)
    edge_attr = torch.tensor(s_edge_weights, dtype=torch.float).view(-1,1)

    memo[row[smile_col]] = (x, edge_index, edge_attr)
  y = torch.tensor(row['y'], dtype=torch.float)
  id = row['ids']
  temps = torch.tensor(row['temps'], dtype=torch.float)
  data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
              y=y, id=id, temps=temps,
              smiles=row[smile_col],
              )
  return data


def add_graphs_to_df(df, tasks):
    df['ids'] = tasks
    graphs_sol_list = []
    for row in df.iterrows():
      row = row[1]
      data = dual_mol_dataframe_row_into_pytorch_geometric_molecular_graph(row, 'smiles', memo)
      data_sol = single_mol_dataframe_row_into_pytorch_geometric_molecular_graph(row, 'smiles_solutes', memo)
      graphs_sol_list.append([data, data_sol])
    df['graphs'] = graphs_sol_list
    return df


class GCN_trad(torch.nn.Module):
    def __init__(self, conv_function, input_channels, input_channels2, embedding_size, linear_size, add_params_num=0):
        super(GCN_trad, self).__init__()
        self.crafted_add_params_num = add_params_num

        self.conv1a = conv_function(input_channels, embedding_size[0])
        self.conv2a = conv_function(embedding_size[0], embedding_size[1])
        self.conv3a = conv_function(embedding_size[1], embedding_size[2])
        self.conv4a = conv_function(embedding_size[2], embedding_size[3])

        self.conv1b = conv_function(input_channels2, embedding_size[0])
        self.conv2b = conv_function(embedding_size[0], embedding_size[1])
        self.conv3b = conv_function(embedding_size[1], embedding_size[2])
        self.conv4b = conv_function(embedding_size[2], embedding_size[3])

        self.dropout1 = torch.nn.Dropout(0.2)

        self.linear1 = Linear(embedding_size[-1]+add_params_num, linear_size[0])
        self.linear2 = Linear(linear_size[0],linear_size[1])

        self.dropout2 = torch.nn.Dropout(0.3)

        self.bnf = torch.nn.BatchNorm1d(linear_size[-1])

        self.out = Linear(linear_size[-1], 1)


    def forward(self, x_l, edge_index_l, edge_weight_l, x_s, edge_index_s, edge_weight_s, batch_index_l, batch_index_s, cond=None):
        hidden1 = self.conv1a(x_l, edge_index_l, edge_weight_l).relu()
        hidden1 = self.dropout1(hidden1)
        hidden1 = self.conv2a(hidden1, edge_index_l, edge_weight_l).relu()
        hidden1 = self.dropout1(hidden1)
        hidden1 = self.conv3a(hidden1, edge_index_l, edge_weight_l).relu()
        hidden1 = self.dropout1(hidden1)
        hidden1 = self.conv4a(hidden1, edge_index_l, edge_weight_l).relu()
        hidden1 = self.dropout1(hidden1)

        hidden2 = self.conv1b(x_s, edge_index_s, edge_weight_s).relu()
        hidden2 = self.dropout1(hidden2)
        hidden2 = self.conv2b(hidden2, edge_index_s, edge_weight_s).relu()
        hidden2 = self.dropout1(hidden2)
        hidden2 = self.conv3b(hidden2, edge_index_s, edge_weight_s).relu()
        hidden2 = self.dropout1(hidden2)
        hidden2 = self.conv4b(hidden2, edge_index_s, edge_weight_s).relu()
        hidden2 = self.dropout1(hidden2)

        hidden = torch.cat([hidden1, hidden2], dim=0)
        batch_index = torch.cat([batch_index_l, batch_index_s], dim=0)
        hidden = gap(hidden, batch_index)

        if self.crafted_add_params_num != 0:
            cond = cond.unsqueeze(1)
            hidden = torch.cat([hidden, cond], dim=1)

        hidden = self.linear1(hidden)
        hidden = self.linear2(hidden)
        hidden = self.dropout2(hidden)
        hidden = self.bnf(hidden)
        hidden = torch.nn.functional.relu(hidden)
        out = self.out(hidden)

        return out, hidden


class TaskBatchSampler(Sampler):
    def __init__(self, dataset, shuffle_tasks=True):
        """
        dataset: list of (dataL, dataS) tuples where dataL.id is the task id
        """
        self.task_to_indices = defaultdict(list)
        for idx, (dataL, _) in enumerate(dataset):
            self.task_to_indices[int(dataL.id)].append(idx)

        self.task_ids = list(self.task_to_indices.keys())
        self.shuffle_tasks = shuffle_tasks

    def __iter__(self):
        task_ids = self.task_ids.copy()
        if self.shuffle_tasks:
            random.shuffle(task_ids)

        for task_id in task_ids:
            yield self.task_to_indices[task_id]

    def __len__(self):
        return len(self.task_ids)

def collate_fn(batch):
    dataL_list = [dataL for dataL, _ in batch]
    dataS_list = [dataS for _, dataS in batch]
    return [Batch.from_data_list(dataL_list), Batch.from_data_list(dataS_list)]

def plot_train_losses_vs_epoch(train_losses, valid_losses, r2s_valid, savedir, flag_verbose):
    plt.figure(figsize=(15,4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(r2s_valid, label='Valid R2', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('R2')
    plt.legend()
    
    if flag_verbose >= 2: plt.show()
    else: plt.savefig(f"{savedir}/train_losses_vs_epoch.svg", format="svg", dpi=600.0)

def plot_r2_scores_pretraining(train_losses, valid_losses, r2s_valid, savedir, flag_verbose):
    plt.figure(figsize=(8, 4))
    r2s_valid = np.array(r2s_valid)
    plt.plot(r2s_valid[r2s_valid > 0], label='Valid R2', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('R2')
    plt.legend()

    if flag_verbose >= 2: plt.show()
    else: plt.savefig(f"{savedir}/r2_scores_pretraining.svg", format="svg", dpi=600.0)


def test_model(model_to_adapt, loader, name, scalers, cond_names, flag_scale_target, flag_two_transforms, device):
    if len(scalers) == 1: 
        scaler = scalers[0]
    if len(scalers) == 2: 
        scaler = scalers[0]
        scaler2 = scalers[1]
    model_to_adapt.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data in loader:
            data[0] = data[0].to(device)
            data[1] = data[1].to(device)
            if cond_names:
                out, _ = model_to_adapt(data[0].x, data[0].edge_index, data[0].edge_attr, 
                                        data[1].x, data[1].edge_index, data[1].edge_attr, 
                                        data[0].batch, data[1].batch, data[0].temps)
            else:
                out, _ = model_to_adapt(data[0].x, data[0].edge_index, data[0].edge_attr, 
                                        data[1].x, data[1].edge_index, data[1].edge_attr, 
                                        data[0].batch, data[1].batch)
            y_true.extend(data[0].y.cpu().numpy())
            y_pred.extend(out.squeeze().cpu().numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if flag_scale_target:
        if flag_two_transforms:
            y_true_unscaled = scaler2.inverse_transform(y_true.reshape(-1, 1)).flatten()
            y_pred_unscaled = scaler2.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            y_true_unscaled = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
            y_pred_unscaled = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        else:
            y_true_unscaled = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
            y_pred_unscaled = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    
    mse_values = (y_true_unscaled - y_pred_unscaled) ** 2
    
    metric = RegressionMetric(y_true_unscaled, y_pred_unscaled)
    rmse = float(metric.get_metric_by_name('RMSE')['RMSE'])
    r2 = float(metric.get_metric_by_name('R2')['R2'])
    mae = float(metric.get_metric_by_name('MAE')['MAE'])
    mare = float(np.mean(np.abs(y_true_unscaled - y_pred_unscaled) / np.abs(y_true_unscaled + 1e-8), axis=0))
    
    print(f"{r2:.4f}")
    print(f"{rmse:.4f}")
    print(f"{mae:.4f}")
    print(f"{mare:.4f}")
    print()
    
    return mse_values, y_true, y_pred, rmse, r2, mae, mare

def plot_parity(y_true, y_pred, flag_verbose, savedir, title="Task Test set Compound parity plot n shot random"):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(title)
    plt.grid()
    if flag_verbose >= 2: plt.show()
    else:
      file_name_parity = title.split()
      plt.savefig(f"{savedir}/parity_{file_name_parity[3]}_{file_name_parity[6]}_{file_name_parity[8]}.svg", format="svg", dpi=600.0) 


def fine_tuning_loop(model_to_adapt, scalers, 
                     adapt_loader, valid_loader, cond_names, flag_scale_target, flag_two_transforms, 
                     epochs, device, savedir='.'):
    
    optimizer = torch.optim.Adam(model_to_adapt.parameters(), lr=1e-6)
    criterion = torch.nn.MSELoss()
    
    train_losses = []
    valid_losses = []
    r2s_valid = []
    best_valid_loss = float('inf')
    if len(scalers) == 1: 
        scaler = scalers[0]
    if len(scalers) == 2: 
        scaler = scalers[0]
        scaler2 = scalers[1]
    
    train_loss, valid_loss, r2 = 0, 0, 0
    
    for epoch in (pbar := tqdm(range(epochs))):
        pbar.set_description(f"Loss: {train_loss:.3f}, valid_loss: {valid_loss:.3f}, R2: {r2:0.3f}")
    
        model_to_adapt.train()
        train_loss = 0
        r2 = 0
        
        total_samples = 0
        for data in adapt_loader:
            samples_in_batch = data[0].y.shape[0]
            total_samples += samples_in_batch
            data[0] = data[0].to(device)
            data[1] = data[1].to(device)
            optimizer.zero_grad()
            if cond_names:
                out, _ = model_to_adapt(data[0].x, data[0].edge_index, data[0].edge_attr, data[1].x, data[1].edge_index, data[1].edge_attr, data[0].batch, data[1].batch, data[0].temps)
            else:
                out, _ = model_to_adapt(data[0].x, data[0].edge_index, data[0].edge_attr, data[1].x, data[1].edge_index, data[1].edge_attr, data[0].batch, data[1].batch)
            loss = criterion(out.squeeze(), data[0].y)
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().cpu().item() * samples_in_batch
        train_loss /= total_samples
    
        model_to_adapt.eval()
        valid_loss = 0
        y_true = []
        y_pred = []
        total_samples = 0
        with torch.no_grad():
            for data in valid_loader:
                samples_in_batch = data[0].y.shape[0]
                total_samples += samples_in_batch
                
                data[0] = data[0].to(device)
                data[1] = data[1].to(device)
                if cond_names:
                    out, _ = model_to_adapt(data[0].x, data[0].edge_index, data[0].edge_attr, data[1].x, data[1].edge_index, data[1].edge_attr, data[0].batch, data[1].batch, data[0].temps)
                else:
                    out, _ = model_to_adapt(data[0].x, data[0].edge_index, data[0].edge_attr, data[1].x, data[1].edge_index, data[1].edge_attr, data[0].batch, data[1].batch)
                loss = criterion(out.squeeze(), data[0].y)
                valid_loss += loss.detach().cpu().item() * samples_in_batch
                y_true.extend(data[0].y.cpu().numpy())
                y_pred.extend(out.squeeze().cpu().numpy())
        valid_loss /= total_samples
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
    
        if flag_scale_target:
            if flag_two_transforms:
              y_true_unscaled = scaler2.inverse_transform(y_true.reshape(-1, 1)).flatten()
              y_pred_unscaled = scaler2.inverse_transform(y_pred.reshape(-1, 1)).flatten()
              y_true_unscaled = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
              y_pred_unscaled = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            else:
              y_true_unscaled = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
              y_pred_unscaled = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    
        metric = RegressionMetric(y_true_unscaled, y_pred_unscaled)
        rmse = metric.get_metric_by_name('RMSE')['RMSE']
        r2 = metric.get_metric_by_name('R2')['R2']
        mae = metric.get_metric_by_name('MAE')['MAE']
        mare = float(np.mean(np.abs(y_true_unscaled - y_pred_unscaled) / np.abs(y_true_unscaled + 1e-8), axis=0))

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        r2s_valid.append(r2)
    
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model_to_adapt.state_dict(), f'{savedir}/best_model.pth')
            best_train_loss = train_loss
            best_valid_loss = valid_loss
            best_r2 = r2
            best_rmse = rmse
            best_mae = mae
            best_mare = mare
            print(f"Saved best model at epoch {epoch+1}")
    
    print(f'\nTrain Loss: {best_train_loss:.4f} | Valid Loss: {best_valid_loss:.4f}', end=' | ')
    print(f'RMSE: {best_rmse:.4f} | R2: {best_r2:.4f} | MAE: {best_mae:.4f} | MARE: {best_mare:.4f}')
    return model_to_adapt, train_losses, valid_losses, r2s_valid


def plot_finetune_losses_vs_epoch(train_losses, valid_losses, r2s_valid, savedir, flag_verbose):

    plt.figure(figsize=(15, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(r2s_valid, label='Valid R2', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('R2')
    plt.legend()
    if flag_verbose >= 2: plt.show()
    else: plt.savefig(f"{savedir}/finetune_losses_vs_epoch.svg", format="svg", dpi=600.0)
