#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EEG–DNN temporal misalignment analysis.

This script:
1. Loads behavioral (action/object) and GIST RDMs.
2. Loads or computes DNN–EEG RSA correlations (per layer, per model).
3. Computes partial correlations controlling for alternative RDMs
   (action, object, GIST) in two variants:
   - Variant 1: partial_corr_results[model][layer][rdm_name]
   - Variant 2: partial_corr_results_2[model][rdm_name][layer]
4. Performs statistical comparisons against behavioral and GIST RDMs.
5. Generates figures for:
   - Highest timepoints per model
   - Group-averaged highest timepoints
   - Slopes across depth
   - Max correlation vs depth
   - Average DNN–EEG correlations across layers
   - Average partial correlations
   - AUC differences of partial vs base correlations

Note: Expects precomputed ERP RDMs and DNN RDMs on disk (paths below).
"""

import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LinearRegression
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu
import pingouin as pg


# ---------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------

# Directory where this script resides
try:
    DIRNAME = os.path.dirname(os.path.abspath(__file__))
except NameError:
    DIRNAME = os.getcwd()


FIGURES_DIR = os.path.join(DIRNAME, "Figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Time axis
TMIN = -0.1
TMAX = 1.0
DOWN_SAMPLE_RATE = 128

DURATION = TMAX - TMIN
N_TIMEPOINTS = int(DURATION * DOWN_SAMPLE_RATE) + 1  # include endpoint
t = np.linspace(TMIN, TMAX, N_TIMEPOINTS)

# Distance metric for RSA
DISTANCE_METRIC = "correlation"

# Paths
ERP_RDM_PATH = "/home/clemens-uva/Desktop/EEG---temporal-dynamics-of-affordance-perception/RDMs/ERP_sliding_window_RDMs/"
DNN_RDM_PATH = "/home/clemens-uva/Github_repos/Visact_fMRI/code/DNN_RDMS/"
CORR_DATA_SAVE_PATH = "/home/clemens-uva/Desktop/EEG_Temporal_misalignment/03_DNN/correlation_data.pkl"
PARTIAL_CORR_SAVE_PATH = "/home/clemens-uva/Desktop/EEG_Temporal_misalignment/03_DNN/partial_corr_results.pkl"
PARTIAL_CORR2_SAVE_PATH = "/home/clemens-uva/Desktop/EEG_Temporal_misalignment/03_DNN/partial_corr_results_2.pkl"

# Stimulus orderings
EEG_LIST_ACTION_SORTED = [
    'outdoor_manmade_0147', 'outdoor_manmade_0148', 'outdoor_natural_0246',
    'outdoor_natural_0062', 'outdoor_natural_0160', 'outdoor_natural_0255',
    'outdoor_natural_0128', 'indoor_0156', 'outdoor_manmade_0173',
    'outdoor_manmade_0089', 'outdoor_natural_0104', 'outdoor_natural_0273',
    'outdoor_natural_0079', 'outdoor_manmade_0175', 'outdoor_natural_0042',
    'outdoor_natural_0198', 'outdoor_manmade_0131', 'outdoor_natural_0091',
    'outdoor_manmade_0152', 'outdoor_natural_0200', 'outdoor_manmade_0157',
    'outdoor_manmade_0155', 'indoor_0282', 'outdoor_manmade_0256',
    'outdoor_manmade_0257', 'outdoor_natural_0011', 'indoor_0066',
    'outdoor_manmade_0119', 'outdoor_manmade_0220', 'outdoor_manmade_0068',
    'outdoor_manmade_0133', 'outdoor_manmade_0258', 'outdoor_manmade_0040',
    'outdoor_natural_0132', 'outdoor_manmade_0064', 'outdoor_manmade_0032',
    'outdoor_manmade_0063', 'outdoor_manmade_0015', 'outdoor_manmade_0110',
    'outdoor_manmade_0167', 'outdoor_manmade_0117', 'outdoor_manmade_0030',
    'outdoor_natural_0207', 'outdoor_natural_0053', 'outdoor_natural_0261',
    'outdoor_natural_0097', 'outdoor_natural_0004', 'outdoor_manmade_0149',
    'outdoor_natural_0034', 'outdoor_manmade_0161', 'indoor_0033',
    'indoor_0163', 'indoor_0235', 'indoor_0100',
    'indoor_0058', 'indoor_0145', 'indoor_0271',
    'indoor_0266', 'indoor_0130', 'outdoor_manmade_0276',
    'indoor_0025', 'indoor_0021', 'outdoor_manmade_0165',
    'indoor_0283', 'indoor_0136', 'indoor_0249',
    'indoor_0279', 'indoor_0215', 'indoor_0221',
    'indoor_0216', 'indoor_0214', 'indoor_0080',
    'indoor_0103', 'indoor_0146', 'indoor_0055',
    'indoor_0212', 'indoor_0281', 'outdoor_manmade_0154',
    'indoor_0270', 'outdoor_natural_0049', 'outdoor_natural_0009',
    'outdoor_natural_0010', 'indoor_0272', 'outdoor_natural_0008',
    'outdoor_natural_0052', 'outdoor_natural_0023', 'outdoor_natural_0250',
    'outdoor_natural_0050', 'outdoor_natural_0017', 'outdoor_natural_0252'
]

FMRI_STIM_ORDERING = [
    'indoor_0021', 'indoor_0025', 'indoor_0033', 'indoor_0055',
    'indoor_0058', 'indoor_0066', 'indoor_0080', 'indoor_0100',
    'indoor_0103', 'indoor_0130', 'indoor_0136', 'indoor_0145',
    'indoor_0146', 'indoor_0156', 'indoor_0163', 'indoor_0212',
    'indoor_0214', 'indoor_0215', 'indoor_0216', 'indoor_0221',
    'indoor_0235', 'indoor_0249', 'indoor_0266', 'indoor_0270',
    'indoor_0271', 'indoor_0272', 'indoor_0279', 'indoor_0281',
    'indoor_0282', 'indoor_0283', 'outdoor_manmade_0015',
    'outdoor_manmade_0030', 'outdoor_manmade_0032', 'outdoor_manmade_0040',
    'outdoor_manmade_0063', 'outdoor_manmade_0064', 'outdoor_manmade_0068',
    'outdoor_manmade_0089', 'outdoor_manmade_0110', 'outdoor_manmade_0117',
    'outdoor_manmade_0119', 'outdoor_manmade_0131', 'outdoor_manmade_0133',
    'outdoor_manmade_0147', 'outdoor_manmade_0148', 'outdoor_manmade_0149',
    'outdoor_manmade_0152', 'outdoor_manmade_0154', 'outdoor_manmade_0155',
    'outdoor_manmade_0157', 'outdoor_manmade_0161', 'outdoor_manmade_0165',
    'outdoor_manmade_0167', 'outdoor_manmade_0173', 'outdoor_manmade_0175',
    'outdoor_manmade_0220', 'outdoor_manmade_0256', 'outdoor_manmade_0257',
    'outdoor_manmade_0258', 'outdoor_manmade_0276', 'outdoor_natural_0004',
    'outdoor_natural_0008', 'outdoor_natural_0009', 'outdoor_natural_0010',
    'outdoor_natural_0011', 'outdoor_natural_0017', 'outdoor_natural_0023',
    'outdoor_natural_0034', 'outdoor_natural_0042', 'outdoor_natural_0049',
    'outdoor_natural_0050', 'outdoor_natural_0052', 'outdoor_natural_0053',
    'outdoor_natural_0062', 'outdoor_natural_0079', 'outdoor_natural_0091',
    'outdoor_natural_0097', 'outdoor_natural_0104', 'outdoor_natural_0128',
    'outdoor_natural_0132', 'outdoor_natural_0160', 'outdoor_natural_0198',
    'outdoor_natural_0200', 'outdoor_natural_0207', 'outdoor_natural_0246',
    'outdoor_natural_0250', 'outdoor_natural_0252', 'outdoor_natural_0255',
    'outdoor_natural_0261', 'outdoor_natural_0273'
]

fmri_ordering = [x.replace("_", "") for x in FMRI_STIM_ORDERING]
images_name = [x.replace("_", "") for x in EEG_LIST_ACTION_SORTED]

# DNN model list and metadata
MODEL_LIST = [
    "AlexNet_VISACT_RDM", "VGG16_VISACT_RDM", "ResNet50_VISACT_RDM",
    "Places365_VISACT_RDM", "SceneParsing_VISACT_RDM", "DINO_VISACT_RDM",
    "CLIP_RN101_VISACT_RDM", "x3d_m_VISACT_RDM", "slowfast_r101_VISACT_RDM",
    "vit_base_patch16_384__VISACT_90_RDM", "DINO_VIT_BASE_P16",
    "CLIP_ViT-B_-_16_VISACT_RDM", "CLIP_ViT-B_-_32_VISACT_90_RDM"
]

MODEL_COLORS = [
    '#7209b7', '#7209b7', '#7209b7', "#480ca8", '#3f37c9',
    '#4361ee', '#4361ee', '#4cc9f0', '#4cc9f0', '#f48c06',
    '#ffba08', '#ffba08', '#ffba08'
]

MODEL_SHORT_NAMES = [
    "AlexNet", "VGG16", "ResNet50",
    "Places365", "SceneParsing",
    "Dino", "CLIP RN101", "x3d", "slowfast RN101",
    "ViT", "DINO ViT", "CLIP ViT-B-16", "CLIP ViT-B-32"
]

df_model_info = pd.DataFrame(
    {"model": MODEL_LIST, "color": MODEL_COLORS, "name": MODEL_SHORT_NAMES}
)

# Grouping of models by architecture / objective
GROUP_INDICES = {
    'CNN object classification': [0, 1, 2],
    'CNN scene classification': [3],
    'CCN scene parsing': [4],
    'CNN alternative training objective': [5, 6],
    'CNN video': [7, 8],
    'ViT object classification': [9],
    'ViT alternative training objective': [10, 11, 12],
}

GROUP_COLORS = ['#7209b7', '#480ca8', '#3f37c9', '#4361ee',
                '#4cc9f0', '#f48c06', '#ffba08']

# For slope plots
LINESTYLES = [
    "dashed", "solid", "dotted", "solid", "solid",
    "solid", "dashed", "solid", "dashed", "solid",
    "dashed", "solid", "dotted"
]

# For layer reduction (partial corr plots etc.)
MODELS_LAYERS = {
    "AlexNet_VISACT_RDM": ["layer1", "layer2", "layer3", "layer4", "layer5"],
    "VGG16_VISACT_RDM": ["layer1", "layer2", "layer3", "layer4", "layer5", "layer6", "layer7", "layer8", "layer9", "layer10", "layer11", "layer12", "layer13"],
    "ResNet50_VISACT_RDM": ["layer1", "layer2", "layer3", "layer4"],
    "Places365_VISACT_RDM": ["layer1", "layer2", "layer3", "layer4", "layer5"],
    "SceneParsing_VISACT_RDM": ["layer1", "layer2", "layer3", "layer4", "layer5"],
    "DINO_VISACT_RDM": ["layer1", "layer2", "layer3", "layer4", "layer5"],
    "CLIP_RN101_VISACT_RDM": ["layer1", "layer2", "layer3", "layer4"],
    "x3d_m_VISACT_RDM": ["layer1", "layer2", "layer3", "layer4", "layer5", "layer6", "layer7", "layer8", "layer9", "layer10"],
    "slowfast_r101_VISACT_RDM": ["layer1", "layer2", "layer3", "layer4", "layer5", "layer6", "layer7", "layer8", "layer9", "layer10", "layer11", "layer12"],
    "vit_base_patch16_384__VISACT_90_RDM": ["layer1", "layer2", "layer3", "layer4", "layer5", "layer6", "layer7", "layer8", "layer9", "layer10", "layer11", "layer12"],
    "DINO_VIT_BASE_P16": ["layer1", "layer2", "layer3", "layer4", "layer5", "layer6", "layer7", "layer8", "layer9", "layer10", "layer11", "layer12"],
    "CLIP_ViT-B_-_16_VISACT_RDM": ["layer1", "layer2", "layer3", "layer4", "layer5", "layer6", "layer7", "layer8", "layer9", "layer10", "layer11", "layer12"],
    "CLIP_ViT-B_-_32_VISACT_90_RDM": ["layer1", "layer2", "layer3", "layer4", "layer5", "layer6", "layer7", "layer8", "layer9", "layer10", "layer11", "layer12"],
}


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def load_and_sort_rdm(rdm_path: str, ordering: List[str]) -> np.ndarray:
    """Load an RDM and sort it according to the given ordering."""
    rdm = np.load(rdm_path)
    if rdm.ndim == 3 and rdm.shape[0] != 90:
        rdm = np.mean(rdm, axis=0)

    rdm_df = pd.DataFrame(rdm, index=fmri_ordering, columns=fmri_ordering)
    sorted_rdm = rdm_df.loc[ordering, ordering].values
    return sorted_rdm


def corr_with_model(rdm1: np.ndarray, model_rdm: np.ndarray) -> Tuple[float, List[float]]:
    """
    Compute Spearman correlation between time-resolved EEG RDMs and a fixed model RDM.
    rdm1: (T, N, N)
    model_rdm: (N, N)
    """
    corrs = []
    rdv2 = squareform(model_rdm.round(10))
    for timepoint in range(rdm1.shape[0]):
        rdv1 = squareform(rdm1[timepoint].round(10))
        corr, _ = spearmanr(rdv1, rdv2)
        corrs.append(corr)
    return float(np.mean(corrs)), corrs


def compute_corrs_average(distance_metric: str,
                          model_rdm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute subject-wise and average EEG–model RSA time courses.
    Returns mean_corr (T,), sem (T,), and subject-wise array (S, T).
    """
    all_sub_corrs = []

    for fname in os.listdir(ERP_RDM_PATH):
        if (distance_metric in fname) and ("_5_" in fname):
            rdms_per_subject = np.load(os.path.join(ERP_RDM_PATH, fname))
            _, corrs = corr_with_model(rdms_per_subject, model_rdm)
            all_sub_corrs.append(corrs)

    all_sub_corrs = np.array(all_sub_corrs)
    mean_corr = np.mean(all_sub_corrs, axis=0)
    sem = np.std(all_sub_corrs, axis=0) / np.sqrt(all_sub_corrs.shape[0])

    return mean_corr, sem, all_sub_corrs


def compute_partial_corrs_sliding(distance_metric: str,
                                  substr: str,
                                  layer_rdm: np.ndarray,
                                  control_rdm: np.ndarray
                                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute partial correlation between EEG RDMs and a DNN layer RDM,
    controlling for another model RDM, across time.

    Uses Spearman partial correlation on vectorized RDMs (upper triangle).
    """
    rdv_x = squareform(layer_rdm.round(10))     # DNN layer RDM
    rdv_z = squareform(control_rdm.round(10))   # control RDM

    all_sub_corrs = []

    for fname in os.listdir(ERP_RDM_PATH):
        if (distance_metric in fname) and (substr in fname):
            eeg_rdms = np.load(os.path.join(ERP_RDM_PATH, fname))  # (T, N, N)
            n_time = eeg_rdms.shape[0]
            corrs = []

            for ti in range(n_time):
                rdv_y = squareform(eeg_rdms[ti].round(10))         # EEG RDM
                df = pd.DataFrame({"x": rdv_x, "y": rdv_y, "z": rdv_z})
                res = pg.partial_corr(data=df, x="x", y="y", covar="z", method="spearman")
                corrs.append(res["r"].iloc[0])

            all_sub_corrs.append(corrs)

    all_sub_corrs = np.array(all_sub_corrs)
    mean_corr = np.mean(all_sub_corrs, axis=0)
    sem_corr = np.std(all_sub_corrs, axis=0) / np.sqrt(all_sub_corrs.shape[0])

    return mean_corr, sem_corr, all_sub_corrs


def normalize_to_zero_one(values) -> List[float]:
    """Normalize a 1D iterable to [0, 1] using first and last as min/max."""
    values = np.array(values, dtype=float)
    min_val = values[0]
    max_val = values[-1]
    normalized_values = (values - min_val) / (max_val - min_val)
    return normalized_values.tolist()


def load_dnn_rdms() -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load DNN RDMs from disk and return a dict:
        {model_name: {layer_name: sorted_rdm}}
    """
    dnn_rdms: Dict[str, Dict[str, np.ndarray]] = {}

    for model in MODEL_LIST:
        model_dir = os.path.join(DNN_RDM_PATH, model)
        layer_list = [x for x in os.listdir(model_dir) if not x.endswith(".json")]

        dnn_rdms[model] = {}
        for idx, layer in enumerate(layer_list):
            layer_name = f"layer{idx + 1}"
            layer_file = np.load(os.path.join(model_dir, layer), allow_pickle=True)
            layer_rdm = layer_file["arr_0"]

            # sort to match EEG ordering
            rdm_df = pd.DataFrame(layer_rdm, index=fmri_ordering, columns=fmri_ordering)
            sorted_rdm = rdm_df.loc[images_name, images_name].values
            dnn_rdms[model][layer_name] = sorted_rdm

    return dnn_rdms


def load_or_compute_correlation_data(save_path: str,
                                     dnn_rdms: Dict[str, Dict[str, np.ndarray]]
                                     ) -> Dict[str, pd.DataFrame]:
    """
    Load precomputed DNN–EEG correlation data if it exists,
    otherwise compute it and save to disk.
    """
    if os.path.exists(save_path):
        print(f"Found {save_path} — loading correlation_data.")
        with open(save_path, "rb") as f:
            return pickle.load(f)

    print("No correlation_data found — computing from RDMs...")

    correlation_data: Dict[str, pd.DataFrame] = {}
    for model in MODEL_LIST:
        layer_corrs = []
        layer_sem = []
        all_layer_names = []
        t_highest_value = []
        highest_value = []
        all_highest_values = []
        array_full = []

        for layer_name, sorted_rdm in dnn_rdms[model].items():
            mean1, sem1, subj_array = compute_corrs_average(DISTANCE_METRIC, sorted_rdm)

            layer_corrs.append(mean1)
            layer_sem.append(sem1)
            all_layer_names.append(layer_name)
            t_highest_value.append(t[np.argmax(mean1)])
            highest_value.append(np.max(mean1))
            all_highest_values.append(t[np.argmax(subj_array, axis=1)])
            array_full.append(subj_array)

        model_df = pd.DataFrame({
            'Layer': all_layer_names,
            'SEM': layer_sem,
            'Correlation': layer_corrs,
            't_highest_corr': t_highest_value,
            'all_highest_values': all_highest_values,
            'higest_corr_value': highest_value,   # keep typo for compatibility
            'position': normalize_to_zero_one(range(len(highest_value))),
            'array': array_full
        })

        correlation_data[model] = model_df

    with open(save_path, "wb") as f:
        pickle.dump(correlation_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Computation finished — correlation_data saved to {save_path}.")
    return correlation_data


def compute_mean_df_and_sem_without_nan(*dfs: pd.DataFrame) -> pd.DataFrame:
    """
    Align models with potentially different depths to a common set of positions
    and compute mean and SEM of 'higest_corr_value' across them.
    """
    min_layers = min(len(df) for df in dfs)
    ref_positions = np.linspace(0, 1, min_layers)

    reindexed = []
    for df in dfs:
        positions = df['position'].values
        values = df['higest_corr_value'].values
        interpolated = np.interp(ref_positions, positions, values)
        reindexed.append(interpolated)

    stacked = np.vstack(reindexed)
    mean_values = np.nanmean(stacked, axis=0)
    sem_values = np.nanstd(stacked, axis=0) / np.sqrt(len(dfs))

    mean_df = pd.DataFrame({
        'position': ref_positions,
        'mean_higest_corr_value': mean_values,
        'sem_higest_corr_value': sem_values
    })
    return mean_df


# ---------------------------------------------------------------------
# Partial correlations (Variant 1): cache / AUC / layer reduction
# Structure: partial_corr_results[model][layer][rdm_name]
# ---------------------------------------------------------------------

def load_or_compute_partial_corr_results(
    output_file: str,
    model_rdms: Dict[str, np.ndarray],
    dnn_rdms: Dict[str, Dict[str, np.ndarray]],
    distance_metric: str
) -> Dict[str, Dict]:
    """
    Load cached partial correlation results if present,
    otherwise compute them and save to disk.

    Structure:
      partial_corr_results[model_name][layer_name]['base_corr']
      partial_corr_results[model_name][layer_name][rdm_name]
    """
    if os.path.exists(output_file):
        print(f"Found cached partial correlations at {output_file} — loading.")
        with open(output_file, "rb") as f:
            return pickle.load(f)

    print("No cached partial correlations found — computing...")

    partial_corr_results: Dict[str, Dict] = {}

    for model_name, layers_dict in dnn_rdms.items():
        partial_corr_results[model_name] = {}

        for layer_name, layer_rdm in layers_dict.items():
            layer_dict = {}

            # Base correlation: DNN layer vs EEG (no partialling)
            mean_corr, sem_corr, base_corrs = compute_corrs_average(distance_metric, layer_rdm)
            layer_dict['base_corr'] = {
                "mean_corr": mean_corr,
                "sem_corr": sem_corr,
                "corrs": base_corrs,
            }

            # Partial correlations controlling for each RDM
            for rdm_name, model_rdm in model_rdms.items():
                mean_p, sem_p, partial_corrs = compute_partial_corrs_sliding(
                    distance_metric, "_5_", layer_rdm, model_rdm
                )
                layer_dict[rdm_name] = {
                    "mean_partial_corr": mean_p,
                    "sem_partial_corr": sem_p,
                    "partial_corrs": partial_corrs,
                }

            partial_corr_results[model_name][layer_name] = layer_dict

    with open(output_file, "wb") as f:
        pickle.dump(partial_corr_results, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved partial correlation data to {output_file}")
    return partial_corr_results


def compute_auc_for_layer_across_models(
    model_list: List[str],
    rdm_name: str,
    layer: str,
    partial_corr_results: Dict[str, Dict]
) -> Dict[str, float]:
    """
    Compute AUC of (partial) correlation timecourses for a given layer
    across all models in model_list.
    """
    auc_results: Dict[str, float] = {}

    for model in model_list:
        if (model in partial_corr_results and
            layer in partial_corr_results[model] and
            rdm_name in partial_corr_results[model][layer]):

            if rdm_name == "base_corr":
                mean_corr = partial_corr_results[model][layer]["base_corr"]["mean_corr"]
            else:
                mean_corr = partial_corr_results[model][layer][rdm_name]['mean_partial_corr']

            auc_results[model] = float(np.trapz(mean_corr))

    return auc_results


def normalize_layers(layers: List[str]) -> List[float]:
    n_layers = len(layers)
    return [i / (n_layers - 1) for i in range(n_layers)] if n_layers > 1 else [0.0]


def reduce_layers_to_common(models_layers: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Reduce layer lists to a common effective depth across models,
    using normalized positions and nearest neighbors.
    """
    min_num_layers = min(len(layers) for layers in models_layers.values())
    target_positions = np.linspace(0, 1, min_num_layers)

    reduced_layers: Dict[str, List[str]] = {}
    for model_name, layers in models_layers.items():
        norm_pos = normalize_layers(layers)
        selected = [
            layers[int(np.argmin([abs(tp - p) for p in norm_pos]))]
            for tp in target_positions
        ]
        reduced_layers[model_name] = selected
    return reduced_layers


def filter_and_rename_layers(
    partial_corr_results: Dict[str, Dict],
    reduced_layers: Dict[str, List[str]]
) -> Dict[str, Dict]:
    """
    Keep only a subset of layers per model (reduced_layers)
    and rename them to layer1, layer2, ... for convenience.

    Works with Variant 1 structure: partial_corr_results[model][layer][rdm_name]
    """
    filtered: Dict[str, Dict] = {}
    for model_name, layers_dict in partial_corr_results.items():
        if model_name not in reduced_layers:
            continue
        filtered[model_name] = {}
        for i, layer_name in enumerate(reduced_layers[model_name], start=1):
            if layer_name in layers_dict:
                filtered[model_name][f'layer{i}'] = layers_dict[layer_name]
    return filtered


# ---------------------------------------------------------------------
# Plotting: original RSA analyses
# ---------------------------------------------------------------------

def plot_highest_timepoints(
    correlation_data: Dict[str, pd.DataFrame],
    action_array: np.ndarray,
    object_array: np.ndarray,
    gist_array: np.ndarray,
    rejected_action: np.ndarray,
    rejected_object: np.ndarray,
    rejected_gist: np.ndarray,
    means: List[float],
    all_values_across_models: List[List[float]],
    models: List[str],
    out_path: str = None,
) -> None:
    """Main 'highest timepoint' plot comparing models with action/object/GIST."""
    fig, ax = plt.subplots(figsize=(7, 5))

    action_vals = t[np.argmax(action_array, axis=1)]
    object_vals = t[np.argmax(object_array, axis=1)]
    gist_vals = t[np.argmax(gist_array, axis=1)]

    # GIST row (y=1)
    jitter_gist = np.random.normal(0, 0.05, size=len(gist_vals))
    ax.scatter(gist_vals, 1 + jitter_gist, color='lightgray', alpha=0.6, s=4)
    ax.plot(np.mean(gist_vals), 1, 'o', color="black", label='GIST',
            markersize=8, markeredgecolor="white", zorder=10)

    # Object row (y=0)
    jitter_object = np.random.normal(0, 0.05, size=len(object_vals))
    ax.scatter(object_vals, 0 + jitter_object, color='lightgray', alpha=0.6, s=4)
    ax.plot(np.mean(object_vals), 0, 'o', color="blue", label='Objects',
            markersize=8, markeredgecolor="white", zorder=10)

    height = 1.0
    for i, (model, all_values) in enumerate(zip(models, all_values_across_models)):
        y_row = i + 2

        subset = df_model_info[df_model_info["model"] == model]
        color = subset["color"].iloc[0]
        name = subset["name"].iloc[0]

        jitter = np.random.normal(0, 0.05, size=len(all_values))
        ax.scatter(all_values, y_row + jitter, color='lightgray', alpha=0.6, s=4)
        ax.plot(means[i], y_row, 'o', color=color, markersize=8,
                label=name, markeredgecolor="white", zorder=10)

        # significance bars
        if rejected_action[i]:
            ax.vlines(x=height, ymin=y_row, ymax=15, color='black', linewidth=1)
            ax.hlines(y=y_row, xmin=height - 0.01, xmax=height, color='black', linewidth=1)
            height += 0.02
        if rejected_object[i]:
            ax.vlines(x=height, ymin=0, ymax=y_row, color='blue', linewidth=1)
            ax.hlines(y=y_row, xmin=height - 0.01, xmax=height, color='blue', linewidth=1)
            height += 0.02
        if rejected_gist[i]:
            ax.vlines(x=height, ymin=1, ymax=y_row, color='green', linewidth=1)
            ax.hlines(y=y_row, xmin=height - 0.01, xmax=height, color='green', linewidth=1)
            height += 0.02

    # Action row at bottom
    jitter_action = np.random.normal(0, 0.05, size=len(action_vals))
    bottom_y = len(models) + 2
    ax.scatter(action_vals, bottom_y + jitter_action, color='lightgray', alpha=0.6, s=4)
    ax.plot(np.mean(action_vals), bottom_y, 'o', color="#ff2c55", label='Affordances',
            markersize=8, markeredgecolor="white", zorder=10)

    # vertical mean lines
    mean_action_time = np.mean(action_vals)
    mean_object_time = np.mean(object_vals)
    mean_gist_time = np.mean(gist_vals)

    max_y = bottom_y + 1
    for y_row in range(max_y):
        ax.plot([mean_action_time, mean_action_time], [y_row - 0.2, y_row + 0.2],
                color="#ff2c55", linewidth=1)
        ax.plot([mean_object_time, mean_object_time], [y_row - 0.2, y_row + 0.2],
                color="blue", linewidth=1)
        ax.plot([mean_gist_time, mean_gist_time], [y_row - 0.2, y_row + 0.2],
                color="black", linewidth=1)

    names_y = ["Objects", "GIST"] + MODEL_SHORT_NAMES + ["Affordances"]
    ax.set_xlim(-0.12, 1.19)
    ax.axvline(0, color='gray', linestyle='--')
    ax.set_yticks(range(len(names_y)))
    ax.set_yticklabels(names_y)
    ax.set_ylabel('DNN models')
    ax.set_xlabel('Time (s)')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path, transparent=True, dpi=300)
    plt.close(fig)


def plot_group_t_highest(
    correlation_data: Dict[str, pd.DataFrame],
    action_array: np.ndarray,
    use_sem: bool,
    out_path: str = None
) -> None:
    """Plot timepoint of highest correlation vs normalized model depth, grouped by architecture."""
    fig, ax = plt.subplots(figsize=(5, 5))

    for idx, (group_name, model_indices) in enumerate(GROUP_INDICES.items()):
        y_values = []
        if use_sem:
            min_layers = min(len(list(correlation_data.values())[i]['t_highest_corr'])
                             for i in model_indices)
            for i_model in model_indices:
                df = list(correlation_data.values())[i_model]
                y_values.append(df['t_highest_corr'].values[:min_layers])
            y_values = np.array(y_values)
            y_avg = np.nanmean(y_values, axis=0)
            y_sem = np.nanstd(y_values, axis=0) / np.sqrt(len(model_indices))
            x_avg = np.linspace(0, 1, len(y_avg))

            ax.plot(x_avg, y_avg, marker='o', label=group_name,
                    color=GROUP_COLORS[idx], markersize=8, markeredgecolor="white")
            ax.plot(x_avg, y_avg, color=GROUP_COLORS[idx])
            ax.fill_between(x_avg, y_avg - y_sem, y_avg + y_sem,
                            color=GROUP_COLORS[idx], alpha=0.2)
        else:
            max_layers = max(len(list(correlation_data.values())[i]['t_highest_corr'])
                             for i in model_indices)
            for i_model in model_indices:
                df = list(correlation_data.values())[i_model]
                y = df['t_highest_corr'].values
                if len(y) < max_layers:
                    y = np.pad(y, (0, max_layers - len(y)), constant_values=np.nan)
                y_values.append(y)
            y_values = np.array(y_values)
            y_avg = np.nanmean(y_values, axis=0)
            x_avg = np.linspace(0, 1, len(y_avg))

            ax.plot(x_avg, y_avg, marker='o', label=group_name,
                    color=GROUP_COLORS[idx], markersize=8, markeredgecolor="white")
            ax.plot(x_avg, y_avg, color=GROUP_COLORS[idx])

    ax.set_ylim(-0.1, 1)
    ax.set_xlim(0, 1)
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.set_xlabel('Normalized model depth')
    ax.set_ylabel('Timepoint of highest Spearman Correlation (s)')
    ax.legend(bbox_to_anchor=(0.8, 0.95))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    action_vals = t[np.argmax(action_array, axis=1)]
    ax.axhline(np.mean(action_vals), color="red", linestyle="--")

    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path, transparent=True, dpi=300)
    plt.close(fig)


def plot_slopes_by_model(
    correlation_data: Dict[str, pd.DataFrame]
) -> Tuple[List[Tuple[str, float]], float]:
    """Fit regression of max correlation vs depth for each model, return slopes."""
    fig, ax = plt.subplots(figsize=(10, 5))

    center_depth = 0.5
    center_corr = 0.05
    slopes = []

    for idx, (model_name, df) in enumerate(correlation_data.items()):
        x = np.array(df['position']).reshape(-1, 1)
        y = df['higest_corr_value']

        lr = LinearRegression().fit(x, y)
        slope = lr.coef_[0]
        slopes.append((model_name, slope))

        adjusted_intercept = center_corr - slope * center_depth
        adjusted_y = slope * x + adjusted_intercept
        ax.plot(x, adjusted_y, color=MODEL_COLORS[idx],
                linestyle=LINESTYLES[idx], linewidth=2)

    slope_values = [s for _, s in slopes]
    avg_slope = float(np.mean(slope_values))
    ax.set_xlabel("Normalized model depth")
    ax.set_ylabel("Spearman Correlation")
    ax.set_title(f"Average slope {avg_slope:.5f}")

    legend_lines = [
        Line2D([0], [0], color=MODEL_COLORS[i], linestyle=LINESTYLES[i])
        for i in range(len(correlation_data))
    ]
    ax.legend(legend_lines, correlation_data.keys(),
              bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()
    plt.close(fig)

    print("Slopes for each model:")
    for model, slope in slopes:
        print(f"{model}: {slope:.6f}")

    return slopes, avg_slope


def plot_positive_negative_slopes(
    slopes: List[Tuple[str, float]],
    out_path: str = None
) -> None:
    """Split models into positive and negative slopes and plot them."""
    df = pd.DataFrame({
        "Model Name": [m for m, _ in slopes],
        "Slope": [s for _, s in slopes],
        "Color": MODEL_COLORS,
        "Linestyle": LINESTYLES
    })

    positive_df = df[df["Slope"] > 0]
    negative_df = df[df["Slope"] < 0]

    avg_pos = positive_df["Slope"].mean()
    avg_neg = negative_df["Slope"].mean()

    center_depth = 0.5
    center_corr = 0.05

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    # Positive
    for _, row in positive_df.iterrows():
        slope = row["Slope"]
        adj_inter = center_corr - slope * center_depth
        x_vals = np.array([0, 1])
        y_vals = slope * x_vals + adj_inter
        ax1.plot(x_vals, y_vals, 'o-', color=row['Color'],
                 linestyle=row['Linestyle'], markersize=5,
                 label=row['Model Name'], markeredgecolor="white")
    ax1.set_title(f"Positive Slopes (Avg: {avg_pos:.4f})")
    ax1.set_xlabel("Normalized Model Depth")
    ax1.set_ylabel("Slope values")
    ax1.legend(loc="lower center", bbox_to_anchor=(0.59, 0), ncol=2)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Negative
    for _, row in negative_df.iterrows():
        slope = row["Slope"]
        adj_inter = center_corr - slope * center_depth
        x_vals = np.array([0, 1])
        y_vals = slope * x_vals + adj_inter
        ax2.plot(x_vals, y_vals, 'o-', color=row['Color'],
                 linestyle=row['Linestyle'], markersize=5,
                 label=row['Model Name'], markeredgecolor="white")
    ax2.set_title(f"Negative Slopes (Avg: {avg_neg:.4f})")
    ax2.set_xlabel("Normalized Model Depth")
    ax2.legend(loc="lower center", bbox_to_anchor=(0.3, 0), ncol=1)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path, transparent=True, dpi=300)
    plt.show()
    plt.close(fig)


def plot_group_highest_corr(
    correlation_data: Dict[str, pd.DataFrame],
    mean_action: np.ndarray,
    use_sem: bool,
    out_path: str = None
) -> None:
    """Plot mean/SEM of max correlation vs depth, grouped by architecture."""
    fig, ax = plt.subplots(figsize=(5, 5))

    for idx, (group_name, model_indices) in enumerate(GROUP_INDICES.items()):
        group_dfs = [list(correlation_data.values())[i] for i in model_indices]
        mean_df = compute_mean_df_and_sem_without_nan(*group_dfs)

        ax.plot(mean_df['position'], mean_df['mean_higest_corr_value'],
                marker='o', label=group_name,
                color=GROUP_COLORS[idx],
                markersize=8, markeredgecolor="white")
        ax.plot(mean_df['position'], mean_df['mean_higest_corr_value'],
                color=GROUP_COLORS[idx])

        if use_sem:
            ax.fill_between(
                mean_df['position'],
                mean_df['mean_higest_corr_value'] - mean_df['sem_higest_corr_value'],
                mean_df['mean_higest_corr_value'] + mean_df['sem_higest_corr_value'],
                color=GROUP_COLORS[idx], alpha=0.2
            )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 0.10)
    ax.set_xlabel('Normalized model depth')
    ax.set_ylabel('Spearman Correlation')

    ax.axhline(np.max(mean_action), color="red", linestyle="--", linewidth=2)

    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path, transparent=True, dpi=300)
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------
# Plotting: averaged base correlations & partial correlations (Variant 1)
# ---------------------------------------------------------------------

def plot_averaged_correlations_per_model(
    model_list: List[str],
    colors: List[str],
    names: List[str],
    filtered_results: Dict[str, Dict],
    savepath: str
) -> None:
    """Average base correlations across layers for each model and plot them."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axvline(x=0, color='lightgray', linestyle='--')
    ax.axhline(y=0, color='lightgray', linestyle='--')

    for i, model_name in enumerate(model_list):
        model_data = filtered_results.get(model_name, {})
        layer_corrs = []
        for layer_name in model_data:
            layer_corrs.append(model_data[layer_name]["base_corr"]["mean_corr"])

        if layer_corrs:
            avg_corr = np.mean(layer_corrs, axis=0)
            ax.plot(t, avg_corr, label=names[i], color=colors[i])

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Spearman Correlation')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(bbox_to_anchor=(1.1, 1.05))
    fig.tight_layout()
    plt.savefig(savepath, transparent=True, dpi=300)
    plt.show()


def plot_averaged_correlations_per_model_with_sem(
    model_list: List[str],
    colors: List[str],
    names: List[str],
    filtered_results: Dict[str, Dict],
    savepath: str
) -> None:
    """Average base correlations across layers with SEM shading."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axvline(x=0, color='lightgray', linestyle='--')
    ax.axhline(y=0, color='lightgray', linestyle='--')

    for i, model_name in enumerate(model_list):
        model_data = filtered_results.get(model_name, {})
        layer_corrs = []
        for layer_name in model_data:
            layer_corrs.append(model_data[layer_name]["base_corr"]["mean_corr"])

        if layer_corrs:
            layer_corrs = np.array(layer_corrs)
            avg_corr = np.mean(layer_corrs, axis=0)
            sem_corr = np.std(layer_corrs, axis=0) / np.sqrt(layer_corrs.shape[0])

            ax.plot(t, avg_corr, label=names[i], color=colors[i])
            ax.fill_between(t, avg_corr - sem_corr, avg_corr + sem_corr,
                            color=colors[i], alpha=0.2)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Spearman Correlation')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(bbox_to_anchor=(1.1, 1.05))
    fig.tight_layout()
    plt.savefig(savepath, transparent=True, dpi=300)
    plt.show()


def plot_auc_and_test_significance(
    layer: str,
    model_list: List[str],
    partial_corr_results: Dict[str, Dict]
) -> None:
    """
    Plot AUCs for base, action, GIST for a given layer and
    run Mann–Whitney U tests with FDR correction.
    """
    def add_significance_bar(ax, x1, x2, y, text):
        ax.plot([x1, x1, x2, x2], [y, y + 0.02, y + 0.02, y], color='black', lw=1.5)
        ax.text((x1 + x2) * 0.5, y + 0.02, text, ha='center', va='bottom', color='black')

    base_auc = list(compute_auc_for_layer_across_models(model_list, 'base_corr', layer, partial_corr_results).values())
    action_auc = list(compute_auc_for_layer_across_models(model_list, 'action_eeg_rdm', layer, partial_corr_results).values())
    gist_auc = list(compute_auc_for_layer_across_models(model_list, 'GIST_265', layer, partial_corr_results).values())

    x_base = np.ones(len(base_auc)) * 1
    x_action = np.ones(len(action_auc)) * 2
    x_gist = np.ones(len(gist_auc)) * 3

    mean_base = np.mean(base_auc)
    mean_action = np.mean(action_auc)
    mean_gist = np.mean(gist_auc)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(x_base + np.random.normal(0, 0.05, size=len(base_auc)), base_auc,
               color='black', alpha=0.7, label='Base Corr')
    ax.scatter(x_action + np.random.normal(0, 0.05, size=len(action_auc)), action_auc,
               color='#ff2c55', alpha=0.7, label='Action EEG RDM')
    ax.scatter(x_gist + np.random.normal(0, 0.05, size=len(gist_auc)), gist_auc,
               color='#ee9b00', alpha=0.7, label='GIST 265 RDM')

    ax.hlines(mean_base, 0.85, 1.15, colors='black', linestyles='dashed', linewidth=2)
    ax.hlines(mean_action, 1.85, 2.15, colors='#ff2c55', linestyles='dashed', linewidth=2)
    ax.hlines(mean_gist, 2.85, 3.15, colors='#ee9b00', linestyles='dashed', linewidth=2)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['Base Corr', 'Action EEG RDM', 'GIST 265 RDM'])
    ax.set_xlabel('Model RDMs')
    ax.set_ylabel('AUC')
    ax.set_title(f'AUC Values for {layer} Across Models')
    ax.legend()

    u_base_act, p_base_act = mannwhitneyu(base_auc, action_auc)
    u_base_gist, p_base_gist = mannwhitneyu(base_auc, gist_auc)
    u_act_gist, p_act_gist = mannwhitneyu(action_auc, gist_auc)

    p_values = [p_base_act, p_base_gist, p_act_gist]
    _, p_corr, _, _ = multipletests(p_values, method='fdr_bh')
    alpha = 0.05
    y_max = max(max(base_auc), max(action_auc), max(gist_auc)) + 0.5

    if p_corr[0] < alpha:
        add_significance_bar(ax, 1, 2, y_max, '*')
        y_max += 0.1
    if p_corr[1] < alpha:
        add_significance_bar(ax, 1, 3, y_max, '*')
        y_max += 0.1
    if p_corr[2] < alpha:
        add_significance_bar(ax, 2, 3, y_max, '*')

    plt.tight_layout()
    plt.show()

    print(f"Base vs Action RDM: U = {u_base_act:.3f}, p (FDR) = {p_corr[0]:.4g}")
    print(f"Base vs GIST  RDM: U = {u_base_gist:.3f}, p (FDR) = {p_corr[1]:.4g}")
    print(f"Action vs GIST   : U = {u_act_gist:.3f}, p (FDR) = {p_corr[2]:.4g}")


def plot_average_partial_correlations(
    models,
    rdm_names,
    filtered_results: Dict[str, Dict],
    distance_metric: str,
    action_eeg_rdm: np.ndarray,
    object_eeg_rdm: np.ndarray,
    GIST_265: np.ndarray,
    savepath: str,
    base_corr_results: str = None,
    colors: List[str] = None,
    names: List[str] = None,
) -> None:
    """
    Plot average partial correlation (over layers) for one or more models
    and one or more RDMs, optionally overlaying the base correlation.

    This version works with Variant 1 structure:
      filtered_results[model][layer][rdm_name]
    """
    if isinstance(models, str):
        models = [models]
    if isinstance(rdm_names, str):
        rdm_names = [rdm_names]

    if colors is None or len(colors) < len(models):
        raise ValueError("colors must be at least as long as models")
    if names is None or len(names) < len(models):
        raise ValueError("names must be at least as long as models")

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.axvline(x=0, color='lightgray', linestyle='--')
    ax.axhline(y=0, color='lightgray', linestyle='--')

    for i, model in enumerate(models):
        color = colors[i]
        name = names[i]

        if model not in filtered_results:
            continue

        for rdm_name in rdm_names:
            layer_corrs = []
            for layer_name in filtered_results[model]:
                layer_dict = filtered_results[model][layer_name]
                if rdm_name in layer_dict and 'mean_partial_corr' in layer_dict[rdm_name]:
                    layer_corrs.append(layer_dict[rdm_name]['mean_partial_corr'])

            if layer_corrs:
                avg_mean_corr = np.mean(layer_corrs, axis=0)
                ax.plot(t, avg_mean_corr, label=f'{name} {rdm_name} (Avg)', color=color)

    # overlay base correlation if requested
    if base_corr_results == "object_eeg_rdm":
        mean_base, _, _ = compute_corrs_average(distance_metric, object_eeg_rdm)
        ax.plot(t, mean_base, label='Base object_eeg_rdm', color="black", linestyle='--')
    elif base_corr_results == "action_eeg_rdm":
        mean_base, _, _ = compute_corrs_average(distance_metric, action_eeg_rdm)
        ax.plot(t, mean_base, label='Base action_eeg_rdm', color="black", linestyle='--')
    elif base_corr_results == "GIST_265":
        mean_base, _, _ = compute_corrs_average(distance_metric, GIST_265)
        ax.plot(t, mean_base, label='Base GIST_265', color="black", linestyle='--')

    ax.set_ylim(-0.015, 0.06)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Partial Correlation')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.legend(bbox_to_anchor=(1.1, 1.05))

    plt.tight_layout()
    plt.savefig(savepath, transparent=True, dpi=300)
    plt.show()


# ---------------------------------------------------------------------
# Partial correlations (Variant 2): model → rdm_name → layer
# and AUC difference plots
# ---------------------------------------------------------------------

def load_or_compute_partial_corr_results_2(
    output_file: str,
    model_rdms: Dict[str, np.ndarray],
    dnn_rdms: Dict[str, Dict[str, np.ndarray]],
    distance_metric: str,
) -> Dict[str, Dict]:
    """
    Load cached partial_corr_results_2 if it exists, otherwise compute and save.

    Structure:
        partial_corr_results_2[model_name]['base_corr'][layer_name] = {...}
        partial_corr_results_2[model_name][rdm_name][layer_name] = {...}
    """
    if os.path.exists(output_file):
        print(f"Found cached partial_corr_results_2 at {output_file} — loading.")
        with open(output_file, "rb") as f:
            return pickle.load(f)

    print("No cached partial_corr_results_2 found — computing...")

    partial_corr_results_2: Dict[str, Dict] = {}

    for model_name, layers_dict in dnn_rdms.items():
        partial_corr_results_2[model_name] = {}

        for layer_name, layer_rdm in layers_dict.items():
            # 1) Base correlation
            mean_corr, sem_corr, base_corrs = compute_corrs_average(distance_metric, layer_rdm)
            base_dict = partial_corr_results_2[model_name].setdefault("base_corr", {})
            base_dict[layer_name] = {
                "mean_corr": mean_corr,
                "sem_corr": sem_corr,
                "corrs": base_corrs,
            }

            # 2) Partial correlations (rdm_name outer key)
            for rdm_name, model_rdm in model_rdms.items():
                rdm_dict = partial_corr_results_2[model_name].setdefault(rdm_name, {})
                mean_p, sem_p, partial_corrs = compute_partial_corrs_sliding(
                    distance_metric, "_5_", layer_rdm, model_rdm
                )
                rdm_dict[layer_name] = {
                    "mean_partial_corr": mean_p,
                    "sem_partial_corr": sem_p,
                    "partial_corrs": partial_corrs,
                }

    with open(output_file, "wb") as f:
        pickle.dump(partial_corr_results_2, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved partial_corr_results_2 to {output_file}")
    return partial_corr_results_2


def filter_and_rename_layers_rdm_first(
    partial_corr_results_2: Dict[str, Dict],
    reduced_layers: Dict[str, List[str]]
) -> Dict[str, Dict]:
    """
    Filters and renames layers to 'layer1', 'layer2', etc. in the
    partial_corr_results_2 dictionary with structure:
      model → rdm_name → layer_name
    """
    filtered_renamed_results: Dict[str, Dict] = {}

    for model_name, rdm_dict in partial_corr_results_2.items():
        if model_name not in reduced_layers:
            continue

        filtered_renamed_results[model_name] = {}
        selected_layers = reduced_layers[model_name]

        for rdm_name, layers_dict in rdm_dict.items():
            filtered_renamed_results[model_name][rdm_name] = {}
            for i, layer_name in enumerate(selected_layers, start=1):
                if layer_name in layers_dict:
                    filtered_renamed_results[model_name][rdm_name][f'layer{i}'] = layers_dict[layer_name]

    return filtered_renamed_results


def compute_auc_differences(
    models: List[str],
    rdm_space_list: List[str],
    filtered_and_renamed_results: Dict[str, Dict],
    base_corr_data: Dict[str, Dict[str, np.ndarray]]
) -> Dict[str, Dict[str, float]]:
    """
    Computes the AUC for the difference between the average mean partial
    correlation across layers and base correlations.

    Works with Variant 2 structure:
        filtered_and_renamed_results[model][rdm_space][layer]
    """
    auc_results: Dict[str, Dict[str, float]] = {}

    for model in models:
        auc_results[model] = {}

        for rdm_space in rdm_space_list:
            if (
                model in filtered_and_renamed_results
                and rdm_space in filtered_and_renamed_results[model]
                and rdm_space in base_corr_data
            ):
                layer_corrs = []
                for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
                    if (layer_name in filtered_and_renamed_results[model][rdm_space] and
                        'mean_partial_corr' in filtered_and_renamed_results[model][rdm_space][layer_name]):
                        mean_corr = filtered_and_renamed_results[model][rdm_space][layer_name]['mean_partial_corr']
                        layer_corrs.append(mean_corr)

                if layer_corrs:
                    average_layer_corr = np.mean(layer_corrs, axis=0)
                    mean_base_corr = base_corr_data[rdm_space]['mean_corr']
                    difference_curve = np.array(mean_base_corr) - np.array(average_layer_corr)
                    auc = np.trapz(difference_curve)
                    auc_results[model][rdm_space] = float(auc)

    return auc_results


def plot_average_partial_correlations_rdm_first(
    models,
    rdm_names,
    filtered_and_renamed_results: Dict[str, Dict],
    distance_metric: str,
    action_eeg_rdm: np.ndarray,
    object_eeg_rdm: np.ndarray,
    GIST_265: np.ndarray,
    savepath: str,
    base_corr_results: str = None,
    colors: List[str] = None,
    names: List[str] = None,
) -> None:
    """
    Plot average partial correlation (over reduced layers) for Variant 2
    structure:
        filtered_and_renamed_results[model][rdm_name][layer]
    """
    if isinstance(models, str):
        models = [models]
    if isinstance(rdm_names, str):
        rdm_names = [rdm_names]

    if colors is None or len(colors) < len(models):
        raise ValueError("colors must be at least as long as models")
    if names is None or len(names) < len(models):
        raise ValueError("names must be at least as long as models")

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.axvline(x=0, color='lightgray', linestyle='--')
    ax.axhline(y=0, color='lightgray', linestyle='--')

    for i, model in enumerate(models):
        color = colors[i]
        name = names[i]

        if model not in filtered_and_renamed_results:
            continue

        for rdm_name in rdm_names:
            if rdm_name not in filtered_and_renamed_results[model]:
                continue

            layer_corrs = []
            for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
                layer_dict = filtered_and_renamed_results[model][rdm_name]
                if layer_name in layer_dict and 'mean_partial_corr' in layer_dict[layer_name]:
                    layer_corrs.append(layer_dict[layer_name]['mean_partial_corr'])

            if layer_corrs:
                avg_mean_corr = np.mean(layer_corrs, axis=0)
                ax.plot(t, avg_mean_corr, label=f'{name} {rdm_name} (Avg)', color=color)

    # overlay base correlation if requested
    if base_corr_results == "object_eeg_rdm":
        mean_base, _, _ = compute_corrs_average(distance_metric, object_eeg_rdm)
        ax.plot(t, mean_base, label='Base object_eeg_rdm', color="black", linestyle='--')
    elif base_corr_results == "action_eeg_rdm":
        mean_base, _, _ = compute_corrs_average(distance_metric, action_eeg_rdm)
        ax.plot(t, mean_base, label='Base action_eeg_rdm', color="black", linestyle='--')
    elif base_corr_results == "GIST_265":
        mean_base, _, _ = compute_corrs_average(distance_metric, GIST_265)
        ax.plot(t, mean_base, label='Base GIST_265', color="black", linestyle='--')

    ax.set_ylim(-0.015, 0.06)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Partial Correlation')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(savepath, transparent=True, dpi=300)
    plt.show()


def plot_auc_differences(auc_results: Dict[str, Dict[str, float]],
                         colors: List[str],
                         names: List[str]) -> None:
    """
    Scatter + group-mean plot of AUC differences for
    action_eeg_rdm vs object_eeg_rdm vs GIST_265, including FDR-corrected tests.
    """
    # Extract data for each RDM space
    action_eeg_rdm_values = [v["action_eeg_rdm"] for v in auc_results.values()]
    object_eeg_rdm_values = [v["object_eeg_rdm"] for v in auc_results.values()]
    gist_values = [v["GIST_265"] for v in auc_results.values()]

    # Means
    mean_action_eeg = np.mean(action_eeg_rdm_values)
    mean_object_eeg = np.mean(object_eeg_rdm_values)
    mean_gist = np.mean(gist_values)

    # t-tests
    t_stat_action_vs_object, p_val_action_vs_object = stats.ttest_ind(
        action_eeg_rdm_values, object_eeg_rdm_values
    )
    t_stat_action_vs_gist, p_val_action_vs_gist = stats.ttest_ind(
        action_eeg_rdm_values, gist_values
    )
    t_stat_object_vs_gist, p_val_object_vs_gist = stats.ttest_ind(
        object_eeg_rdm_values, gist_values
    )
    print("Action vs GIST t, p:", t_stat_action_vs_gist, p_val_action_vs_gist)
    print("Object vs GIST t, p:", t_stat_object_vs_gist, p_val_object_vs_gist)

    _, corrected_p_vals, _, _ = multipletests(
        [p_val_action_vs_object, p_val_object_vs_gist, p_val_action_vs_gist],
        method="fdr_bh",
    )
    print("FDR-corrected p-values:", corrected_p_vals)

    # Plotting
    fig, ax = plt.subplots(figsize=(3, 5))

    x_positions_action = np.ones(len(action_eeg_rdm_values)) * 1
    x_positions_object = np.ones(len(object_eeg_rdm_values)) * 2
    x_positions_gist = np.ones(len(gist_values)) * 3

    for i in range(len(action_eeg_rdm_values)):
        jitter_a = np.random.normal(0, 0.05)
        jitter_o = np.random.normal(0, 0.05)
        jitter_g = np.random.normal(0, 0.05)

        ax.scatter(
            x_positions_action[i] + jitter_a,
            action_eeg_rdm_values[i],
            color=colors[i],
            alpha=0.7,
            label=names[i] if i == 0 else "",
        )
        ax.scatter(
            x_positions_object[i] + jitter_o,
            object_eeg_rdm_values[i],
            color=colors[i],
            alpha=0.7,
        )
        ax.scatter(
            x_positions_gist[i] + jitter_g,
            gist_values[i],
            color=colors[i],
            alpha=0.7,
        )

    # Horizontal mean lines
    ax.hlines(mean_action_eeg, 0.85, 1.15, colors="black", linestyles="dashed", linewidth=2)
    ax.hlines(mean_object_eeg, 1.85, 2.15, colors="black", linestyles="dashed", linewidth=2)
    ax.hlines(mean_gist, 2.85, 3.15, colors="black", linestyles="dashed", linewidth=2)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Affordance", "Object", "GIST"], fontsize=12)
    ax.set_ylabel("AUC of Difference from Base correlation", fontsize=12)

    # Significance bars
    y_max = max(
        [max(action_eeg_rdm_values), max(object_eeg_rdm_values), max(gist_values)]
    ) + 0.5

    # Action vs Object
    if corrected_p_vals[0] < 0.05:
        ax.plot([1, 1, 2, 2], [y_max, y_max + 0.05, y_max + 0.05, y_max], color="black", lw=1.5)
        ax.text(1.5, y_max + 0.05, "*", ha="center", va="bottom", color="black")

    # Object vs GIST
    if corrected_p_vals[1] < 0.05:
        ax.plot(
            [1, 1, 3, 3],
            [y_max + 0.1, y_max + 0.15, y_max + 0.15, y_max + 0.1],
            color="black",
            lw=1.5,
        )
        ax.text(2, y_max + 0.15, "*", ha="center", va="bottom", color="black")

    # Action vs GIST
    if corrected_p_vals[2] < 0.05:
        ax.plot([2, 2, 3, 3], [y_max, y_max + 0.05, y_max + 0.05, y_max], color="black", lw=1.5)
        ax.text(2.5, y_max + 0.03, "*", ha="center", va="bottom", color="black")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    # Example: save if you want
    # plt.savefig("/path/to/AUC_affordance_object_gist_colored.svg", transparent=True, dpi=300)
    plt.show()


def run_partial_corr_block_variant2(
    dnn_rdms: Dict[str, Dict[str, np.ndarray]],
    action_eeg_rdm: np.ndarray,
    object_eeg_rdm: np.ndarray,
    GIST_265: np.ndarray,
) -> None:
    """
    High-level convenience function for Variant 2 partial correlations
    and AUC difference plots.
    """
    # Model RDMs for partialling out
    model_rdms = {
        "action_eeg_rdm": action_eeg_rdm,
        "object_eeg_rdm": object_eeg_rdm,
        "GIST_265": GIST_265,
    }

    # 1) Load / compute
    partial_corr_results_2 = load_or_compute_partial_corr_results_2(
        output_file=PARTIAL_CORR2_SAVE_PATH,
        model_rdms=model_rdms,
        dnn_rdms=dnn_rdms,
        distance_metric=DISTANCE_METRIC,
    )

    # 2) Reduce layers & rename
    reduced_layers = reduce_layers_to_common(MODELS_LAYERS)
    filtered_and_renamed_results_2 = filter_and_rename_layers_rdm_first(
        partial_corr_results_2, reduced_layers
    )

    # 3) Plot average partial correlations for each RDM space
    plot_average_partial_correlations_rdm_first(
        MODEL_LIST,
        "action_eeg_rdm",
        filtered_and_renamed_results_2,
        DISTANCE_METRIC,
        action_eeg_rdm,
        object_eeg_rdm,
        GIST_265,
        os.path.join(FIGURES_DIR, 'Panel_C_average_Affordance_all_models.svg'),
        base_corr_results="action_eeg_rdm",
        colors=MODEL_COLORS,
        names=MODEL_SHORT_NAMES,
    )

    plot_average_partial_correlations_rdm_first(
        MODEL_LIST,
        "GIST_265",
        filtered_and_renamed_results_2,
        DISTANCE_METRIC,
        action_eeg_rdm,
        object_eeg_rdm,
        GIST_265,
        os.path.join(FIGURES_DIR , 'Panel_C_average_GIST_all_models.svg'),
        base_corr_results="GIST_265",
        colors=MODEL_COLORS,
        names=MODEL_SHORT_NAMES,
    )

    plot_average_partial_correlations_rdm_first(
        MODEL_LIST,
        "object_eeg_rdm",
        filtered_and_renamed_results_2,
        DISTANCE_METRIC,
        action_eeg_rdm,
        object_eeg_rdm,
        GIST_265,
        os.path.join(FIGURES_DIR , 'Panel_C_average_Object_all_models.svg'),
        base_corr_results="object_eeg_rdm",
        colors=MODEL_COLORS,
        names=MODEL_SHORT_NAMES,
    )

    # 4) Base correlations for AUC differences
    mean_action, _, _ = compute_corrs_average(DISTANCE_METRIC, action_eeg_rdm)
    mean_object, _, _ = compute_corrs_average(DISTANCE_METRIC, object_eeg_rdm)
    mean_gist, _, _ = compute_corrs_average(DISTANCE_METRIC, GIST_265)

    base_corr_data = {
        'object_eeg_rdm': {'mean_corr': mean_object},
        'action_eeg_rdm': {'mean_corr': mean_action},
        'GIST_265': {'mean_corr': mean_gist},
    }

    # 5) AUC differences and plot
    rdm_space_list = ["action_eeg_rdm", "object_eeg_rdm", "GIST_265"]
    auc_results = compute_auc_differences(
        MODEL_LIST, rdm_space_list, filtered_and_renamed_results_2, base_corr_data
    )
    plot_auc_differences(auc_results, MODEL_COLORS, MODEL_SHORT_NAMES)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    # 1) Load behavioral & GIST RDMs
    action_eeg_rdm = np.load(
        "/home/clemens-uva/Github_repos/EEG/DATA/Behavioral_annotations/RDMs/action_average_RDM_euclidean.npy"
    )
    object_eeg_rdm = np.load(
        "/home/clemens-uva/Github_repos/EEG/DATA/Behavioral_annotations/RDMs/object_average_RDM_euclidean.npy"
    )
    GIST_265 = load_and_sort_rdm(
        "/home/clemens-uva/Github_repos/Visact_fMRI/fMRI_folder/VISACT_RDM_collection/GIST/VISACT_fMRI/GIST_256_RDM_fMRI.npy",
        images_name
    )

    # 2) Load DNN RDMs (once)
    dnn_rdms = load_dnn_rdms()

    # 3) Load or compute model correlation data
    correlation_data = load_or_compute_correlation_data(CORR_DATA_SAVE_PATH, dnn_rdms)

    # 4) RSA with behavioral & GIST RDMs
    mean_action, sem_action, action_array = compute_corrs_average(DISTANCE_METRIC, action_eeg_rdm)
    mean_object, sem_object, object_array = compute_corrs_average(DISTANCE_METRIC, object_eeg_rdm)
    mean_gist, sem_gist, gist_array = compute_corrs_average(DISTANCE_METRIC, GIST_265)

    action_vals = t[np.argmax(action_array, axis=1)]
    object_vals = t[np.argmax(object_array, axis=1)]
    gist_vals = t[np.argmax(gist_array, axis=1)]

    # 5) Model-wise statistics vs behavioral / GIST
    all_values_across_models = []
    means = []
    models = []
    p_values_action = []
    p_values_object = []
    p_values_gist = []

    for model_name, df in correlation_data.items():
        all_values = []
        for row in df['all_highest_values']:
            all_values.extend(row)

        models.append(model_name)
        means.append(np.mean(all_values))
        all_values_across_models.append(all_values)

        t_stat, p_val = stats.ttest_ind(all_values, action_vals, equal_var=False)
        p_values_action.append(p_val)
        t_stat, p_val = stats.ttest_ind(all_values, object_vals, equal_var=False)
        p_values_object.append(p_val)
        t_stat, p_val = stats.ttest_ind(all_values, gist_vals, equal_var=False)
        p_values_gist.append(p_val)

    rejected_action, _ = multipletests(p_values_action, alpha=0.05, method='fdr_bh')[:2]
    rejected_object, _ = multipletests(p_values_object, alpha=0.05, method='fdr_bh')[:2]
    rejected_gist, _ = multipletests(p_values_gist, alpha=0.05, method='fdr_bh')[:2]

    # 6) Figures: highest timepoints + groups + slopes + max corr
    plot_highest_timepoints(
        correlation_data=correlation_data,
        action_array=action_array,
        object_array=object_array,
        gist_array=gist_array,
        rejected_action=rejected_action,
        rejected_object=rejected_object,
        rejected_gist=rejected_gist,
        means=means,
        all_values_across_models=all_values_across_models,
        models=models,
        out_path=os.path.join(FIGURES_DIR , 'Panel_B_highest_timepoints_DNNs.svg'),
    )

    plot_group_t_highest(
        correlation_data=correlation_data,
        action_array=action_array,
        use_sem=False,
        out_path=os.path.join(FIGURES_DIR , 'Supplementary_highest_timepoint_normalized_model_depth.svg')
    )
    plot_group_t_highest(
        correlation_data=correlation_data,
        action_array=action_array,
        use_sem=True,
        out_path=os.path.join(FIGURES_DIR , 'Supplementary_highest_timepoint_normalized_model_depth_SEM.svg')
    )

    slopes, avg_slope = plot_slopes_by_model(correlation_data)
    plot_positive_negative_slopes(
        slopes,
        out_path=os.path.join(FIGURES_DIR , 'Supplementary_Slopes.svg')
    )

    plot_group_highest_corr(
        correlation_data=correlation_data,
        mean_action=mean_action,
        use_sem=False,
        out_path=os.path.join(FIGURES_DIR , 'Supplementary_highest_correlation_normalized_model_depth.svg')
    )
    plot_group_highest_corr(
        correlation_data=correlation_data,
        mean_action=mean_action,
        use_sem=True,
        out_path=os.path.join(FIGURES_DIR , 'Supplementary_highest_correlation_normalized_model_depth_SEM.svg')
    )

    # 7) Partial correlations Variant 1: load/compute + layer reduction
    model_rdms = {
        "action_eeg_rdm": action_eeg_rdm,
        "object_eeg_rdm": object_eeg_rdm,
        "GIST_265": GIST_265,
    }

    partial_corr_results = load_or_compute_partial_corr_results(
        PARTIAL_CORR_SAVE_PATH,
        model_rdms=model_rdms,
        dnn_rdms=dnn_rdms,
        distance_metric=DISTANCE_METRIC,
    )

    reduced_layers = reduce_layers_to_common(MODELS_LAYERS)
    filtered_results = filter_and_rename_layers(partial_corr_results, reduced_layers)

    # 8) Average base correlations across layers (all models)
    plot_averaged_correlations_per_model(
        MODEL_LIST, MODEL_COLORS, MODEL_SHORT_NAMES,
        filtered_results,
        savepath=os.path.join(FIGURES_DIR, 'Additional_Panel_A_average_DNN_corrs.svg'),
    )

    plot_averaged_correlations_per_model_with_sem(
        MODEL_LIST, MODEL_COLORS, MODEL_SHORT_NAMES,
        filtered_results,
        savepath=os.path.join(FIGURES_DIR, 'Panel_A_average_DNN_corrs_SEM.svg'),
    )

    # 10) Partial correlations Variant 2 + AUC difference plots
    run_partial_corr_block_variant2(
        dnn_rdms=dnn_rdms,
        action_eeg_rdm=action_eeg_rdm,
        object_eeg_rdm=object_eeg_rdm,
        GIST_265=GIST_265,
    )


if __name__ == "__main__":
    main()
