import numpy as np
import os
import mne
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests
from sklearn import linear_model
# for RSA toolbox
import rsatoolbox
from rsatoolbox.rdm import calc_rdm_movie

FIGURE_DIR = "/home/clemens-uva/Desktop/EEG_Temporal_misalignment/02_Spatiotemporalfusion/Figures"
OSF_DATA_DIR = "/home/clemens-uva/Desktop/DATA_OSF_external_storage"
PREPROCESSED_DATA_DIR = os.path.join(OSF_DATA_DIR, "pre_processed_data")
AVERAGE_RDM_DIR = os.path.join(OSF_DATA_DIR, "average_RDMs")
BEHAVIOR_RDM_DIR = os.path.join(OSF_DATA_DIR, "Behavioral_annotations")
MODEL_RDM_DIR = os.path.join(OSF_DATA_DIR, "Model_RDMs")
GIST_RDM_PATH = os.path.join(OSF_DATA_DIR, "GIST", "GIST_256_RDM_fMRI.npy")
ROI_RDM_DIR = os.path.join(OSF_DATA_DIR, "ROI_RDMs")
RDM_OUTPUT_DIR = os.path.join(OSF_DATA_DIR, "ERP_sliding_window_RDMs")
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(RDM_OUTPUT_DIR, exist_ok=True)

dirname = FIGURE_DIR

EEG_list_action_sorted = ['outdoor_manmade_0147',  'outdoor_manmade_0148',  'outdoor_natural_0246',
 'outdoor_natural_0062',   'outdoor_natural_0160',   'outdoor_natural_0255',
 'outdoor_natural_0128',   'indoor_0156',   'outdoor_manmade_0173',
 'outdoor_manmade_0089',   'outdoor_natural_0104', 'outdoor_natural_0273',
 'outdoor_natural_0079',  'outdoor_manmade_0175',  'outdoor_natural_0042',
 'outdoor_natural_0198',  'outdoor_manmade_0131',  'outdoor_natural_0091',
 'outdoor_manmade_0152',  'outdoor_natural_0200',  'outdoor_manmade_0157',
 'outdoor_manmade_0155',  'indoor_0282',  'outdoor_manmade_0256',
 'outdoor_manmade_0257',  'outdoor_natural_0011',  'indoor_0066',
 'outdoor_manmade_0119',  'outdoor_manmade_0220',  'outdoor_manmade_0068',
 'outdoor_manmade_0133',  'outdoor_manmade_0258',  'outdoor_manmade_0040',
 'outdoor_natural_0132',  'outdoor_manmade_0064',  'outdoor_manmade_0032',
 'outdoor_manmade_0063',  'outdoor_manmade_0015',  'outdoor_manmade_0110',
 'outdoor_manmade_0167',  'outdoor_manmade_0117',  'outdoor_manmade_0030',
 'outdoor_natural_0207',  'outdoor_natural_0053',  'outdoor_natural_0261',
 'outdoor_natural_0097',  'outdoor_natural_0004',  'outdoor_manmade_0149',
 'outdoor_natural_0034',  'outdoor_manmade_0161',  'indoor_0033',
 'indoor_0163',  'indoor_0235',  'indoor_0100',
 'indoor_0058',  'indoor_0145',  'indoor_0271',
 'indoor_0266',  'indoor_0130',  'outdoor_manmade_0276',
 'indoor_0025',  'indoor_0021',  'outdoor_manmade_0165',
 'indoor_0283',  'indoor_0136',  'indoor_0249',
 'indoor_0279',  'indoor_0215',  'indoor_0221',
 'indoor_0216',  'indoor_0214',  'indoor_0080',
 'indoor_0103',  'indoor_0146',  'indoor_0055',
 'indoor_0212',  'indoor_0281',  'outdoor_manmade_0154',
 'indoor_0270',  'outdoor_natural_0049',  'outdoor_natural_0009',
 'outdoor_natural_0010',  'indoor_0272',  'outdoor_natural_0008',
 'outdoor_natural_0052',  'outdoor_natural_0023',  'outdoor_natural_0250',
 'outdoor_natural_0050',  'outdoor_natural_0017',  'outdoor_natural_0252']

fMRI_stim_ordering = ['indoor_0021', 'indoor_0025', 'indoor_0033', 'indoor_0055',
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
       'outdoor_natural_0261', 'outdoor_natural_0273']

# remove "_" from each string in the list
fmri_ordering = [x.replace("_", "") for x in fMRI_stim_ordering]

# remove "_" from each string in the list
images_name = [x.replace("_", "") for x in EEG_list_action_sorted]

def load_and_sort_rdm_neuro(rdm_path, ordering):
    """
    Load an RDM from a file, convert it to a DataFrame, and sort it according to the given ordering.
    
    Parameters:
    rdm_path (str): Path to the RDM file.
    ordering (list): List of image names in the desired order.
    
    Returns:
    sorted_rdm (np.array): Sorted RDM.
    """
    fMRI_stim_ordering = ['indoor_0021', 'indoor_0025', 'indoor_0033', 'indoor_0055',
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
       'outdoor_natural_0261', 'outdoor_natural_0273']
    
    fmri_ordering = [x.replace("_", "") for x in fMRI_stim_ordering]

    rdm = np.load(rdm_path)["arr_0"]
    if rdm.shape[0] != 90:
        rdm = np.mean(rdm, axis=0)
        
    rdm_df = pd.DataFrame(rdm)
    rdm_df.index = fmri_ordering
    rdm_df.columns = fmri_ordering
    sorted_rdm = rdm_df.loc[ordering, ordering].values
    return sorted_rdm

def load_and_sort_rdm(rdm_path, ordering):
    """
    Load an RDM from a file, convert it to a DataFrame, and sort it according to the given ordering.
    
    Parameters:
    rdm_path (str): Path to the RDM file.
    ordering (list): List of image names in the desired order.
    
    Returns:
    sorted_rdm (np.array): Sorted RDM.
    """
    fMRI_stim_ordering = ['indoor_0021', 'indoor_0025', 'indoor_0033', 'indoor_0055',
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
       'outdoor_natural_0261', 'outdoor_natural_0273']
    
    fmri_ordering = [x.replace("_", "") for x in fMRI_stim_ordering]

    rdm = np.load(rdm_path)
    if rdm.shape[0] != 90:
        rdm = np.mean(rdm, axis=0)
        
    rdm_df = pd.DataFrame(rdm)
    rdm_df.index = fmri_ordering
    rdm_df.columns = fmri_ordering
    sorted_rdm = rdm_df.loc[ordering, ordering].values
    return sorted_rdm

#event dictionary
event_dict = { 'indoor0156' : 4033, 'indoor0282' : 3852, 'indoor0270' : 4064, 'indoor0272' : 4007, 'indoor0066' : 4023, 'indoor0283' : 3898, 'indoor0214' : 3953, 'indoor0080' : 4055, 'indoor0215' : 3964, 'indoor0216': 3931, 'indoor0146' : 4074, 'indoor0221' : 4045, 'indoor0235': 4071, 'indoor0212' : 3960, 'indoor0058' : 4047, 'indoor0145' : 3989, 'indoor0136' : 4018, 'indoor0130' : 4088, 'indoor0163' : 3894, 'indoor0103': 4017,'indoor0100' : 3842, 'indoor0055' : 3858, 'indoor0021' : 3888, 'indoor0266': 3853, 'indoor0025' : 4062, 'indoor0279' : 4027, 'indoor0281' : 3873, 'indoor0271' : 4014, 'indoor0249' : 4002, 'indoor0033' : 4085, 'outdoornatural0010' : 4020, 'outdoornatural0009' : 3981, 'outdoornatural0049' : 3942, 'outdoornatural0008' : 3903, 'outdoornatural0052' : 4076, 'outdoornatural0050' : 4072, 'outdoornatural0132' : 3914, 'outdoornatural0053' : 3930, 'outdoornatural0004' : 3984, 'outdoornatural0207' : 3997, 'outdoornatural0097' : 4003, 'outdoornatural0261' : 4056, 'outdoornatural0011' : 4075, 'outdoornatural0198' : 4063, 'outdoornatural0128' : 3971, 'outdoornatural0255' : 3955, 'outdoornatural0062' : 3925, 'outdoornatural0246' : 3994, 'outdoornatural0160' : 3940, 'outdoornatural0091' : 4030, 'outdoornatural0104' : 4000, 'outdoornatural0200' : 3902, 'outdoornatural0273' : 4043, 'outdoornatural0079' : 3944, 'outdoornatural0042' : 3986, 'outdoornatural0034' : 4061, 'outdoornatural0017' : 3950, 'outdoornatural0023' : 3859, 'outdoornatural0252' : 3870, 'outdoornatural0250' : 3884, 'outdoormanmade0167' : 4059, 'outdoormanmade0040' : 3851, 'outdoormanmade0110' : 3841, 'outdoormanmade0117' : 4077, 'outdoormanmade0030': 3891, 'outdoormanmade0258' : 4081, 'outdoormanmade0064' : 3926, 'outdoormanmade0068' : 4038, 'outdoormanmade0063' : 3845, 'outdoormanmade0015' : 3871, 'outdoormanmade0257': 4078, 'outdoormanmade0032' : 3878, 'outdoormanmade0256': 3918, 'outdoormanmade0220' : 4052, 'outdoormanmade0133' : 4013, 'outdoormanmade0119' : 3886, 'outdoormanmade0152' : 4001, 'outdoormanmade0148' : 4083, 'outdoormanmade0155' : 3899, 'outdoormanmade0157' : 3843, 'outdoormanmade0175' : 4048, 'outdoormanmade0173': 3907, 'outdoormanmade0089' : 3862, 'outdoormanmade0147': 4060, 'outdoormanmade0131' : 3874, 'outdoormanmade0161' : 3869, 'outdoormanmade0154' : 4041, 'outdoormanmade0165' : 3854, 'outdoormanmade0276': 3976, 'outdoormanmade0149' : 3866}
#images_name = [ 'indoor0156' , 'indoor0282', 'indoor0270' , 'indoor0272', 'indoor0066', 'indoor0283', 'indoor0214', 'indoor0080' , 'indoor0215', 'indoor0216' , 'indoor0146', 'indoor0221' , 'indoor0235', 'indoor0212' , 'indoor0058' , 'indoor0145', 'indoor0136', 'indoor0130' , 'indoor0163', 'indoor0103','indoor0100' , 'indoor0055', 'indoor0021', 'indoor0266', 'indoor0025', 'indoor0279' , 'indoor0281', 'indoor0271', 'indoor0249' , 'indoor0033', 'outdoornatural0010', 'outdoornatural0009', 'outdoornatural0049' , 'outdoornatural0008' , 'outdoornatural0052' , 'outdoornatural0050' , 'outdoornatural0132' , 'outdoornatural0053' , 'outdoornatural0004' , 'outdoornatural0207', 'outdoornatural0097', 'outdoornatural0261', 'outdoornatural0011' , 'outdoornatural0198' , 'outdoornatural0128' , 'outdoornatural0255', 'outdoornatural0062' , 'outdoornatural0246' , 'outdoornatural0160', 'outdoornatural0091' , 'outdoornatural0104' , 'outdoornatural0200' , 'outdoornatural0273' , 'outdoornatural0079', 'outdoornatural0042' , 'outdoornatural0034' , 'outdoornatural0017', 'outdoornatural0023' , 'outdoornatural0252', 'outdoornatural0250' , 'outdoormanmade0167' , 'outdoormanmade0040' , 'outdoormanmade0110' , 'outdoormanmade0117' , 'outdoormanmade0030', 'outdoormanmade0258' , 'outdoormanmade0064' , 'outdoormanmade0068' , 'outdoormanmade0063', 'outdoormanmade0015' , 'outdoormanmade0257', 'outdoormanmade0032' , 'outdoormanmade0256' , 'outdoormanmade0220'  , 'outdoormanmade0133' , 'outdoormanmade0119' , 'outdoormanmade0152' , 'outdoormanmade0148' , 'outdoormanmade0155', 'outdoormanmade0157', 'outdoormanmade0175' , 'outdoormanmade0173', 'outdoormanmade0089' , 'outdoormanmade0147', 'outdoormanmade0131', 'outdoormanmade0161', 'outdoormanmade0154' , 'outdoormanmade0165' , 'outdoormanmade0276', 'outdoormanmade0149' ]
#event_dict = { 'indoor_0156' : 4033, 'indoor_0282' : 3852, 'indoor_0270' : 4064, 'indoor_0272' : 4007, 'indoor_0066' : 4023, 'indoor_0283' : 3898, 'indoor_0214' : 3953, 'indoor_0080' : 4055, 'indoor_0215' : 3964, 'indoor_0216': 3931, 'indoor_0146' : 4074, 'indoor_0221' : 4045, 'indoor_0235': 4071, 'indoor_0212' : 3960, 'indoor_0058' : 4047, 'indoor_0145' : 3989, 'indoor_0136' : 4018, 'indoor_0130' : 4088, 'indoor_0163' : 3894, 'indoor_0103': 4017,'indoor_0100' : 3842, 'indoor_0055' : 3858, 'indoor_0021' : 3888, 'indoor_0266': 3853, 'indoor_0025' : 4062, 'indoor_0279' : 4027, 'indoor_0281' : 3873, 'indoor_0271' : 4014, 'indoor_0249' : 4002, 'indoor_0033' : 4085, 'outdoor_natural_0010' : 4020, 'outdoor_natural_0009' : 3981, 'outdoor_natural_0049' : 3942, 'outdoor_natural_0008' : 3903, 'outdoor_natural_0052' : 4076, 'outdoor_natural_0050' : 4072, 'outdoor_natural_0132' : 3914, 'outdoor_natural_0053' : 3930, 'outdoor_natural_0004' : 3984, 'outdoor_natural_0207' : 3997, 'outdoor_natural_0097' : 4003, 'outdoor_natural_0261' : 4056, 'outdoor_natural_0011' : 4075, 'outdoor_natural_0198' : 4063, 'outdoor_natural_0128' : 3971, 'outdoor_natural_0255' : 3955, 'outdoor_natural_0062' : 3925, 'outdoor_natural_0246' : 3994, 'outdoor_natural_0160' : 3940, 'outdoor_natural_0091' : 4030, 'outdoor_natural_0104' : 4000, 'outdoor_natural_0200' : 3902, 'outdoor_natural_0273' : 4043, 'outdoor_natural_0079' : 3944, 'outdoor_natural_0042' : 3986, 'outdoor_natural_0034' : 4061, 'outdoor_natural_0017' : 3950, 'outdoor_natural_0023' : 3859, 'outdoor_natural_0252' : 3870, 'outdoor_natural_0250' : 3884, 'outdoor_manmade_0167' : 4059, 'outdoor_manmade_0040' : 3851, 'outdoor_manmade_0110' : 3841, 'outdoor_manmade_0117' : 4077, 'outdoor_manmade_0030': 3891, 'outdoor_manmade_0258' : 4081, 'outdoor_manmade_0064' : 3926, 'outdoor_manmade_0068' : 4038, 'outdoor_manmade_0063' : 3845, 'outdoor_manmade_0015' : 3871, 'outdoor_manmade_0257': 4078, 'outdoor_manmade_0032' : 3878, 'outdoor_manmade_0256': 3918, 'outdoor_manmade_0220' : 4052, 'outdoor_manmade_0133' : 4013, 'outdoor_manmade_0119' : 3886, 'outdoor_manmade_0152' : 4001, 'outdoor_manmade_0148' : 4083, 'outdoor_manmade_0155' : 3899, 'outdoor_manmade_0157' : 3843, 'outdoor_manmade_0175' : 4048, 'outdoor_manmade_0173': 3907, 'outdoor_manmade_0089' : 3862, 'outdoor_manmade_0147': 4060, 'outdoor_manmade_0131' : 3874, 'outdoor_manmade_0161' : 3869, 'outdoor_manmade_0154' : 4041, 'outdoor_manmade_0165' : 3854, 'outdoor_manmade_0276': 3976, 'outdoor_manmade_0149' : 3866}


event_dict = { 'indoor0156' : 4033, 'indoor0282' : 3852, 'indoor0270' : 4064, 'indoor0272' : 4007, 'indoor0066' : 4023, 'indoor0283' : 3898, 'indoor0214' : 3953, 'indoor0080' : 4055, 'indoor0215' : 3964, 'indoor0216': 3931, 'indoor0146' : 4074, 'indoor0221' : 4045, 'indoor0235': 4071, 'indoor0212' : 3960, 'indoor0058' : 4047, 'indoor0145' : 3989, 'indoor0136' : 4018, 'indoor0130' : 4088, 'indoor0163' : 3894, 'indoor0103': 4017,'indoor0100' : 3842, 'indoor0055' : 3858, 'indoor0021' : 3888, 'indoor0266': 3853, 'indoor0025' : 4062, 'indoor0279' : 4027, 'indoor0281' : 3873, 'indoor0271' : 4014, 'indoor0249' : 4002, 'indoor0033' : 4085, 'outdoornatural0010' : 4020, 'outdoornatural0009' : 3981, 'outdoornatural0049' : 3942, 'outdoornatural0008' : 3903, 'outdoornatural0052' : 4076, 'outdoornatural0050' : 4072, 'outdoornatural0132' : 3914, 'outdoornatural0053' : 3930, 'outdoornatural0004' : 3984, 'outdoornatural0207' : 3997, 'outdoornatural0097' : 4003, 'outdoornatural0261' : 4056, 'outdoornatural0011' : 4075, 'outdoornatural0198' : 4063, 'outdoornatural0128' : 3971, 'outdoornatural0255' : 3955, 'outdoornatural0062' : 3925, 'outdoornatural0246' : 3994, 'outdoornatural0160' : 3940, 'outdoornatural0091' : 4030, 'outdoornatural0104' : 4000, 'outdoornatural0200' : 3902, 'outdoornatural0273' : 4043, 'outdoornatural0079' : 3944, 'outdoornatural0042' : 3986, 'outdoornatural0034' : 4061, 'outdoornatural0017' : 3950, 'outdoornatural0023' : 3859, 'outdoornatural0252' : 3870, 'outdoornatural0250' : 3884, 'outdoormanmade0167' : 4059, 'outdoormanmade0040' : 3851, 'outdoormanmade0110' : 3841, 'outdoormanmade0117' : 4077, 'outdoormanmade0030': 3891, 'outdoormanmade0258' : 4081, 'outdoormanmade0064' : 3926, 'outdoormanmade0068' : 4038, 'outdoormanmade0063' : 3845, 'outdoormanmade0015' : 3871, 'outdoormanmade0257': 4078, 'outdoormanmade0032' : 3878, 'outdoormanmade0256': 3918, 'outdoormanmade0220' : 4052, 'outdoormanmade0133' : 4013, 'outdoormanmade0119' : 3886, 'outdoormanmade0152' : 4001, 'outdoormanmade0148' : 4083, 'outdoormanmade0155' : 3899, 'outdoormanmade0157' : 3843, 'outdoormanmade0175' : 4048, 'outdoormanmade0173': 3907, 'outdoormanmade0089' : 3862, 'outdoormanmade0147': 4060, 'outdoormanmade0131' : 3874, 'outdoormanmade0161' : 3869, 'outdoormanmade0154' : 4041, 'outdoormanmade0165' : 3854, 'outdoormanmade0276': 3976, 'outdoormanmade0149' : 3866}
images_name = [x.replace("_", "") for x in EEG_list_action_sorted]


participants_list = ['sapaj', 'ppnjn', 'azrfp', 'cuvfl', 'domdz', 'npcrj', 'hoxev','kuupm',
                    'rxsrg', 'pflzs', 'kktpp', 'pyyor', 'liirj', 'qmrlx', 'jpdoy', 'hapql', 'ghldo', 'fgljq'] # pwixa


#####
# Path to preprocessed data 
DATA_path = PREPROCESSED_DATA_DIR + "/"

file_substring = sorted(os.listdir(DATA_path))[0]

# settings
tmin                = -0.1
tmax                = 1.0
down_sample_rate    = 128 # might need to be adapted depending on the preprocessing

# Calculate the duration
duration = tmax - tmin

# Calculate the number of time points
n_timepoints = int(duration * down_sample_rate) + 1  # +1 to include both endpoints

# Generate downsampled time points
t = np.linspace(tmin, tmax, n_timepoints)

# channel info
n_channels      = 64

# Epochs
n_epochs        = 540


# sets of electrodes
occipital_electrodes = ['P1',  'P3',  'P5',  'P7',  'P9',  'PO7',  'PO3',  'O1', 'Oz',
 'POz',  'Pz', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2']
frontal_electrodes = ['Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F3', 'Fz', 'F4', 'F8']

all_electrodes = ['Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F3', 'Fz', 'F4', 'F8', 'P1',  'P3',  'P5',  'P7',  'P9',  'PO7',  'PO3',  'O1', 'Oz',
 'POz',  'Pz', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2']

def corr_with_model(rdm1, model_rdm):
    corrs = []
    for timepoint in range(rdm1.shape[0]):
        rdv1 = squareform(rdm1[timepoint].round(10))
        rdv2 = squareform(model_rdm.round(10))
        corr, p = spearmanr(rdv1, rdv2)
        corrs.append(corr)

    mean = np.mean(corrs)

    return mean, corrs


def compute_corrs_sliding(distance_metric, n, model_rdm):
    path = RDM_OUTPUT_DIR

    all_sub_corrs = []
    for file in os.listdir(path):
        if (distance_metric in file) and (n in file):
            rdms_per_subject = np.load(os.path.join(path, file))
            mean_corr, corrs = corr_with_model(rdms_per_subject, model_rdm)
            all_sub_corrs.append(corrs)
    
    mean_corr = np.mean(np.array(all_sub_corrs), axis = 0)
    sem = np.std(all_sub_corrs, axis=0) / np.sqrt(len(all_sub_corrs))

    return mean_corr, sem, np.array(all_sub_corrs)

'''
def significant_against_zero(array):
    
    t_values, p_values = ttest_1samp(array, 0, axis=0)
    # Adjust p-values for FDR using Benjamini-Hochberg procedure
    alpha = 0.05
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
    # Output the results
    significant_timepoints = np.where(reject)[0]


    return significant_timepoints

    '''
def significant_against_zero(array):
    t_values, p_values = ttest_1samp(array, 0, axis=0)
    
    # Adjust p-values for FDR using Benjamini-Hochberg procedure
    alpha = 0.05
    reject, pvals_corrected, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
    
    # Get indices of significant time points
    significant_timepoints = np.where(reject)[0]

    # Extract t-values and p-values for significant time points
    significant_t_values = t_values[significant_timepoints]
    significant_p_values = p_values[significant_timepoints]

    # Print results only for significant time points
    for i, idx in enumerate(significant_timepoints):
        print(f"Timepoint {idx}: t = {significant_t_values[i]:.2f}, p = {significant_p_values[i]:.3f}")

    return significant_timepoints

def lowest_value(array1, array2):
    """
    Finds the lowest value between two arrays and returns it.

    Parameters:
    - array1: First numpy array.
    - array2: Second numpy array.

    Returns:
    - The lowest value found in both arrays.
    """

    # Find the minimum of each array
    min1 = np.min(array1)
    min2 = np.min(array2)
    
    # Return the minimum of both values
    return min(min1, min2)

def mean_and_sem(list):
    mean = np.mean(np.array(list), axis = 0)
    sem = np.std(list, axis=0) / np.sqrt(len(list))

    return mean, sem

PPA = load_and_sort_rdm_neuro(os.path.join(ROI_RDM_DIR, "fmri_PPA_mean.npz"), images_name)
OPA = load_and_sort_rdm_neuro(os.path.join(ROI_RDM_DIR, "fmri_OPA_mean.npz"), images_name)
RSC = load_and_sort_rdm_neuro(os.path.join(ROI_RDM_DIR, "fmri_RSC_mean.npz"), images_name)


action_eeg_rdm = np.load(os.path.join(BEHAVIOR_RDM_DIR, "action_average_RDM_euclidean.npy"))
object_eeg_rdm = np.load(os.path.join(BEHAVIOR_RDM_DIR, "object_average_RDM_euclidean.npy"))

GIST_265 = load_and_sort_rdm(GIST_RDM_PATH, images_name)

action_eeg_rdm.shape


#task = "action"
distance_metric = "correlation"

color1 = "#bfd200"
color2 = "#348aa7"
color3 = "#5dd39e"


line_1 = "-"
line_2 = "-"
line_3 = "-"
alpha_line = 1
alpha_shades = 0.1
lw = 2

# create Figure
fig, ax = plt.subplots()

ax.axvline(x=0, color='lightgray', linestyle='--')
ax.axhline(y=0, color='lightgray', linestyle='--')


mean_PPA, sem_PPA, PPA_array = compute_corrs_sliding(distance_metric, "_5_", PPA)
mean_OPA, sem_OPA, OPA_array = compute_corrs_sliding(distance_metric, "_5_", OPA)
mean_RSC, sem_RSC, RSC_array = compute_corrs_sliding(distance_metric, "_5_", RSC)



plt.plot(t, mean_PPA, color = color1, label="PPA", alpha = alpha_line, lw  = lw)
plt.fill_between(t, mean_PPA - sem_PPA, mean_PPA + sem_PPA, alpha=alpha_shades, color = color1)

plt.plot(t, mean_OPA, color = color2, label="OPA", alpha = alpha_line, lw = lw)
plt.fill_between(t, mean_OPA - sem_OPA, mean_OPA + sem_OPA, alpha=alpha_shades, color = color2)

plt.plot(t, mean_RSC, color = color3, label="MPA", alpha = alpha_line, lw = lw)
plt.fill_between(t, mean_RSC - sem_RSC, mean_RSC + sem_RSC, alpha=alpha_shades, color = color3)


min_value = -0.002
# test if correlation is different from zero
PPA_sig_timepoints = significant_against_zero(PPA_array)
for timepoint in PPA_sig_timepoints:
    plt.text(t[timepoint], min_value - 0.005, color = color1, s =".", fontsize=20)

OPA_sig_timepoints = significant_against_zero(OPA_array)
for timepoint in OPA_sig_timepoints:
    plt.text(t[timepoint], min_value - 0.001, color = color2, s =".", fontsize=20)

RSC_sig_timepoints = significant_against_zero(RSC_array)
for timepoint in RSC_sig_timepoints:
    plt.text(t[timepoint], min_value - 0.003, color = color3, s =".", fontsize=20)

# pairwise comparison

#pairwise = action_array - object_array
#pairwise_sig_timepoints = significant_against_zero(pairwise)
#for timepoint in pairwise_sig_timepoints:
    #plt.text(t[timepoint], min_value - 0.004, color = "black", s =".", fontsize=15)

# plot peak timepoints
ax.axvline(x= t[np.argmax(mean_PPA)], color=color1, linestyle='--')
ax.axvline(x= t[np.argmax(mean_OPA)], color=color2, linestyle='--')
ax.axvline(x= t[np.argmax(mean_RSC)], color=color3, linestyle='--')



plt.ylim(-0.015, 0.125)
plt.xlabel('Time (s)')
plt.ylabel('Spearman Correlation', fontsize = 15)
plt.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig(os.path.join(dirname, 'Panel_A_EEG_brain_correlation.svg'), transparent = True, dpi = 300)
plt.savefig(os.path.join(dirname, 'Panel_A_EEG_brain_correlation.png'), transparent = True, dpi = 300)
plt.show()

print("PPA:" + str(t[np.argmax(mean_PPA)]))
print("OPA:" + str(t[np.argmax(mean_OPA)]))
print("MPA:" + str(t[np.argmax(mean_RSC)]))

highest_value_action =  t[np.argmax(PPA_array, axis=1)]
highest_value_object = t[np.argmax(OPA_array, axis=1)]

from scipy.stats import ttest_ind, wilcoxon
t_stat, p_value_ttest = ttest_ind(highest_value_action, highest_value_object)
print(f"Paired t-test: t-statistic = {t_stat}, p-value = {p_value_ttest}")
# Check if p-value from t-test is below 0.05
if p_value_ttest < 0.05:
    print("Paired t-test: The difference is significant (p < 0.05).")
else:
    print("Paired t-test: The difference is not significant (p >= 0.05).")

# Perform Wilcoxon signed-rank test
stat, p_value_wilcoxon = wilcoxon(highest_value_action, highest_value_object)
print(f"Wilcoxon signed-rank test: statistic = {stat}, p-value = {p_value_wilcoxon}")
if p_value_wilcoxon < 0.05:
    print("Wilcoxon signed-rank test: The difference is significant (p < 0.05).")
else:
    print("Wilcoxon signed-rank test: The difference is not significant (p >= 0.05).")

# create a function for variance partitioning

def VarPart_3(Y, X1, X2, X3):

    # initalize model
    lm = linear_model.LinearRegression()

    # Calculate R-squared values for each independent variable
    R1 = lm.fit(X1.reshape(-1, 1), Y).score(X1.reshape(-1, 1), Y)
    R2 = lm.fit(X2.reshape(-1, 1), Y).score(X2.reshape(-1, 1), Y)
    R3 = lm.fit(X3.reshape(-1, 1), Y).score(X3.reshape(-1, 1), Y)

    # Calculate R-squared values for combinations of independent variables
    R12 = lm.fit(np.vstack((X1, X2)).T, Y).score(np.vstack((X1, X2)).T, Y)
    R13 = lm.fit(np.vstack((X1, X3)).T, Y).score(np.vstack((X1, X3)).T, Y)
    R23 = lm.fit(np.vstack((X2, X3)).T, Y).score(np.vstack((X2, X3)).T, Y)

    R123 = lm.fit(np.vstack((X1, X2, X3)).T, Y).score(np.vstack((X1, X2, X3)).T, Y)

    # Calculate variance partitioning components
    y123 = R1 + R2 + R3 - R12 - R13 - R23 + R123
    y12 = R1 + R2 - R12 - y123
    y13 = R1 + R3 - R13 - y123
    y23 = R2 + R3 - R23 - y123
    y1 = R1 - y12 - y13 - y123
    y2 = R2 - y12 - y23 - y123
    y3  = R3 - y13 - y23 - y123

    return y1, y2, y3, y123

path = RDM_OUTPUT_DIR
distance_metric = "correlation"

X1 = squareform(PPA.round(10))
X2 = squareform(OPA.round(10))
X3 = squareform(RSC.round(10))

full_y1 = []
full_y2 = []
full_y3 = []
full_y123 = []

for file in os.listdir(path):
    if (distance_metric in file) and ("_5_" in file):
        rdms_per_subject = np.load(os.path.join(path, file))
        all_y1 = []
        all_y2 = []
        all_y3 = []
        all_y123 = []
        for timepoint in rdms_per_subject:
            Y = squareform(timepoint.round(10))
            y1, y2, y3, y123 = VarPart_3(Y, X1, X2, X3)
            all_y1.append(y1)
            all_y2.append(y2)
            all_y3.append(y3)
            all_y123.append(y123)
        full_y1.append(all_y1 - np.mean(all_y1[:13]))
        full_y2.append(all_y2 - np.mean(all_y2[:13]))
        full_y3.append(all_y3 - np.mean(all_y3[:13]))
        full_y123.append(all_y123)

mean_y1, sem_y1 = mean_and_sem(full_y1)
mean_y2, sem_y2 = mean_and_sem(full_y2)
mean_y3, sem_y3 = mean_and_sem(full_y3)
mean_y123, sem_y123 = mean_and_sem(full_y123)

highest_value_action =  t[np.argmax(full_y1, axis=1)]
highest_value_object = t[np.argmax(full_y2, axis=1)]

from scipy.stats import ttest_rel, wilcoxon
t_stat, p_value_ttest = ttest_rel(highest_value_action, highest_value_object)
print(f"Paired t-test: t-statistic = {t_stat}, p-value = {p_value_ttest}")
# Check if p-value from t-test is below 0.05
if p_value_ttest < 0.05:
    print("Paired t-test: The difference is significant (p < 0.05).")
else:
    print("Paired t-test: The difference is not significant (p >= 0.05).")

# Perform Wilcoxon signed-rank test
stat, p_value_wilcoxon = wilcoxon(highest_value_action, highest_value_object)
print(f"Wilcoxon signed-rank test: statistic = {stat}, p-value = {p_value_wilcoxon}")
if p_value_wilcoxon < 0.05:
    print("Wilcoxon signed-rank test: The difference is significant (p < 0.05).")
else:
    print("Wilcoxon signed-rank test: The difference is not significant (p >= 0.05).")

# create Figure
fig, ax = plt.subplots()

ax.axvline(x=0, color='lightgray', linestyle='--')
ax.axhline(y=0, color='lightgray', linestyle='--')

color1 = "#bfd200"
color2 = "#348aa7"
color3 = "#5dd39e"


line_1 = "-"
line_2 = "-"
line_3 = "-"
alpha_line = 1
alpha_shades = 0.1
lw = 2


min_value = 0

plt.plot(t, mean_y1, color = color1, label="PPA",  lw = lw)
plt.fill_between(t, mean_y1 - sem_y1, mean_y1 + sem_y1, alpha=0.1, color = color1)

plt.plot(t, mean_y2, color = color2, label="OPA",  lw = lw)
plt.fill_between(t, mean_y2 - sem_y2, mean_y2 + sem_y2, alpha=0.1, color = color2)

plt.plot(t, mean_y3, color = color3, label="MPA", lw = lw)
plt.fill_between(t, mean_y3 - sem_y3, mean_y3 + sem_y3, alpha=0.1, color = color3)

plt.plot(t, mean_y123, color = "black", label="shared by all", linestyle = "--", lw = lw)
plt.fill_between(t, mean_y123 - sem_y123, mean_y123 + sem_y123, alpha=0.1, color = "black")

# test if correlation is different from zero
mean_y1_sig_timepoints = significant_against_zero(np.array(full_y1))
for timepoint in mean_y1_sig_timepoints:
    plt.text(t[timepoint], min_value - 0.001 , color = color1, s =".", fontsize=20)

# test if correlation is different from zero
mean_y2_sig_timepoints = significant_against_zero(np.array(full_y2))
for timepoint in mean_y2_sig_timepoints:
    plt.text(t[timepoint],  min_value - 0.0015, color = color2, s =".", fontsize=20)

mean_y3_sig_timepoints = significant_against_zero(np.array(full_y3))
for timepoint in mean_y3_sig_timepoints:
    plt.text(t[timepoint],  min_value - 0.002, color = color3, s =".", fontsize=20)


mean_y123_sig_timepoints = significant_against_zero(np.array(full_y123))
for timepoint in mean_y123_sig_timepoints:
    plt.text(t[timepoint],  min_value , color = "black", s =".", fontsize=20)

# plot peak timepoints
ax.axvline(x= t[np.argmax(mean_y1)], color=color1, linestyle='--')
ax.axvline(x= t[np.argmax(mean_y2)], color=color2, linestyle='--')
ax.axvline(x= t[np.argmax(mean_y3)], color=color3, linestyle='--')

#ax.plot([t[np.argmax(mean_y1)], t[np.argmax(mean_y2)]], [0.0058, 0.0058], color="black", linestyle='-')
#ax.plot([t[np.argmax(mean_y1)], t[np.argmax(mean_y1)]], [0.0055, 0.0058], color="black", linestyle='-')
#ax.plot([t[np.argmax(mean_y2)], t[np.argmax(mean_y2)]], [0.0055, 0.0058], color="black", linestyle='-')
#ax.text((t[np.argmax(mean_y1)] + t[np.argmax(mean_y2)])/2, 0.0058, '*', ha='center', fontsize=12)



plt.ylim(-0.003, 0.040)
plt.xlabel('Time (s)')
plt.ylabel('Unique Variance', fontsize = 15)
plt.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig(os.path.join(dirname, 'Panel_B_VarPart_unique_ROI_corrs.svg'), transparent = True, dpi = 300)
plt.savefig(os.path.join(dirname, 'Panel_B_VarPart_unique_ROI_corrs.png'), transparent = True, dpi = 300)
plt.show()

import numpy as np
from scipy.spatial.distance import squareform
import pingouin as pg

def partial_corr_with_model_pingouin(rdm1, model_rdm1, model_rdm2):
    corrs = []
    for timepoint in range(rdm1.shape[0]):
        rdv1 = squareform(rdm1[timepoint].round(10))   # ERP RDM (timepoint specific)
        rdv_model1 = squareform(model_rdm1.round(10))  # Model RDM1 (e.g., action)
        rdv_model2 = squareform(model_rdm2.round(10))  # Model RDM2 (e.g., GIST_256)
        
        # Prepare the data for Pingouin's partial correlation
        data = np.column_stack([rdv1, rdv_model1, rdv_model2])
        df = pd.DataFrame(data, columns=['RDM1', 'Model1', 'Model2'])
        
        # Calculate partial correlation (between RDM1 and Model1, controlling for Model2)
        partial_corr_result = pg.partial_corr(data=df, x='Model1', y='RDM1', covar='Model2', method='spearman')
        
        # Extract partial correlation coefficient
        partial_corr = partial_corr_result['r'].values[0]
        corrs.append(partial_corr)
    
    # Compute the mean correlation across timepoints
    mean_corr = np.mean(corrs)

    return mean_corr, corrs


def compute_partial_corrs_sliding(distance_metric, n, model_rdm1, model_rdm2):
    path = RDM_OUTPUT_DIR
    
    all_sub_corrs = []
    for file in os.listdir(path):
        if (distance_metric in file) and (n in file):
            rdms_per_subject = np.load(os.path.join(path, file))
            mean_corr, corrs = partial_corr_with_model_pingouin(rdms_per_subject, model_rdm1, model_rdm2)
            all_sub_corrs.append(corrs)
    
    mean_corr = np.mean(np.array(all_sub_corrs), axis=0)
    sem = np.std(all_sub_corrs, axis=0) / np.sqrt(len(all_sub_corrs))

    return mean_corr, sem, np.array(all_sub_corrs)

distance_metric = "correlation"

color1 = "#bfd200"
color2 = "#348aa7"

line_1 = "-"
line_2 = "-"
alpha_line = 1
alpha_shades = 0.1
lw = 2

# create Figure
fig, ax = plt.subplots()

ax.axvline(x=0, color='lightgray', linestyle='--')
ax.axhline(y=0, color='lightgray', linestyle='--')

# Compute partial correlation for action and object EEG RDMs
mean_action, sem_action, action_array = compute_partial_corrs_sliding(distance_metric, "_5_", PPA, OPA)
mean_object, sem_object, object_array = compute_partial_corrs_sliding(distance_metric, "_5_", OPA, PPA)

# Plot action RDM with partial correlation
plt.plot(t, mean_action, color=color1, label="PPA (partialed out OPA)", alpha=alpha_line, lw=lw)
plt.fill_between(t, mean_action - sem_action, mean_action + sem_action, alpha=alpha_shades, color=color1)

# Plot object RDM with partial correlation
plt.plot(t, mean_object, color=color2, label="OPA (partialed out PPA)", alpha=alpha_line, lw=lw)
plt.fill_between(t, mean_object - sem_object, mean_object + sem_object, alpha=alpha_shades, color=color2)

min_value = lowest_value((mean_action - sem_action), (mean_object - sem_object))

# test if correlation is different from zero
action_sig_timepoints = significant_against_zero(action_array)
for timepoint in action_sig_timepoints:
    plt.text(t[timepoint], min_value - 0.001, color=color1, s=".", fontsize=20)

object_sig_timepoints = significant_against_zero(object_array)
for timepoint in object_sig_timepoints:
    plt.text(t[timepoint], min_value - 0.003, color=color2, s=".", fontsize=20)

# pairwise comparison
pairwise = action_array - object_array
pairwise_sig_timepoints = significant_against_zero(pairwise)
for timepoint in pairwise_sig_timepoints:
    plt.text(t[timepoint], min_value - 0.004, color="black", s=".", fontsize=20)

# plot peak timepoints
ax.axvline(x= t[np.argmax(mean_action)], color=color1, linestyle='--')
ax.axvline(x= t[np.argmax(mean_object)], color=color2, linestyle='--')

#ax.plot([t[np.argmax(mean_action)], t[np.argmax(mean_object)]], [0.065, 0.065], color="black", linestyle='-')
#ax.plot([t[np.argmax(mean_action)], t[np.argmax(mean_action)]], [0.063, 0.065], color="black", linestyle='-')
#ax.plot([t[np.argmax(mean_object)], t[np.argmax(mean_object)]], [0.063, 0.065], color="black", linestyle='-')
#ax.text((t[np.argmax(mean_action)] + t[np.argmax(mean_object)])/2, 0.066, '*', ha='center', fontsize=12)

plt.ylim(-0.015, 0.125)
plt.xlabel('Time (s)')
plt.ylabel('Partial Spearman Correlation')
plt.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig(os.path.join(dirname, 'Panel_C_PPA_OPA_partialCorr.svg'), transparent = True, dpi = 300)
plt.savefig(os.path.join(dirname, 'Panel_C_PPA_OPA_partialCorr.png'), transparent = True, dpi = 300)
plt.show()



def corr_with_model(rdm1, model_rdm):
    corrs = []
    for timepoint in range(rdm1.shape[0]):
        rdv1 = squareform(rdm1[timepoint].round(10))
        rdv2 = squareform(model_rdm.round(10))
        corr, p = spearmanr(rdv1, rdv2)
        corrs.append(corr)

    mean = np.mean(corrs)

    return mean, corrs


def compute_corrs_average(distance_metric, model_rdm):
    path = RDM_OUTPUT_DIR

    all_sub_corrs = []
    for file in os.listdir(path):
        if (distance_metric in file) and ("_5_" in file):
            rdms_per_subject = np.load(os.path.join(path, file))
            mean_corr, corrs = corr_with_model(rdms_per_subject, model_rdm)
            all_sub_corrs.append(corrs)
    
    mean_corr = np.mean(np.array(all_sub_corrs), axis = 0)
    sem = np.std(all_sub_corrs, axis=0) / np.sqrt(len(all_sub_corrs))

    return mean_corr, sem, np.array(all_sub_corrs)

# Define the model RDMs
model_rdms = [action_eeg_rdm, object_eeg_rdm, GIST_265]
model_names = ["action_eeg_rdm", "object_eeg_rdm", "GIST_265"]

# Initialize an empty dictionary to store correlation results
correlation_results = {}

# Define your distance metric and regions
distance_metric = 'correlation'  # Replace with the actual distance metric you are using

# Loop over each model RDM to compute and store the correlations
for idx, model_rdm in enumerate(model_rdms):
    # Compute base correlation
    mean_base, sem_base, base_array = compute_corrs_average(distance_metric, model_rdm)
    
    # Compute partial correlations for PPA and OPA
    mean_partial_ppa, sem_partial_ppa, partial_corrs_ppa = compute_partial_corrs_sliding(distance_metric, "_5_", model_rdm, PPA)
    mean_partial_opa, sem_partial_opa, partial_corrs_opa = compute_partial_corrs_sliding(distance_metric, "_5_", model_rdm, OPA)
    
    # Store results in the dictionary
    correlation_results[model_names[idx]] = {
        'base': {
            'mean': mean_base,
            'sem': sem_base,
            'array': base_array
        },
        'PPA': {
            'mean_partial_corr': mean_partial_ppa,
            'sem_partial_corr': sem_partial_ppa,
            'partial_corrs': partial_corrs_ppa
        },
        'OPA': {
            'mean_partial_corr': mean_partial_opa,
            'sem_partial_corr': sem_partial_opa,
            'partial_corrs': partial_corrs_opa
        }
    }

def plot_partial_correlations(correlation_results, model_rdm_to_plot, model_name, colors, savepath):
    """
    Plots the base and partial correlation results for a specified model RDM.
    
    Parameters:
        correlation_results (dict): Dictionary containing the correlation data for each model RDM.
        model_rdm_to_plot (str): The model RDM to plot.
        colors (list): List of 3 colors [base_color, ppa_color, opa_color].
    """
    if model_rdm_to_plot not in correlation_results:
        print(f"{model_rdm_to_plot} not found in correlation results.")
        return
    
    # Extract colors for the base, PPA, and OPA plots
    base_color, ppa_color, opa_color = colors
    
    lw = 2
    
    # Initialize the figure and axis for this model RDM
    fig, ax = plt.subplots() #figsize=(4, 4)

    # Add horizontal and vertical reference lines
    ax.axvline(x=0, color='lightgray', linestyle='--')
    ax.axhline(y=0, color='lightgray', linestyle='--')
    
    # Plot the base correlation with specified color
    mean_base = correlation_results[model_rdm_to_plot]['base']['mean']
    sem_base = correlation_results[model_rdm_to_plot]['base']['sem']
    ax.plot(t, mean_base, label=f'{model_name}', color=base_color, linestyle='--', lw=lw)
    ax.fill_between(t, mean_base - sem_base, mean_base + sem_base, color=base_color, alpha=0.1)

    # Plot the partial correlation for PPA with specified color
    mean_partial_ppa = correlation_results[model_rdm_to_plot]['PPA']['mean_partial_corr']
    sem_partial_ppa = correlation_results[model_rdm_to_plot]['PPA']['sem_partial_corr']
    ax.plot(t, mean_partial_ppa, label=f'{model_name} partialed out PPA', color=ppa_color, lw=lw)
    ax.fill_between(t, mean_partial_ppa - sem_partial_ppa, mean_partial_ppa + sem_partial_ppa, color=ppa_color, alpha=0.1)

    # Plot the partial correlation for OPA with specified color
    mean_partial_opa = correlation_results[model_rdm_to_plot]['OPA']['mean_partial_corr']
    sem_partial_opa = correlation_results[model_rdm_to_plot]['OPA']['sem_partial_corr']
    ax.plot(t, mean_partial_opa, label=f'{model_name} partialed out OPA', color=opa_color, lw=lw)
    ax.fill_between(t, mean_partial_opa - sem_partial_opa, mean_partial_opa + sem_partial_opa, color=opa_color, alpha=0.1)

    # Labeling the plot
    ax.set_ylim(-0.02, 0.06)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Correlation')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Clean up the plot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set the title for the figure
   
    plt.savefig(savepath, transparent=True, dpi=300)
    # Show the plot
    plt.show()

plot_partial_correlations(correlation_results, "action_eeg_rdm", "Affordance", ['#ff2c55', '#bfd200', '#348aa7'],
                          os.path.join(dirname, 'Panel_D_Affordance_OPA_PPA_partial.png'))

plot_partial_correlations(correlation_results, "object_eeg_rdm", "Objects", ['#0974f1', '#bfd200', '#348aa7'], 
                          os.path.join(dirname, 'Panel_D_Objects_OPA_PPA_partial.png'))

plot_partial_correlations(correlation_results, "GIST_265", "GIST", ['#ee9b00', '#bfd200', '#348aa7'],
                          os.path.join(dirname, 'Panel_D_GIST_OPA_PPA_partial'))

def plot_difference_and_auc(correlation_results, model_rdm_1, model_rdm_2):
    """
    Plots the difference between base and partial correlations for two model spaces,
    computes the AUC for each difference line, and displays the difference in AUC values.
    
    Parameters:
        correlation_results (dict): Dictionary containing the correlation data for each model RDM.
        model_rdm_1 (str): The first model RDM to compare.
        model_rdm_2 (str): The second model RDM to compare.
        colors (list): List of 2 colors [ppa_color, opa_color].
        savepath (str): Path to save the plot.
    """
    
    def compute_diff_auc(correlation_results, model_rdm, region):
        """ Helper function to compute difference array and AUC for a given model and region. """
        base_array = correlation_results[model_rdm]['base']['array']
        partial_corrs = correlation_results[model_rdm][region]['partial_corrs']
        diff = base_array - partial_corrs
        auc = np.trapz(diff) 
        return auc

    # Compute differences, SEMs, AUCs, and significant points for each model and region
    model1_ppa_auc = compute_diff_auc(correlation_results, model_rdm_1, 'PPA')
    model1_opa_auc = compute_diff_auc(correlation_results, model_rdm_1, 'OPA')
    
    model2_ppa_auc = compute_diff_auc(correlation_results, model_rdm_2, 'PPA')
    model2_opa_auc = compute_diff_auc(correlation_results, model_rdm_2, 'OPA')
    


    return model1_ppa_auc, model1_opa_auc, model2_ppa_auc, model2_opa_auc

from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_rel

model1_ppa_auc, model1_opa_auc, model2_ppa_auc, model2_opa_auc = plot_difference_and_auc(correlation_results, "action_eeg_rdm", "GIST_265")
# Data organization for plotting and statistical tests
data = [model1_ppa_auc, model1_opa_auc, model2_ppa_auc, model2_opa_auc]
labels = ["Action PPA", "Action OPA", "GIST PPA", "GIST OPA"]
positions = [0, 1, 2, 3]
colors = ['#bfd200', '#348aa7', '#bfd200', '#348aa7']

# Paired t-tests for specific comparisons and storing t and p values
comparisons = [
    ("Action PPA vs Action OPA", model1_ppa_auc, model1_opa_auc),
    ("GIST PPA vs GIST OPA", model2_ppa_auc, model2_opa_auc),
    ("Action PPA vs GIST PPA", model1_ppa_auc, model2_ppa_auc),
    ("Action OPA vs GIST OPA", model1_opa_auc, model2_opa_auc)
]

t_values = []
p_values = []

for label, group1, group2 in comparisons:
    t_stat, p_val = ttest_rel(group1, group2)
    t_values.append(t_stat)
    p_values.append(p_val)
    # Print the t and p values rounded to 3 decimal places
    print(f"{label}: t = {t_stat:.3f}, p = {p_val:.3f}")

# Apply Bonferroni correction
_, corrected_p_values, _, _ = multipletests(p_values, alpha=0.05, method='bonferroni')

# Plotting
fig, ax = plt.subplots(figsize=(4, 5))

for i, (pos, values, color) in enumerate(zip(positions, data, colors)):
    # Scatter individual points with jitter
    jitter = np.random.normal(0, 0.05, size=len(values))  # Adds slight jitter for visibility
    ax.scatter(np.full_like(values, pos) + jitter, values, color=color, alpha=0.7, s = 50)

    # Plot the mean as a horizontal line
    mean_value = np.mean(values)
    ax.hlines(mean_value, pos - 0.2, pos + 0.2, color='black', linewidth=2, linestyles="--")

# Set labels and title
ax.set_xticks(positions)
ax.set_xticklabels(labels)
ax.set_ylabel("AUC of Difference from Base Correlation")
ax.set_ylim(-1, 5)

# Clean up the plot
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add significance markers based on corrected p-values
significance_positions = [(0, 1), (2, 3), (0, 2), (1, 3)]
for (x1, x2), p_val in zip(significance_positions, corrected_p_values):
    if p_val < 0.05:
        y_max = max(np.max(data[x1]), np.max(data[x2])) + 0.5
        ax.plot([x1, x2], [y_max, y_max], color="black", lw = 2)
        ax.text((x1 + x2) / 2, y_max + 0.1, "*", ha='center', color="black", fontsize=20)

# Save the figure
plt.savefig(os.path.join(dirname, 'Panel_E_AUC_Affordance_GIST.png'), transparent = True, dpi = 300)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

# Assuming you have already defined the necessary functions and variables:
# compute_corrs_sliding, action_eeg_rdm, OPA, PPA, t, etc.
color1 = "#ff2c55"
color2 = "#bfd200"
color3 = "#348aa7"

# Create Figure
fig, ax = plt.subplots(figsize=(4, 2))  # Adjust figure size for the new orientation

ax.axvline(x=0, color='lightgray', linestyle='--')  # Change horizontal line to vertical

# Compute mean and individual data points for each RDM
mean_action, sem_action, array_action = compute_corrs_sliding(distance_metric, "_5_", action_eeg_rdm)
mean_OPA, sem_OPA, array_OPA = compute_corrs_sliding(distance_metric, "_5_", OPA)
mean_PPA, sem_PPA, array_PPA = compute_corrs_sliding(distance_metric, "_5_", PPA)

# Extract highest values from the arrays
highest_values_action = t[np.argmax(array_action, axis=1)]
highest_values_OPA = t[np.argmax(array_OPA, axis=1)]
highest_values_PPA = t[np.argmax(array_PPA, axis=1)]

# Define jitter parameters
jitter_amount = 0.1  # Adjust the amount of jitter if needed

# Add jittered scatter plot for individual data points with updated order (Action, PPA, OPA)
ax.scatter(highest_values_action, 
           np.ones_like(highest_values_action) + np.random.uniform(-jitter_amount, jitter_amount, size=len(highest_values_action)), 
           color='lightgray', alpha=0.5, s=15, label='Affordances')
ax.scatter(highest_values_PPA, 
           np.ones_like(highest_values_PPA) * 2 + np.random.uniform(-jitter_amount, jitter_amount, size=len(highest_values_PPA)), 
           color='lightgray', alpha=0.5, s=15, label='PPA')
ax.scatter(highest_values_OPA, 
           np.ones_like(highest_values_OPA) * 3 + np.random.uniform(-jitter_amount, jitter_amount, size=len(highest_values_OPA)), 
           color='lightgray', alpha=0.5, s=15, label='OPA')

# Plot mean values with larger, colored dots
ax.scatter(np.mean(highest_values_action), 1, color=color1, s=70, edgecolor='white', zorder=3)
ax.scatter(np.mean(highest_values_PPA), 2, color=color2, s=70, edgecolor='white', zorder=3)
ax.scatter(np.mean(highest_values_OPA), 3, color=color3, s=70, edgecolor='white', zorder=3)

# Add vertical lines for the mean "Action" at the y-positions for "PPA" and "OPA"
action_mean = np.mean(highest_values_action)
ax.plot([action_mean, action_mean], [1.9, 2.1], color=color1, linewidth=1, linestyle='-')  # Line for PPA
ax.plot([action_mean, action_mean], [2.9, 3.1], color=color1, linewidth=1, linestyle='-')  # Line for OPA

# Set labels and title
ax.set_xlim(-0.1, 1)  # Adjust limits for the x-axis (formerly y-axis)
ax.set_yticks([1, 2, 3])
ax.set_yticklabels(['Affordance', 'PPA', 'OPA'])
ax.set_xlabel('Highest correlation timepoint')  # Update label for the x-axis

# Statistical testing between pairs with updated order
groups = [highest_values_action, highest_values_PPA, highest_values_OPA]
group_names = ['Action', 'PPA', 'OPA']

x_max = 0.85  # Set starting point for significance line above the highest group
significance_level = 0.05
y_max = 0.85
# Loop through all pairs for t-tests
for i in range(len(groups)):
    for j in range(i + 1, len(groups)):
        # Perform t-test
        t_stat, p_value = ttest_ind(groups[i], groups[j])
        print(group_names[i] + " vs " + group_names[j] + ": t=" + str(t_stat))
        print(group_names[i] + " vs " + group_names[j] + ": p=" + str(p_value))
        
        # If significant, plot a horizontal line above the points
        #if p_value < significance_level:
            # Determine y positions for the line (above the compared groups)
            #y = (i + 1 + j + 1) / 2  # Midpoint between the two groups for line
            #x1, x2 = np.mean(groups[i]), np.mean(groups[j])
            #h, col = 0.02, 'black'
            
            # Draw horizontal line and asterisk for significance
            #ax.plot([x1, x2], [y_max, y_max], lw=1.5, color=col)
            #ax.text((x1 + x2) / 2, y_max + h, f'*', ha='center', va='bottom', color=col)
            #x_max += 0.1  # Increment x_max for stacking lines if needed

# Remove the top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig(os.path.join(dirname, 'Additional_highest_timepoints_PPA_OPA_affordance.svg'), transparent = True, dpi = 300)
# Show the plot
plt.show()

def VarPart_2(Y, X1, X2):
    from sklearn import linear_model
    # initalize model
    lm = linear_model.LinearRegression()

    # Calculate R-squared values for each independent variable
    R1 = lm.fit(X1.reshape(-1, 1), Y).score(X1.reshape(-1, 1), Y)
    R2 = lm.fit(X2.reshape(-1, 1), Y).score(X2.reshape(-1, 1), Y)

    # Calculate R-squared values for combinations of independent variables
    R12 = lm.fit(np.vstack((X1, X2)).T, Y).score(np.vstack((X1, X2)).T, Y)

    # Calculate variance partitioning components
    y12 = R1 + R2 - R12
    y1 = R1 - y12 
    y2 = R2 - y12 


    return y1, y2, y12

path = RDM_OUTPUT_DIR
distance_metric = "correlation"

X1 = squareform(PPA.round(5))
X2 = squareform(OPA.round(5))

full_y1 = []
full_y2 = []
full_y12 = []

for file in os.listdir(path):
        if (distance_metric in file) and ("_5_" in file):
            rdms_per_subject = np.load(os.path.join(path, file))
            all_y1 = []
            all_y2 = []
            all_y12 = []
            for timepoint in rdms_per_subject:
                Y = squareform(timepoint.round(10))
                y1, y2, y12 = VarPart_2(Y, X1, X2)
                all_y1.append(y1)
                all_y2.append(y2)
                all_y12.append(y12)

        

            #full_y1.append(all_y1)
            #full_y2.append(all_y2)
            full_y1.append(all_y1 - np.mean(all_y1[:13]))
            full_y2.append(all_y2 - np.mean(all_y2[:13]))
            full_y12.append(all_y12)


mean_y1, sem_y1 = mean_and_sem(full_y1)
mean_y2, sem_y2 = mean_and_sem(full_y2)
mean_y12, sem_y12 = mean_and_sem(full_y12)


# create Figure
fig, ax = plt.subplots()

ax.axvline(x=0, color='lightgray', linestyle='--')
ax.axhline(y=0, color='lightgray', linestyle='--')

color1 = "#bfd200"
color2 = "#348aa7"


new_t = t[:len(mean_y1)]

plt.plot(new_t, mean_y1, color = color1, label="PPA",  lw = lw)
plt.fill_between(new_t, mean_y1 - sem_y1, mean_y1 + sem_y1, alpha=0.1, color = color1)

plt.plot(new_t, mean_y2, color = color2, label="OPA",  lw = lw)
plt.fill_between(new_t, mean_y2 - sem_y2, mean_y2 + sem_y2, alpha=0.1, color = color2)

#plt.plot(new_t, mean_y12, color = "black", label="Shared by all", linestyle = "--", lw = 1)
#plt.fill_between(new_t, mean_y12 - sem_y12, mean_y12 + sem_y12, alpha=0.1, color = "black")

min_value = lowest_value((mean_y1 - sem_y1), (mean_y2 - sem_y2))

# test if correlation is different from zero
mean_y1_sig_timepoints = significant_against_zero(np.array(full_y1))
for timepoint in mean_y1_sig_timepoints:
    plt.text(new_t[timepoint], min_value - 0.0003, color = color1, s =".", fontsize=15)

# test if correlation is different from zero
mean_y2_sig_timepoints = significant_against_zero(np.array(full_y2))
for timepoint in mean_y2_sig_timepoints:
    plt.text(new_t[timepoint],  min_value - 0.0008, color = color2, s =".", fontsize=15)


# test if correlation is different from zero
mean_y12_sig_timepoints = significant_against_zero(np.array(full_y12))
for timepoint in mean_y12_sig_timepoints:
    plt.text(new_t[timepoint],  min_value - 0.00012, color = "black", s =".", fontsize=15)

# plot peak timepoints
ax.axvline(x= new_t[np.argmax(mean_y1)], color=color1, linestyle='--')
ax.axvline(x= new_t[np.argmax(mean_y2)], color=color2, linestyle='--')

#ax.plot([t[np.argmax(mean_y1)], t[np.argmax(mean_y2)]], [0.0058, 0.0058], color="black", linestyle='-')
#ax.plot([t[np.argmax(mean_y1)], t[np.argmax(mean_y1)]], [0.0055, 0.0058], color="black", linestyle='-')
#ax.plot([t[np.argmax(mean_y2)], t[np.argmax(mean_y2)]], [0.0055, 0.0058], color="black", linestyle='-')
#ax.text((t[np.argmax(mean_y1)] + t[np.argmax(mean_y2)])/2, 0.0058, '*', ha='center', fontsize=12)


#plt.ylim(-0.001, 0.006)
plt.xlabel('Time (s)')
plt.ylabel('Unique Variance')
plt.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig(os.path.join(dirname, 'Additional_VaPart_PPA_OPA.svg'), transparent = True, dpi = 300)
plt.show()

distance_metric = "correlation"

X1 = squareform(PPA.round(5))
X2 = squareform(OPA.round(5))

full_PPA = []
full_action = []
full_action_PPA = []

for file in os.listdir(path):
    if (distance_metric in file) and ("_5_" in file):
        rdms_per_subject = np.load(os.path.join(path, file))
        all_y1 = []
        all_y2 = []
        all_y12 = []
        for timepoint in rdms_per_subject:
            Y = squareform(timepoint.round(10))
            y1, y2, y12 = VarPart_2(Y, X1, X2)
            all_y1.append(y1)
            all_y2.append(y2)
            all_y12.append(y12)

        full_PPA.append(all_y1 - np.mean(all_y1[:13]))
        full_action.append(all_y2 - np.mean(all_y2[:13]))
        full_action_PPA.append(all_y12)


PPA_noOPA_mean, PPA_noOPA_sem = mean_and_sem(full_PPA)
OPA_noPPA_mean, OPA_noPPA_sem = mean_and_sem(full_action)
PPA_OPA_mean, PPA_OPA_sem = mean_and_sem(full_action_PPA)

fig, ax = plt.subplots()

ax.axvline(x=0, color='lightgray', linestyle='--')
ax.axhline(y=0, color='lightgray', linestyle='--')

color1 = "#bfd200"
color2 = "#348aa7"


plt.plot(t, PPA_noOPA_mean, color = color1, label="PPA (no OPA)",  lw = lw)
plt.fill_between(new_t, PPA_noOPA_mean - PPA_noOPA_sem, PPA_noOPA_mean + PPA_noOPA_sem, alpha=0.1, color = color1)

plt.plot(t, OPA_noPPA_mean, color = color2, label="OPA (no PPA)",  lw = lw)
plt.fill_between(t, OPA_noPPA_mean - OPA_noPPA_sem, OPA_noPPA_mean + OPA_noPPA_sem, alpha=0.1, color = color2)

plt.plot(t, PPA_OPA_mean, color = "black", label="Shared by both", linestyle = "--", lw = 1)
plt.fill_between(t, PPA_OPA_mean - PPA_OPA_sem, PPA_OPA_mean + PPA_OPA_sem, alpha=0.1, color = "black")

min_value = 0

# test if correlation is different from zero
mean_y1_sig_timepoints = significant_against_zero(np.array(full_PPA))
for timepoint in mean_y1_sig_timepoints:
    plt.text(t[timepoint], min_value - 0.0005, color = color1, s =".", fontsize=15)

# test if correlation is different from zero
mean_y2_sig_timepoints = significant_against_zero(np.array(full_action))
for timepoint in mean_y2_sig_timepoints:
    plt.text(t[timepoint],  min_value - 0.0009, color = color2, s =".", fontsize=15)#


# test if correlation is different from zero
mean_shared_sig_timepoints = significant_against_zero(np.array(full_action_PPA))
for timepoint in mean_shared_sig_timepoints:
    plt.text(t[timepoint],  min_value - 0.002, color = "black", s =".", fontsize=15)


# plot peak timepoints
ax.axvline(x= t[np.argmax(PPA_noOPA_mean)], color=color1, linestyle='--')
ax.axvline(x= t[np.argmax(OPA_noPPA_mean)], color=color2, linestyle='--')

#ax.plot([t[np.argmax(mean_y1)], t[np.argmax(mean_y2)]], [0.0058, 0.0058], color="black", linestyle='-')
#ax.plot([t[np.argmax(mean_y1)], t[np.argmax(mean_y1)]], [0.0055, 0.0058], color="black", linestyle='-')
#ax.plot([t[np.argmax(mean_y2)], t[np.argmax(mean_y2)]], [0.0055, 0.0058], color="black", linestyle='-')
#ax.text((t[np.argmax(mean_y1)] + t[np.argmax(mean_y2)])/2, 0.0058, '*', ha='center', fontsize=12)


#plt.ylim(-0.001, 0.006)
plt.xlabel('Time (s)')
plt.ylabel('Unique Variance')
plt.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig(os.path.join(dirname, 'Additional_VaPart_PPA_OPA_and_shared.svg'), transparent = True, dpi = 300)
#plt.savefig(

highest_value_action =  t[np.argmax(full_PPA, axis=1)]
highest_value_object = t[np.argmax(full_action, axis=1)]

from scipy.stats import ttest_rel, wilcoxon
t_stat, p_value_ttest = ttest_rel(highest_value_action, highest_value_object)
print(f"Paired t-test: t-statistic = {t_stat}, p-value = {p_value_ttest}")
# Check if p-value from t-test is below 0.05
if p_value_ttest < 0.05:
    print("Paired t-test: The difference is significant (p < 0.05).")
else:
    print("Paired t-test: The difference is not significant (p >= 0.05).")

# Perform Wilcoxon signed-rank test
stat, p_value_wilcoxon = wilcoxon(highest_value_action, highest_value_object)
print(f"Wilcoxon signed-rank test: statistic = {stat}, p-value = {p_value_wilcoxon}")
if p_value_wilcoxon < 0.05:
    print("Wilcoxon signed-rank test: The difference is significant (p < 0.05).")
else:
    print("Wilcoxon signed-rank test: The difference is not significant (p >= 0.05).")

#task = "action"
distance_metric = "correlation"

color1 = "#ff2c55"
color2 = "#348aa7"


line_1 = "-"
line_2 = "-"
alpha_line = 1
alpha_shades = 0.1
lw = 1

# create Figure
fig, ax = plt.subplots()

ax.axvline(x=0, color='lightgray', linestyle='--')
ax.axhline(y=0, color='lightgray', linestyle='--')


mean_action, sem_action, action_array = compute_corrs_sliding(distance_metric, "_5_", action_eeg_rdm)
mean_object, sem_object, object_array = compute_corrs_sliding(distance_metric, "_5_", OPA)



plt.plot(t, mean_action, color = color1, label="Affordances (EEG)", alpha = alpha_line, lw  = lw)
plt.fill_between(t, mean_action - sem_action, mean_action + sem_action, alpha=alpha_shades, color = color1)

plt.plot(t, mean_object, color = color2, label="OPA", alpha = alpha_line, lw = lw)
plt.fill_between(t, mean_object - sem_object, mean_object + sem_object, alpha=alpha_shades, color = color2)


min_value = lowest_value((mean_action - sem_action), (mean_object - sem_object))

# test if correlation is different from zero
action_sig_timepoints = significant_against_zero(action_array)
for timepoint in action_sig_timepoints:
    plt.text(t[timepoint], min_value, color = color1, s =".", fontsize=15)

object_sig_timepoints = significant_against_zero(object_array)
for timepoint in object_sig_timepoints:
    plt.text(t[timepoint], min_value - 0.003, color = color2, s =".", fontsize=15)


# pairwise comparison

pairwise = action_array - object_array
pairwise_sig_timepoints = significant_against_zero(pairwise)
for timepoint in pairwise_sig_timepoints:
    plt.text(t[timepoint], min_value - 0.006, color = "black", s =".", fontsize=15)

# plot peak timepoints
ax.axvline(x= t[np.argmax(mean_action)], color=color1, linestyle='--')
ax.axvline(x= t[np.argmax(mean_object)], color=color2, linestyle='--')

#ax.plot([t[np.argmax(mean_action)], t[np.argmax(mean_object)]], [0.065, 0.065], color="black", linestyle='-')
#ax.plot([t[np.argmax(mean_action)], t[np.argmax(mean_action)]], [0.063, 0.065], color="black", linestyle='-')
#ax.plot([t[np.argmax(mean_object)], t[np.argmax(mean_object)]], [0.063, 0.065], color="black", linestyle='-')
#ax.text((t[np.argmax(mean_action)] + t[np.argmax(mean_object)])/2, 0.066, '*', ha='center', fontsize=12)


#plt.ylim(-0.015, 0.07)
plt.xlabel('Time (s)', fontsize = 15)
plt.ylabel('Spearman Correlation', fontsize = 15)
plt.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig(os.path.join(dirname, 'Supplementary_Affordance_OPA.svg'), transparent = True, dpi = 300)
plt.show()

#task = "action"
distance_metric = "correlation"

color1 = "#ff2c55"
color2 = "#bfd200"


line_1 = "-"
line_2 = "-"
alpha_line = 1
alpha_shades = 0.1
lw = 1

# create Figure
fig, ax = plt.subplots()

ax.axvline(x=0, color='lightgray', linestyle='--')
ax.axhline(y=0, color='lightgray', linestyle='--')


mean_action, sem_action, action_array = compute_corrs_sliding(distance_metric, "_5_", action_eeg_rdm)
mean_object, sem_object, object_array = compute_corrs_sliding(distance_metric, "_5_", PPA)



plt.plot(t, mean_action, color = color1, label="Affordances (EEG)", alpha = alpha_line, lw  = lw)
plt.fill_between(t, mean_action - sem_action, mean_action + sem_action, alpha=alpha_shades, color = color1)

plt.plot(t, mean_object, color = color2, label="PPA", alpha = alpha_line, lw = lw)
plt.fill_between(t, mean_object - sem_object, mean_object + sem_object, alpha=alpha_shades, color = color2)


min_value = lowest_value((mean_action - sem_action), (mean_object - sem_object))

# test if correlation is different from zero
action_sig_timepoints = significant_against_zero(action_array)
for timepoint in action_sig_timepoints:
    plt.text(t[timepoint], min_value, color = color1, s =".", fontsize=15)

object_sig_timepoints = significant_against_zero(object_array)
for timepoint in object_sig_timepoints:
    plt.text(t[timepoint], min_value - 0.003, color = color2, s =".", fontsize=15)


# pairwise comparison

pairwise = action_array - object_array
pairwise_sig_timepoints = significant_against_zero(pairwise)
for timepoint in pairwise_sig_timepoints:
    plt.text(t[timepoint], min_value - 0.006, color = "black", s =".", fontsize=15)

# plot peak timepoints
ax.axvline(x= t[np.argmax(mean_action)], color=color1, linestyle='--')
ax.axvline(x= t[np.argmax(mean_object)], color=color2, linestyle='--')

#ax.plot([t[np.argmax(mean_action)], t[np.argmax(mean_object)]], [0.065, 0.065], color="black", linestyle='-')
#ax.plot([t[np.argmax(mean_action)], t[np.argmax(mean_action)]], [0.063, 0.065], color="black", linestyle='-')
#ax.plot([t[np.argmax(mean_object)], t[np.argmax(mean_object)]], [0.063, 0.065], color="black", linestyle='-')
#ax.text((t[np.argmax(mean_action)] + t[np.argmax(mean_object)])/2, 0.066, '*', ha='center', fontsize=12)


#plt.ylim(-0.015, 0.07)
plt.xlabel('Time (s)', fontsize = 15)
plt.ylabel('Spearman Correlation', fontsize = 15)
plt.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig(os.path.join(dirname, 'Supplementary_Affordance_PPA.svg'), transparent = True, dpi = 300)
plt.show()


def compute_3Var(model1, model2, model3):
        path = RDM_OUTPUT_DIR
        distance_metric = "correlation"

        X1 = squareform(model1.round(10))
        X2 = squareform(model2.round(10))
        X3 = squareform(model3.round(10))

        full_y1 = []
        full_y2 = []
        full_y3 = []
        full_y123 = []

        for file in os.listdir(path):
                if (distance_metric in file) and ("_5_" in file):
                        rdms_per_subject = np.load(os.path.join(path, file))
                        all_y1 = []
                        all_y2 = []
                        all_y3 = []
                        all_y123 = []
                        for timepoint in rdms_per_subject:
                                Y = squareform(timepoint.round(10))
                                y1, y2, y3, y123 = VarPart_3(Y, X1, X2, X3)
                                all_y1.append(y1)
                                all_y2.append(y2)
                                all_y3.append(y3)
                                all_y123.append(y123)
                        full_y1.append(all_y1 - np.mean(all_y1[:13]))
                        full_y2.append(all_y2 - np.mean(all_y2[:13]))
                        full_y3.append(all_y3 - np.mean(all_y3[:13]))
                        full_y123.append(all_y123)

        mean_y1, sem_y1 = mean_and_sem(full_y1)
        mean_y2, sem_y2 = mean_and_sem(full_y2)
        mean_y3, sem_y3 = mean_and_sem(full_y3)
        mean_y123, sem_y123 = mean_and_sem(full_y123)

        return mean_y1, mean_y2, mean_y3, mean_y123, full_y1, full_y2, full_y3, full_y123

mean_PPA_GIST, mean_OPA_GIST, mean_GIST_GIST, mean_all_GIST, full_PPA_GIST, full_OPA_GIST, full_GIST_GIST, full_all_GIST = compute_3Var(PPA, OPA, GIST_265)
mean_PPA_ACT, mean_OPA_ACT, mean_ACT_ACT, mean_all_ACT, full_PPA_ACT, full_OPA_ACT, full_ACT_ACT, full_all_ACT = compute_3Var(PPA, OPA, action_eeg_rdm)

# create Figure
fig, ax = plt.subplots()

ax.axvline(x=0, color='lightgray', linestyle='--')
ax.axhline(y=0, color='lightgray', linestyle='--')

color1 = "#bfd200"
color2 = "#348aa7"
color3 = "blue"


line_1 = "-"
line_2 = "-"
line_3 = "-"
alpha_line = 1
alpha_shades = 0.1
lw = 1


min_value = 0

plt.plot(t, mean_y1, color = color1, label="PPA",  lw = lw)
plt.fill_between(t, mean_y1 - sem_y1, mean_y1 + sem_y1, alpha=0.1, color = color1)

plt.plot(t, mean_y2, color = color2, label="OPA",  lw = lw)
plt.fill_between(t, mean_y2 - sem_y2, mean_y2 + sem_y2, alpha=0.1, color = color2)

plt.plot(t, mean_y3, color = color3, label="object", lw = 1)
plt.fill_between(t, mean_y3 - sem_y3, mean_y3 + sem_y3, alpha=0.1, color = color3)

plt.plot(t, mean_y123, color = "black", label="shared by all", linestyle = "--", lw = 1)
plt.fill_between(t, mean_y123 - sem_y123, mean_y123 + sem_y123, alpha=0.1, color = "black")

# test if correlation is different from zero
mean_y1_sig_timepoints = significant_against_zero(np.array(full_y1))
for timepoint in mean_y1_sig_timepoints:
    plt.text(t[timepoint], min_value - 0.001 , color = color1, s =".", fontsize=15)

# test if correlation is different from zero
mean_y2_sig_timepoints = significant_against_zero(np.array(full_y2))
for timepoint in mean_y2_sig_timepoints:
    plt.text(t[timepoint],  min_value - 0.0015, color = color2, s =".", fontsize=15)

mean_y3_sig_timepoints = significant_against_zero(np.array(full_y3))
for timepoint in mean_y3_sig_timepoints:
    plt.text(t[timepoint],  min_value - 0.002, color = color3, s =".", fontsize=15)


mean_y123_sig_timepoints = significant_against_zero(np.array(full_y123))
for timepoint in mean_y123_sig_timepoints:
    plt.text(t[timepoint],  min_value , color = "black", s =".", fontsize=15)

# plot peak timepoints
ax.axvline(x= t[np.argmax(mean_y1)], color=color1, linestyle='--')
ax.axvline(x= t[np.argmax(mean_y2)], color=color2, linestyle='--')
ax.axvline(x= t[np.argmax(mean_y3)], color=color3, linestyle='--')

#ax.plot([t[np.argmax(mean_y1)], t[np.argmax(mean_y2)]], [0.0058, 0.0058], color="black", linestyle='-')
#ax.plot([t[np.argmax(mean_y1)], t[np.argmax(mean_y1)]], [0.0055, 0.0058], color="black", linestyle='-')
#ax.plot([t[np.argmax(mean_y2)], t[np.argmax(mean_y2)]], [0.0055, 0.0058], color="black", linestyle='-')
#ax.text((t[np.argmax(mean_y1)] + t[np.argmax(mean_y2)])/2, 0.0058, '*', ha='center', fontsize=12)



plt.ylim(-0.003, 0.040)
plt.xlabel('Time (s)')
plt.ylabel('Unique Variance', fontsize = 15)
plt.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig(os.path.join(dirname, 'Supplementary_VarPart_Object_PPA_OPA.svg'), transparent = True, dpi = 300)
plt.show()

X1 = squareform(PPA.round(10))
X2 = squareform(OPA.round(10))
X3 = squareform(action_eeg_rdm.round(10))

full_y1 = []
full_y2 = []
full_y3 = []
full_y123 = []

for file in os.listdir(path):
    if (distance_metric in file) and ("_5_" in file):
        rdms_per_subject = np.load(os.path.join(path, file))
        all_y1 = []
        all_y2 = []
        all_y3 = []
        all_y123 = []
        for timepoint in rdms_per_subject:
            Y = squareform(timepoint.round(10))
            y1, y2, y3, y123 = VarPart_3(Y, X1, X2, X3)
            all_y1.append(y1)
            all_y2.append(y2)
            all_y3.append(y3)
            all_y123.append(y123)
        full_y1.append(all_y1 - np.mean(all_y1[:13]))
        full_y2.append(all_y2 - np.mean(all_y2[:13]))
        full_y3.append(all_y3 - np.mean(all_y3[:13]))
        full_y123.append(all_y123)

mean_y1, sem_y1 = mean_and_sem(full_y1)
mean_y2, sem_y2 = mean_and_sem(full_y2)
mean_y3, sem_y3 = mean_and_sem(full_y3)
mean_y123, sem_y123 = mean_and_sem(full_y123)

# create Figure
fig, ax = plt.subplots()

ax.axvline(x=0, color='lightgray', linestyle='--')
ax.axhline(y=0, color='lightgray', linestyle='--')

color1 = "#bfd200"
color2 = "#348aa7"
color3 = "red"


line_1 = "-"
line_2 = "-"
line_3 = "-"
alpha_line = 1
alpha_shades = 0.1
lw = 1


min_value = 0

plt.plot(t, mean_y1, color = color1, label="PPA",  lw = lw)
plt.fill_between(t, mean_y1 - sem_y1, mean_y1 + sem_y1, alpha=0.1, color = color1)

plt.plot(t, mean_y2, color = color2, label="OPA",  lw = lw)
plt.fill_between(t, mean_y2 - sem_y2, mean_y2 + sem_y2, alpha=0.1, color = color2)

plt.plot(t, mean_y3, color = color3, label="Affordance", lw = 1)
plt.fill_between(t, mean_y3 - sem_y3, mean_y3 + sem_y3, alpha=0.1, color = color3)

plt.plot(t, mean_y123, color = "black", label="shared by all", linestyle = "--", lw = 1)
plt.fill_between(t, mean_y123 - sem_y123, mean_y123 + sem_y123, alpha=0.1, color = "black")

# test if correlation is different from zero
mean_y1_sig_timepoints = significant_against_zero(np.array(full_y1))
for timepoint in mean_y1_sig_timepoints:
    plt.text(t[timepoint], min_value - 0.001 , color = color1, s =".", fontsize=15)

# test if correlation is different from zero
mean_y2_sig_timepoints = significant_against_zero(np.array(full_y2))
for timepoint in mean_y2_sig_timepoints:
    plt.text(t[timepoint],  min_value - 0.0015, color = color2, s =".", fontsize=15)

mean_y3_sig_timepoints = significant_against_zero(np.array(full_y3))
for timepoint in mean_y3_sig_timepoints:
    plt.text(t[timepoint],  min_value - 0.002, color = color3, s =".", fontsize=15)


mean_y123_sig_timepoints = significant_against_zero(np.array(full_y123))
for timepoint in mean_y123_sig_timepoints:
    plt.text(t[timepoint],  min_value , color = "black", s =".", fontsize=15)

# plot peak timepoints
ax.axvline(x= t[np.argmax(mean_y1)], color=color1, linestyle='--')
ax.axvline(x= t[np.argmax(mean_y2)], color=color2, linestyle='--')
ax.axvline(x= t[np.argmax(mean_y3)], color=color3, linestyle='--')

#ax.plot([t[np.argmax(mean_y1)], t[np.argmax(mean_y2)]], [0.0058, 0.0058], color="black", linestyle='-')
#ax.plot([t[np.argmax(mean_y1)], t[np.argmax(mean_y1)]], [0.0055, 0.0058], color="black", linestyle='-')
#ax.plot([t[np.argmax(mean_y2)], t[np.argmax(mean_y2)]], [0.0055, 0.0058], color="black", linestyle='-')
#ax.text((t[np.argmax(mean_y1)] + t[np.argmax(mean_y2)])/2, 0.0058, '*', ha='center', fontsize=12)



plt.ylim(-0.003, 0.025)
plt.xlabel('Time (s)')
plt.ylabel('Unique Variance', fontsize = 15)
plt.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig(os.path.join(dirname, 'Supplementary_VarPart_Affordance_PPA_OPA.svg'), transparent = True, dpi = 300)
plt.show()

X1 = squareform(PPA.round(5))
X2 = squareform(OPA.round(5))
X3 = squareform(GIST_265.round(5))

full_y1 = []
full_y2 = []
full_y3 = []
full_y123 = []

for file in os.listdir(path):
    if (distance_metric in file) and ("_5_" in file):
        rdms_per_subject = np.load(os.path.join(path, file))
        all_y1 = []
        all_y2 = []
        all_y3 = []
        all_y123 = []
        for timepoint in rdms_per_subject:
            Y = squareform(timepoint.round(10))
            y1, y2, y3, y123 = VarPart_3(Y, X1, X2, X3)
            all_y1.append(y1)
            all_y2.append(y2)
            all_y3.append(y3)
            all_y123.append(y123)
        full_y1.append(all_y1 - np.mean(all_y1[:13]))
        full_y2.append(all_y2 - np.mean(all_y2[:13]))
        full_y3.append(all_y3 - np.mean(all_y3[:13]))
        full_y123.append(all_y123)

mean_y1, sem_y1 = mean_and_sem(full_y1)
mean_y2, sem_y2 = mean_and_sem(full_y2)
mean_y3, sem_y3 = mean_and_sem(full_y3)
mean_y123, sem_y123 = mean_and_sem(full_y123)

# create Figure
fig, ax = plt.subplots()

ax.axvline(x=0, color='lightgray', linestyle='--')
ax.axhline(y=0, color='lightgray', linestyle='--')

color1 = "#bfd200"
color2 = "#348aa7"
color3 = "orange"


line_1 = "-"
line_2 = "-"
line_3 = "-"
alpha_line = 1
alpha_shades = 0.1
lw = 1


min_value = 0

plt.plot(t, mean_y1, color = color1, label="PPA",  lw = lw)
plt.fill_between(t, mean_y1 - sem_y1, mean_y1 + sem_y1, alpha=0.1, color = color1)

plt.plot(t, mean_y2, color = color2, label="OPA",  lw = lw)
plt.fill_between(t, mean_y2 - sem_y2, mean_y2 + sem_y2, alpha=0.1, color = color2)

plt.plot(t, mean_y3, color = color3, label="GIST", lw = 1)
plt.fill_between(t, mean_y3 - sem_y3, mean_y3 + sem_y3, alpha=0.1, color = color3)

plt.plot(t, mean_y123, color = "black", label="shared by all", linestyle = "--", lw = 1)
plt.fill_between(t, mean_y123 - sem_y123, mean_y123 + sem_y123, alpha=0.1, color = "black")

# test if correlation is different from zero
mean_y1_sig_timepoints = significant_against_zero(np.array(full_y1))
for timepoint in mean_y1_sig_timepoints:
    plt.text(t[timepoint], min_value - 0.001 , color = color1, s =".", fontsize=15)

# test if correlation is different from zero
mean_y2_sig_timepoints = significant_against_zero(np.array(full_y2))
for timepoint in mean_y2_sig_timepoints:
    plt.text(t[timepoint],  min_value - 0.0015, color = color2, s =".", fontsize=15)

mean_y3_sig_timepoints = significant_against_zero(np.array(full_y3))
for timepoint in mean_y3_sig_timepoints:
    plt.text(t[timepoint],  min_value - 0.002, color = color3, s =".", fontsize=15)


mean_y123_sig_timepoints = significant_against_zero(np.array(full_y123))
for timepoint in mean_y123_sig_timepoints:
    plt.text(t[timepoint],  min_value , color = "black", s =".", fontsize=15)

# plot peak timepoints
ax.axvline(x= t[np.argmax(mean_y1)], color=color1, linestyle='--')
ax.axvline(x= t[np.argmax(mean_y2)], color=color2, linestyle='--')
ax.axvline(x= t[np.argmax(mean_y3)], color=color3, linestyle='--')

#ax.plot([t[np.argmax(mean_y1)], t[np.argmax(mean_y2)]], [0.0058, 0.0058], color="black", linestyle='-')
#ax.plot([t[np.argmax(mean_y1)], t[np.argmax(mean_y1)]], [0.0055, 0.0058], color="black", linestyle='-')
#ax.plot([t[np.argmax(mean_y2)], t[np.argmax(mean_y2)]], [0.0055, 0.0058], color="black", linestyle='-')
#ax.text((t[np.argmax(mean_y1)] + t[np.argmax(mean_y2)])/2, 0.0058, '*', ha='center', fontsize=12)



plt.ylim(-0.003, 0.025)
plt.xlabel('Time (s)')
plt.ylabel('Unique Variance', fontsize = 15)
plt.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig(os.path.join(dirname, 'Supplementary_VarPart_GIST_PPA_OPA.svg'), transparent = True, dpi = 300)
plt.show()
