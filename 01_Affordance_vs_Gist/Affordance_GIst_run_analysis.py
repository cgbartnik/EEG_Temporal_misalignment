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
from scipy.stats import ttest_ind, wilcoxon, ttest_rel

FIGURE_DIR = "/home/clemens-uva/Desktop/EEG_Temporal_misalignment/01_Affordance_vs_Gist/Figures"
os.makedirs(FIGURE_DIR, exist_ok=True)

OSF_DATA_DIR = "/home/clemens-uva/Desktop/DATA_OSF_external_storage"
RDM_OUTPUT_DIR = os.path.join(OSF_DATA_DIR, "ERP_sliding_window_RDMs")
PREPROCESSED_DATA_DIR = os.path.join(OSF_DATA_DIR, "preprocessed_data")
AVERAGE_RDM_DIR = os.path.join(OSF_DATA_DIR, "average_RDMs")
BEHAVIOR_RDM_DIR = os.path.join(OSF_DATA_DIR, "Behavioral_annotations")
MODEL_RDM_DIR = os.path.join(OSF_DATA_DIR, "Model_RDMs")
VISACT_BEHAVIOR_MEANS_DIR = os.path.join(OSF_DATA_DIR, "Behavioral_annotations_mean_dfs")
GIST_RDM_PATH = os.path.join(OSF_DATA_DIR, "GIST", "GIST_256_RDM_fMRI.npy")
os.makedirs(OSF_DATA_DIR, exist_ok=True)
os.makedirs(RDM_OUTPUT_DIR, exist_ok=True)


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

def compute_sliding_window_rdms_with_padding(DATA_path, subname, n, metric):
    output_path = os.path.join(RDM_OUTPUT_DIR, f"{subname}_{str(n)}_sliding_RDMs_{metric}.npy")
    alt_output_path = os.path.join(RDM_OUTPUT_DIR, f"{subname}_{str(n)}_sliding_RDMs_{metric}_repro.npy")
    if os.path.exists(output_path):
        return np.load(output_path)
    if os.path.exists(alt_output_path):
        return np.load(alt_output_path)

    # Load epochs
    epochs_action = mne.read_epochs(f"{DATA_path}{subname}_action_" + "_".join(file_substring.split("_")[2:]))
    epochs_object = mne.read_epochs(f"{DATA_path}{subname}_object_" + "_".join(file_substring.split("_")[2:]))
    epochs_fixation = mne.read_epochs(f"{DATA_path}{subname}_fixation_" + "_".join(file_substring.split("_")[2:]))
    
    # Initialize averaged_epochs
    averaged_epochs = np.zeros((len(images_name), len(occipital_electrodes), len(t)))
    image_list = []
    for ii, image in enumerate(images_name):
        act = epochs_action[image].get_data(picks=occipital_electrodes)
        obj = epochs_object[image].get_data(picks=occipital_electrodes)
        fix = epochs_fixation[image].get_data(picks=occipital_electrodes)

        full = np.concatenate((act, obj, fix), axis=0)
        averaged_epochs[ii, :, :] = np.mean(full, axis=0)
        image_list.append(image)

    n_timepoints = averaged_epochs.shape[2]  # Total number of timepoints

    # Add padding to the averaged_epochs data
    padding_width = ((0, 0), (0, 0), (n//2, n//2))
    padded_epochs = np.pad(averaged_epochs, pad_width=padding_width, mode='edge')

    # Initialize an empty list to store the RDMs
    all_rdms = []

    # Calculate the RDM using a sliding window approach with padding
    for idx3 in range(n_timepoints):
        # Average the data within the window of n timepoints
        window_data = np.mean(padded_epochs[:, :, idx3:idx3 + n], axis=2)
        # Calculate the RDM for the averaged data in the window
        rdm = pairwise_distances(window_data, metric=metric)
        all_rdms.append(rdm)  # Append the RDM to the list

    # Convert the list of RDMs to a numpy array
    rdms_2 = np.array(all_rdms)

    # Save RDMs
    np.save(os.path.join(RDM_OUTPUT_DIR, f"{subname}_{str(n)}_sliding_RDMs_{metric}.npy"), rdms_2)

    return rdms_2


for participant in participants_list:
    rdm_3 = compute_sliding_window_rdms_with_padding(DATA_path, participant, 5, "correlation")


def corr_with_model(rdm1, model_rdm):
    corrs = []
    for timepoint in range(rdm1.shape[0]):
        rdv1 = squareform(rdm1[timepoint].round(10))
        rdv2 = squareform(model_rdm.round(10))
        corr, p = spearmanr(rdv1, rdv2)
        corrs.append(corr)

    mean = np.mean(corrs)

    return mean, corrs

metric = "euclidean"

action_eeg_rdm = np.load(os.path.join(BEHAVIOR_RDM_DIR, f"action_average_RDM_{metric}.npy"))
object_eeg_rdm = np.load(os.path.join(BEHAVIOR_RDM_DIR, f"object_average_RDM_{metric}.npy"))

action_online_rdm = np.load(os.path.join(MODEL_RDM_DIR, f"online_action_rdm_EGG_action_sorted_{metric}.npy"))
object_online_rdm = np.load(os.path.join(MODEL_RDM_DIR, f"online_object_rdm_EGG_action_sorted_{metric}.npy"))

GIST_265 = load_and_sort_rdm(GIST_RDM_PATH, images_name)

spearmanr(squareform(action_eeg_rdm.round(5)),squareform(GIST_265.round(5)))[0].round(2
                                                                                      )

import numpy as np
import matplotlib.pyplot as plt
import mne

# Assuming 'corr_with_model' and 'compute_sliding_window_rdms_with_padding' are already defined
mne.set_log_level('WARNING')

# Initialize lists to store correlation values for each participant
all_corrs_base = []
all_corrs_3 = []
all_corrs_5 = []
all_corrs_10 = []

for participant in participants_list:
    # Compute RDMs with sliding windows of different sizes
    rdm_3 = compute_sliding_window_rdms_with_padding(DATA_path, participant, 3, "correlation")
    rdm_5 = compute_sliding_window_rdms_with_padding(DATA_path, participant, 5, "correlation")
    rdm_10 = compute_sliding_window_rdms_with_padding(DATA_path, participant, 10, "correlation")
    rdm_15 = compute_sliding_window_rdms_with_padding(DATA_path, participant, 15, "correlation")
    
    # Load the base RDM
    base_rdm = np.load(os.path.join(AVERAGE_RDM_DIR, f"sub-{participant}_metric-correlation_rdms.npy"))
    
    # Compute correlations for the base RDM and the sliding window RDMs
    mean_base, corrs_base = corr_with_model(base_rdm, GIST_265)
    mean_3, corrs_3 = corr_with_model(rdm_3, GIST_265)
    mean_5, corrs_5 = corr_with_model(rdm_5, GIST_265)
    mean_10, corrs_10 = corr_with_model(rdm_10, GIST_265)
    
    # Store the correlation values for each participant
    all_corrs_base.append(corrs_base)
    all_corrs_3.append(corrs_3)
    all_corrs_5.append(corrs_5)
    all_corrs_10.append(corrs_10)

# Convert to numpy arrays and calculate mean across participants
all_corrs_base = np.array(all_corrs_base)
all_corrs_3 = np.array(all_corrs_3)
all_corrs_5 = np.array(all_corrs_5)
all_corrs_10 = np.array(all_corrs_10)

mean_corrs_base = np.mean(all_corrs_base, axis=0)
mean_corrs_3 = np.mean(all_corrs_3, axis=0)
mean_corrs_5 = np.mean(all_corrs_5, axis=0)
mean_corrs_10 = np.mean(all_corrs_10, axis=0)


# Compute the standard error of the mean (SEM) for each condition
n_participants = len(participants_list)

sem_corrs_base = np.std(all_corrs_base, axis=0) / np.sqrt(n_participants)
sem_corrs_3 = np.std(all_corrs_3, axis=0) / np.sqrt(n_participants)
sem_corrs_5 = np.std(all_corrs_5, axis=0) / np.sqrt(n_participants)
sem_corrs_10 = np.std(all_corrs_10, axis=0) / np.sqrt(n_participants)

fig, ax = plt.subplots(figsize=(4, 4))

ax.axvline(x=0, color='lightgray', linestyle='--')
ax.axhline(y=0, color='lightgray', linestyle='--')

# Plot the average correlations
ax.plot(t, mean_corrs_base, color="#13594e", label="n = 0 (Base)")
#ax.fill_between(t[:len(mean_corrs_base)], mean_corrs_base - sem_corrs_base, mean_corrs_base + sem_corrs_base, 
  #              color="#13594e", alpha=0.1)
ax.plot(t[:len(mean_corrs_3)], mean_corrs_3, color="#1d8676", label="n = 3")
#ax.fill_between(t[:len(mean_corrs_3)], mean_corrs_3 - sem_corrs_3, mean_corrs_3 + sem_corrs_3, 
 #               color="#1d8676", alpha=0.1)
ax.plot(t[:len(mean_corrs_5)], mean_corrs_5, color="#59e5d0", label="n = 5")
#ax.fill_between(t[:len(mean_corrs_5)], mean_corrs_5 - sem_corrs_5, mean_corrs_5 + sem_corrs_5, 
               # color="#59e5d0", alpha=0.1)
#ax.plot(t[:len(mean_corrs_10)], mean_corrs_10, color="#59e5d0", label="n = 10")

# Add axis labels and title
ax.set_xlabel("Time (s)", fontsize=12)
ax.set_ylabel("Spearman Correlation", fontsize=12)
#ax.set_title("Average Correlation across Participants", fontsize=14)

# Add a legend
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# Display the plot
plt.tight_layout()
plt.savefig("/home/clemens-uva/Desktop/EEG_Temporal_misalignment/01_Affordance_vs_Gist/Figures/Supplementary_Sliding_window_figure.svg", transparent = True, dpi = 300)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import mne

mne.set_log_level('WARNING')

# Initialize lists to store correlation values for each participant
all_corrs_base = []
all_corrs_3 = []
all_corrs_5 = []
all_corrs_10 = []

for participant in participants_list:
    # Compute RDMs with sliding windows of different sizes
    rdm_3 = compute_sliding_window_rdms_with_padding(DATA_path, participant, 5, "euclidean")
    rdm_5 = compute_sliding_window_rdms_with_padding(DATA_path, participant, 5, "correlation")
    rdm_10 = compute_sliding_window_rdms_with_padding(DATA_path, participant, 5, "mahalanobis")
    
    # Load the base RDM
    base_rdm = np.load(os.path.join(AVERAGE_RDM_DIR, f"sub-{participant}_metric-correlation_rdms.npy"))
    
    # Compute correlations for the base RDM and the sliding window RDMs
    mean_base, corrs_base = corr_with_model(base_rdm, GIST_265)
    mean_3, corrs_3 = corr_with_model(rdm_3, GIST_265)
    mean_5, corrs_5 = corr_with_model(rdm_5, GIST_265)
    mean_10, corrs_10 = corr_with_model(rdm_10, GIST_265)
    
    # Store the correlation values for each participant
    all_corrs_base.append(corrs_base)
    all_corrs_3.append(corrs_3)
    all_corrs_5.append(corrs_5)
    all_corrs_10.append(corrs_10)

# Convert to numpy arrays and calculate mean across participants
all_corrs_base = np.array(all_corrs_base)
all_corrs_3 = np.array(all_corrs_3)
all_corrs_5 = np.array(all_corrs_5)
all_corrs_10 = np.array(all_corrs_10)

# Compute the mean for each condition
mean_corrs_base = np.mean(all_corrs_base, axis=0)
mean_corrs_3 = np.mean(all_corrs_3, axis=0)
mean_corrs_5 = np.mean(all_corrs_5, axis=0)
mean_corrs_10 = np.mean(all_corrs_10, axis=0)

# Compute the standard error of the mean (SEM) for each condition
n_participants = len(participants_list)

sem_corrs_base = np.std(all_corrs_base, axis=0) / np.sqrt(n_participants)
sem_corrs_3 = np.std(all_corrs_3, axis=0) / np.sqrt(n_participants)
sem_corrs_5 = np.std(all_corrs_5, axis=0) / np.sqrt(n_participants)
sem_corrs_10 = np.std(all_corrs_10, axis=0) / np.sqrt(n_participants)



fig, ax = plt.subplots(figsize=(5, 5))

# Add a reference grid with dashed lines
ax.axvline(x=0, color='lightgray', linestyle='--')
ax.axhline(y=0, color='lightgray', linestyle='--')

# Plot the mean correlations with SEM
ax.plot(t[:len(mean_corrs_3)], mean_corrs_3, color="#0974f1", label="Euclidean", lw=1)
ax.fill_between(t[:len(mean_corrs_3)], mean_corrs_3 - sem_corrs_3, mean_corrs_3 + sem_corrs_3, 
                color="#0974f1", alpha=0.3)

ax.plot(t[:len(mean_corrs_10)], mean_corrs_10, color="#103783", label="Mahalanobis", lw=1)
ax.fill_between(t[:len(mean_corrs_10)], mean_corrs_10 - sem_corrs_10, mean_corrs_10 + sem_corrs_10, 
                color="#103783", alpha=0.3)

ax.plot(t[:len(mean_corrs_5)], mean_corrs_5, color="#99f2d1", label="Correlation", lw=2)
ax.fill_between(t[:len(mean_corrs_5)], mean_corrs_5 - sem_corrs_5, mean_corrs_5 + sem_corrs_5, 
                color="#99f2d1", alpha=0.5)

# Add axis labels and title
ax.set_xlabel("Time (s)", fontsize=12)
ax.set_ylabel("Spearman Correlation", fontsize=12)
# ax.set_title("Average Correlation across Participants", fontsize=14)

# Add a legend
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# Display the plot with a tight layout
plt.tight_layout()
plt.title("GIST")

# Save the figure to a file
plt.savefig("/home/clemens-uva/Desktop/EEG_Temporal_misalignment/01_Affordance_vs_Gist/Figures/Supplementary_distance_metrics_figure_GIST.svg", transparent=True, dpi=300)

# Show the plot
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import mne

mne.set_log_level('WARNING')

# Initialize lists to store correlation values for each participant
all_corrs_base = []
all_corrs_3 = []
all_corrs_5 = []
all_corrs_10 = []

for participant in participants_list:
    # Compute RDMs with sliding windows of different sizes
    rdm_3 = compute_sliding_window_rdms_with_padding(DATA_path, participant, 5, "euclidean")
    rdm_5 = compute_sliding_window_rdms_with_padding(DATA_path, participant, 5, "correlation")
    rdm_10 = compute_sliding_window_rdms_with_padding(DATA_path, participant, 5, "mahalanobis")
    
    # Load the base RDM
    base_rdm = np.load(os.path.join(AVERAGE_RDM_DIR, f"sub-{participant}_metric-correlation_rdms.npy"))
    
    # Compute correlations for the base RDM and the sliding window RDMs
    mean_base, corrs_base = corr_with_model(base_rdm, action_eeg_rdm)
    mean_3, corrs_3 = corr_with_model(rdm_3, action_eeg_rdm)
    mean_5, corrs_5 = corr_with_model(rdm_5, action_eeg_rdm)
    mean_10, corrs_10 = corr_with_model(rdm_10, action_eeg_rdm)
    
    # Store the correlation values for each participant
    all_corrs_base.append(corrs_base)
    all_corrs_3.append(corrs_3)
    all_corrs_5.append(corrs_5)
    all_corrs_10.append(corrs_10)

# Convert to numpy arrays and calculate mean across participants
all_corrs_base = np.array(all_corrs_base)
all_corrs_3 = np.array(all_corrs_3)
all_corrs_5 = np.array(all_corrs_5)
all_corrs_10 = np.array(all_corrs_10)

# Compute the mean for each condition
mean_corrs_base = np.mean(all_corrs_base, axis=0)
mean_corrs_3 = np.mean(all_corrs_3, axis=0)
mean_corrs_5 = np.mean(all_corrs_5, axis=0)
mean_corrs_10 = np.mean(all_corrs_10, axis=0)

# Compute the standard error of the mean (SEM) for each condition
n_participants = len(participants_list)

sem_corrs_base = np.std(all_corrs_base, axis=0) / np.sqrt(n_participants)
sem_corrs_3 = np.std(all_corrs_3, axis=0) / np.sqrt(n_participants)
sem_corrs_5 = np.std(all_corrs_5, axis=0) / np.sqrt(n_participants)
sem_corrs_10 = np.std(all_corrs_10, axis=0) / np.sqrt(n_participants)

fig, ax = plt.subplots(figsize=(5, 5))

# Add a reference grid with dashed lines
ax.axvline(x=0, color='lightgray', linestyle='--')
ax.axhline(y=0, color='lightgray', linestyle='--')

# Plot the mean correlations with SEM
ax.plot(t[:len(mean_corrs_3)], mean_corrs_3, color="#0974f1", label="Euclidean", lw=1)
ax.fill_between(t[:len(mean_corrs_3)], mean_corrs_3 - sem_corrs_3, mean_corrs_3 + sem_corrs_3, 
                color="#0974f1", alpha=0.3)

ax.plot(t[:len(mean_corrs_10)], mean_corrs_10, color="#103783", label="Mahalanobis", lw=1)
ax.fill_between(t[:len(mean_corrs_10)], mean_corrs_10 - sem_corrs_10, mean_corrs_10 + sem_corrs_10, 
                color="#103783", alpha=0.3)

ax.plot(t[:len(mean_corrs_5)], mean_corrs_5, color="#99f2d1", label="Correlation", lw=2)
ax.fill_between(t[:len(mean_corrs_5)], mean_corrs_5 - sem_corrs_5, mean_corrs_5 + sem_corrs_5, 
                color="#99f2d1", alpha=0.5)

# Add axis labels and title
ax.set_xlabel("Time (s)", fontsize=12)
ax.set_ylabel("Spearman Correlation", fontsize=12)
# ax.set_title("Average Correlation across Participants", fontsize=14)

# Add a legend
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# Display the plot with a tight layout
plt.tight_layout()
plt.title("Affordance")

# Save the figure to a file

# Show the plot
plt.show()

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

#task = "action"
distance_metric = "correlation"

color1 = "#ff2c55"
color2 = "#0974f1"


line_1 = "-"
line_2 = "-"
alpha_line = 1
alpha_shades = 0.1
lw = 2

# create Figure
fig, ax = plt.subplots()

ax.axvline(x=0, color='lightgray', linestyle='--')
ax.axhline(y=0, color='lightgray', linestyle='--')


mean_action, sem_action, action_array = compute_corrs_sliding(distance_metric, "_5_", action_eeg_rdm)
mean_object, sem_object, object_array = compute_corrs_sliding(distance_metric, "_5_", object_eeg_rdm)



plt.plot(t, mean_action, color = color1, label="Affordances (EEG)", alpha = alpha_line, lw  = lw)
plt.fill_between(t, mean_action - sem_action, mean_action + sem_action, alpha=alpha_shades, color = color1)

plt.plot(t, mean_object, color = color2, label="Objects (EEG)", alpha = alpha_line, lw = lw)
plt.fill_between(t, mean_object - sem_object, mean_object + sem_object, alpha=alpha_shades, color = color2)


min_value = lowest_value((mean_action - sem_action), (mean_object - sem_object))

# test if correlation is different from zero
action_sig_timepoints = significant_against_zero(action_array)
for timepoint in action_sig_timepoints:
    plt.text(t[timepoint], min_value - 0.001, color = color1, s =".", fontsize=20)

object_sig_timepoints = significant_against_zero(object_array)
for timepoint in object_sig_timepoints:
    plt.text(t[timepoint], min_value - 0.003, color = color2, s =".", fontsize=20)


# pairwise comparison

pairwise = action_array - object_array
pairwise_sig_timepoints = significant_against_zero(pairwise)
for timepoint in pairwise_sig_timepoints:
    plt.text(t[timepoint], min_value - 0.004, color = "black", s =".", fontsize=20)

# plot peak timepoints
ax.axvline(x= t[np.argmax(mean_action)], color=color1, linestyle='--')
ax.axvline(x= t[np.argmax(mean_object)], color=color2, linestyle='--')



highest_value_action =  t[np.argmax(action_array, axis=1)]
highest_value_object = t[np.argmax(object_array, axis=1)]



t_stat, p_value_ttest = ttest_rel(highest_value_action, highest_value_object)
print(f"Paired t-test: t-statistic = {t_stat}, p-value = {p_value_ttest}")
# Check if p-value from t-test is below 0.05
if p_value_ttest < 0.05:

    ax.plot([t[np.argmax(mean_action)], t[np.argmax(mean_object)]], [0.065, 0.065], color="black", linestyle='-', lw = 2)
    ax.plot([t[np.argmax(mean_action)], t[np.argmax(mean_action)]], [0.063, 0.065], color="black", linestyle='-', lw = 2)
    ax.plot([t[np.argmax(mean_object)], t[np.argmax(mean_object)]], [0.063, 0.065], color="black", linestyle='-', lw = 2)
    ax.text((t[np.argmax(mean_action)] + t[np.argmax(mean_object)])/2, 0.066, '*', ha='center', fontsize=20)

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




plt.ylim(-0.02, 0.07)
plt.xlabel('Time (s)')
plt.ylabel('Spearman Correlation')
plt.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig("/home/clemens-uva/Desktop/EEG_Temporal_misalignment/01_Affordance_vs_Gist/Figures/Figure_2_panal_A.svg", transparent = True, dpi = 300)
plt.savefig("/home/clemens-uva/Desktop/EEG_Temporal_misalignment/01_Affordance_vs_Gist/Figures/Figure_2_panal_A.png", transparent = True, dpi = 300)
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

def mean_and_sem(list):
    mean = np.mean(np.array(list), axis = 0)
    sem = np.std(list, axis=0) / np.sqrt(len(list))

    return mean, sem

path = RDM_OUTPUT_DIR
distance_metric = "correlation"
path = RDM_OUTPUT_DIR

X1 = squareform(action_eeg_rdm.round(5))
X2 = squareform(object_eeg_rdm.round(5))

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

color1 = "#ff2c55"
color2 = "#0974f1"

new_t = t[:len(mean_y1)]

plt.plot(new_t, mean_y1, color = color1, label="Affordance (EEG)",  lw = lw)
plt.fill_between(new_t, mean_y1 - sem_y1, mean_y1 + sem_y1, alpha=0.1, color = color1)

plt.plot(new_t, mean_y2, color = color2, label="Objects (EEG)",  lw = lw)
plt.fill_between(new_t, mean_y2 - sem_y2, mean_y2 + sem_y2, alpha=0.1, color = color2)

#plt.plot(new_t, mean_y12, color = "black", label="Shared by all", linestyle = "--")
#plt.fill_between(new_t, mean_y12 - sem_y12, mean_y12 + sem_y12, alpha=0.3, color = "black")

min_value = lowest_value((mean_y1 - sem_y1), (mean_y2 - sem_y2))

# test if correlation is different from zero
mean_y1_sig_timepoints = significant_against_zero(np.array(full_y1))
for timepoint in mean_y1_sig_timepoints:
    plt.text(new_t[timepoint], min_value - 0.00005 , color = color1, s =".", fontsize=20)

# test if correlation is different from zero
mean_y2_sig_timepoints = significant_against_zero(np.array(full_y2))
for timepoint in mean_y2_sig_timepoints:
    plt.text(new_t[timepoint],  min_value - 0.00020, color = color2, s =".", fontsize=20)

# plot peak timepoints
ax.axvline(x= new_t[np.argmax(mean_y1)], color=color1, linestyle='--')
ax.axvline(x= new_t[np.argmax(mean_y2)], color=color2, linestyle='--')


t_stat, p_value_ttest = ttest_rel(highest_value_action, highest_value_object)
print(f"Paired t-test: t-statistic = {t_stat}, p-value = {p_value_ttest}")
# Check if p-value from t-test is below 0.05
if p_value_ttest < 0.05:

    #ax.plot([t[np.argmax(mean_y1)], t[np.argmax(mean_y2)]], [0.0058, 0.0058], color="black", linestyle='-')
    #ax.plot([t[np.argmax(mean_y1)], t[np.argmax(mean_y1)]], [0.0055, 0.0058], color="black", linestyle='-')
    #ax.plot([t[np.argmax(mean_y2)], t[np.argmax(mean_y2)]], [0.0055, 0.0058], color="black", linestyle='-')
    #ax.text((t[np.argmax(mean_y1)] + t[np.argmax(mean_y2)])/2, 0.0058, '*', ha='center', fontsize=12)

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


#plt.ylim(-0.001, 0.006)
plt.xlabel('Time (s)')
plt.ylabel('Unique Variance')
plt.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#plt.savefig("/home/clemens-uva/Desktop/EEG_Temporal_misalignment/01_Affordance_vs_Gist/Figures/Unique_variance_affordance_object.svg", transparent = True, dpi = 300)
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

import os

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

color1 = "#ff2c55"
color2 = "#0974f1"

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
mean_action, sem_action, action_array = compute_partial_corrs_sliding(distance_metric, "_5_", action_eeg_rdm, object_eeg_rdm)
mean_object, sem_object, object_array = compute_partial_corrs_sliding(distance_metric, "_5_", object_eeg_rdm, action_eeg_rdm)

# Plot action RDM with partial correlation
plt.plot(t, mean_action, color=color1, label="Affordance (partialed out objects)", alpha=alpha_line, lw=lw)
plt.fill_between(t, mean_action - sem_action, mean_action + sem_action, alpha=alpha_shades, color=color1)

# Plot object RDM with partial correlation
plt.plot(t, mean_object, color=color2, label="Objects (partialed out affordances)", alpha=alpha_line, lw=lw)
plt.fill_between(t, mean_object - sem_object, mean_object + sem_object, alpha=alpha_shades, color=color2)

min_value = lowest_value((mean_action - sem_action), (mean_object - sem_object))

# test if correlation is different from zero
action_sig_timepoints = significant_against_zero(action_array)
for timepoint in action_sig_timepoints:
    plt.text(t[timepoint], min_value - 0.001, color=color1, s=".", fontsize=20)

object_sig_timepoints = significant_against_zero(object_array)
for timepoint in object_sig_timepoints:
    plt.text(t[timepoint], min_value - 0.001, color=color2, s=".", fontsize=20)

# pairwise comparison
pairwise = action_array - object_array
pairwise_sig_timepoints = significant_against_zero(pairwise)
for timepoint in pairwise_sig_timepoints:
    plt.text(t[timepoint], min_value - 0.004, color="black", s=".", fontsize=20)

# plot peak timepoints
ax.axvline(x= t[np.argmax(mean_action)], color=color1, linestyle='--')
ax.axvline(x= t[np.argmax(mean_object)], color=color2, linestyle='--')

highest_value_action =  t[np.argmax(action_array, axis=1)]
highest_value_object = t[np.argmax(object_array, axis=1)]



t_stat, p_value_ttest = ttest_rel(highest_value_action, highest_value_object)
print(f"Paired t-test: t-statistic = {t_stat}, p-value = {p_value_ttest}")
# Check if p-value from t-test is below 0.05
if p_value_ttest < 0.05:

    ax.plot([t[np.argmax(mean_action)], t[np.argmax(mean_object)]], [0.065, 0.065], color="black", linestyle='-', lw = lw)
    ax.plot([t[np.argmax(mean_action)], t[np.argmax(mean_action)]], [0.063, 0.065], color="black", linestyle='-', lw = lw)
    ax.plot([t[np.argmax(mean_object)], t[np.argmax(mean_object)]], [0.063, 0.065], color="black", linestyle='-', lw = lw)
    ax.text((t[np.argmax(mean_action)] + t[np.argmax(mean_object)])/2, 0.066, '*', ha='center', fontsize=20)

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



plt.ylim(-0.02, 0.07)
plt.xlabel('Time (s)')
plt.ylabel('Partial Spearman Correlation')
plt.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig("/home/clemens-uva/Desktop/EEG_Temporal_misalignment/01_Affordance_vs_Gist/Figures/Figure_2_panel_B_partial_affordnace_object.svg", transparent=True, dpi=300)
plt.savefig("/home/clemens-uva/Desktop/EEG_Temporal_misalignment/01_Affordance_vs_Gist/Figures/Figure_2_panel_B_partial_affordnace_object.png", transparent=True, dpi=300)
plt.show()

task = "action"
distance_metric = "correlation"

color1 = "#ff2c55"
color2 = "#0974f1"
color3 = "#ee9b00"


line_1 = "-"
line_2 = "-"
alpha_line = 1
alpha_shades = 0.1
lw = 2

# create Figure
fig, ax = plt.subplots()

ax.axvline(x=0, color='lightgray', linestyle='--')
ax.axhline(y=0, color='lightgray', linestyle='--')


mean_action, sem_action, action_array = compute_corrs_sliding(distance_metric, "_5_", action_eeg_rdm.round(5))
mean_object, sem_object, object_array = compute_corrs_sliding(distance_metric, "_5_", object_eeg_rdm.round(5))
mean_GIST, sem_GIST, gist_array = compute_corrs_sliding(distance_metric, "_5_", GIST_265.round(5))



plt.plot(t, mean_action, color = color1, label="Affordances (EEG)", alpha = alpha_line, lw  = lw)
plt.fill_between(t, mean_action - sem_action, mean_action + sem_action, alpha=alpha_shades, color = color1)

#plt.plot(t, mean_object, color = color2, label="Object (EEG)", alpha = alpha_line, lw = lw)
#plt.fill_between(t, mean_object - sem_object, mean_object + sem_object, alpha=alpha_shades, color = color2)

plt.plot(t, mean_GIST, color = color3, label="GIST", alpha = alpha_line, lw = lw)
plt.fill_between(t, mean_GIST - sem_GIST, mean_GIST + sem_GIST, alpha=alpha_shades, color = color3)


min_value = lowest_value((mean_action - sem_action), (mean_object - sem_object))

# test if correlation is different from zero
action_sig_timepoints = significant_against_zero(action_array)
for timepoint in action_sig_timepoints:
    plt.text(t[timepoint], min_value - 0.0005, color = color1, s =".", fontsize=20)

#object_sig_timepoints = significant_against_zero(object_array)
#for timepoint in object_sig_timepoints:
#    plt.text(t[timepoint], min_value - 0.0020, color = color2, s =".", fontsize=15)

gist_sig_timepoints = significant_against_zero(gist_array)
for timepoint in gist_sig_timepoints:
    plt.text(t[timepoint], min_value - 0.0025, color = color3, s =".", fontsize=20)

# pairwise comparison

#pairwise = action_array - object_array
#pairwise_sig_timepoints = significant_against_zero(pairwise)
#for timepoint in pairwise_sig_timepoints:
    #plt.text(t[timepoint], min_value - 0.0035, color = "black", s =".", fontsize=15)

# plot peak timepoints
ax.axvline(x= t[np.argmax(mean_action)], color=color1, linestyle='--')
#ax.axvline(x= t[np.argmax(mean_object)], color=color2, linestyle='--')
ax.axvline(x= t[np.argmax(mean_GIST)], color=color3, linestyle='--')


highest_value_action =  t[np.argmax(action_array, axis=1)]
highest_value_GIST = t[np.argmax(gist_array, axis=1)]



t_stat, p_value_ttest = ttest_rel(highest_value_action, highest_value_GIST)
print(f"Paired t-test: t-statistic = {t_stat}, p-value = {p_value_ttest}")
# Check if p-value from t-test is below 0.05
if p_value_ttest < 0.05:

    ax.plot([t[np.argmax(mean_action)], t[np.argmax(mean_GIST)]], [0.065, 0.065], color="black", linestyle='-', lw = lw)
    ax.plot([t[np.argmax(mean_action)], t[np.argmax(mean_action)]], [0.063, 0.065], color="black", linestyle='-', lw = lw)
    ax.plot([t[np.argmax(mean_GIST)], t[np.argmax(mean_GIST)]], [0.063, 0.065], color="black", linestyle='-', lw = lw)
    ax.text((t[np.argmax(mean_action)] + t[np.argmax(mean_GIST)])/2, 0.066, '*', ha='center', fontsize=20)

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


plt.ylim(-0.02, 0.07)
plt.xlabel('Time (s)')
plt.ylabel('Spearman Correlation')
plt.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig("/home/clemens-uva/Desktop/EEG_Temporal_misalignment/01_Affordance_vs_Gist/Figures/Figure_2_panal_C.svg", transparent = True, dpi = 300)
plt.savefig("/home/clemens-uva/Desktop/EEG_Temporal_misalignment/01_Affordance_vs_Gist/Figures/Figure_2_panal_C.png", transparent = True, dpi = 300)
plt.show()

# create a function for variance partitioning

def VarPart_3_all(Y, X1, X2, X3):

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

    return y1, y2, y3, y12, y13, y23, y123

distance_metric = "correlation"

X1 = squareform(action_eeg_rdm.round(5))
X2 = squareform(object_eeg_rdm.round(5))
X3 = squareform(GIST_265.round(5))

full_y1 = []
full_y2 = []
full_y3 = []
full_y12 = []
full_y13 = []
full_y23 = []
full_y123 = []

for file in os.listdir(path):
        if (distance_metric in file) and ("_5_" in file):
            rdms_per_subject = np.load(os.path.join(path, file))
            all_y1 = []
            all_y2 = []
            all_y3 = []
            all_y12 = []
            all_y13 = []
            all_y23 = []
            all_y123 = []
            for timepoint in rdms_per_subject:
                    Y = squareform(timepoint.round(10))
                    y1, y2, y3, y12, y13, y23, y123 = VarPart_3_all(Y, X1, X2, X3)
                    all_y1.append(y1)
                    all_y2.append(y2)
                    all_y3.append(y3)
                    all_y12.append(y12)
                    all_y13.append(y13)
                    all_y23.append(y23)
                    all_y123.append(y123)
            #full_y1.append(all_y1)
            #full_y2.append(all_y2)
            #full_y3.append(all_y3)
            full_y1.append(all_y1 - np.mean(all_y1[:13]))
            full_y2.append(all_y2 - np.mean(all_y2[:13]))
            full_y3.append(all_y3 - np.mean(all_y3[:13]))
            full_y12.append(all_y12 - np.mean(all_y12[:13]))
            full_y13.append(all_y13 - np.mean(all_y13[:13]))
            full_y23.append(all_y23 - np.mean(all_y23[:13]))
            full_y123.append(all_y123)

mean_y1, sem_y1 = mean_and_sem(full_y1)
mean_y2, sem_y2 = mean_and_sem(full_y2)
mean_y3, sem_y3 = mean_and_sem(full_y3)
mean_y12, sem_y12 = mean_and_sem(full_y12)
mean_y13, sem_y13 = mean_and_sem(full_y13)
mean_y23, sem_y23 = mean_and_sem(full_y23)
mean_y123, sem_y123 = mean_and_sem(full_y123)

# create Figure
fig, ax = plt.subplots()

ax.axvline(x=0, color='lightgray', linestyle='--')
ax.axhline(y=0, color='lightgray', linestyle='--')

color1 = "#ff2c55"
color2 = "#0974f1"
color3 = "#ee9b00"


line_1 = "-"
line_2 = "-"
line_3 = "-"
alpha_line = 1
alpha_shades = 0.1
lw = 1

new_t = t[:len(mean_y1)]

min_value = 0

plt.plot(new_t, mean_y1, color = color1, label="Affordance (EEG)",  lw = lw)
plt.fill_between(new_t, mean_y1 - sem_y1, mean_y1 + sem_y1, alpha=0.1, color = color1)

plt.plot(new_t, mean_y2, color = color2, label="Object (EEG)",  lw = lw)
plt.fill_between(new_t, mean_y2 - sem_y2, mean_y2 + sem_y2, alpha=0.1, color = color2)

plt.plot(new_t, mean_y3, color = color3, label="GIST", lw = 1)
plt.fill_between(new_t, mean_y3 - sem_y3, mean_y3 + sem_y3, alpha=0.1, color = color3)

plt.plot(new_t, mean_y123, color = "black", label="shared by all", linestyle = "--", lw = 1)
plt.fill_between(new_t, mean_y123 - sem_y123, mean_y123 + sem_y123, alpha=0.1, color = "black")

# test if correlation is different from zero
mean_y1_sig_timepoints = significant_against_zero(np.array(full_y1))
for timepoint in mean_y1_sig_timepoints:
    plt.text(new_t[timepoint], min_value - 0.00015 , color = color1, s =".", fontsize=15)

# test if correlation is different from zero
mean_y2_sig_timepoints = significant_against_zero(np.array(full_y2))
for timepoint in mean_y2_sig_timepoints:
    plt.text(new_t[timepoint],  min_value - 0.00025, color = color2, s =".", fontsize=15)

mean_y3_sig_timepoints = significant_against_zero(np.array(full_y3))
for timepoint in mean_y3_sig_timepoints:
    plt.text(new_t[timepoint],  min_value - 0.00035, color = color3, s =".", fontsize=15)

# plot peak timepoints
ax.axvline(x= new_t[np.argmax(mean_y1)], color=color1, linestyle='--')
ax.axvline(x= new_t[np.argmax(mean_y2)], color=color2, linestyle='--')
ax.axvline(x= new_t[np.argmax(mean_y3)], color=color3, linestyle='--')

highest_value_action =  new_t[np.argmax(full_y1, axis=1)]
highest_value_object = new_t[np.argmax(full_y2, axis=1)]

t_stat, p_value_ttest = ttest_ind(highest_value_action, highest_value_object)
print(f"Paired t-test: t-statistic = {t_stat}, p-value = {p_value_ttest}")
# Check if p-value from t-test is below 0.05
if p_value_ttest < 0.05:

    ax.plot([t[np.argmax(mean_y1)], t[np.argmax(mean_y2)]], [0.0058, 0.0058], color="black", linestyle='-')
    ax.plot([t[np.argmax(mean_y1)], t[np.argmax(mean_y1)]], [0.0055, 0.0058], color="black", linestyle='-')
    ax.plot([t[np.argmax(mean_y2)], t[np.argmax(mean_y2)]], [0.0055, 0.0058], color="black", linestyle='-')
    ax.text((t[np.argmax(mean_y1)] + t[np.argmax(mean_y2)])/2, 0.0058, '*', ha='center', fontsize=12)

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



#plt.ylim(-0.003, 0.040)
plt.xlabel('Time (s)')
plt.ylabel('Unique Variance', fontsize = 15)
plt.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#plt.savefig("/home/clemens-uva/Github_repos/EEG/EEG_final/Figures/final/brain_VPA.svg", transparent = True, dpi = 300)
plt.show()

distance_metric = "correlation"

X1 = squareform(action_eeg_rdm.round(5))
X2 = squareform(GIST_265.round(5))

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


color1 = "#ff2c55"
color2 = "#ee9b00"

new_t = t[:len(mean_y1)]

plt.plot(new_t, mean_y1, color = color1, label="Affordance (EEG)",  lw = lw)
plt.fill_between(new_t, mean_y1 - sem_y1, mean_y1 + sem_y1, alpha=0.1, color = color1)

plt.plot(new_t, mean_y2, color = color2, label="GIST",  lw = lw)
plt.fill_between(new_t, mean_y2 - sem_y2, mean_y2 + sem_y2, alpha=0.1, color = color2)

#plt.plot(new_t, mean_y12, color = "black", label="Shared by all", linestyle = "--")
#plt.fill_between(new_t, mean_y12 - sem_y12, mean_y12 + sem_y12, alpha=0.3, color = "black")

min_value = lowest_value((mean_y1 - sem_y1), (mean_y2 - sem_y2))

# test if correlation is different from zero
mean_y1_sig_timepoints = significant_against_zero(np.array(full_y1))
for timepoint in mean_y1_sig_timepoints:
    plt.text(new_t[timepoint], min_value - 0.00005 , color = color1, s =".", fontsize=20)

# test if correlation is different from zero
mean_y2_sig_timepoints = significant_against_zero(np.array(full_y2))
for timepoint in mean_y2_sig_timepoints:
    plt.text(new_t[timepoint],  min_value - 0.00020, color = color2, s =".", fontsize=20)

# plot peak timepoints
ax.axvline(x= new_t[np.argmax(mean_y1)], color=color1, linestyle='--')
ax.axvline(x= new_t[np.argmax(mean_y2)], color=color2, linestyle='--')



highest_value_action =  t[np.argmax(full_y1, axis=1)]
highest_value_object = t[np.argmax(full_y2, axis=1)]


t_stat, p_value_ttest = ttest_ind(highest_value_action, highest_value_object)
print(f"Paired t-test: t-statistic = {t_stat}, p-value = {p_value_ttest}")
# Check if p-value from t-test is below 0.05
if p_value_ttest < 0.05:

    ax.plot([t[np.argmax(mean_y1)], t[np.argmax(mean_y2)]], [0.0058, 0.0058], color="black", linestyle='-', lw = lw)
    ax.plot([t[np.argmax(mean_y1)], t[np.argmax(mean_y1)]], [0.0055, 0.0058], color="black", linestyle='-', lw = lw)
    ax.plot([t[np.argmax(mean_y2)], t[np.argmax(mean_y2)]], [0.0055, 0.0058], color="black", linestyle='-', lw = lw)
    ax.text((t[np.argmax(mean_y1)] + t[np.argmax(mean_y2)])/2, 0.0058, '*', ha='center', fontsize=20)

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


#plt.ylim(-0.001, 0.006)
plt.xlabel('Time (s)')
plt.ylabel('Unique Variance')
plt.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig("/home/clemens-uva/Desktop/EEG_Temporal_misalignment/01_Affordance_vs_Gist/Figures/Figure_2_panal_D_with_significant.svg", transparent = True, dpi = 300)
plt.savefig("/home/clemens-uva/Desktop/EEG_Temporal_misalignment/01_Affordance_vs_Gist/Figures/Figure_2_panal_D_with_significant.png", transparent = True, dpi = 300)
plt.show()

#task = "action"
distance_metric = "correlation"

color1 = "#ff2c55"
color2 = "#ee9b00"

line_1 = "-"
line_2 = "-"
alpha_line = 1
alpha_shades = 0.1
lw = 1

# create Figure
fig, ax = plt.subplots()

ax.axvline(x=0, color='lightgray', linestyle='--')
ax.axhline(y=0, color='lightgray', linestyle='--')

# Compute partial correlation for action and object EEG RDMs
mean_action, sem_action, action_array = compute_partial_corrs_sliding(distance_metric, "_5_", action_eeg_rdm, GIST_265)
mean_object, sem_object, object_array = compute_partial_corrs_sliding(distance_metric, "_5_", GIST_265, action_eeg_rdm)

# Plot action RDM with partial correlation
plt.plot(t, mean_action, color=color1, label="Affordance (partialed out GIST)", alpha=alpha_line, lw=lw)
plt.fill_between(t, mean_action - sem_action, mean_action + sem_action, alpha=alpha_shades, color=color1)

# Plot object RDM with partial correlation
plt.plot(t, mean_object, color=color2, label="GIST (partialed out affordances)", alpha=alpha_line, lw=lw)
plt.fill_between(t, mean_object - sem_object, mean_object + sem_object, alpha=alpha_shades, color=color2)

min_value = lowest_value((mean_action - sem_action), (mean_object - sem_object))

# test if correlation is different from zero
action_sig_timepoints = significant_against_zero(action_array)
for timepoint in action_sig_timepoints:
    plt.text(t[timepoint], min_value - 0.001, color=color1, s=".", fontsize=15)

object_sig_timepoints = significant_against_zero(object_array)
for timepoint in object_sig_timepoints:
    plt.text(t[timepoint], min_value - 0.003, color=color2, s=".", fontsize=15)

# pairwise comparison
pairwise = action_array - object_array
pairwise_sig_timepoints = significant_against_zero(pairwise)
for timepoint in pairwise_sig_timepoints:
    plt.text(t[timepoint], min_value - 0.004, color="black", s=".", fontsize=15)

# plot peak timepoints
ax.axvline(x= t[np.argmax(mean_action)], color=color1, linestyle='--')
ax.axvline(x= t[np.argmax(mean_object)], color=color2, linestyle='--')

print(t[np.argmax(mean_action)])
print(t[np.argmax(mean_object)])

highest_value_action =  t[np.argmax(action_array, axis=1)]
highest_value_object = t[np.argmax(object_array, axis=1)]



t_stat, p_value_ttest = ttest_rel(highest_value_action, highest_value_object)
print(f"Paired t-test: t-statistic = {t_stat}, p-value = {p_value_ttest}")
# Check if p-value from t-test is below 0.05
if p_value_ttest < 0.05:

    ax.plot([t[np.argmax(mean_action)], t[np.argmax(mean_object)]], [0.065, 0.065], color="black", linestyle='-')
    ax.plot([t[np.argmax(mean_action)], t[np.argmax(mean_action)]], [0.063, 0.065], color="black", linestyle='-')
    ax.plot([t[np.argmax(mean_object)], t[np.argmax(mean_object)]], [0.063, 0.065], color="black", linestyle='-')
    ax.text((t[np.argmax(mean_action)] + t[np.argmax(mean_object)])/2, 0.066, '*', ha='center', fontsize=12)

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

#plt.ylim(-0.015, 0.07)
plt.xlabel('Time (s)')
plt.ylabel('Partial Spearman Correlation')
plt.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig("/home/clemens-uva/Desktop/EEG_Temporal_misalignment/01_Affordance_vs_Gist/Figures/Supplementary_partial_affordance_GIST.svg", transparent=True, dpi=300)
plt.show()

# load behavior dataframes 
action_df = pd.read_csv(os.path.join(VISACT_BEHAVIOR_MEANS_DIR, 'mean_action_df.csv'), index_col=0)
material_df = pd.read_csv(os.path.join(VISACT_BEHAVIOR_MEANS_DIR, 'mean_material_df.csv'), index_col=0)
categories_df = pd.read_csv(os.path.join(VISACT_BEHAVIOR_MEANS_DIR, 'mean_categories_df.csv'), index_col=0)
attributes_df = pd.read_csv(os.path.join(VISACT_BEHAVIOR_MEANS_DIR, 'mean_attributes_df.csv'), index_col=0)
objects_df = pd.read_csv(os.path.join(VISACT_BEHAVIOR_MEANS_DIR, 'mean_objects_df.csv'), index_col=0)

index_list = []
for i in list(action_df.index):
    if "indoor" in i:
        num = i.split("_")[0]
        new_name = "indoor_" + num
    elif "outdoor_manmade" in i:
        num = i.split("_")[0]
        new_name = "outdoor_manmade_" + num
    elif "outdoor_natural" in i:
        num = i.split("_")[0]
        new_name = "outdoor_natural_" + num
    index_list.append(new_name) 

action_df.index = index_list
material_df.index = index_list
categories_df.index = index_list
attributes_df.index = index_list
objects_df.index = index_list

action_subset = action_df.loc[EEG_list_action_sorted]
material_subset = material_df.loc[EEG_list_action_sorted]
categories_subset = categories_df.loc[EEG_list_action_sorted]
attributes_subset = attributes_df.loc[EEG_list_action_sorted]
objects_subset = objects_df.loc[EEG_list_action_sorted]

from sklearn.metrics import pairwise_distances

action_RDM_corr = pairwise_distances(action_subset, metric='correlation')
material_RDM_corr = pairwise_distances(material_subset, metric='correlation')
categories_RDM_corr = pairwise_distances(categories_subset, metric='correlation')
attributes_RDM_corr = pairwise_distances(attributes_subset, metric='correlation')
objects_RDM_corr = pairwise_distances(objects_subset, metric='correlation')


action_RDM_euc = pairwise_distances(action_subset, metric='euclidean')
material_RDM_euc = pairwise_distances(material_subset, metric='euclidean')
categories_RDM_euc = pairwise_distances(categories_subset, metric='euclidean')
attributes_RDM_euc = pairwise_distances(attributes_subset, metric='euclidean')
objects_RDM_euc = pairwise_distances(objects_subset, metric='euclidean')

#task = "action"
distance_metric = "correlation"

color1 = "#ff2c55"
color2 = "#0974f1"


line_1 = "-"
line_2 = "-"
alpha_line = 1
alpha_shades = 0.1
lw = 1

# create Figure
fig, ax = plt.subplots()

ax.axvline(x=0, color='lightgray', linestyle='--')
ax.axhline(y=0, color='lightgray', linestyle='--')


mean_action, sem_action, action_array = compute_corrs_sliding(distance_metric, "_5_", action_RDM_corr)
mean_object, sem_object, object_array = compute_corrs_sliding(distance_metric, "_5_", objects_RDM_corr)



plt.plot(t, mean_action, color = color1, label="Affordances (online)", alpha = alpha_line, lw  = lw)
plt.fill_between(t, mean_action - sem_action, mean_action + sem_action, alpha=alpha_shades, color = color1)

plt.plot(t, mean_object, color = color2, label="Objects (online)", alpha = alpha_line, lw = lw)
plt.fill_between(t, mean_object - sem_object, mean_object + sem_object, alpha=alpha_shades, color = color2)


min_value = lowest_value((mean_action - sem_action), (mean_object - sem_object))

# test if correlation is different from zero
action_sig_timepoints = significant_against_zero(action_array)
for timepoint in action_sig_timepoints:
    plt.text(t[timepoint], min_value - 0.0005, color = color1, s =".", fontsize=15)

object_sig_timepoints = significant_against_zero(object_array)
for timepoint in object_sig_timepoints:
    plt.text(t[timepoint], min_value - 0.0020, color = color2, s =".", fontsize=15)


# pairwise comparison

pairwise = action_array - object_array
pairwise_sig_timepoints = significant_against_zero(pairwise)
for timepoint in pairwise_sig_timepoints:
    plt.text(t[timepoint], min_value - 0.0035, color = "black", s =".", fontsize=15)

# plot peak timepoints
ax.axvline(x= t[np.argmax(mean_action)], color=color1, linestyle='--')
ax.axvline(x= t[np.argmax(mean_object)], color=color2, linestyle='--')

highest_value_action =  t[np.argmax(action_array, axis=1)]
highest_value_object = t[np.argmax(object_array, axis=1)]

t_stat, p_value_ttest = ttest_rel(highest_value_action, highest_value_object)
print(f"Paired t-test: t-statistic = {t_stat}, p-value = {p_value_ttest}")
# Check if p-value from t-test is below 0.05
if p_value_ttest < 0.05:

    ax.plot([t[np.argmax(mean_action)], t[np.argmax(mean_object)]], [0.065, 0.065], color="black", linestyle='-')
    ax.plot([t[np.argmax(mean_action)], t[np.argmax(mean_action)]], [0.063, 0.065], color="black", linestyle='-')
    ax.plot([t[np.argmax(mean_object)], t[np.argmax(mean_object)]], [0.063, 0.065], color="black", linestyle='-')
    ax.text((t[np.argmax(mean_action)] + t[np.argmax(mean_object)])/2, 0.066, '*', ha='center', fontsize=12)

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

#plt.ylim(-0.015, 0.07)
plt.xlabel('Time (s)')
plt.ylabel('Spearman Correlation')
plt.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig("/home/clemens-uva/Desktop/EEG_Temporal_misalignment/01_Affordance_vs_Gist/Figures/Supplementary_Figure2_PanelA_Aff_obj_online_corr.svg", transparent = True, dpi = 300)
plt.show()

#task = "action"
distance_metric = "correlation"

color1 = "#ff2c55"
color2 = "#0974f1"


line_1 = "-"
line_2 = "-"
alpha_line = 1
alpha_shades = 0.1
lw = 1

# create Figure
fig, ax = plt.subplots()

ax.axvline(x=0, color='lightgray', linestyle='--')
ax.axhline(y=0, color='lightgray', linestyle='--')


mean_action, sem_action, action_array = compute_corrs_sliding(distance_metric, "_5_", action_RDM_euc.round(5))
mean_object, sem_object, object_array = compute_corrs_sliding(distance_metric, "_5_", objects_RDM_euc.round(5))



plt.plot(t, mean_action, color = color1, label="Affordances (online)", alpha = alpha_line, lw  = lw)
plt.fill_between(t, mean_action - sem_action, mean_action + sem_action, alpha=alpha_shades, color = color1)

plt.plot(t, mean_object, color = color2, label="Objects (online)", alpha = alpha_line, lw = lw)
plt.fill_between(t, mean_object - sem_object, mean_object + sem_object, alpha=alpha_shades, color = color2)


min_value = lowest_value((mean_action - sem_action), (mean_object - sem_object))

# test if correlation is different from zero
action_sig_timepoints = significant_against_zero(action_array)
for timepoint in action_sig_timepoints:
    plt.text(t[timepoint], min_value - 0.0005, color = color1, s =".", fontsize=15)

object_sig_timepoints = significant_against_zero(object_array)
for timepoint in object_sig_timepoints:
    plt.text(t[timepoint], min_value - 0.0020, color = color2, s =".", fontsize=15)


# pairwise comparison

pairwise = action_array - object_array
pairwise_sig_timepoints = significant_against_zero(pairwise)
for timepoint in pairwise_sig_timepoints:
    plt.text(t[timepoint], min_value - 0.0035, color = "black", s =".", fontsize=15)

# plot peak timepoints
ax.axvline(x= t[np.argmax(mean_action)], color=color1, linestyle='--')
ax.axvline(x= t[np.argmax(mean_object)], color=color2, linestyle='--')

highest_value_action =  t[np.argmax(action_array, axis=1)]
highest_value_object = t[np.argmax(object_array, axis=1)]

t_stat, p_value_ttest = ttest_rel(highest_value_action, highest_value_object)
print(f"Paired t-test: t-statistic = {t_stat}, p-value = {p_value_ttest}")
# Check if p-value from t-test is below 0.05
if p_value_ttest < 0.05:

    ax.plot([t[np.argmax(mean_action)], t[np.argmax(mean_object)]], [0.065, 0.065], color="black", linestyle='-')
    ax.plot([t[np.argmax(mean_action)], t[np.argmax(mean_action)]], [0.063, 0.065], color="black", linestyle='-')
    ax.plot([t[np.argmax(mean_object)], t[np.argmax(mean_object)]], [0.063, 0.065], color="black", linestyle='-')
    ax.text((t[np.argmax(mean_action)] + t[np.argmax(mean_object)])/2, 0.066, '*', ha='center', fontsize=12)

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


#plt.ylim(-0.015, 0.07)
plt.xlabel('Time (s)')
plt.ylabel('Spearman Correlation')
plt.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig("/home/clemens-uva/Desktop/EEG_Temporal_misalignment/01_Affordance_vs_Gist/Figures/Supplementary_Figure2_PanelB_Aff_obj_online_euc.svg", transparent = True, dpi = 300)
plt.show()

distance_metric = "correlation"

X1 = squareform(action_RDM_corr.round(5))
X2 = squareform(objects_RDM_corr.round(5))

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

color1 = "#ff2c55"
color2 = "#0974f1"

new_t = t[:len(mean_y1)]

plt.plot(new_t, mean_y1, color = color1, label="Affordance (online)",  lw = lw)
plt.fill_between(new_t, mean_y1 - sem_y1, mean_y1 + sem_y1, alpha=0.1, color = color1)

plt.plot(new_t, mean_y2, color = color2, label="Objects (online)",  lw = lw)
plt.fill_between(new_t, mean_y2 - sem_y2, mean_y2 + sem_y2, alpha=0.1, color = color2)

#plt.plot(new_t, mean_y12, color = "black", label="Shared by all", linestyle = "--")
#plt.fill_between(new_t, mean_y12 - sem_y12, mean_y12 + sem_y12, alpha=0.3, color = "black")

min_value = lowest_value((mean_y1 - sem_y1), (mean_y2 - sem_y2))

# test if correlation is different from zero
mean_y1_sig_timepoints = significant_against_zero(np.array(full_y1))
for timepoint in mean_y1_sig_timepoints:
    plt.text(new_t[timepoint], min_value - 0.00005 , color = color1, s =".", fontsize=15)

# test if correlation is different from zero
mean_y2_sig_timepoints = significant_against_zero(np.array(full_y2))
for timepoint in mean_y2_sig_timepoints:
    plt.text(new_t[timepoint],  min_value - 0.00020, color = color2, s =".", fontsize=15)

# plot peak timepoints
#ax.axvline(x= new_t[np.argmax(mean_y1)], color=color1, linestyle='--')
#ax.axvline(x= new_t[np.argmax(mean_y2)], color=color2, linestyle='--')


highest_value_action =  new_t[np.argmax(full_y1, axis=1)]
highest_value_object = new_t[np.argmax(full_y2, axis=1)]

t_stat, p_value_ttest = ttest_ind(highest_value_action, highest_value_object)
print(f"Paired t-test: t-statistic = {t_stat}, p-value = {p_value_ttest}")
# Check if p-value from t-test is below 0.05
if p_value_ttest < 0.05:

    ax.plot([t[np.argmax(mean_y1)], t[np.argmax(mean_y2)]], [0.0058, 0.0058], color="black", linestyle='-')
    ax.plot([t[np.argmax(mean_y1)], t[np.argmax(mean_y1)]], [0.0055, 0.0058], color="black", linestyle='-')
    ax.plot([t[np.argmax(mean_y2)], t[np.argmax(mean_y2)]], [0.0055, 0.0058], color="black", linestyle='-')
    ax.text((t[np.argmax(mean_y1)] + t[np.argmax(mean_y2)])/2, 0.0058, '*', ha='center', fontsize=12)

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


#plt.ylim(-0.001, 0.006)
plt.xlabel('Time (s)')
plt.ylabel('Unique Variance')
plt.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig("/home/clemens-uva/Desktop/EEG_Temporal_misalignment/01_Affordance_vs_Gist/Figures/Supplementary_Figure2_PanelC_uniqueVAR_Aff_obj_online_corr.svg", transparent = True, dpi = 300)
plt.show()

distance_metric = "correlation"

X1 = squareform(action_RDM_euc.round(5))
X2 = squareform(objects_RDM_euc.round(5))

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

color1 = "#ff2c55"
color2 = "#0974f1"

new_t = t[:len(mean_y1)]

plt.plot(new_t, mean_y1, color = color1, label="Affordance (online)",  lw = lw)
plt.fill_between(new_t, mean_y1 - sem_y1, mean_y1 + sem_y1, alpha=0.1, color = color1)

plt.plot(new_t, mean_y2, color = color2, label="Objects (online)",  lw = lw)
plt.fill_between(new_t, mean_y2 - sem_y2, mean_y2 + sem_y2, alpha=0.1, color = color2)

#plt.plot(new_t, mean_y12, color = "black", label="Shared by all", linestyle = "--")
#plt.fill_between(new_t, mean_y12 - sem_y12, mean_y12 + sem_y12, alpha=0.3, color = "black")

min_value = lowest_value((mean_y1 - sem_y1), (mean_y2 - sem_y2))

# test if correlation is different from zero
mean_y1_sig_timepoints = significant_against_zero(np.array(full_y1))
for timepoint in mean_y1_sig_timepoints:
    plt.text(new_t[timepoint], min_value - 0.00005 , color = color1, s =".", fontsize=15)

# test if correlation is different from zero
mean_y2_sig_timepoints = significant_against_zero(np.array(full_y2))
for timepoint in mean_y2_sig_timepoints:
    plt.text(new_t[timepoint],  min_value - 0.00020, color = color2, s =".", fontsize=15)

# plot peak timepoints
ax.axvline(x= new_t[np.argmax(mean_y1)], color=color1, linestyle='--')
ax.axvline(x= new_t[np.argmax(mean_y2)], color=color2, linestyle='--')

highest_value_action =  new_t[np.argmax(full_y1, axis=1)]
highest_value_object = new_t[np.argmax(full_y2, axis=1)]

t_stat, p_value_ttest = ttest_ind(highest_value_action, highest_value_object)
print(f"Paired t-test: t-statistic = {t_stat}, p-value = {p_value_ttest}")
# Check if p-value from t-test is below 0.05
if p_value_ttest < 0.05:

    ax.plot([t[np.argmax(mean_y1)], t[np.argmax(mean_y2)]], [0.0058, 0.0058], color="black", linestyle='-')
    ax.plot([t[np.argmax(mean_y1)], t[np.argmax(mean_y1)]], [0.0055, 0.0058], color="black", linestyle='-')
    ax.plot([t[np.argmax(mean_y2)], t[np.argmax(mean_y2)]], [0.0055, 0.0058], color="black", linestyle='-')
    ax.text((t[np.argmax(mean_y1)] + t[np.argmax(mean_y2)])/2, 0.0058, '*', ha='center', fontsize=12)

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


#plt.ylim(-0.001, 0.006)
plt.xlabel('Time (s)')
plt.ylabel('Unique Variance')
plt.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig("/home/clemens-uva/Desktop/EEG_Temporal_misalignment/01_Affordance_vs_Gist/Figures/Supplementary_Figure2_PanelC_uniqueVAR_Aff_obj_online_euc_with_highest.svg", transparent = True, dpi = 300)
plt.show()

# Define the RDMs and their properties
rdms = {
    "Affordances": {
        "data": action_RDM_corr,
        "color": "#ff2c55",
        "line_style": "-",
        "alpha_line": 1,
        "alpha_shades": 0.1
    },
    "Material": {
        "data": material_RDM_corr,
        "color": "#33a02c",
        "line_style": "-",
        "alpha_line": 1,
        "alpha_shades": 0.1
    },
    "Categories": {
        "data": categories_RDM_corr,
        "color": "#6a3d9a",
        "line_style": "-",
        "alpha_line": 1,
        "alpha_shades": 0.1
    },
    "Attributes": {
        "data": attributes_RDM_corr,
        "color": "#ff7f00",
        "line_style": "-",
        "alpha_line": 1,
        "alpha_shades": 0.1
    },
    "Objects": {
        "data": objects_RDM_corr,
        "color": "#0974f1",
        "line_style": "-",
        "alpha_line": 1,
        "alpha_shades": 0.1
    }
}

# Initialize the plot
fig, ax = plt.subplots()

# Axes settings
ax.axvline(x=0, color='lightgray', linestyle='--')
ax.axhline(y=0, color='lightgray', linestyle='--')

# Store all the min values to find the lowest one
height = -0.005

# Iterate over each RDM and plot the correlation
for label, properties in rdms.items():
    # Compute correlations
    mean_corr, sem_corr, corr_array = compute_corrs_sliding(distance_metric, "_5_", properties['data'])
    
    # Plot the correlation line
    plt.plot(t, mean_corr, color=properties['color'], label=label, alpha=properties['alpha_line'], lw=lw, linestyle=properties['line_style'])
    
    # Plot the shaded area for SEM
    plt.fill_between(t, mean_corr - sem_corr, mean_corr + sem_corr, alpha=properties['alpha_shades'], color=properties['color'])
    
   
    # Plot significance markers
    sig_timepoints = significant_against_zero(corr_array)
    for timepoint in sig_timepoints:
        plt.text(t[timepoint], height, color=properties['color'], s=".", fontsize=15)
    
    # Plot peak timepoints
    ax.axvline(x=t[np.argmax(mean_corr)], color=properties['color'], linestyle='--')

    height = height - 0.002

# Plot settings
plt.xlabel('Time (s)')
plt.ylabel('Spearman Correlation')
plt.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Save and show the plot
plt.savefig("/home/clemens-uva/Desktop/EEG_Temporal_misalignment/01_Affordance_vs_Gist/Figures/Supplementary_Figure2_all_spaces.svg", transparent=True, dpi=300)
plt.show()

action_fmri_behavior = np.load(os.path.join(MODEL_RDM_DIR, "fmri_action_rdm_EGG_action_sorted_correlation.npy"))
object_fmri_behavior = np.load(os.path.join(MODEL_RDM_DIR, "fmri_object_rdm_EGG_action_sorted_correlation.npy"))

action_online_behavior = np.load(os.path.join(MODEL_RDM_DIR, "online_action_rdm_EGG_action_sorted_correlation.npy"))
object_online_behavior = np.load(os.path.join(MODEL_RDM_DIR, "online_object_rdm_EGG_action_sorted_correlation.npy"))

eeg_behavior_action = np.load(os.path.join(BEHAVIOR_RDM_DIR, "action_average_RDM_correlation.npy"))
eeg_behavior_object = np.load(os.path.join(BEHAVIOR_RDM_DIR, "object_average_RDM_correlation.npy"))

#rdms = [action_eeg_rdm, object_eeg_rdm, action_RDM_euc, objects_RDM_euc, action_fmri, object_fmri]
#rdm_labels = ['Action EEG', 'Object EEG', 'Action RDM Euc', 'Object RDM Euc', 'Action fMRI', 'Object fMRI']
rdms = [eeg_behavior_action, action_online_behavior, action_fmri_behavior, eeg_behavior_object, object_online_behavior, object_fmri_behavior]
rdm_labels = ['Affordance (EEG)', 'Affordance (online)', 'Affordance (fMRI)','Objects (EEG)', 'Objects (online)', 'Objects (fMRI)']

num_rdms = len(rdms)
correlation_matrix = np.zeros((num_rdms, num_rdms))

for idx1, i in enumerate(rdms):
    for idx2, j in enumerate(rdms):
        corr, _ = spearmanr(squareform(i.round(5)), squareform(j.round(5)))
        correlation_matrix[idx1, idx2] = corr

plt.figure(figsize=(8, 6))
plt.imshow(correlation_matrix, cmap='coolwarm', vmin=0, vmax=1)
plt.colorbar(label='Spearman Correlation')

# Set the tick labels
plt.xticks(np.arange(num_rdms), rdm_labels, rotation=90, ha='center')
plt.yticks(np.arange(num_rdms), rdm_labels)

# Add the correlation values in each cell
for i in range(num_rdms):
    for j in range(num_rdms):
        if i !=j:
            plt.text(j, i, f'{correlation_matrix[i, j]:.2f}', ha='center', va='center', color='black')

plt.axhline(2.5, color = "white")
plt.axvline(2.5, color = "white")
# Show the plot
plt.tight_layout()
plt.savefig("/home/clemens-uva/Desktop/EEG_Temporal_misalignment/01_Affordance_vs_Gist/Figures/Supplemtary_Corr_overview_correlation.svg", transparent = True, dpi = 300)
plt.show()

action_fmri_behavior = np.load(os.path.join(MODEL_RDM_DIR, "fmri_action_rdm_EGG_action_sorted_euclidean.npy"))
object_fmri_behavior = np.load(os.path.join(MODEL_RDM_DIR, "fmri_object_rdm_EGG_action_sorted_euclidean.npy"))

action_online_behavior = np.load(os.path.join(MODEL_RDM_DIR, "online_action_rdm_EGG_action_sorted_euclidean.npy"))
object_online_behavior = np.load(os.path.join(MODEL_RDM_DIR, "online_object_rdm_EGG_action_sorted_euclidean.npy"))

eeg_behavior_action = np.load(os.path.join(BEHAVIOR_RDM_DIR, "action_average_RDM_euclidean.npy"))
eeg_behavior_object = np.load(os.path.join(BEHAVIOR_RDM_DIR, "object_average_RDM_euclidean.npy"))

#rdms = [action_eeg_rdm, object_eeg_rdm, action_RDM_euc, objects_RDM_euc, action_fmri, object_fmri]
#rdm_labels = ['Action EEG', 'Object EEG', 'Action RDM Euc', 'Object RDM Euc', 'Action fMRI', 'Object fMRI']
rdms = [eeg_behavior_action, action_online_behavior, action_fmri_behavior, eeg_behavior_object, object_online_behavior, object_fmri_behavior]
rdm_labels = ['Affordance (EEG)', 'Affordance (online)', 'Affordance (fMRI)','Objects (EEG)', 'Objects (online)', 'Objects (fMRI)']

num_rdms = len(rdms)
correlation_matrix = np.zeros((num_rdms, num_rdms))

for idx1, i in enumerate(rdms):
    for idx2, j in enumerate(rdms):
        corr, _ = spearmanr(squareform(i.round(5)), squareform(j.round(5)))
        correlation_matrix[idx1, idx2] = corr

plt.figure(figsize=(8, 6))
plt.imshow(correlation_matrix, cmap='coolwarm', vmin=0, vmax=1)
plt.colorbar(label='Spearman Correlation')

# Set the tick labels
plt.xticks(np.arange(num_rdms), rdm_labels, rotation=90, ha='center')
plt.yticks(np.arange(num_rdms), rdm_labels)

# Add the correlation values in each cell
for i in range(num_rdms):
    for j in range(num_rdms):
        if i !=j:
            plt.text(j, i, f'{correlation_matrix[i, j]:.2f}', ha='center', va='center', color='black')

plt.axhline(2.5, color = "white")
plt.axvline(2.5, color = "white")
# Show the plot
plt.tight_layout()
plt.savefig("/home/clemens-uva/Desktop/EEG_Temporal_misalignment/01_Affordance_vs_Gist/Figures/Supplemtary_Corr_overview_euclidean.svg", transparent = True, dpi = 300)
plt.show()
