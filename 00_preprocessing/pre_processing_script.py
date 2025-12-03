import os
import numpy as np
import mne
from mne.preprocessing import EOGRegression
from mne.defaults import HEAD_SIZE_DEFAULT
from mne.channels._standard_montage_utils import _read_theta_phi_in_degrees
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Base directories (edit these when you move data)
# ---------------------------------------------------------------------

# Root directory for OSF-shared data
OSF_DATA_DIR = "/home/clemens-uva/Desktop/DATA_OSF_external_storage"

# Current location of raw EEG data (e.g., on a server / cluster)
# When you move the raw data into OSF_DATA_DIR, change this to e.g.:
# RAW_DATA_BASE = os.path.join(OSF_DATA_DIR, "EEG_DATA", "Participants")
RAW_DATA_BASE = "/data/EEG/EEG_DATA/Participants"

# Directory where preprocessed FIF files will be saved
PREPROC_DIR = os.path.join(OSF_DATA_DIR, "preprocessed_data")
os.makedirs(PREPROC_DIR, exist_ok=True)

# Montage TSV path (so it also lives inside the OSF data tree)
os.makedirs(OSF_DATA_DIR, exist_ok=True)
MONTAGE_TSV = os.path.join(OSF_DATA_DIR, "chs.tsv")

# ---------------------------------------------------------------------
# Helper functions for paths
# ---------------------------------------------------------------------

def raw_bdf_path(participant: str, condition: str) -> str:
    """
    Construct the path to the raw .bdf file for a given participant/condition.

    Only this function (or RAW_DATA_BASE above) needs to change when
    the raw data location changes.
    """
    return os.path.join(
        RAW_DATA_BASE,
        participant,
        "raw",
        f"{participant}_{condition}.bdf",
    )


def preproc_save_path(participant: str, condition: str) -> str:
    """
    Construct the path to the preprocessed .fif file for a given
    participant/condition inside PREPROC_DIR.

    File name is set to match the original script:
        {participant}_{condition}_only_75_128_downsample_epo.fif
    """
    base_name = f"{participant}_{condition}_only_75_128_downsample_epo.fif"
    return os.path.join(PREPROC_DIR, base_name)


# ---------------------------------------------------------------------
# Event dictionary & stimulus names (unchanged)
# ---------------------------------------------------------------------

event_dict = {
    'indoor0156': 4033, 'indoor0282': 3852, 'indoor0270': 4064, 'indoor0272': 4007, 'indoor0066': 4023,
    'indoor0283': 3898, 'indoor0214': 3953, 'indoor0080': 4055, 'indoor0215': 3964, 'indoor0216': 3931,
    'indoor0146': 4074, 'indoor0221': 4045, 'indoor0235': 4071, 'indoor0212': 3960, 'indoor0058': 4047,
    'indoor0145': 3989, 'indoor0136': 4018, 'indoor0130': 4088, 'indoor0163': 3894, 'indoor0103': 4017,
    'indoor0100': 3842, 'indoor0055': 3858, 'indoor0021': 3888, 'indoor0266': 3853, 'indoor0025': 4062,
    'indoor0279': 4027, 'indoor0281': 3873, 'indoor0271': 4014, 'indoor0249': 4002, 'indoor0033': 4085,
    'outdoornatural0010': 4020, 'outdoornatural0009': 3981, 'outdoornatural0049': 3942, 'outdoornatural0008': 3903,
    'outdoornatural0052': 4076, 'outdoornatural0050': 4072, 'outdoornatural0132': 3914, 'outdoornatural0053': 3930,
    'outdoornatural0004': 3984, 'outdoornatural0207': 3997, 'outdoornatural0097': 4003, 'outdoornatural0261': 4056,
    'outdoornatural0011': 4075, 'outdoornatural0198': 4063, 'outdoornatural0128': 3971, 'outdoornatural0255': 3955,
    'outdoornatural0062': 3925, 'outdoornatural0246': 3994, 'outdoornatural0160': 3940, 'outdoornatural0091': 4030,
    'outdoornatural0104': 4000, 'outdoornatural0200': 3902, 'outdoornatural0273': 4043, 'outdoornatural0079': 3944,
    'outdoornatural0042': 3986, 'outdoornatural0034': 4061, 'outdoornatural0017': 3950, 'outdoornatural0023': 3859,
    'outdoornatural0252': 3870, 'outdoornatural0250': 3884, 'outdoormanmade0167': 4059, 'outdoormanmade0040': 3851,
    'outdoormanmade0110': 3841, 'outdoormanmade0117': 4077, 'outdoormanmade0030': 3891, 'outdoormanmade0258': 4081,
    'outdoormanmade0064': 3926, 'outdoormanmade0068': 4038, 'outdoormanmade0063': 3845, 'outdoormanmade0015': 3871,
    'outdoormanmade0257': 4078, 'outdoormanmade0032': 3878, 'outdoormanmade0256': 3918, 'outdoormanmade0220': 4052,
    'outdoormanmade0133': 4013, 'outdoormanmade0119': 3886, 'outdoormanmade0152': 4001, 'outdoormanmade0148': 4083,
    'outdoormanmade0155': 3899, 'outdoormanmade0157': 3843, 'outdoormanmade0175': 4048, 'outdoormanmade0173': 3907,
    'outdoormanmade0089': 3862, 'outdoormanmade0147': 4060, 'outdoormanmade0131': 3874, 'outdoormanmade0161': 3869,
    'outdoormanmade0154': 4041, 'outdoormanmade0165': 3854, 'outdoormanmade0276': 3976, 'outdoormanmade0149': 3866
}

images_name = [
    'indoor0156', 'indoor0282', 'indoor0270', 'indoor0272', 'indoor0066', 'indoor0283', 'indoor0214', 'indoor0080',
    'indoor0215', 'indoor0216', 'indoor0146', 'indoor0221', 'indoor0235', 'indoor0212', 'indoor0058', 'indoor0145',
    'indoor0136', 'indoor0130', 'indoor0163', 'indoor0103', 'indoor0100', 'indoor0055', 'indoor0021', 'indoor0266',
    'indoor0025', 'indoor0279', 'indoor0281', 'indoor0271', 'indoor0249', 'indoor0033', 'outdoornatural0010',
    'outdoornatural0009', 'outdoornatural0049', 'outdoornatural0008', 'outdoornatural0052', 'outdoornatural0050',
    'outdoornatural0132', 'outdoornatural0053', 'outdoornatural0004', 'outdoornatural0207', 'outdoornatural0097',
    'outdoornatural0261', 'outdoornatural0011', 'outdoornatural0198', 'outdoornatural0128', 'outdoornatural0255',
    'outdoornatural0062', 'outdoornatural0246', 'outdoornatural0160', 'outdoornatural0091', 'outdoornatural0104',
    'outdoornatural0200', 'outdoornatural0273', 'outdoornatural0079', 'outdoornatural0042', 'outdoornatural0034',
    'outdoornatural0017', 'outdoornatural0023', 'outdoornatural0252', 'outdoornatural0250', 'outdoormanmade0167',
    'outdoormanmade0040', 'outdoormanmade0110', 'outdoormanmade0117', 'outdoormanmade0030', 'outdoormanmade0258',
    'outdoormanmade0064', 'outdoormanmade0068', 'outdoormanmade0063', 'outdoormanmade0015', 'outdoormanmade0257',
    'outdoormanmade0032', 'outdoormanmade0256', 'outdoormanmade0220', 'outdoormanmade0133', 'outdoormanmade0119',
    'outdoormanmade0152', 'outdoormanmade0148', 'outdoormanmade0155', 'outdoormanmade0157', 'outdoormanmade0175',
    'outdoormanmade0173', 'outdoormanmade0089', 'outdoormanmade0147', 'outdoormanmade0131', 'outdoormanmade0161',
    'outdoormanmade0154', 'outdoormanmade0165', 'outdoormanmade0276', 'outdoormanmade0149'
]

# ---------------------------------------------------------------------
# Montage (electrode positions) – same as original script
# ---------------------------------------------------------------------

DATA = """
Fp1 -92 -72
AF7 -92 -54
AF3 -74 -65
F1 -50 -68
F3 -60 -51
F5 -115 72
F7 -92 -36
FT7 -92 -18
FC5 -72 -21
FC3 -50 -28
FC1 -32 -45
C1 -23 0
C3 -46 0
C5 -69 0
T7 -92 0
TP7 -92 18
CP5 -72 21
CP3 -50 28
CP1 -32 45
P1 -50 68
P3 -60 51
P5 -75 41
P7 -92 36
P9 -115 36
PO7 -92 54
PO3 -74 65
O1 -92 72
Iz 115 -90
Oz 92 -90
POz 69 -90
Pz 46 -90
CPz 23 -90
Fpz 92 90
Fp2 92 72
AF8 92 54
AF4 74 65
AFz 69 90
Fz 46 90
F2 50 68
F4 60 51
F6 115 -72
F8 92 36
FT8 92 18
FC6 72 21
FC4 50 28
FC2 32 45
FCz 23 90
Cz 0 0
C2 23 0
C4 46 0
C6 69 0
T8 92 0
TP8 92 -18
CP6 72 -21
CP4 50 -28
CP2 32 -45
P2 50 -68
P4 60 -51
P6 75 -41
P8 92 -36
P10 115 -36
PO8 92 -54
PO4 74 -65
O2 92 -72
Nz 115 90
LPA -115 0
RPA 115 0
"""

with open(MONTAGE_TSV, 'w') as fout:
    fout.write(DATA)

montage = _read_theta_phi_in_degrees(
    fname=MONTAGE_TSV,
    head_size=HEAD_SIZE_DEFAULT,
    fid_names=['Nz', 'LPA', 'RPA'],
    add_fiducials=False,
)

# ---------------------------------------------------------------------
# Participants & conditions
# ---------------------------------------------------------------------

participants_list = [
    'sapaj', 'ppnjn', 'azrfp', 'cuvfl', 'domdz', 'npcrj', 'hoxev', 'kuupm',
    'rxsrg', 'pflzs', 'kktpp', 'pyyor', 'liirj', 'qmrlx', 'pwixa', 'jpdoy',
    'hapql', 'ghldo', 'fgljq'
]

conditions = ['action', 'object', 'fixation']

# ---------------------------------------------------------------------
# Processing settings – EXACTLY as in your original code
# ---------------------------------------------------------------------

tmin, tmax = -0.1, 1.0
sample_rate = 2048
down_sample_rate = 128
f_low, f_high = 0.1, 30
f_notch1, f_notch2 = 50, 60
duration = tmax - tmin
n_timepoints = int(duration * down_sample_rate) + 1
t = np.linspace(tmin, tmax, n_timepoints)

reject_criteria = dict(eeg=3.6e-4)
flat_criteria = dict(eeg=50e-6)
n_channels = 64
reference = ['Mlef', 'Mrig']
eog = ['Left', 'Righ', 'Up', 'Down']
exc = ['EXG7', 'EXG8', 'GSR1', 'GSR2', 'Erg1', 'Erg2', 'Resp', 'Plet', 'Temp']

# ---------------------------------------------------------------------
# Preprocessing function – matched to your original Iris_preprocessing
# ---------------------------------------------------------------------

def Iris_preprocessing(participants, conditions):
    """
    Loop over participants × conditions, load raw BDF, preprocess,
    and save FIF to PREPROC_DIR if not already present.

    The preprocessing steps are matched to the original script:
      - BDF import with eog/misc/exclude
      - Mastoids reference
      - Notch (50 & 60 Hz) + 0.1–30 Hz band-pass
      - Montage
      - Epoching (-0.1 to 1.0 s), baseline (-0.1, 0)
      - Downsample to 128 Hz
      - EOGRegression + baseline
      - Manual Oz > 75 µV rejection
      - CSD transform
      - Save as *_only_75_128_downsample_epo.fif
    """
    for participant in participants:
        for condition in conditions:
            raw_path = raw_bdf_path(participant, condition)
            save_path = preproc_save_path(participant, condition)

            if os.path.isfile(save_path):
                print(f"[SKIP] Preprocessed file already exists: {save_path}")
                continue

            if not os.path.isfile(raw_path):
                print(f"[MISSING] Raw file not found: {raw_path}")
                continue

            print(f"[PROCESS] {participant} – {condition}")
            print(f"  Raw:  {raw_path}")
            print(f"  Save: {save_path}")

            try:
                # Import raw data (same as original)
                raw = mne.io.read_raw_bdf(
                    raw_path,
                    eog=eog,
                    misc=reference,
                    exclude=exc,
                    preload=True,
                )

                # Optional: keep this to exactly mirror your old code
                channel_names = raw.info['ch_names'][:n_channels]

                # Reference signal to mastoids
                raw.set_eeg_reference(reference)

                # Filters: notch then band-pass
                raw.notch_filter(freqs=(f_notch1, f_notch2))
                raw.filter(l_freq=f_low, h_freq=f_high)

                # Montage
                raw.set_montage(montage)

                # Events
                events = mne.find_events(raw)

                # Epochs
                epochs = mne.Epochs(
                    raw,
                    events,
                    baseline=(tmin, 0),
                    picks=['eeg', 'eog'],
                    event_id=event_dict,
                    tmin=tmin,
                    tmax=tmax,
                    preload=True,
                )
                epochs.resample(down_sample_rate, npad='auto')

                # EOG regression + baseline (same as original)
                model_plain = EOGRegression(picks='eeg', picks_artifact='eog').fit(epochs)
                epochs_clean_plain = model_plain.apply(epochs)
                epochs_clean_plain.apply_baseline(baseline=(tmin, 0))

                # Manual rejection based on "Oz" amplitude exceeding 75 µV
                # ORIGINAL CODE: used epochs.ch_names.index('Oz') for index
                oz_channel_index = epochs.ch_names.index('Oz')
                epochs_data = epochs_clean_plain.get_data(copy=True)
                epochs_to_drop = [
                    i for i, epoch in enumerate(epochs_data)
                    if (epoch[oz_channel_index].max() - epoch[oz_channel_index].min()) > 75e-6
                ]
                if len(epochs_to_drop) > 0:
                    print(f"  Dropping {len(epochs_to_drop)} epochs (Oz > 75 µV).")
                epochs_clean_plain.drop(
                    indices=epochs_to_drop,
                    reason='Oz amplitude exceeds 75 µV'
                )

                # Transform to current source density (CSD)
                epochs_clean_plain_csd = mne.preprocessing.compute_current_source_density(
                    epochs_clean_plain
                )

                # Save preprocessed epochs
                epochs_clean_plain_csd.save(save_path, overwrite=True)
                print(f"  Saved: {save_path}")

            except Exception as e:
                print(f"[ERROR] {participant} – {condition}: {e}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # mne.set_log_level('WARNING')
    Iris_preprocessing(participants_list, conditions)
