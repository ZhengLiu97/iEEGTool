# -*- coding: UTF-8 -*-
"""
@Project ：EpiLocker 
@File    ：_epi_index.py
@Author  ：Barry
@Date    ：2022/1/11 22:55 
"""
import mne
import numpy as np
import pandas as pd

from scipy.signal import argrelextrema



def calc_psd_welch(raw, freqs, window=1, step=0.25):
    """Calculating PSD using welch
    Parameters
    ----------
    raw : mne.io.Raw
        raw data of SEEG
    freqs
    window : float | int
        window size of welch in second
    step : float
        percentage of window to move, should between 0 ~ 1

    Returns
    -------
    psds : ndarray, shape (n_channels, n_freqs, n_segments).
    """

    samples_per_seg = int(raw.info['sfreq'] * window)
    fmin = freqs[0]
    fmax = freqs[1]
    step_points = int(step * raw.info['sfreq'])
    overlap = raw.info['sfreq'] - step_points
    print(f"sampling rate {raw.info['sfreq']}")
    print(f"step {step}")
    print(f"overlap {overlap}")
    print(f"samples_per_seg {samples_per_seg}")
    psd, freqs = mne.time_frequency.psd_welch(raw, fmin=fmin, fmax=fmax, n_fft=samples_per_seg,
                                              n_per_seg=samples_per_seg,
                                              n_overlap=overlap, average=None, window='hamming')
    # psd_hann, freqs = mne.time_frequency.psd_welch(raw, fmin=fmin, fmax=fmax, n_fft=samples_per_seg,
    #                                                n_per_seg=samples_per_seg,
    #                                                n_overlap=overlap, average=None, window='hann')
    # psd += psd_hann
    # psd /= 2

    print(f'PSD shape = {psd.shape}')
    return psd, freqs



def calc_ER(raw, low=(4, 12), high=(12, 127), window=1, step=0.25):
    """Calculating energy ratio
    Parameters
    ----------
    raw : mne.io.Raw
        raw data of SEEG
    low : tuple | list  default (4, 12)
        low frequency band
    high : tuple | list  default (12, 127)
        high frequency band
    window : float | int  default 1
        window size to calculate PSD
    step : float  default 0.25
        step size to calculate PSD  should be 0 ~ 1
    Returns
    -------
    Energy ratio  shape (n_channels, n_segments)
    """
    lpsd, lfreqs = calc_psd_welch(raw, low, window, step)
    hpsd, hfreqs = calc_psd_welch(raw, high, window, step)
    # lpsd, lfreqs = calc_psd_multitaper(raw, low, window, step)
    # hpsd, hfreqs = calc_psd_multitaper(raw, high, window, step)

    lfreq_band = np.sum(lpsd, axis=1)
    hfreq_band = np.sum(hpsd, axis=1)
    print(f"lfreq_band shape {lfreq_band.shape}")
    print(f"hfreq_band shape {hfreq_band.shape}")

    ER = hfreq_band / lfreq_band
    return ER


def page_hinkley(data, bias=0.1, threshold=1):
    """Page-Hinkley algorithm for concept drift detection
    Parameters
    ----------
    data : np.ndarray
        data to detect concept drift
    bias : float  default 0.1
     bias factor for Page-Hinkley test
    threshold : float  default 1
        the concept drift threshold

    Returns
    -------
    drift_index : list
        index of the data when drifting
    U_n : list
        the cusum of data

    """
    bias = bias
    threshold = threshold

    x_mean = 0
    sample_count = 1
    x_sum = 0
    U_n = []
    U_n_min = []
    all_mean = []

    concept_drift = True
    drift_index = []

    for index, x in enumerate(data):
        if concept_drift:
            x_mean = 0
            sample_count = 1
            x_sum = 0
            U_n_min = []

        x_mean = x_mean + (x - x_mean) / float(sample_count)
        x_sum = x_sum + (x - x_mean - bias)
        U_n.append(x_sum)
        U_n_min.append(x_sum)
        all_mean.append(x_mean)

        concept_drift = False
        if x_sum - min(U_n_min) > threshold:
            concept_drift = True
            drift_index.append(index)

        sample_count += 1

    if not len(drift_index):
        drift_index = [np.nan]

    return drift_index, U_n


def calc_EI(raw, low=(4, 12), high=(12, 127), window=1, step=0.25,
            bias=0.1, threshold=1, tau=1, H=5):
    """
    Parameters
    ----------
    raw
    low
    high
    window
    step
    bias
    threshold
    tau
    H

    Returns
    -------

    Note
    EI is calculated as
    .. math:: EI_i=\frac{1}{N_{di} - N_0 + \tau}\sum_{n=N_{di}}^{N_{di}+H}ER[n],\quad \tau>0
    delta  δ < 3Hz
    theta  θ  3-7 Hz
    alpha  α  7-12 Hz
    beta   β  12-30 Hz
    gamma  γ  > 30 Hz
    """
    ch_names = raw.ch_names
    EI_window = int(H / step)
    ER = calc_ER(raw, low=low, high=high, window=window, step=step)

    start = window / 2

    ei_df = pd.DataFrame(columns=['Channel', 'detection_idx', 'detection_time', 'alarm_idx',
                                  'alarm_time', 'ER', 'norm_ER', 'EI', 'norm_EI'], dtype=np.float64)
    ei_df['Channel'] = ch_names

    drift_idx = []
    U_n = []
    for data in ER:
        result = page_hinkley(data, bias, threshold)
        drift_idx.append(result[0][0]) # only use the first drift
        U_n.append(result[1])
    U_n = np.asarray(U_n)

    # calculate the alarm time
    ei_df['alarm_idx'] = drift_idx
    ei_df['alarm_time'] = start + step * ei_df.alarm_idx

    # calculate the detection time
    detection_idx = [np.nan] * len(ch_names)
    for num, idx in enumerate(drift_idx):
        if not np.isnan(idx):
            detect_idx = argrelextrema(U_n[num, :idx + 1], np.less)[0]
            if len(detect_idx):
                detection_idx[num] = detect_idx[-1]
            else:
                detection_idx[num] = np.argmin(U_n[num, :idx + 1])
    ei_df['detection_idx'] = detection_idx
    ei_df['detection_time'] = start + step * ei_df.detection_idx

    ei_df['EI'] = np.zeros((len(ch_names)))
    N0 = ei_df.detection_time.min(skipna=True)

    for i in range(len(ch_names)):
        N_di = ei_df.detection_time[i]
        if not np.isnan(N_di):
            denom = N_di - N0 + tau
            N_di_idx = int(ei_df.detection_idx[i])
            end = int(N_di_idx + EI_window)
            if end > ER.shape[-1]:
                ei_df.loc[i, 'ER'] = np.sum(ER[i, N_di_idx:])
            else:
                ei_df.loc[i, 'ER'] = np.sum(ER[i, N_di_idx: end + 1])
            ei_df.loc[i, 'EI'] = ei_df.loc[i, 'ER'] / denom

    ER_max = ei_df['ER'].max()
    ei_df['norm_ER'] = ei_df.ER / ER_max

    EI_max = ei_df['EI'].max()
    ei_df['norm_EI'] = ei_df.EI / EI_max

    ei_df = ei_df.round(3)

    return ei_df, U_n




