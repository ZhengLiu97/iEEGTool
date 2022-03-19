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

import matplotlib.pyplot as plt
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector


class PageHinkley(BaseDriftDetector):
    """ Page-Hinkley method for concept drift detection.
    References
    ----------
    .. [1] E. S. Page. 1954. Continuous Inspection Schemes.
       Biometrika 41, 1/2 (1954), 100–115.

    Parameters
    ----------
    min_instances: int (default=30)
        The minimum number of instances before detecting change.
    delta: float (default=0.005)
        The delta factor for the Page Hinkley test.
    threshold: int (default=50)
        The change detection threshold (lambda).
    alpha: float (default=1 - 0.0001)
        The forgetting factor, used to weight the observed value
        and the mean.
    """

    def __init__(self, delta=0.1, threshold=1):
        super().__init__()
        self.delta = delta
        self.threshold = threshold
        self.x_mean = None
        self.sample_count = None
        self.sum = None
        self.U_n = []
        self.U_n_min = []
        self._mean = []
        self.reset()

    def reset(self):
        """ reset

        Resets the change detector parameters.

        """
        super().reset()
        self.sample_count = 1
        self.x_mean = 0.0
        self.sum = 0.0
        self.U_n_min = []

    def add_element(self, x):
        """ Add a new element to the statistics

        Parameters
        ----------
        x: numeric value
            The observed value, from which we want to detect the
            concept change.

        Notes
        -----
        After calling this method, to verify if change was detected, one
        should call the super method detected_change, which returns True
        if concept drift was detected and False otherwise.

        """
        if self.in_concept_change:
            self.reset()

        self.x_mean = self.x_mean + (x - self.x_mean) / float(self.sample_count)
        self._mean.append(self.x_mean)
        self.sum = self.sum + (x - self.x_mean - self.delta)
        self.U_n.append(self.sum)
        self.U_n_min.append(self.sum)
        self.sample_count += 1

        self.estimation = self.x_mean
        self.in_concept_change = False
        self.in_warning_zone = False

        self.delay = 0

        # print(f'sum {self.sum}  threshold {self.threshold}')
        if self.sum - min(self.U_n_min) > self.threshold:
            # print(self.sum - min(self.U_n_min))
            # print(self.threshold)
            self.in_concept_change = True


def calc_psd_multitaper(ieeg, freqs, window=1, step=0.25):
    fmin = freqs[0]
    fmax = freqs[1]

    start = 0
    samples_per_seg = int(ieeg.info['sfreq'] * window)
    step = samples_per_seg * step

    data = ieeg.get_data()
    sfreq = ieeg.info['sfreq']
    ch_len = data.shape[0]
    n_segs = int(data.shape[1] // step)
    multitaper_psd = np.zeros((ch_len, int(fmax - fmin) + 1, n_segs))
    print(multitaper_psd.shape)
    for i in range(n_segs):
        end = start + samples_per_seg
        if end > data.shape[-1]:
            return multitaper_psd, freqs
        seg_data = data[:, start: end]
        psd, freqs = mne.time_frequency.psd_array_multitaper(seg_data, sfreq=sfreq, fmin=fmin, fmax=fmax, adaptive=True,
                                                             n_jobs=10, verbose='error')
        multitaper_psd[:, :, i] = psd
        start = int(start + step)
    return multitaper_psd, freqs

def calc_psd_welch(raw, freqs, window=1, step=0.25):
    """Calculating PSD using welch morlet
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

    ER =hfreq_band[:, :-3] / lfreq_band[:, :-3]
    return ER


def page_hinkley(ch_names, ER, start, step, threshold=1, bias=1):
    """Calculating detection time and alarm time using Page-Hinkley algorithm
    Parameters
    ----------
    ch_names : list
        channels' name
    ER : np.array  shape (n_channels, n_segments)
        Energy ratio of channels
    start : float | int  default 0
        start time in second
    step : float
        step of PSD calculation
    threshold : float | int
        number of deviations to use in Page-Hinkley(lambda)
    bias: float | int
        the bias/delta factor for the Page Hinkley test

    Returns
    -------
    ei_df : pd.DataFrame
        DataFrame consists of Channel, detection_idx, detection_time, alarm_idx,
        alarm_time, ER, norm_ER, EI and norm_EI
    U_n : np.array shape (n_channels, step)
        cusum of ER in each step of channels
    """
    from scipy.signal import argrelextrema

    ei_df = pd.DataFrame(columns=['Channel', 'detection_idx', 'detection_time', 'alarm_idx',
                                  'alarm_time', 'ER', 'norm_ER', 'EI', 'norm_EI'], dtype=np.float64)
    ei_df['Channel'] = ch_names

    drift_idx = [np.nan] * len(ch_names)
    U_n = []
    for i in range(len(ch_names)):
        ch_drift_idx = []
        ph = PageHinkley(threshold=threshold, delta=bias)
        for j in range(len(ER[i, :])):
            ph.add_element(ER[i, j])
            if ph.detected_change():
                ch_drift_idx.append(j)
                # print(f'Change has been detected in data: {ER[i, j]} of index: {j} in channel {ch_names[i]}')
        # print('\n')
        drift_idx[i] = int(ch_drift_idx[0]) if len(ch_drift_idx) else np.nan
        U_n.append(ph.U_n)
    U_n = np.asarray(U_n)

    ei_df['alarm_idx'] = drift_idx
    ei_df['alarm_time'] = start + step * ei_df.alarm_idx

    detection_idx = [np.nan] * len(ch_names)
    for num, idx in enumerate(drift_idx):
        if not np.isnan(idx):
            detect_idx = argrelextrema(U_n[num, :idx+1], np.less)[0]
            if len(detect_idx):
                detection_idx[num] = detect_idx[-1]
            else:
                detection_idx[num] = np.argmin(U_n[num, :idx+1])
    ei_df['detection_idx'] = detection_idx
    ei_df['detection_time'] = start + step * ei_df.detection_idx
    print(f"The first time which gets the threshold is the {ei_df.alarm_time.min()}th second")

    return ei_df, U_n


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
    ei_df, U_n = page_hinkley(ch_names, ER=ER, start=start, step=step, threshold=threshold, bias=bias)

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



