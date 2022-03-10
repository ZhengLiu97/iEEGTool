# -*- coding: UTF-8 -*-
"""
@Project ：iEEGTool 
@File    ：thread.py
@Author  ：Barry
@Date    ：2022/2/18 2:07 
"""
import mne
import numpy as np
import pandas as pd

from mne.io import BaseRaw
from mne.time_frequency import AverageTFR
from PyQt5.QtCore import QThread, pyqtSignal

from utils.log_config import create_logger

logger = create_logger(filename='iEEGTool.log')


class ImportSEEG(QThread):
    # Thread for importing iEEG
    LOAD_SIGNAL = pyqtSignal(BaseRaw)

    def __init__(self, path):
        super(ImportSEEG, self).__init__()
        # 数据路径S
        self.path = path
        self.SEEG = None

    def run(self):
        # rewrite run method
        print(f'From {self.path} load iEEG')
        ieeg_fmt = self.path[self.path.rfind('.') + 1:]
        if ieeg_fmt == 'set':
            self.SEEG = mne.io.read_raw_eeglab(self.path, preload=True)
        elif ieeg_fmt == 'edf':
            self.SEEG = mne.io.read_raw_edf(self.path, preload=True)
        elif ieeg_fmt == 'fif':
            self.SEEG = mne.io.read_raw_fif(self.path, preload=True, verbose='error')
        elif ieeg_fmt == 'vhdr':
            self.SEEG = mne.io.read_raw_brainvision(self.path, preload=True)
        self.SEEG.set_channel_types({ch_name: 'seeg' for ch_name in self.SEEG.ch_names})
        self.LOAD_SIGNAL.emit(self.SEEG)


class ResampleiEEG(QThread):
    RESAMPLE_SIGNAL = pyqtSignal(BaseRaw)

    def __init__(self, ieeg, sampling_rate):
        super(ResampleiEEG, self).__init__()
        self.ieeg = ieeg
        self.sampling_rate = sampling_rate

    def run(self) -> None:
        self.ieeg.resample(self.sampling_rate, 2)
        self.RESAMPLE_SIGNAL.emit(self.ieeg)


class FIRFilter(QThread):
    IEEG_SIGNAL = pyqtSignal(object)

    def __init__(self, ieeg, lfreq, hfreq, notch_freqs, params):
        super(FIRFilter, self).__init__()

        self.ieeg = ieeg

        self.lfreq = lfreq
        self.hfreq = hfreq
        self.notch_freqs = notch_freqs
        self.params = params

    def run(self) -> None:
        if self.lfreq is not None or self.hfreq is not None:
            logger.info(f"Start running FIR filter {self.lfreq}-{self.hfreq}Hz")
            self.ieeg.filter(self.lfreq, self.hfreq, **self.params)
        if self.notch_freqs is not None:
            logger.info(f"Start running FIR notch filter at {self.notch_freqs[0]} "
                        f"with level {len(self.notch_freqs)}")
            self.ieeg.notch_filter(self.notch_freqs, **self.params)
        logger.info("Finish filtering")
        self.IEEG_SIGNAL.emit(self.ieeg)


class ComputeSpectralCon(QThread):
    _FINISH_SIGNAL = pyqtSignal(list)

    def __init__(self, data, params):
        super(CalculateSpectralCon, self).__init__()
        self.data = np.expand_dims(data, axis=0)
        self.methods = params['methods']
        self.sfreq = params['sfreq']
        self.fmin = params['fmin']
        self.fmax = params['fmax']
        self.faverage = params['faverage']
        self.mt_bandwidth = params['mt_bandwidth']
        self.mt_adaptive = params['mt_adaptive']
        self.mt_low_bias = params['mt_low_bias']
        self.block_size = params['block_size']
        self.use_morlet = params['use_morlet']

    def run(self) -> None:
        from mne_connectivity import spectral_connectivity
        mt_con = spectral_connectivity(self.data, mode='multitaper',
                                       method=self.methods,
                                       freq=self.sfreq,
                                       fmin=self.fmin, fmax=self.fmax,
                                       mt_adaptive=self.mt_adaptive,
                                       mt_bandwidth=self.mt_bandwidth,
                                       mt_low_bias=self.mt_low_bias,
                                       faverage=self.faverage,
                                       block_size=self.block_size)
        if self.use_morlet:
            cwt_freqs = np.arange(self.fmin, self.fmax, 1)
            cwt_n_cycles = cwt_freqs / 7.
            morlet_con = spectral_connectivity(self.data, mode='cwt_morlet',
                                               method=self.methods,
                                               sfreq=self.sfreq,
                                               cwt_freqs=cwt_freqs,
                                               cwt_n_cycles=cwt_n_cycles,
                                               block_size=self.block_size)
        else:
            morlet_con = None
            mt_con = mt_con[:, :, 0]
        self._FINISH_SIGNAL.emit([mt_con, morlet_con])

    @property
    def FINISH_SIGNAL(self):
        return self._FINISH_SIGNAL


class AlignCTMRI(QThread):
    _ALIGN_SIGNAL = pyqtSignal(list)

    def __init__(self, ct, t1, pipeline=None, mode='ants', transform='DenseRigid'):
        super(AlignCTMRI, self).__init__()
        self._ct = ct
        self._t1 = t1
        self._pipeline = pipeline
        self._mode = mode
        self._transform = transform

    def run(self) -> None:
        if self._mode == 'ants':
            import ants
            from dipy.align.imaffine import MutualInformationMetric
            from dipy.align.transforms import RigidTransform3D
            print('Start registration')
            result = ants.registration(fixed=self._t1, moving=self._ct, type_of_transform=self._transform)
            ct_aligned = ants.apply_transforms(fixed=self._t1, moving=self._ct,
                                               transformlist=result['fwdtransforms'],
                                               interpolator='linear')
            print('Start calculating Mutual Information')
            metric = MutualInformationMetric()
            transform = RigidTransform3D()

            static = self._t1.to_nibabel()
            move = self._ct.to_nibabel()
            move_align = ct_aligned.to_nibabel()

            metric.setup(transform, np.array(static.dataobj), np.array(move.dataobj),
                         static.affine, move.affine)
            metric._update_mutual_information(transform.get_identity_parameters())
            orig_mutual_info = metric.metric_val

            metric.setup(transform, np.array(static.dataobj), np.array(move_align.dataobj),
                         static.affine, move_align.affine)
            metric._update_mutual_information(transform.get_identity_parameters())
            align_mutual_info = metric.metric_val
        else:
            import mne
            reg_affine, _ = mne.transforms.compute_volume_registration(self._ct, self._t1,
                                                                       pipeline=self._pipeline)
            print('reg_affine: \n', reg_affine)
            ct_aligned = mne.transforms.apply_volume_registration(self._ct, self._t1, reg_affine)
        print(ct_aligned)
        self._ALIGN_SIGNAL.emit([ct_aligned, orig_mutual_info, align_mutual_info])


class ReconAll(QThread):
    _RECON_SIGNAL = pyqtSignal(int)

    def __init__(self, t1_path, subject_id, method):
        super(ReconAll, self).__init__()
        self._t1_path = t1_path
        self._subject_id = subject_id
        self._method = method
        self._output_dir = 'data/freesurfer/subjects'

    def run(self) -> None:
        import os
        if not os.path.exists(self._output_dir):
            os.mkdir(self._output_dir)
        # TODO recon-all
        result = 0
        _subject_id = self._subject_id
        index = self._t1_path.rfind('.')
        if self._method == 'recon-all':
            if self._t1_path[index+1:] not in ['nii', 'mgz', 'gz']:
                print(self._t1_path[index+1:])
                order = "mri_convert {:} {:}".format(self._t1_path, self._t1_path[:index]+'.mgz')
                print(order)
                result = os.system(order)
                print(result)
                if result:
                    self._RECON_SIGNAL.emit(result)
            if not os.path.exists(os.path.join(self._output_dir, _subject_id)):
                recon_order = "recon-all -i {:} -s {:}  -sd {:} -all -qcache -cw256".format(self._t1_path, self._subject_id,
                                                                       self._output_dir)
                print(recon_order)
                result = os.system(recon_order)
                print('Start recon-all')
            print('Finish recon-all')
            if not os.path.exists(os.path.join(self._output_dir, _subject_id, 'mri', 'aparc+aseg.vep.mgz')):
                create_vep_order = "bash VEP_atlas/create_vep_parc_without_reconall.sh"
                print(create_vep_order)
                result = os.system(create_vep_order)
            print('Finish create VEP atlas')
        elif self._method == 'fastsurfer':
            pass
        elif self._method =='deepcsr':
            pass
        result = not result
        self._RECON_SIGNAL.emit(result)


class RoiMapping(QThread):
    _COMPUTE_SIGNAL = pyqtSignal(list)

    def __init__(self, elec_df, seg, subject, subjects_dir, ct_aligned, calc_mni):
        super(RoiMapping, self).__init__()
        self._elec_df = elec_df
        frame = 'Surface_RAS' if 'Surface_RAS' in list(elec_df.columns) else 'World'
        self._mri_coord = np.asarray(elec_df[frame].to_list())
        self._seg = seg
        self._subject = subject
        self._subjects_dir = subjects_dir
        self._ct_aligned = ct_aligned
        self._calc_mni = calc_mni

    def run(self) -> None:
        import os
        import mne
        import nibabel as nib
        from mne.transforms import apply_trans
        from utils._roi_mapping import roi_mapping
        from utils._process import get_montage
        from utils._get_anatomy import get_aal_anatomy
        from utils._computed_mni import compute_mni

        subject = self._subject
        subjects_dir = self._subjects_dir

        ch_names = self._elec_df['Channel'].to_list()
        mri_coord = self._mri_coord / 1000.
        ch_pos = dict(zip(ch_names, mri_coord))

        # ch_pos_df = pd.DataFrame(dict(zip(ch_names, self._mri_coord)).items(), columns=['Channel', 'Surface_RAS'])
        montage_mri, montage_head = get_montage(ch_pos, subject, subjects_dir)

        roi_color_df, roi_df = roi_mapping(montage=montage_mri, parcellation=self._seg, subject_id=self._subject,
                                           fs_dir=self._subjects_dir)

        elec_coord_roi_df = pd.DataFrame()
        elec_coord_roi_df['Channel'] = self._elec_df['Channel']
        mri_coord_str = [','.join(str(item) for item in coord) for coord in self._mri_coord]
        elec_coord_roi_df['Surface_RAS'] = mri_coord_str
        elec_coord_roi_df['ROI'] = roi_color_df['ROI']
        elec_coord_roi_df['Color'] = roi_color_df['Color']

        if self._calc_mni:
            if self._ct_aligned is None:
                mni_coord, mri_mni_t = compute_mni(mri_coord*1000, subject, subjects_dir)
            else:
                mni_coord, mri_mni_t = compute_mni(mri_coord*1000, subject, subjects_dir, ct_aligned, montage_head)
            mni_coord_str = [','.join(str(item) for item in coord) for coord in mni_coord]
            elec_coord_roi_df['MNI'] = mni_coord_str
            aal_anatomy = get_aal_anatomy(ch_names, self._mri_coord, replace_bad='Not found')
            elec_coord_roi_df['AAL3'] = aal_anatomy['AAL3'].to_list()

        self._COMPUTE_SIGNAL.emit([elec_coord_roi_df, roi_df, mri_mni_t])


class ComputePSD(QThread):
    PSD_SIGNAL = pyqtSignal(dict)

    def __init__(self, ieeg, compute_method, params):
        super(ComputePSD, self).__init__()
        self.ieeg = ieeg
        self.compute_method = compute_method
        self.params = params

    def run(self) -> None:
        from mne.time_frequency import psd_multitaper, psd_welch
        if self.compute_method == 'multitaper':
            print('Start Calculating Multitaper PSD')
            psd, freqs = psd_multitaper(self.ieeg, **self.params)
            print(psd.shape)
            print(freqs.shape)
            print('Finish Calculating Multitaper PSD')
            result = {'psd': psd, 'freqs': freqs}
            self.PSD_SIGNAL.emit(result)
        elif self.compute_method == 'welch':
            print('Start Calculating Welch PSD')
            psd, freqs = psd_welch(self.ieeg, **self.params)
            print('Finish Calculating Welch PSD')
            result = {'psd': psd, 'freqs': freqs}
            self.PSD_SIGNAL.emit(result)


class ComputeTFR(QThread):

    COMPUTE_SIGNAL = pyqtSignal(AverageTFR)

    def __init__(self, epoch, compute_method, params):
        super(ComputeTFR, self).__init__()
        self.epoch = epoch
        self.compute_method = compute_method
        self.params = params

    def run(self) -> None:
        from mne.time_frequency import tfr_multitaper, tfr_morlet, tfr_stockwell

        if self.compute_method == 'multitaper':
            logger.info('Start Calculating Multitaper Time-Frequency Response')
            power = tfr_multitaper(self.epoch, return_itc=False, **self.params)
            logger.info('Finish Calculating Multitaper Time-Frequency Response')
            self.COMPUTE_SIGNAL.emit(power)
        elif self.compute_method == 'morlet':
            logger.info('Start Calculating Morlet Time-Frequency Response')
            power = tfr_morlet(self.epoch, return_itc=False, **self.params)
            logger.info('Finish Calculating Morlet Time-Frequency Response')
            self.COMPUTE_SIGNAL.emit(power)
        elif self.compute_method == 'stockwell':
            logger.info('Start Calculating Stockwell Time-Frequency Response')
            power = tfr_stockwell(self.epoch, return_itc=False, **self.params)
            logger.info('Finish Calculating Stockwell Time-Frequency Response')
            self.COMPUTE_SIGNAL.emit(power)


class ComputeSpectralConnectivity(QThread):
    _COMPUTE_SIGNAL = pyqtSignal(object)

    def __init__(self, epoch, params):
        super(ComputeSpectralConnectivity, self).__init__()
        self._epoch = epoch
        self._params = params

    def run(self) -> None:
        from mne_connectivity import spectral_connectivity_epochs
        con = spectral_connectivity_epochs(self._epoch, **self._params)
        self._COMPUTE_SIGNAL.emit(con)


class ComputeHFOsRate(QThread):
    HFO_SIGNAL = pyqtSignal(object)

    def __init__(self, ieeg, detector):
        super(ComputeHFOsRate, self).__init__()
        self.ieeg = ieeg
        self.detector = detector

    def run(self) -> None:
        detector = self.detector.fit(self.ieeg)
        self.HFO_SIGNAL.emit(detector)


class ComputeEI(QThread):
    EI_SIGNAL = pyqtSignal(list)

    def __init__(self, ieeg, params):
        super(ComputeEI, self).__init__()
        self.ieeg = ieeg
        self._params = params

    def run(self) -> None:
        from utils.epi_index import calc_EI

        print('Start calculating EI')
        ei_df, U_n = calc_EI(self.ieeg, **self._params)
        print('Finish calculating EI')
        self.EI_SIGNAL.emit([ei_df, U_n])
