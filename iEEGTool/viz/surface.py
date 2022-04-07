# -*- coding: UTF-8 -*-
"""
@Project ：iEEGTool 
@File    ：surface.py
@Author  ：Barry
@Date    ：2022/3/16 17:00 
"""
import os.path as op
import numpy as np
import pyvista as pv
import nibabel as nib

from mne.transforms import apply_trans
from utils.marching_cubes import marching_cubes


def check_hemi(hemi):
    if hemi == 'Both':
        return ['lh', 'rh']
    elif hemi == 'Left':
        return 'lh'
    else:
        return 'rh'

def inflated_offset(coords, subject, subjects_dir, hemi):
    from mne.transforms import apply_trans, invert_transform
    from mne._freesurfer import read_talxfm
    from mne.coreg import fit_matched_points, _trans_from_params

    rigid = np.eye(4)
    xfm = read_talxfm(subject, subjects_dir)
    # XYZ+origin + halfway
    pts_tal = np.concatenate([np.eye(4)[:, :3], np.eye(3) * 0.5])
    pts_subj = apply_trans(invert_transform(xfm), pts_tal)
    # we fit with scaling enabled, but then discard it (we just need
    # the rigid-body components)
    params = fit_matched_points(pts_subj, pts_tal, scale=3, out='params')
    rigid[:] = _trans_from_params((True, True, False), params[:6])
    x_dir = rigid[0, :3]
    if hemi == 'lh':
        left_x_ = coords @ x_dir
        coords -= (np.max(left_x_) + 0) * x_dir
    else:
        right_x_ = coords @ x_dir
        coords -= (np.min(right_x_) + 0) * x_dir
    return coords

def read_fs_surface(subject, subjects_dir, hemi, surf='pial'):
    print(f'Loading surface files from {subject}/{subjects_dir}/surf/{hemi}.{surf}')
    coords, faces = nib.freesurfer.read_geometry(op.join(subjects_dir, subject, 'surf', f'{hemi}.{surf}'))

    face_nums = np.ones(faces.shape[0]) * 3
    faces = np.c_[face_nums, faces].astype(np.int32)

    if surf == 'inflated':
        coords = inflated_offset(coords, subject, subjects_dir, hemi)
    return coords, faces

def create_chs_sphere(ch_coords, radius=1.):
    sphere = []
    print("Creating Channels' sphere")
    for ch_coord in ch_coords:
        sphere.append(pv.Sphere(radius=radius, center=ch_coord))

    return sphere

def create_roi_surface(subject, subjects_dir, aseg, rois):
    from utils.freesurfer import read_freesurfer_lut

    aseg_file = aseg + '.mgz'
    aseg_path = op.join(subjects_dir, subject, 'mri', aseg_file)

    aseg_mgz = nib.load(aseg_path)
    aseg_data = np.asarray(aseg_mgz.dataobj)
    vox_mri_t = aseg_mgz.header.get_vox2ras_tkr()
    print(f'loading segment file from {aseg_file}')

    if 'vep' not in aseg:
        lut_name = 'utils/FreeSurferColorLUT.txt'
    else:
        lut_name = 'utils/VepFreeSurferColorLut.txt'

    lut, fs_colors = read_freesurfer_lut(lut_name)
    if isinstance(rois, str):
        rois = [rois]
    roi2color = {roi: fs_colors[roi][:-1] / 255 for roi in rois}
    idx = [lut[roi] for roi in rois]

    print('Running marching cubes')
    if len(idx):
        surfs, _ = marching_cubes(aseg_data, idx, smooth=0.85)

        roi_mesh_color = {}
        for roi, (verts, faces) in zip(rois, surfs):
            roi_color = roi2color[roi]
            verts = apply_trans(vox_mri_t, verts)
            nums = np.ones(faces.shape[0]) * 3
            faces = np.c_[nums, faces].astype(np.int32)
            roi_mesh_color[roi] = [pv.PolyData(verts, faces), roi_color]
        return roi_mesh_color
    else:
        return None

