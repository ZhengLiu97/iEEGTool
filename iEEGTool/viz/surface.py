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
    import nibabel as nib

    coords, faces = nib.freesurfer.read_geometry(op.join(subjects_dir, subject, 'surf', f'{hemi}.{surf}'))

    face_nums = np.ones(faces.shape[0]) * 3
    faces = np.c_[face_nums, faces].astype(np.int32)

    if surf == 'inflated':
        coords = inflated_offset(coords, subject, subjects_dir, hemi)
    return coords, faces

def create_chs_sphere(ch_coords):
    sphere = []
    for ch_coord in ch_coords:
        sphere.append(pv.Sphere(radius=1., center=ch_coord))

    return sphere