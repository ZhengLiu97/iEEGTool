# -*- coding: UTF-8 -*-
"""
@Project ：iEEGTool 
@File    ：get_anatomical_labels.py
@Author  ：Barry
@Date    ：2022/3/8 23:24 
"""
import os
import os.path as op

import numpy as np
import nibabel as nib

from utils.freesurfer import read_freesurfer_lut, _get_lut


def apply_transform(tr, xyz, inverse=False):
    """Apply the transformation to coordinates.

    Parameters
    ----------
    tr : array_like
        The (4, 4) transformation array
    xyz : array_like
        Array of coordinates (e.g vertices, sEEG contacts etc.)
    inverse : bool | False
        Inverse transformation

    Returns
    -------
    xyz_m : array_like
        Transformed coordinates
    """
    assert tr.shape == (4, 4)
    assert xyz.shape[1] == 3
    n_xyz = xyz.shape[0]
    if inverse:
        tr = np.linalg.inv(tr)
    return tr.dot(np.c_[xyz, np.ones(n_xyz)].T).T[:, 0:-1]


def get_contact_label_vol(vol, tab_idx, tab_labels, xyz, radius=5.,
                          wm_idx=None, vs=None, bad_label='none'):
    """Get the label of a single contact in a volume.

    Parameters
    ----------
    vol : array_like
        The full volume that contains the indices (3D array)
    tab_idx : array_like
        Array of unique indices contained in the volume
    tab_labels : array_like
        Array of labels where each label is associated to the indices in
        `tab_idx`
    xyz : array_like
        Array of contacts' coordinates of shape (3,). The coordinates should be
        in the same voxel space as the volume
    radius : float | 5.
        Use the voxels that are contained in a sphere centered arround each
        contact
    wm_idx : array_like | None
        List of white matter indices
    vs : array_like | None
        Voxel sizes. Should be a list (or array) of length 3 describing the
        voxel sizes along the (x, y, z) axes.
    bad_label : string | 'none'
        Label to use for contacts that have no close roi

    Returns
    -------
    label : string
        Label associate to the contact's coordinates
    """
    assert len(tab_idx) == len(tab_labels)
    if tab_labels.ndim == 1:
        tab_labels = tab_labels.reshape(-1, 1)
    if vs is None:
        vs = [1, 1, 1]
    n_labs = tab_labels.shape[-1]
    bad_labels = np.full((1, n_labs), bad_label)
    vs_x, vs_y, vs_z = vs[0], vs[1], vs[2]
    # build distances along (x, y, z) axes
    d_x = int(np.round(radius / vs_x))
    d_y = int(np.round(radius / vs_y))
    d_z = int(np.round(radius / vs_z))
    # build the voxel mask (under `radius`)
    mask_x = np.arange(-d_x, d_x + 1)
    mask_y = np.arange(-d_y, d_y + 1)
    mask_z = np.arange(-d_z, d_z + 1)
    [x_m, y_m, z_m] = np.meshgrid(mask_x, mask_y, mask_z)
    mask_vol = np.sqrt(x_m ** 2 + y_m ** 2 + z_m ** 2) <= radius
    # get indices for selecting voxels under `radius`
    x_m = x_m[mask_vol] + int(np.round(xyz[0]))
    y_m = y_m[mask_vol] + int(np.round(xyz[1]))
    z_m = z_m[mask_vol] + int(np.round(xyz[2]))
    subvol_idx = vol[x_m, y_m, z_m]
    # get indices and number of voxels contained in the selected subvolume
    unique, counts = np.unique(subvol_idx, return_counts=True)
    if (len(unique) == 1) and (unique[0] == 0):
        return bad_labels  # skip 'Unknown' only
    else:
        # if there's 'Unknown' + something else, skip 'Unknown'
        counts[unique == 0] = 0
    # white matter indices
    if isinstance(wm_idx, list):
        for wm in wm_idx:
            if (len(unique) == 1) and (unique[0] == wm):
                return tab_labels[wm]
            else:
                counts[unique == wm] = 0
    # infer the label
    u_vol_idx = unique[counts.argmax()]
    is_index = tab_idx == u_vol_idx
    if is_index.any():
        return tab_labels[is_index, :][0].reshape(1, -1)
    else:
        return bad_labels


def labelling_contacts_vol_fs_mgz(subject, subjects_dir, xyz, radius=2., file='aseg',
                                  bad_label='Unknown', lut_path=None):
    """Labelling contacts using Freesurfer mgz volume.

    This function should be used with files like aseg.mgz, aparc+aseg.mgz
    or aparc.a2009s+aseg.mgz

    Parameters
    ----------
    subject : string
        Subject name (e.g 'subject_01')
    subjects_dir : string
        Path to the Freesurfer folder where subject are stored
    xyz : array_like
        Array of contacts' coordinates of shape (n_contacts, 3)
    radius : float | 5.
        Use the voxels that are contained in a sphere centered around each
        contact
    file : string | 'aseg'
        The volume to consider. Use either :

            * 'aseg'
            * 'aparc+aseg'
            * 'aparc.a2009s+aseg'
            * 'aparc+aseg.vep'
    bad_label : string | 'none'
        Label to use for contacts that have no close roi
    lut_path : str
        Path of ColorLut

    Returns
    -------
    labels : array_like
        Array of labels of shape (n_contacts,)
    """
    # -------------------------------------------------------------------------
    # build path to the volume file
    mri_path = op.join(subjects_dir, subject, 'mri')
    mgz_path = op.join(mri_path, f"{file}.mgz")
    print(mgz_path)
    if not op.isfile(mgz_path):
        raise IOError(f"File {mgz_path} doesn't exist in the /mri/ Freesurfer "
                      "subfolder.")
    n_contacts = xyz.shape[0]
    print(f"-> Localize {n_contacts} using {file}.mgz")
    # white matter indices
    wm_idx = [2, 41]

    # -------------------------------------------------------------------------
    # load volume and transformation
    arch = nib.load(mgz_path)
    vol = arch.get_data()
    tr = arch.header.get_vox2ras_tkr()
    vs = nib.affines.voxel_sizes(tr)
    # load freesurfer LUT table using mne
    if lut_path is None:
        lut = _get_lut()
    else:
        lut = _get_lut(lut_path)
    fs_labels = np.array(lut['name'])
    fs_idx = np.array(lut['id'])

    # -------------------------------------------------------------------------
    # transform coordinates into the voxel space
    xyz_tr = apply_transform(tr, xyz, inverse=True)
    # localize contacts
    labels = []
    for k in range(n_contacts):
        _lab = get_contact_label_vol(vol, fs_idx, fs_labels, xyz_tr[k, :],
                                     radius=radius, bad_label=bad_label,
                                     wm_idx=wm_idx, vs=vs)
        labels.append(_lab.item())

    return labels