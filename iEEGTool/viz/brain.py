# -*- coding: UTF-8 -*-
"""
@Project ：iEEGTool 
@File    ：brain.py
@Author  ：Barry
@Date    ：2022/3/16 15:27 
"""
import pyvista as pv
from pyvistaqt import QtInteractor

from viz.surface import read_fs_surface, create_chs_sphere, create_roi_surface
from utils.config import color, brain_kwargs, contact_kwargs, text_kwargs, roi_kwargs
from utils.process import get_chan_group


class Brain(QtInteractor):
    def __init__(self, parent=None):
        # Must Add this parent=None is this format
        # otherwise error shows up
        # for this widget would be added to a Layout
        # it needs a parent, no change of this!
        super(Brain, self).__init__(parent)
        self.background_color = 'w'
        self.enable_depth_peeling()
        self.enable_anti_aliasing()
        self.line_smoothing =True
        self.point_smoothing =True

        self.brain_surface = {}
        self.actors = {}
        self.legend_actors = {}

    def add_brain(self, subject, subjects_dir, hemi, surf, opacity):
        if isinstance(hemi, str):
            coords, faces = read_fs_surface(subject, subjects_dir, hemi, surf)
            self.brain_surface[hemi] = pv.PolyData(coords, faces)
        else:
            for h in hemi:
                coords, faces = read_fs_surface(subject, subjects_dir, h, surf)
                self.brain_surface[h] = pv.PolyData(coords, faces)
        brain_kwargs['opacity'] = opacity
        for h in self.brain_surface:
            self.actors[h] = self.add_mesh(self.brain_surface[h], name=h, **brain_kwargs)

    def enable_brain_viz(self, viz, hemi):
        if isinstance(hemi, list):
            [self.actors[name].SetVisibility(viz) for name in hemi]
        else:
            self.actors[hemi].SetVisibility(viz)

    def set_background_color(self, bk_color):
        self.set_background(color=bk_color)

    def set_brain_color(self, brain_color):
        hemi = ['lh', 'rh']
        [self.actors[name].GetProperty().SetColor(brain_color) for name in hemi]

    def set_brain_opacity(self, opacity):
        hemi = ['lh', 'rh']
        [self.actors[name].GetProperty().SetOpacity(opacity) for name in hemi]

    def set_brain_hemi(self, hemi):
        if isinstance(hemi, list):
            [self.actors[name].SetVisibility(True) for name in hemi]
        else:
            nviz_hemi = ['lh', 'rh']
            nviz_hemi.remove(hemi)
            self.actors[hemi].SetVisibility(True)
            self.actors[nviz_hemi[0]].SetVisibility(False)

    def add_chs(self, ch_names, ch_coords):
        spheres = create_chs_sphere(ch_coords)
        ch_pos = dict(zip(ch_names, ch_coords))
        ch_spheres = dict(zip(ch_names, spheres))
        ch_groups = get_chan_group(ch_names)
        ch_color = {}
        for index, group in enumerate(ch_groups):
            for ch_name in ch_groups[group]:
                ch_color[ch_name] = color[index]
        for ch_name in ch_spheres:
            self.actors[ch_name] = self.add_mesh(ch_spheres[ch_name], name=ch_name,
                                                 color=ch_color[ch_name], **contact_kwargs)
            self.actors[f'{ch_name} name'] = self.add_point_labels(ch_pos[ch_name] + 1, [ch_name],
                                                                   name=f'{ch_name} name',
                                                                   **text_kwargs)

        for group in ch_groups:
            ch_name = ch_groups[group][-1]
            group_label_pos = ch_pos[ch_name] + 1
            self.actors[group] = self.add_point_labels(group_label_pos, [group],
                                                       name=group, **text_kwargs)

    def enable_chs_viz(self, ch_names, viz):
        [self.actors[name].SetVisibility(viz) for name in ch_names]

    def enable_group_label_viz(self, group, viz):
        [self.actors[name].SetVisibility(viz) for name in group]

    def enable_ch_name_viz(self, ch_names, viz):
        [self.actors[f'{ch_name} name'].SetVisibility(viz) for ch_name in ch_names]

    def add_rois(self, subject, subjects_dir, rois, aseg):
        roi_mesh_color = create_roi_surface(subject, subjects_dir, aseg, rois)
        if roi_mesh_color is not None:
            for idx, roi in enumerate(roi_mesh_color):
                polydata = roi_mesh_color[roi][0]
                roi_color = roi_mesh_color[roi][1]
                self.actors[roi] = self.add_mesh(polydata, name=roi, label=roi,
                                                 color=roi_color, **roi_kwargs)
                # self.legend_actors[roi] = self.add_legend(labels=[[roi, roi_color]], name=roi+'legend')
                # self.legend_actors[roi] = self.add_legend()

    def enable_rois_viz(self, roi, viz):
        self.actors[roi].SetVisibility(viz)
        # if not viz:
        #     self.remove_legend()
        # else:
        #     self.add_actor(self.legend_actors[roi])

