# -*- coding: UTF-8 -*-
"""
@Project ：iEEGTool 
@File    ：brain.py
@Author  ：Barry
@Date    ：2022/3/16 15:27 
"""
import pyvista as pv
from pyvistaqt import QtInteractor, BackgroundPlotter

from viz.surface import read_fs_surface, create_chs_sphere
from utils.config import color, brain_kwargs, contact_kwargs, text_kwargs, roi_kwargs
from utils.process import get_chan_group


class Brain(pv.Plotter):
    def __init__(self):
        super().__init__(line_smoothing=True, point_smoothing=True)
        # super().__init__(menu_bar=False, toolbar=False, update_app_icon=False,
        # line_smoothing=True, point_smoothing=True)
        self.background_color = 'w'
        self.enable_depth_peeling()
        self.enable_anti_aliasing()
        self.isometric_view_interactive()

        self.brain_surface = {}
        self.brain_actors = {}

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
            self.add_mesh(self.brain_surface[h], name=h, **brain_kwargs)

    def enable_brain_viz(self, viz):
        hemi = ['lh', 'rh']
        actors = self.renderer.actors
        [actors[name].SetVisibility(viz) for name in actors if name in hemi]

    def set_brain_color(self, brain_color):
        hemi = ['lh', 'rh']
        actors = self.renderer.actors
        [actors[name].GetProperty().SetColor(brain_color) for name in actors if name in hemi]

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
            self.add_mesh(ch_spheres[ch_name], name=ch_name + '_sphere',
                          color=ch_color[ch_name], **contact_kwargs)
        for group in ch_groups:
            ch_name = ch_groups[group][-1]
            group_label_pos = ch_pos[ch_name] + 1
            self.add_point_labels(group_label_pos, [group], name=group, **text_kwargs)

    def enable_chs_viz(self, ch_names):
        ch_names = [f'{ch_name}_sphere' for ch_name in ch_names]
        actors = self.renderer.actors
        [actors[name].SetVisibility(viz) for name in actors if name in ch_names]

    def enable_group_label_viz(self, group, viz):
        actors = self.renderer.actors
        print(actors.keys())
        [actors[name].SetVisibility(viz) for name in actors if name == f'{group}-labels']

    def add_rois(self):
        # compute all coords of the regions related in a dict
        # then use the dict
        pass