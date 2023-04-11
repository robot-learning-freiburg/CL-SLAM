# pylint: skip-file

from subprocess import call
from typing import List, Tuple, Union

import cv2
import matplotlib.cm
import numpy as np
from tqdm import trange

DEFAULT_COLOR = (0, 0, 0)


class MeshlabInf:
    @staticmethod
    def show_multi_layer(**kwargs):
        """
        usage: given a set of MeshlabInf objects and their names, displays them in layer format.
        example:
        o1 = MeshlabInf()
        o2 = MeshlabInf()
        ...
        MeshlabInf().show_multi_layer(layerA=o1,layerB=o2,...)
        """
        d = dict(**kwargs)
        for k, v in d.items():
            v.write(k + ".obj")

        call(["meshlab"] + [x + ".obj" for x in d.keys()])

    @staticmethod
    def plot3d(pts, false_color=False):
        oo = MeshlabInf()
        oo.add_points(pts)
        oo.show(false_color)

    @staticmethod
    def get_colormap(sz, cmap_name="jet"):
        colormap = matplotlib.cm.get_cmap(cmap_name)(np.linspace(0, 1, sz))[:, 0:3]
        return colormap

    def __init__(self, global_transformation=np.eye(4)):
        self.global_transformation = global_transformation
        self._xyzrgb = np.empty((0, 6))
        self._faces = []
        self._lines = []

    def clear(self):
        self.__init__()

    def add_camera(self, p, color=DEFAULT_COLOR, scale=0.1, rotation=np.eye(3), camera_matrix=np.eye(3)):
        q = 1 / camera_matrix[0, 0]
        b = 1 / camera_matrix[1, 1]
        u = camera_matrix[-1, 0] / q
        v = camera_matrix[-1, 1] / b
        pts = (
                np.array(
                    [+u, +v, +0, -q, -b, +1, -q, +b, +1, +q, +b, +1, +q, -b, +1, +q, +v, +0, +u, +b, +0]
                ).reshape(-1, 3)
                * scale
        )
        zup_from_zfwd = np.array([[0.0, -1.0, -0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
        pts = pts @ zup_from_zfwd
        pts = pts @ rotation.T
        pts = pts + np.array(p).flatten()

        n = self._xyzrgb.shape[0]
        self.add_points(pts, color)
        f = np.array([0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 1, 0, 5, 6]).reshape(-1, 3)
        self.add_faces(list(f + n))

    def add_line(self, p1, p2, c=None):
        self.add_points(p1, c)
        self.add_points(p2, c)
        self.add_points(p2, c)
        n = self._xyzrgb.shape[0]
        self._lines.append([n - 3, n - 2, n - 1])

    def add_mesh(self, xyz, c=None):
        m, n, _ = xyz.shape
        if c is None:
            c = np.ones((m, n, 3))
        elif len(c.shape) == 2:
            c = np.repeat(np.expand_dims(c, 2), 3, axis=2)
        elif len(c.shape) == 3:
            pass
        else:
            raise Exception("unknown color size")

        good_ind = np.arange(0, m * n).reshape(m, n)
        va = np.vstack((good_ind[:-1:, :-1:].flatten(), good_ind[:-1:, 1::].flatten(),
                        good_ind[1::, :-1:].flatten())).T

        vb = np.vstack(
            (good_ind[1::,
                      1::].flatten(), good_ind[1::, :-1:].flatten(), good_ind[:-1:,
                                                                              1::].flatten())).T
        v = np.vstack((va, vb))
        v = v[:, [1, 0, 2]]
        xyzrgb = np.hstack((xyz.reshape(m * n, 3), c.reshape(m * n, 3)))
        ok = np.all(~np.isnan(xyzrgb), axis=1)
        good_ind = np.arange(0, xyzrgb.shape[0])[ok]
        bad_ind = np.arange(0, xyzrgb.shape[0])[~ok]
        xyzrgb = xyzrgb[good_ind, :]

        for i in bad_ind:
            v = v[np.all(v != i, axis=1), :]

        for i, idx in enumerate(good_ind):
            v[v == idx] = i
        n = self._xyzrgb.shape[0]
        self.add_points(xyzrgb)

        self.add_faces(list(v + n))

    def add_points(self, xyz, color=None):
        xyz_ = xyz.copy()
        if len(xyz_.shape) == 1:
            xyz_ = xyz_.reshape(1, -1)
        xyz_ = xyz_[~np.any(np.isnan(xyz_), axis=1), :]
        n = xyz_.shape[0]
        if color is None:
            if xyz_.shape[1] == 3:
                xyz_ = np.hstack((xyz_, np.ones((n, 3)) * xyz_[:, 2:]))
            elif xyz_.shape[1] == 4:
                xyz_ = np.hstack((xyz_, np.ones((n, 2)) * xyz_[:, 3:]))
            elif xyz_.shape[1] == 6:
                pass
            else:
                raise Exception("unknown points dimension")
        elif isinstance(color,
                        np.ndarray) and color.shape[0] == xyz.shape[0] and color.shape[1] == 3:
            xyz_ = np.c_[xyz_[:, :3], color]
        else:
            if len(color) != 3:
                raise Exception("color should be 3 elem vector")
            xyz_ = np.c_[xyz_[:, :3], np.ones((n, 1)) * color]

        self._xyzrgb = np.vstack((self._xyzrgb, xyz_))

    def add_pgon(self, xyz, color=None):
        xyz = xyz[~np.any(np.isnan(xyz), axis=1), :]
        n = self._xyzrgb.shape[0]
        self.add_points(xyz, color)
        self.add_faces([list(np.arange(n, n + xyz.shape[0]))])

    def add_faces(self, verts):
        self._faces += verts

    def read(self, fname):
        raise NotImplementedError("function 'read' is not yet implemented")

    def show(self, false_color=False):
        fn = "temp.obj"
        self.write(fn, false_color)
        call(["meshlab", fn])
        # call(['rm', fn])

    def write(self, fname, false_color=False, verbose=True):

        xyzrgb = self._xyzrgb.copy()

        if false_color:
            colind = np.mean(self._xyzrgb[:, 3:], axis=1)
            mm = norm_range_01(colind, (1, 99))
            colind = (mm * 255).astype(int)

            colormap = self.get_colormap(256)
            for i in range(xyzrgb.shape[0]):
                z = xyzrgb[i, 2]
                if np.isnan(z):
                    xyzrgb[i, 3:] = np.array([0, 0, 0])
                else:
                    xyzrgb[i, 3:] = colormap[colind[i], :]

            colormap2 = self.get_colormap(len(self._faces))
            for j, p in enumerate(self._faces):
                for i in p:
                    xyzrgb[i, 3:] = colormap2[j, :]
        else:

            xyzrgb[:, 3:] = norm_range_01(xyzrgb[:, 3:])
            xyzrgb[:, 3:] += 1 - np.max(xyzrgb[:, 3:])
        with open(fname, "w", encoding='utf-8') as f:

            f.write("# OBJ file\n")
            for i in trange(xyzrgb.shape[0],
                            desc="Writing points to meshlab file",
                            disable=not verbose):
                x = xyzrgb[i, :]
                x[:3] = self.global_transformation[:3, :3] @ x[:3] + self.global_transformation[:3,
                                                                                                -1]
                f.write("v %.4f %.4f %.4f %.4f %.4f %.4f\n" % (x[0], x[1], x[2], x[3], x[4], x[5]))

            for p in self._faces:
                f.write("f")
                for i in p:
                    f.write(" %d" % (i + 1))
                f.write("\n")

            for p in self._lines:
                f.write("f")
                for i in p:
                    f.write(" %d" % (i + 1))
                f.write("\n")
            f.close()


def norm_range_01(v: np.ndarray, prcnt: tuple = None) -> np.ndarray:
    """
    normalize the input values in range of [0-1].
    """
    if np.all(np.isnan(v)):
        return v
    if prcnt is None:
        min_value = np.nanmin(v)
        max_value = np.nanmax(v)
    else:
        if len(prcnt) != 2 or prcnt[0] >= prcnt[1] or prcnt[0] < 0 or prcnt[1] > 100:
            raise Exception("input #2 should contain hi and low percentage within [0-100]")

        min_value, max_value = np.nanpercentile(v, prcnt)

    d = max_value - min_value
    if d < np.finfo(type(min_value)).eps:
        d = 1
    v_out = (v - min_value) / d
    v_out = np.clip(v_out, 0, 1)
    return v_out


def rotation_matrix_from_to(v_from: Union[List, Tuple, np.ndarray],
                            v_to: Union[List, Tuple, np.ndarray],
                            output4x4: bool = False) -> np.ndarray:
    """
    calculate the rotation matrix the describes the rotation
    from input vector v_from to input vector v_to.
    input vectors must be of length 3
    """
    assert len(v_from) == 3
    assert len(v_to) == 3
    v_to = (v_to / np.linalg.norm(v_to)).flatten()
    v_from = (v_from / np.linalg.norm(v_from)).flatten()

    if np.all(np.abs(v_to - v_from) < np.finfo(float).eps * 1e3):
        axis = v_to
        angle = 0
    else:
        axis = np.cross(v_from, v_to)
        nrm = np.linalg.norm(axis)
        if nrm == 0:  # co-linear
            rr = np.random.randn(3)
            axis = rr - (v_from.T @ rr) * v_from
            axis = axis / np.linalg.norm(axis)
            angle = np.pi
        else:
            axis /= nrm
            angle = np.arccos(min(1.0, v_to @ v_from))

    r, _ = cv2.Rodrigues(axis * angle)
    if output4x4:
        outmat = np.eye(4)
        outmat[:3, :3] = r
    else:

        outmat = r
    return outmat
