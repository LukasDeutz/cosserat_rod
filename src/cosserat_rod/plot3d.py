# Build-in imports
import os
import time
from typing import List, Dict, Tuple, Union

# Thid party imports
from fenics import *
import matplotlib.animation as manimation
import matplotlib.colors as colors
import numpy as np
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import proj3d

# Local imports
from cosserat_rod.controls import ControlSequenceNumpy
from cosserat_rod.controls import ControlsNumpy
from cosserat_rod.frame import FrameNumpy, FrameSequenceNumpy, FRAME_COMPONENT_KEYS
from cosserat_rod.util import expand_numpy, f2n, v2f
from cosserat_rod.rod import Rod

FPS = 5
MIDLINE_CMAP_DEFAULT = 'plasma'


def interactive():
    import matplotlib
    gui_backend = 'Qt5Agg'
    matplotlib.use(gui_backend, force=True)


def cla(ax):
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.get_zaxis().set_ticks([])


class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work either side from a prescribed midpoint value
    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100)
    """

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


class Arrow3D(FancyArrowPatch):
    def __init__(self, origin, vec, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self.set_verts(origin, vec)

    def set_verts(self, origin, vec):
        xs = [origin[0], origin[0] + vec[0]]
        ys = [origin[1], origin[1] + vec[1]]
        zs = [origin[2], origin[2] + vec[2]]
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


class FrameArtist:
    """
    Draw midlines and frame component vectors.
    """

    midline_opt_defaults = {
        's': 20,
        'alpha': 0.9,
        'depthshade': True
    }
    arrow_opt_defaults = {
        'mutation_scale': 10,
        'arrowstyle': '-|>',
        'linewidth': 1,
        'alpha': 0.7
    }
    arrow_colour_defaults = {
        'e3': 'red',
        'e1': 'blue',
        'e2': 'green',
    }

    def __init__(
            self,
            F: FrameNumpy,
            midline_opts: Dict = None,
            arrow_opts: Dict = None,
            arrow_colours: Dict = None,
            arrow_scale: float = 0.1,
            n_arrows: int = 0,
            alpha_max: float = None,
            beta_max: float = None
    ):
        self.F = F
        self.N = self.F.x.shape[-1]
        self.arrows = {}
        self.scat = None
        if midline_opts is None:
            midline_opts = {}
        self.midline_opts = {**FrameArtist.midline_opt_defaults, **midline_opts}
        if arrow_opts is None:
            arrow_opts = {}
        self.arrow_opts = {**FrameArtist.arrow_opt_defaults, **arrow_opts}
        if arrow_colours is None:
            arrow_colours = {}
        self.arrow_colours = {**FrameArtist.arrow_colour_defaults, **arrow_colours}
        self.arrow_scale = arrow_scale

        # Show a subset of the arrows
        if 0 < n_arrows < self.N:
            idxs = np.round(np.linspace(0, self.N - 1, n_arrows)).astype(int)
        else:
            idxs = range(self.N)
        self.arrow_idxs = idxs

        # Calculate the worm length just once and assume it stays pretty constant
        self.worm_length = self.F.get_worm_length()

        # Controls colour-maps
        self.cmaps = {
            'alpha': cm.get_cmap('OrRd'),
            'beta': cm.get_cmap('BuPu'),
        }

        # Controls fixed maximums
        self.max_vals = {
            'alpha': alpha_max,
            'beta': beta_max
        }

    def add_midline(self, ax: Axes, cmap_name: str = MIDLINE_CMAP_DEFAULT):
        """
        Add the initial midline scatter plot.
        """

        # Colourmap / facecolors
        cmap = cm.get_cmap(cmap_name)
        fc = cmap((np.arange(self.N) + 0.5) / self.N)

        # Scatter plot of midline
        X = self.F.x
        self.scat = ax.scatter(X[0], X[1], X[2], c=fc, **self.midline_opts)

    def add_component_vectors(
            self,
            ax: Axes,
            draw_e3: bool = True,
            C: ControlsNumpy = None,
    ):
        """
        Add the initial component/force vectors.
        """
        arrows = {}
        keys = FRAME_COMPONENT_KEYS.copy()
        if not draw_e3:
            keys.remove('e3')
        for k in keys:
            arrows[k] = []
            vec, colours = self._get_vectors_and_colours(k, C)
            for i in self.arrow_idxs:
                arrow = Arrow3D(
                    origin=self.F.x[:, i],
                    vec=vec[:, i],
                    color=colours[i],
                    **self.arrow_opts
                )
                aa = ax.add_artist(arrow)
                arrows[k].append(aa)
        self.arrows = arrows

    def update(self, F: FrameNumpy, C: ControlsNumpy = None):
        """
        Update the midline and the component vectors.
        """
        self.F = F
        X = self.F.x

        # Update midline
        if self.scat is not None:
            self.scat.set_offsets(X[:2].T)
            self.scat.set_3d_properties(X[2], zdir='z')

        # Update component vectors
        if len(self.arrows) is not None:
            for k in self.arrows:
                vec, colours = self._get_vectors_and_colours(k, C)
                for i, j in enumerate(self.arrow_idxs):
                    self.arrows[k][i].set_verts(origin=X[:, j], vec=vec[:, j])
                    self.arrows[k][i].set_color(colours[j])

    def _get_vectors_and_colours(self, k: str, C: ControlsNumpy = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the component vectors for the frame.
        These are scaled and coloured according to the controls if passed.
        """
        vectors = getattr(self.F, k) * self.worm_length * self.arrow_scale
        if C is not None and k != 'e3':                        
            fk = 'alpha' if k == 'e1' else 'beta'
            force = getattr(C, fk)
            max_val = self.max_vals[fk]
            if max_val is None:
                max_val = np.abs(force).max()
            vectors *= force
            colours = self.cmaps[fk](np.abs(force) / max_val)
        else:
            colours = [self.arrow_colours[k] for _ in range(self.N)]

        return vectors, colours


class FrameDiffArtist:
    """
    Draw lines showing the difference between midline points.
    """
    line_opt_defaults = {
        'linewidth': 1,
        'alpha': 0.7,
        'color': 'red'
    }

    def __init__(
            self,
            F0: FrameNumpy,
            F1: FrameNumpy,
            line_opts: Dict = None,
    ):
        self.F0 = F0
        self.F1 = F1
        self.N = self.F0.x.shape[-1]
        self.arrows = {}
        if line_opts is None:
            line_opts = {}
        self.line_opts = {**FrameDiffArtist.line_opt_defaults, **line_opts}

    def add_diff_arrows(
            self,
            ax,
    ):
        arrows = []
        for i in range(self.N):
            p0 = self.F0.x[:, i]
            p1 = self.F1.x[:, i]

            arrow, = ax.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                zs=[p0[2], p1[2]],
                **self.line_opt_defaults
            )

            aa = ax.add_artist(arrow)
            arrows.append(aa)
        self.arrows = arrows

    def update(self, F0: FrameNumpy, F1: FrameNumpy):
        self.F0 = F0
        self.F1 = F1

        # Update diff arrows
        for i in range(self.N):
            p0 = self.F0.x[:, i]
            p1 = self.F1.x[:, i]
            self.arrows[i].set_data_3d(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                [p0[2], p1[2]]
            )


def plot_frame(F , arrow_scale: float = 0.1) -> Figure:
    """
    Single 3D frame plot with component arrows.
    """

    # Set up figure and 3d axes
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(projection='3d')
    cla(ax)

    # Add frame arrows and midline
    fa = FrameArtist(F, arrow_scale=arrow_scale)
    fa.add_component_vectors(ax)
    fa.add_midline(ax)

    # Fix axes range
    mins, maxs = F.get_bounding_box()
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])

    fig.tight_layout()
    plt.show()

    return fig


def plot_frame_3d(
        F0: FrameNumpy
) -> Figure:
    """
    Plot a 3x3 grid of 3D plots of the same worm frame from different angles.
    """
    mins, maxs = F0.get_bounding_box()
    elevs = [-60, 0, 60]
    azims = [-60, 0, 60]

    # Set up figure and 3d axes
    fig = plt.figure(facecolor='white', figsize=(10, 10))
    gs = gridspec.GridSpec(3, 3)
    for row_idx in range(3):
        for col_idx in range(3):
            ax = fig.add_subplot(gs[row_idx, col_idx], projection='3d', elev=elevs[row_idx], azim=azims[col_idx])
            cla(ax)
            fa = FrameArtist(F=F0, midline_opts={'s': 10}, n_arrows=12)
            fa.add_component_vectors(ax, draw_e0=False)
            fa.add_midline(ax)
            ax.set_xlim(mins[0], maxs[0])
            ax.set_ylim(mins[1], maxs[1])
            ax.set_zlim(mins[2], maxs[2])

    fig.tight_layout()
    fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0)

    return fig


def generate_interactive_scatter_clip(
        FS: FrameSequenceNumpy,
        fps: float,
        show: bool = True,
        perspectives = {}
):
    """
    Generate an interactive video clip from a FrameSequence
    """
    if show:
        interactive()

    # Show from 3 different perspectives
    if perspectives == 'yz':
        perspectives = {'elev': 0, 'azim': 0}
    elif perspectives == 'xz':
        perspectives = {'elev': 0, 'azim': -90}
    elif perspectives == 'xy':
        perspectives = {'elev': 90, 'azim': -90}


    # Set up figure and 3d axes
    fig = plt.figure(facecolor='white', figsize=(12, 12))
    ax = plt.axes([0.05, 0.15, 0.9, 0.80], projection='3d', **perspectives)
    ax_slider = plt.axes([0.05, 0.02, 0.9, 0.03])
    mins, maxs = FS.get_bounding_box(zoom=1)
    cla(ax)
    fa = FrameArtist(FS[0])
    fa.add_component_vectors(ax)
    fa.add_midline(ax)

    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])

    # Animation controls
    time_slider = Slider(ax_slider, 'Frame', 0, FS.n_timesteps, valinit=0, valstep=1)
    is_manual = False  # True if user has taken control of the animation

    def update(i):
        fa.update(FS[int(i)])
        return ()

    def update_slider(val):
        nonlocal is_manual
        is_manual = True
        update(val)

    def update_plot(num):
        nonlocal is_manual
        if is_manual:
            return ()
        val = num % time_slider.valmax
        time_slider.set_val(val)
        is_manual = False
        return ()

    def on_click(event):
        # Check where the click happened
        (xm, ym), (xM, yM) = time_slider.label.clipbox.get_points()
        if xm < event.x < xM and ym < event.y < yM:
            # Event happened within the slider, ignore since it is handled in update_slider
            return
        else:
            # user clicked somewhere else on canvas = toggle pause
            nonlocal is_manual
            if is_manual:
                is_manual = False
                ani.event_source.start()
            else:
                is_manual = True
                ani.event_source.stop()

    # Call update function on slider value change
    time_slider.on_changed(update_slider)
    fig.canvas.mpl_connect('button_press_event', on_click)
    ani = manimation.FuncAnimation(fig, update_plot, interval=1 / fps)

    if show:
        plt.show()

    return ani
    
def generate_scatter_clip(
        clips: List[FrameSequenceNumpy],
        save_dir,
        save_fn=None,
        labels=None,
        fps=FPS):
    """
    Generate a video clip from a list of FrameSequences
    """
    os.makedirs(save_dir, exist_ok=True)
    save_fn = save_fn if save_fn is not None else time.strftime('%Y-%m-%d_%H%M%S')
    save_path = save_dir + '/' + save_fn + '.mp4'
    # print('save_path', save_path)

    if labels is None:
        labels = [f'X_{i + 1}' for i in range(len(clips))]

    # Show from 3 different perspectives
    perspectives = [
        {'elev': 30, 'azim': -60},
        {'elev': 30, 'azim': 60},
        {'elev': -30, 'azim': -45},
    ]

    # Set up figure and 3d axes
    fig = plt.figure(facecolor='white', figsize=(len(clips) * 4, 12))
    gs = gridspec.GridSpec(len(perspectives), len(clips))
    axes = [[], [], []]
    artists = [[], [], []]
    for col_idx, FS in enumerate(clips):
        mins, maxs = FS.get_bounding_box(zoom=1)
        for row_idx in range(3):
            ax = fig.add_subplot(gs[row_idx, col_idx], projection='3d', **perspectives[row_idx])
            cla(ax)

            fa = FrameArtist(FS[0])
            fa.add_component_vectors(ax)
            fa.add_midline(ax)
            artists[row_idx].append(fa)

            if row_idx == 0:
                ax.set_title(labels[col_idx])

            ax.set_xlim(mins[0], maxs[0])
            ax.set_ylim(mins[1], maxs[1])
            ax.set_zlim(mins[2], maxs[2])
            axes[row_idx].append(ax)

    def update(i):
        for col_idx, FS in enumerate(clips):
            for row_idx in range(3):
                artists[row_idx][col_idx].update(FS[i])
        return ()

    fig.tight_layout()
    fig.subplots_adjust(left=0, right=1, top=0.95, bottom=0)

    ani = manimation.FuncAnimation(
        fig,
        update,
        frames=clips[0].n_timesteps,
        blit=True
    )

    # Save
    metadata = dict(title=save_path, artist='WormLab Leeds')
    ani.save(save_path, writer='ffmpeg', fps=fps, metadata=metadata)
    plt.close(fig)


def generate_scatter_diff_clip(
        FS_target: FrameSequenceNumpy,
        FS_attempt: FrameSequenceNumpy,
        save_dir: str,
        save_fn: str = None,
        arrow_scale: float = 0.2,
        n_arrows: int = 12,
        n_perspectives: int = 3
):
    """
    Generate a video clip showing a target sequence, an attempt and the difference.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_fn = save_fn if save_fn is not None else time.strftime('%Y-%m-%d_%H%M%S')
    save_path = save_dir + '/' + save_fn + '.mp4'
    labels = ['Target', 'Error', 'Attempt']

    # Show from 3 different perspectives
    perspectives = [
        {'elev': 30, 'azim': -60},
        {'elev': 30, 'azim': 60},
        {'elev': -30, 'azim': -45},
    ]

    # Set up figure and 3d axes
    fig = plt.figure(facecolor='white', figsize=(12, 12))
    gs = gridspec.GridSpec(n_perspectives, 3)
    axes = [[], [], []]
    artists = [[], [], []]
    for col_idx in range(3):
        if col_idx in [0, 2]:
            if col_idx == 0:
                FS = FS_target
            else:
                FS = FS_attempt
            mins, maxs = FS.get_bounding_box(zoom=1)
        else:
            mins_target, maxs_target = FS_attempt.get_bounding_box(zoom=1)
            mins_attempt, maxs_attempt = FS_target.get_bounding_box(zoom=1)
            mins = np.min((mins_target, mins_attempt), axis=0)
            maxs = np.max((maxs_target, maxs_attempt), axis=0)

        for row_idx in range(n_perspectives):
            ax = fig.add_subplot(gs[row_idx, col_idx], projection='3d', **perspectives[row_idx])
            cla(ax)

            if col_idx in [0, 2]:
                fa = FrameArtist(FS[0], arrow_scale=arrow_scale, n_arrows=n_arrows)
                fa.add_component_vectors(ax, draw_e0=False)
                fa.add_midline(ax)
            else:
                fa = FrameDiffArtist(FS_target[0], FS_attempt[0])
                fa.add_diff_arrows(ax)

            artists[row_idx].append(fa)

            if row_idx == 0:
                ax.set_title(labels[col_idx])

            ax.set_xlim(mins[0], maxs[0])
            ax.set_ylim(mins[1], maxs[1])
            ax.set_zlim(mins[2], maxs[2])
            axes[row_idx].append(ax)

    def update(i):
        for col_idx in range(3):
            for row_idx in range(3):
                if col_idx in [0, 2]:
                    if col_idx == 0:
                        FS = FS_target
                    else:
                        FS = FS_attempt
                    artists[row_idx][col_idx].update(FS[i])
                else:
                    artists[row_idx][col_idx].update(FS_target[i], FS_attempt[i])
        return ()

    fig.tight_layout()
    fig.subplots_adjust(left=0, right=1, top=0.95, bottom=0)

    ani = manimation.FuncAnimation(
        fig,
        update,
        frames=FS_target.n_timesteps,
        blit=True
    )

    # Save
    metadata = dict(title=save_path, artist='WormLab Leeds')
    ani.save(save_path, writer='ffmpeg', fps=FPS, metadata=metadata)
    plt.close(fig)


def plot_X_vs_target(
        FS: FrameSequenceNumpy,
        FS_target: FrameSequenceNumpy
) -> Figure:
    """
    Plot the x,y,z midline positions over time as matrices.
    """
    X = FS.x
    X_target = FS_target.x

    # Get MSE losses
    LX = np.square(X - X_target)

    # Determine common scales
    X_vmin = min(X.min(), X_target.min())
    X_vmax = max(X.max(), X_target.max())
    L_vmax = LX.max()

    fig, axes = plt.subplots(3, 3, figsize=(14, 8))

    for row_idx, M in enumerate([X, LX, X_target]):
        for col_idx in range(3):
            ax = axes[row_idx, col_idx]

            # Select xyz component and transpose so columns are frames
            Mc = M[:, col_idx].T
            if row_idx in [0, 2]:
                vmin = X_vmin
                vmax = X_vmax
                m = ax.matshow(
                    Mc,
                    cmap=plt.cm.PRGn,
                    clim=(vmin, vmax),
                    norm=MidpointNormalize(midpoint=0, vmin=vmin, vmax=vmax),
                    aspect='auto'
                )
            else:
                if L_vmax > 0:
                    norm = colors.LogNorm(vmin=1e-7, vmax=L_vmax)
                else:
                    norm = None
                m = ax.matshow(Mc, cmap=plt.cm.Reds, aspect='auto', norm=norm)

            if row_idx == 0:
                ax.set_title(['x', 'y', 'z'][col_idx])
            if col_idx == 0:
                ax.set_ylabel(['X', 'LX', 'X_target'][row_idx])
            if col_idx == 2:
                fig.colorbar(m, ax=ax, format='%.3E')

            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

    fig.tight_layout()

    return fig

def plot_frame_components(
        F: FrameNumpy,
) -> Figure:
    """
    Plot the psi/e0/e1/e2 frame components as matrices.
    """

    # Expand the psi to two dims
    psi0 = expand_numpy(F.psi)

    # Determine common scales
    e_vmin = min(F.e0.min(), F.e1.min(), F.e2.min())
    e_vmax = max(F.e0.max(), F.e1.max(), F.e2.max())

    fig, axes = plt.subplots(1, 4, figsize=(12, 7), squeeze=False)

    Ms = [psi0, F.e0, F.e1, F.e2]

    for col_idx in range(4):
        ax = axes[0, col_idx]
        if col_idx == 0:
            # Use a cyclic colormap with a fixed scale for psi as 0=2pi
            cmap = plt.cm.twilight
            vmin = 0
            vmax = 2 * np.pi
            norm = None
        else:
            cmap = plt.cm.PRGn
            vmin = e_vmin
            vmax = e_vmax
            norm = MidpointNormalize(midpoint=0, vmin=vmin, vmax=vmax)

        m = ax.matshow(
            Ms[col_idx],
            cmap=cmap,
            clim=(vmin, vmax),
            norm=norm,
            aspect='auto'
        )

        ax.set_title(['$\psi$', '$e_0$', '$e_1$', '$e_2$'][col_idx])
        if col_idx in [0, 3]:
            fig.colorbar(m, ax=ax)

        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

    fig.tight_layout()
    return fig


def plot_frame_components_vs_target(
        F: FrameNumpy,
        F_target: FrameNumpy,
) -> Figure:
    """
    Plot the psi/e0/e1/e2 frame components as matrices against a target, along with errors.
    """

    # Expand the psi to two dims
    psi0 = expand_numpy(F.psi)
    psi0_target = expand_numpy(F_target.psi)

    # Get MSE losses
    Lp = np.square(psi0 - psi0_target)
    Le0 = np.square(F.e0 - F_target.e0)
    Le1 = np.square(F.e1 - F_target.e1)
    Le2 = np.square(F.e2 - F_target.e2)

    # Determine common scales
    e_vmin = min(F.e0.min(), F.e1.min(), F.e2.min(),
                 F_target.e0.min(), F_target.e1.min(), F_target.e2.min())
    e_vmax = max(F.e0.max(), F.e1.max(), F.e2.max(),
                 F_target.e0.max(), F_target.e1.max(), F_target.e2.max())
    L_vmin = min(Le0.min(), Le1.min(), Le2.min())
    L_vmax = max(Le0.max(), Le1.max(), Le2.max())

    fig, axes = plt.subplots(3, 4, figsize=(12, 7))

    Mp = [psi0, Lp, psi0_target]
    M0 = [F.e0, Le0, F_target.e0]
    M1 = [F.e1, Le1, F_target.e1]
    M2 = [F.e2, Le2, F_target.e2]

    for row_idx in range(3):
        for col_idx in range(4):
            ax = axes[row_idx, col_idx]
            M = [Mp, M0, M1, M2][col_idx][row_idx]

            if row_idx in [0, 2]:
                if col_idx == 0:
                    # Use a cyclic colormap with a fixed scale for psi as 0=2pi
                    cmap = plt.cm.twilight
                    vmin = 0
                    vmax = 2 * np.pi
                    norm = None
                else:
                    cmap = plt.cm.PRGn
                    vmin = e_vmin
                    vmax = e_vmax
                    norm = MidpointNormalize(midpoint=0, vmin=vmin, vmax=vmax)
                m = ax.matshow(
                    M,
                    cmap=cmap,
                    clim=(vmin, vmax),
                    norm=norm,
                    aspect='auto'
                )
            else:
                m = ax.matshow(M, cmap=plt.cm.Reds, aspect='auto', vmin=L_vmin, vmax=L_vmax)

            if row_idx == 0:
                ax.set_title(['$\psi$', '$e_0$', '$e_1$', '$e_2$'][col_idx])
            if col_idx == 0:
                ax.set_ylabel(['Component', 'MSE', 'Target'][row_idx])
            if col_idx in [0, 3]:
                fig.colorbar(m, ax=ax)

            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

    fig.tight_layout()
    return fig


def plot_FS_orthogonality(
        FS: FrameSequenceNumpy
) -> Figure:
    """
    Plot the angles between the frame components over time.
    """

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    kps = [('e0', 'e1'), ('e0', 'e2'), ('e1', 'e2')]

    # Calculate angles between components
    N = FS[0].x.shape[-1]
    angles = np.zeros((3, len(FS), N))
    for i, (k1, k2) in enumerate(kps):
        for t, F in enumerate(FS):
            u = getattr(F, k1)
            v = getattr(F, k2)
            cosA = np.diag(np.matmul(u.T, v)) / \
                   (np.linalg.norm(u, axis=0) * np.linalg.norm(v, axis=0))
            angle = np.arccos(cosA)
            angles[i, t] = angle

    # Plot the angles averaged over space and time
    for i, (k1, k2) in enumerate(kps):
        # Space-averaged
        ax = axes[0, i]
        mu = angles[i].mean(axis=1)
        std = angles[i].std(axis=1)
        ax.fill_between(mu, mu - 2 * std, mu + 2 * std, alpha=0.25)
        ax.plot(mu)
        ax.set_title(f'Space-averaged {{{k1},{k2}}}')
        ax.set_xlabel('Time')

        # Time-averaged
        ax = axes[1, i]
        mu = angles[i].mean(axis=0)
        std = angles[i].std(axis=0)
        ax.fill_between(mu, mu - 2 * std, mu + 2 * std, alpha=0.25)
        ax.plot(mu)
        ax.set_title(f'Time-averaged {{{k1},{k2}}}')
        ax.set_xlabel('Head-Tail')

    fig.tight_layout()
    return fig


def plot_FS_normality(
        FS: FrameSequenceNumpy
) -> Figure:
    """
    Plot the magnitude of the frame components over time.
    """

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))

    # Calculate magnitudes of components
    N = FS[0].x.shape[-1]
    magnitudes = np.zeros((3, len(FS), N))
    for i, k in enumerate(FRAME_COMPONENT_KEYS):
        for t, F in enumerate(FS):
            u = getattr(F, k)
            magnitudes[i, t] = np.linalg.norm(u, axis=0)

    # Plot the magnitudes averaged over space and time
    for i, k in enumerate(FRAME_COMPONENT_KEYS):
        # Space-averaged
        ax = axes[0, i]
        mu = magnitudes[i].mean(axis=1)
        std = magnitudes[i].std(axis=1)
        ax.fill_between(mu, mu - 2 * std, mu + 2 * std, alpha=0.25)
        ax.plot(mu)
        ax.set_title(f'Space-averaged {k}')
        ax.set_xlabel('Time')

        # Time-averaged
        ax = axes[1, i]
        mu = magnitudes[i].mean(axis=0)
        std = magnitudes[i].std(axis=0)
        ax.fill_between(mu, mu - 2 * std, mu + 2 * std, alpha=0.25)
        ax.plot(mu)
        ax.set_title(f'Time-averaged {k}')
        ax.set_xlabel('Head-Tail')

    fig.tight_layout()
    return fig


def plot_CS(
        CS: ControlSequenceNumpy,
        dt: float = None
) -> Figure:
    """
    Plot the control sequences as matrices.
    """
    fig = plot_ome_sig()
    fig.suptitle('Control Sequence')
    fig.tight_layout()
    return fig


def plot_CS_vs_target(
        CS: ControlSequenceNumpy,
        CS_target: ControlSequenceNumpy,
) -> Figure:
    # Get MSE losses
    La = np.square(CS.alpha - CS_target.alpha)
    Lb = np.square(CS.beta - CS_target.beta)
    Lg = np.square(CS.gamma - CS_target.gamma)

    # Determine common scales
    X_vmin = min(CS.alpha.min(), CS_target.alpha.min(), CS.beta.min(),
                 CS_target.beta.min(), CS.gamma.min(), CS_target.gamma.min())
    X_vmax = max(CS.alpha.max(), CS_target.alpha.max(), CS.beta.max(),
                 CS_target.beta.max(), CS.gamma.max(), CS_target.gamma.max())
    L_vmax = max(La.max(), Lb.max(), Lg.max())

    fig, axes = plt.subplots(3, 3, figsize=(12, 7))

    for row_idx, Ms in enumerate([
        [CS.alpha, CS.beta, CS.gamma],
        [La, Lb, Lg],
        [CS_target.alpha, CS_target.beta, CS_target.gamma]
    ]):
        for col_idx in range(3):
            M = Ms[col_idx].T
            ax = axes[row_idx, col_idx]
            if row_idx in [0, 2]:
                vmin = X_vmin
                vmax = X_vmax
                m = ax.matshow(
                    M,
                    cmap=plt.cm.PRGn,
                    clim=(vmin, vmax),
                    norm=MidpointNormalize(midpoint=0, vmin=vmin, vmax=vmax),
                    aspect='auto'
                )
            else:
                if L_vmax > 0:
                    norm = colors.LogNorm(vmin=1e-7, vmax=L_vmax)
                else:
                    norm = None
                m = ax.matshow(M, cmap=plt.cm.Reds, aspect='auto', norm=norm)

            if row_idx == 0:
                ax.set_title(['$\\alpha^0$', '$\\beta^0$', '$\gamma^0$'][col_idx])
            if col_idx == 0:
                ax.set_ylabel(['attempt', 'L', 'target'][row_idx])
            if col_idx == 2:
                fig.colorbar(m, ax=ax, format='%.3f')

            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

    fig.tight_layout()
    return fig


def plot_curvature_and_twist(
        FS: FrameSequenceNumpy,
        dt: float = None
) -> Figure:
    """
    Plot the curvatures (alpha/beta) and the twist (gamma).
    """
    fig = plot_abgm(FS, dt)
    fig.suptitle('Curvature and twist')
    fig.tight_layout()
    return fig

def plot_CS_vs_output(
        CS: ControlSequenceNumpy,
        FS: FrameSequenceNumpy,
        dt: float = None,
        plot_kappa = True,
        scale_curvature = True
) -> Figure:
    
    # Scale alpha and beta
    if scale_curvature:    
        
        # number vertices
        N = CS.alpha.shape[1]        
        mesh = UnitIntervalMesh(N-1)
        
        V = FunctionSpace(mesh, 'Lagrange', 1)
        Q = FunctionSpace(mesh, 'DP', 0)
                                            
        alpha_scaled = np.zeros(CS.alpha.shape) 
        beta_scaled  = np.zeros(CS.beta.shape) 
                      
        alpha_func = Function(V)
        beta_func = Function(V)
        mu_func = Function(Q)
                        
        for i, (alpha_t, beta_t, mu_t) in enumerate(zip(CS.alpha, CS.beta, CS.mu)):
        
            alpha_func.vector()[:] = alpha_t
            beta_func.vector()[:] = beta_t
            mu_func.vector()[:] = mu_t
        
            alpha_t = project(alpha_func/mu_func, V)
            beta_t  = project(beta_func/mu_func, V)
                
            alpha_scaled[i, :] = alpha_t.vector().get_local()
            beta_scaled[i, :] = beta_t.vector().get_local()

        CS.alpha = alpha_scaled
        CS.beta = beta_scaled
                                
    ab_vmin = min(CS.alpha.min(), FS.alpha.min(), CS.beta.min(), FS.beta.min())
    g_vmin = min(CS.gamma.min(), FS.gamma.min())
    m_vmin = min(CS.mu.min(), FS.mu.min(), CS.mu.min(), FS.mu.min())
    ab_vmax = max(CS.alpha.max(), FS.alpha.max(), CS.beta.max(), FS.beta.max())
    g_vmax = max(CS.gamma.max(), FS.gamma.max())
    m_vmax = min(CS.mu.max(), FS.mu.max(), CS.mu.max(), FS.mu.max())
    
    fig = plt.figure(figsize=(16, 8))
    gs_top = gridspec.GridSpec(2, 4)
    
    if plot_kappa:
        gs_bottom = gridspec.GridSpec(3, 4, right=1.108)
    # else:
    #     gs_bottom = gridspec.GridSpec(2, 4, right=1.108)

    for row_idx, Ms in enumerate([
        [CS.alpha, CS.beta, CS.gamma, CS.mu],
        [FS.alpha, FS.beta, FS.gamma, FS.mu]
    ]):
        for col_idx in range(4):
            M = Ms[col_idx].T
            ax = fig.add_subplot(gs_top[row_idx, col_idx])
            if col_idx == 2:
                cmap = plt.cm.BrBG
                vmin = g_vmin
                vmax = g_vmax
                cbar_format = '%.2f'
            if col_idx == 3:
                cmap = plt.cm.BrBG
                vmin = m_vmin
                vmax = m_vmax
                cbar_format = '%.2f'            
            else:
                cmap = plt.cm.PRGn
                vmin = ab_vmin
                vmax = ab_vmax
                cbar_format = '%.1f'
            m = ax.matshow(
                M,
                cmap=cmap,
                clim=(vmin, vmax),
                norm=MidpointNormalize(midpoint=0, vmin=vmin, vmax=vmax),
                aspect='auto'
            )
            fig.colorbar(m, ax=ax, format=cbar_format)

            if row_idx == 0:
                ax.set_title(['$\\alpha^0$', '$\\beta^0$', '$\gamma^0$', '$\mu^0$'][col_idx])
            else:
                ax.set_title(['$\\alpha$', '$\\beta$', '$\gamma$', '$\mu$'][col_idx])
            if col_idx == 0:
                ax.set_ylabel(['Inputs', 'Outputs'][row_idx])
            ax.text(-0.02, 1, 'H', transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
                    fontweight='bold')
            ax.text(-0.02, 0, 'T', transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='right',
                    fontweight='bold')
            if dt is not None:
                
                f_s = ("{:.%if}" % len(str(dt).split('.')[-1]))             
                                
                ax.text(0, -0.01, '0', transform=ax.transAxes, verticalalignment='top', horizontalalignment='left',
                        fontweight='bold')
                ax.text(1, -0.01, f'{f_s.format(CS.n_timesteps * dt)}s', transform=ax.transAxes, verticalalignment='top',
                        horizontalalignment='right', fontweight='bold')

            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
    
    
    
    
    if plot_kappa: 
        # Plot kymogram showing absolute curvature
        ax = fig.add_subplot(gs_bottom[2, :])
        M = (np.abs(FS.alpha) + np.abs(FS.beta)).T
        m = ax.matshow(
            M,
            cmap=plt.cm.YlOrRd,
            clim=(0, M.max()),
            aspect='auto'
        )
        fig.colorbar(m, ax=ax, format='%.1f', pad=0.015)
        ax.set_title('Combined curvature')
        ax.text(-0.005, 1, 'H', transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
                fontweight='bold')
        ax.text(-0.005, 0, 'T', transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='right',
                fontweight='bold')
        if dt is not None:
            ax.text(0, -0.01, '0', transform=ax.transAxes, verticalalignment='top', horizontalalignment='left',
                    fontweight='bold')
            ax.text(1, -0.01, f'{CS.n_timesteps * dt:.2f}s', transform=ax.transAxes, verticalalignment='top',
                    horizontalalignment='right', fontweight='bold')
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])            
            
    fig.tight_layout()
    return fig


def plot_strains(
        S: Union[FrameSequenceNumpy, ControlSequenceNumpy],
        dt: float = None,
        T_max = None
) -> Figure:
    """
    Plot the sequence of alpha, beta and gamma over time.
    """
    # Determine common scales
    if dt is not None and T_max is not None:    
        n_timesteps = int(T_max/dt)
        alpha = S.alpha[:n_timesteps,:]
        beta = S.beta[:n_timesteps,:]
        gamma = S.gamma[:n_timesteps,:]
        mu = S.mu[:n_timesteps,:]
    else:
        n_timesteps = S.n_timesteps
        sigma = S.Omega
        sigma = S.sigma
    
    Omega1 = S.Omega[:, 0, :]
    Omega2 = S.Omega[:, 1, :]
    Omega3 = S.Omega[:, 2, :]
    
    sigma1 = S.sigma[:, 0, :]
    sigma2 = S.sigma[:, 1, :]
    sigma3 = S.sigma[:, 2, :]
    
    o1_min = Omega1.min()
    o1_max = Omega1.max()
    o2_min = Omega2.min()
    o2_max = Omega2.max()
    o3_min = Omega3.min()
    o3_max = Omega3.max()

    s1_min = sigma1.min()
    s1_max = sigma1.max()
    s2_min = sigma2.min()
    s2_max = sigma2.max()
    s3_min = sigma3.min()
    s3_max = sigma3.max()
            
    fig, axes = plt.subplots(1, 6, figsize=(12, 7), squeeze=False)
    
    Ms = [Omega1, Omega2, Omega3, sigma1, sigma2, sigma3]
    s_mins = [o1_min, o2_min, o3_min, s1_min, s2_min, s3_min]
    s_maxs = [o1_max, o2_max, o3_max, s1_max, s2_max, s3_max]
    cmaps  = [plt.cm.BrBG, plt.cm.BrBG, plt.cm.PRGn, plt.cm.BrBG, plt.cm.BrBG, plt.cm.PRGn]
    cbar_formats = ['%.4f', '%.4f', '%.3f', '%.2f', '%.2f', '%.2f']
    
    for col_idx, (M, s_min, s_max, cmap, cbar_format)  in enumerate(zip(Ms, s_mins, s_maxs, cmaps, cbar_formats)):
        
        ax = axes[0, col_idx]
        
        m = ax.matshow(
            M.T,
            cmap=cmap,
            clim=(s_min, s_max),
            norm=MidpointNormalize(midpoint=0, vmin=s_min, vmax= s_max),
            aspect='auto'
        )

        ax.set_title([r'$\Omega_1$', r'$\Omega_2$', r'$\Omega_3$', r'$\sigma_1$', r'$\sigma_2$', r'$\sigma_3$'][col_idx])
        ax.text(-0.02, 1, 'H', transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
                fontweight='bold')
        ax.text(-0.02, 0, 'T', transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='right',
                fontweight='bold')
        if dt is not None:            
            f_s = ("{:.%if}" % len(str(dt).split('.')[-1]))             
            
            ax.text(0, -0.01, '0', transform=ax.transAxes, verticalalignment='top', horizontalalignment='left',
                    fontweight='bold')
            ax.text(1, -0.01, f'{f_s.format(n_timesteps * dt)}s', transform=ax.transAxes, verticalalignment='top',
                    horizontalalignment='right', fontweight='bold')
        
        fig.colorbar(m, ax=ax, format=cbar_format)

        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

    fig.tight_layout()
    return fig

def plot_controls_CS_vs_FS(
        CS: Union[FrameSequenceNumpy, ControlSequenceNumpy],
        FS: Union[FrameSequenceNumpy, ControlSequenceNumpy],
        dt: float = None,
        T_max = None
) -> Figure:
    """
    Plot the sequence of alpha, beta and gamma over time.
    """
    if dt is not None and T_max is not None:    
        n_timesteps = int(T_max/dt)
    else:
        n_timesteps = FS.n_timesteps

    # Determine common scales    
    o_min = np.zeros(3)
    o_max = np.zeros(3)

    s_min = np.zeros(3)
    s_max = np.zeros(3)

    Omega_FS_list = []
    sigma_FS_list = []

    Omega_CS_list = []
    sigma_CS_list = []

    # Omega and sigma vector have three coordinates
    for i in range(3):
                
        oi_min = np.min([FS.Omega[:, i, :].min(), CS.Omega[:, i, :].min()])
        oi_max = np.max([FS.Omega[:, i, :].max(), CS.Omega[:, i, :].max()])
                        
        si_min = np.min([FS.sigma[:, i, :].min(), CS.sigma[:, i, :].min()])
        si_max = np.max([FS.sigma[:, i, :].max(), CS.sigma[:, i, :].max()])
        
        o_min[i] = oi_min
        o_max[i] = oi_max
        
        s_min[i] = si_min
        s_max[i] = si_max
                    
        Omega_FS_list.append(FS.Omega[:, i, :])
        sigma_FS_list.append(FS.sigma[:, i, :])
        
        Omega_CS_list.append(CS.Omega[:, i, :])
        sigma_CS_list.append(CS.sigma[:, i, :])
                    

    v_min = np.hstack((o_min, s_min))
    v_max = np.hstack((o_max, s_max))
    
    M_FS_list = Omega_FS_list + sigma_FS_list
    M_CS_list = Omega_CS_list + sigma_CS_list

    fig, axes = plt.subplots(2, 6, figsize=(18, 7), squeeze=False)
    
    cmaps  = [plt.cm.BrBG, plt.cm.BrBG, plt.cm.PRGn, plt.cm.RdGy, plt.cm.RdGy, plt.cm.RdBu]
    cbar_formats = ['%.4f', '%.4f', '%.3f', '%.2f', '%.2f', '%.2f']

    for i, (M_FS, M_CS) in enumerate(zip(M_FS_list, M_CS_list)):
                
        ax_CS = axes[0, i]
        ax_FS = axes[1, i]
        
        m = ax_FS.matshow(
            M_FS.T,
            cmap=cmaps[i],
            clim=(v_min[i], v_max[i]),
            norm=MidpointNormalize(midpoint=0, vmin=v_min[i], vmax= v_max[i]),
            aspect='auto'
        )

        m = ax_CS.matshow(
            M_CS.T,
            cmap=cmaps[i],
            clim=(v_min[i], v_max[i]),
            norm=MidpointNormalize(midpoint=0, vmin=v_min[i], vmax= v_max[i]),
            aspect='auto'
        )

        ax_CS.set_title([r'$\Omega^0_1$', r'$\Omega^0_2$', r'$\Omega^0_3$', r'$\sigma^0_1$', r'$\sigma^0_2$', r'$\sigma^0_3$'][i])        
        ax_FS.set_title([r'$\Omega_1$', r'$\Omega_2$', r'$\Omega_3$', r'$\sigma_1$', r'$\sigma_2$', r'$\sigma_3$'][i])
        
        for ax in [ax_CS, ax_FS]:
        
            ax.text(-0.02, 1, 'H', transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
                    fontweight='bold')
            ax.text(-0.02, 0, 'T', transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='right',
                    fontweight='bold')
    
            ax.text(-0.02, 1, 'H', transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
                    fontweight='bold')
            ax.text(-0.02, 0, 'T', transform=ax.transAxes, verticalalignment='bottom', horizontalalignment='right',
                    fontweight='bold')

            fig.colorbar(m, ax=ax, format=cbar_formats[i])

            if dt is not None:            
                f_s = ("{:.%if}" % len(str(dt).split('.')[-1]))             
                
                ax.text(0, -0.01, '0', transform=ax.transAxes, verticalalignment='top', horizontalalignment='left',
                        fontweight='bold')
                ax.text(1, -0.01, f'{f_s.format(n_timesteps * dt)}s', transform=ax.transAxes, verticalalignment='top',
                        horizontalalignment='right', fontweight='bold')
        

            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

    fig.tight_layout()
    
    return fig


def plot_FS_3d(
        FSs: List[FrameSequenceNumpy],
        CSs: List[ControlSequenceNumpy],
        labels=None
) -> Figure:
    # interactive()
    n_frames = len(FSs[0])
    if labels is None:
        labels = [f'X_{i + 1}' for i in range(len(FSs))]

    # Get common scale
    mins, maxs = get_bounding_box(FSs)

    alpha_max = 0
    beta_max = 0
    for C in CSs:
        if C is None:
            continue
        alpha_max = max(alpha_max, C.alpha.max())
        beta_max = max(beta_max, C.beta.max())

    # Set up figure and 3d axes
    fig = plt.figure(facecolor='white', figsize=(n_frames * 4, len(FSs) * 5))
    gs = gridspec.GridSpec(len(FSs), n_frames)
    for row_idx in range(len(FSs)):
        FS = FSs[row_idx]
        CS = CSs[row_idx]
        for col_idx in range(n_frames):
            ax = fig.add_subplot(gs[row_idx, col_idx], projection='3d')
            cla(ax)
            if row_idx == 0:
                ax.set_title(f'frame={col_idx + 1}')
            if col_idx == 0:
                ax.text2D(-0.1, 0.5, labels[row_idx], transform=ax.transAxes, rotation='vertical')

            fa = FrameArtist(F=FS[col_idx], alpha_max=alpha_max, beta_max=beta_max)
            C = CS[col_idx] if CS is not None else None
            fa.add_component_vectors(ax, draw_e0=False, C=C)
            fa.add_midline(ax)

            ax.set_xlim(mins[0], maxs[0])
            ax.set_ylim(mins[1], maxs[1])
            ax.set_zlim(mins[2], maxs[2])

    fig.tight_layout()
    fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0)

    return fig

def get_bounding_box(FSs: List[FrameSequenceNumpy], zoom=1):
    # Get common scale
    mins = np.array([np.inf, np.inf, np.inf])
    maxs = np.array([-np.inf, -np.inf, -np.inf])
    for FS in FSs:
        FS_mins, FS_maxs = FS.get_range()
        mins = np.minimum(mins, FS_mins)
        maxs = np.maximum(maxs, FS_maxs)
    max_range = max(maxs - mins)
    means = mins + (maxs - mins) / 2
    mins = means - max_range * zoom / 2
    maxs = means + max_range * zoom / 2
    return mins, maxs
