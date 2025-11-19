"""
Plotting utilities for UPT model visualization.
Based on GAOT plotting functions, adapted for UPT's architecture.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Union, Tuple, Dict
import matplotlib.colors as mcolors
import matplotlib

########################################################
# Plotting settings (matching GAOT style)
########################################################
C_BLACK = '#000000'
C_WHITE = '#ffffff'
C_BLUE = '#093691'
C_RED = '#911b09'
C_BLACK_BLUEISH = '#011745'
C_BLACK_REDDISH = '#380801'
C_WHITE_BLUEISH = '#dce5f5'
C_WHITE_REDDISH = '#f5dcdc'

CMAP_BWR = matplotlib.colors.LinearSegmentedColormap.from_list(
    'blue_white_red',
    [C_BLACK_BLUEISH, C_BLUE, C_WHITE, C_RED, C_BLACK_REDDISH],
    N=200,
)
CMAP_WRB = matplotlib.colors.LinearSegmentedColormap.from_list(
    'white_red_black',
    [C_WHITE, C_RED, C_BLACK],
    N=200,
)

SCATTER_SETTINGS = dict(marker='s', s=1, alpha=1, linewidth=0)
HATCH_SETTINGS = dict(facecolor='#b8b8b8', edgecolor='#4f4f4f', linewidth=.0)


########################################################
# Main plotting functions
########################################################
def plot_estimates(
    u_inp: np.ndarray,
    u_gtr: np.ndarray,
    u_prd: np.ndarray,
    x_inp: np.ndarray,
    x_out: np.ndarray,
    symmetric: Union[bool, List[bool]] = True,
    names: Optional[List[str]] = None,
    domain: Tuple[List[float], List[float]] = None,
    colorbar_type: str = "light",
    show_error: bool = True
) -> plt.Figure:
    """
    Plot input, ground-truth, predictions, and absolute error for UPT model outputs.
    
    This creates a figure with 3-4 columns for each variable:
    1) Input data
    2) Ground-truth values
    3) Model predictions  
    4) Absolute error (optional)
    
    Parameters
    ----------
    u_inp : np.ndarray
        Input data, shape (N_inp, n_vars)
    u_gtr : np.ndarray
        Ground-truth data, shape (N_out, n_vars)
    u_prd : np.ndarray
        Model predictions, shape (N_out, n_vars)
    x_inp : np.ndarray
        Input coordinates, shape (N_inp, 2)
    x_out : np.ndarray
        Output coordinates, shape (N_out, 2)
    symmetric : bool or list of bool
        Whether to use symmetric colorscale per variable
    names : list of str, optional
        Variable names for labels
    domain : tuple of list, optional
        Plot domain ([x_min, y_min], [x_max, y_max])
    colorbar_type : str
        "light" or "dark" colorbar style
    show_error : bool
        Whether to show error column
        
    Returns
    -------
    fig : plt.Figure
        The generated figure
    """
    _HEIGHT_PER_ROW = 1.9
    _HEIGHT_MARGIN = .2
    _SCATTER_SETTINGS = SCATTER_SETTINGS.copy()
    _SCATTER_SETTINGS['s'] = _SCATTER_SETTINGS['s'] * .4 * _HEIGHT_PER_ROW
    _SCATTER_SETTINGS['s'] = _SCATTER_SETTINGS['s'] * 128 / (x_inp.shape[0] ** .5)

    n_vars = u_gtr.shape[-1]
    if isinstance(symmetric, bool):
        symmetric = [symmetric] * n_vars

    # Auto-detect domain if not provided
    if domain is None:
        domain = ([x_out[:, 0].min(), x_out[:, 1].min()],
                  [x_out[:, 0].max(), x_out[:, 1].max()])

    # Calculate number of columns
    n_cols = 4 if show_error else 3
    base_width = 8.6
    figsize = (base_width * n_cols / 4.0, _HEIGHT_PER_ROW*n_vars+_HEIGHT_MARGIN)
    fig = plt.figure(figsize=figsize)
    g_fig = fig.add_gridspec(nrows=n_vars, ncols=1, wspace=0, hspace=0)

    figs = []
    for ivar in range(n_vars):
        figs.append(fig.add_subfigure(g_fig[ivar], frameon=False))
    
    # Add axes
    axs_inp = []
    axs_gtr = []
    axs_prd = []
    axs_err = []
    axs_cb_inp = []
    axs_cb_out = []
    axs_cb_err = []
    
    for ivar in range(n_vars):
        g = figs[ivar].add_gridspec(
            nrows=2,
            ncols=n_cols,
            height_ratios=[1, .05],
            wspace=0.20,
            hspace=0.05,
        )
        axs_inp.append(figs[ivar].add_subplot(g[0, 0]))
        axs_gtr.append(figs[ivar].add_subplot(g[0, 1]))
        axs_prd.append(figs[ivar].add_subplot(g[0, 2]))
        if show_error:
            axs_err.append(figs[ivar].add_subplot(g[0, 3]))
        else:
            axs_err.append(None)
        
        axs_cb_inp.append(figs[ivar].add_subplot(g[1, 0]))
        if show_error:
            axs_cb_out.append(figs[ivar].add_subplot(g[1, 1:3]))
            axs_cb_err.append(figs[ivar].add_subplot(g[1, 3]))
        else:
            axs_cb_out.append(figs[ivar].add_subplot(g[1, 1:3]))
            axs_cb_err.append(None)
    
    # Configure all axes
    all_axes = [axs_inp, axs_gtr, axs_prd]
    if show_error:
        all_axes.append(axs_err)
    for ax in [ax for axs in all_axes for ax in axs if ax is not None]:
        ax: plt.Axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([domain[0][0], domain[1][0]])
        ax.set_ylim([domain[0][1], domain[1][1]])
        ax.fill_between(
            x=[domain[0][0], domain[1][0]], 
            y1=domain[0][1], y2=domain[1][1],
            **HATCH_SETTINGS,
        )

    # Get prediction error
    u_err = np.abs(u_gtr - u_prd)

    # Choose colormap
    if colorbar_type == "light":
        cmap_symmetric = plt.cm.jet
        cmap_asymmetric = plt.cm.jet
    else:
        cmap_symmetric = CMAP_BWR
        cmap_asymmetric = CMAP_WRB

    # Plot each variable
    for ivar in range(n_vars):
        # Get ranges
        vmax_inp = np.max(u_inp[:, ivar])
        vmax_gtr = np.max(u_gtr[:, ivar])
        vmax_prd = np.max(u_prd[:, ivar])
        vmax_out = max(vmax_gtr, vmax_prd)
        vmin_inp = np.min(u_inp[:, ivar])
        vmin_gtr = np.min(u_gtr[:, ivar])
        vmin_prd = np.min(u_prd[:, ivar])
        vmin_out = min(vmin_gtr, vmin_prd)
        abs_vmax_inp = max(np.abs(vmax_inp), np.abs(vmin_inp))
        abs_vmax_out = max(np.abs(vmax_out), np.abs(vmin_out))

        # Plot input
        h = axs_inp[ivar].scatter(
            x=x_inp[:, 0],
            y=x_inp[:, 1],
            c=u_inp[:, ivar],
            cmap=(cmap_symmetric if symmetric[ivar] else cmap_asymmetric),
            vmax=(abs_vmax_inp if symmetric[ivar] else vmax_inp),
            vmin=(-abs_vmax_inp if symmetric[ivar] else vmin_inp),
            **_SCATTER_SETTINGS,
        )
        cb = plt.colorbar(h, cax=axs_cb_inp[ivar], orientation='horizontal')
        cb.formatter.set_powerlimits((-0, 0))
        
        # Plot ground truth
        h = axs_gtr[ivar].scatter(
            x=x_out[:, 0],
            y=x_out[:, 1],
            c=u_gtr[:, ivar],
            cmap=(cmap_symmetric if symmetric[ivar] else cmap_asymmetric),
            vmax=(abs_vmax_out if symmetric[ivar] else vmax_out),
            vmin=(-abs_vmax_out if symmetric[ivar] else vmin_out),
            **_SCATTER_SETTINGS,
        )
        
        # Plot prediction
        h = axs_prd[ivar].scatter(
            x=x_out[:, 0],
            y=x_out[:, 1],
            c=u_prd[:, ivar],
            cmap=(cmap_symmetric if symmetric[ivar] else cmap_asymmetric),
            vmax=(abs_vmax_out if symmetric[ivar] else vmax_out),
            vmin=(-abs_vmax_out if symmetric[ivar] else vmin_out),
            **_SCATTER_SETTINGS,
        )
        cb = plt.colorbar(h, cax=axs_cb_out[ivar], orientation='horizontal')
        cb.formatter.set_powerlimits((-0, 0))

        # Plot RELATIVE error (as percentage 0-100%)
        if show_error:
            # Compute relative error: |pred - gt| / |gt| * 100
            # Avoid division by zero by adding small epsilon
            rel_err = np.abs(u_err[:, ivar]) / (np.abs(u_gtr[:, ivar]) + 1e-10) * 100
            # Clip to reasonable range (0-100%)
            rel_err = np.clip(rel_err, 0, 100)
            
            h = axs_err[ivar].scatter(
                x=x_out[:, 0],
                y=x_out[:, 1],
                c=rel_err,
                cmap=cmap_asymmetric,
                vmin=0,
                vmax=100,  # FIXED scale: 0-100%
                **_SCATTER_SETTINGS,
            )
            cb = plt.colorbar(h, cax=axs_cb_err[ivar], orientation='horizontal')
            cb.set_label('Rel. Error (%)', fontsize=8)
            cb.ax.xaxis.set_tick_params(labelsize=8)

    # Set titles
    axs_inp[0].set(title='Input')
    axs_gtr[0].set(title='Ground-truth')
    axs_prd[0].set(title='Model estimate')
    if show_error:
        axs_err[0].set(title='Relative error (%)')

    # Set variable names
    for ivar in range(n_vars):
        label = names[ivar] if names else f'Variable {ivar:02d}'
        axs_inp[ivar].set(ylabel=label)

    # Format colorbars
    cb_axes = [axs_cb_inp, axs_cb_out]
    if show_error:
        cb_axes.append(axs_cb_err)
    for ax in [ax for axs in cb_axes for ax in axs if ax is not None]:
        ax: plt.Axes
        ax.xaxis.get_offset_text().set(size=8)
        ax.xaxis.set_tick_params(labelsize=8)

    return fig


def create_sequential_animation(
    gt_sequence: np.ndarray,
    pred_sequence: np.ndarray,
    coords: np.ndarray,
    save_path: str,
    input_data: np.ndarray = None,
    time_values: List[float] = None,
    interval: int = 500,
    symmetric: Union[bool, List[bool]] = True,
    domain: Tuple[List[float], List[float]] = None,
    names: List[str] = None,
    colorbar_type: str = "light",
    show_error: bool = True,
    dynamic_colorscale: bool = False,
) -> None:
    """
    Create animation comparing ground truth and predictions over time.
    
    Parameters
    ----------
    gt_sequence : np.ndarray
        Ground truth sequence, shape (n_timesteps, n_points, n_channels)
    pred_sequence : np.ndarray
        Prediction sequence, shape (n_timesteps, n_points, n_channels)
    coords : np.ndarray
        Spatial coordinates, shape (n_points, 2)
    save_path : str
        Path to save animation (should end with .gif or .mp4)
    input_data : np.ndarray, optional
        Initial input data, shape (n_points, n_channels)
    time_values : list of float, optional
        Time values for each frame
    interval : int
        Delay between frames in milliseconds
    symmetric : bool or list of bool
        Whether to use symmetric colorscale per variable
    domain : tuple, optional
        Plot domain ([x_min, y_min], [x_max, y_max])
    names : list of str, optional
        Variable names
    colorbar_type : str
        "light" or "dark"
    show_error : bool
        Whether to show error column
    dynamic_colorscale : bool
        Whether to update colorscale per frame
    """
    try:
        from matplotlib.animation import FuncAnimation
    except ImportError:
        print("Matplotlib animation not available")
        return
    
    if coords.shape[1] != 2:
        print("Animation currently only supports 2D coordinates")
        return
    
    n_timesteps, n_points, n_channels = gt_sequence.shape
    
    _HEIGHT_PER_ROW = 1.9
    _HEIGHT_MARGIN = .2
    _SCATTER_SETTINGS = SCATTER_SETTINGS.copy()
    _SCATTER_SETTINGS['s'] = _SCATTER_SETTINGS['s'] * .4 * _HEIGHT_PER_ROW
    _SCATTER_SETTINGS['s'] = _SCATTER_SETTINGS['s'] * 128 / (coords.shape[0] ** .5)
    
    if isinstance(symmetric, bool):
        symmetric = [symmetric] * n_channels
    
    if colorbar_type == "light":
        cmap_symmetric = plt.cm.jet
        cmap_asymmetric = plt.cm.jet
    else:
        cmap_symmetric = CMAP_BWR
        cmap_asymmetric = CMAP_WRB
    
    # Auto-detect domain
    if domain is None:
        domain = ([coords[:, 0].min(), coords[:, 1].min()],
                  [coords[:, 0].max(), coords[:, 1].max()])
    
    n_cols = 4 if show_error else 3
    base_width = 8.6
    figsize = (base_width * n_cols / 4.0, _HEIGHT_PER_ROW*n_channels+_HEIGHT_MARGIN+1.5)
    fig = plt.figure(figsize=figsize)
    g_fig = fig.add_gridspec(nrows=n_channels, ncols=1, wspace=0, hspace=0)

    figs = []
    for ivar in range(n_channels):
        figs.append(fig.add_subfigure(g_fig[ivar], frameon=False))
    
    scatter_objects = {'inp': [], 'gt': [], 'pred': [], 'error': []}
    axes_inp = []
    axes_gt = []
    axes_pred = []
    axes_err = []
    axes_cb_inp = []
    axes_cb_gt = []
    axes_cb_err = []
    
    # Create axes for each channel
    for ivar in range(n_channels):
        g = figs[ivar].add_gridspec(
            nrows=2,
            ncols=n_cols,
            height_ratios=[1, .05],
            wspace=0.20,
            hspace=0.05,
        )
        axes_inp.append(figs[ivar].add_subplot(g[0, 0]))
        axes_gt.append(figs[ivar].add_subplot(g[0, 1]))
        axes_pred.append(figs[ivar].add_subplot(g[0, 2]))
        if show_error:
            axes_err.append(figs[ivar].add_subplot(g[0, 3]))
        else:
            axes_err.append(None)
            
        axes_cb_inp.append(figs[ivar].add_subplot(g[1, 0]))
        if show_error:
            axes_cb_gt.append(figs[ivar].add_subplot(g[1, 1:3]))
            axes_cb_err.append(figs[ivar].add_subplot(g[1, 3]))
        else:
            axes_cb_gt.append(figs[ivar].add_subplot(g[1, 1:3]))
            axes_cb_err.append(None)
    
    # Configure axes
    all_axes = [axes_inp, axes_gt, axes_pred]
    if show_error:
        all_axes.append(axes_err)
    for ax in [ax for axs in all_axes for ax in axs if ax is not None]:
        ax: plt.Axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim([domain[0][0], domain[1][0]])
        ax.set_ylim([domain[0][1], domain[1][1]])
        ax.fill_between(
            x=[domain[0][0], domain[1][0]], 
            y1=domain[0][1], y2=domain[1][1],
            **HATCH_SETTINGS,
        )
    
    # Initialize scatter plots
    for ivar in range(n_channels):
        gt_all = gt_sequence[:, :, ivar]
        pred_all = pred_sequence[:, :, ivar]
        
        vmax_gtr = np.max(gt_all)
        vmax_prd = np.max(pred_all)
        vmax_out = max(vmax_gtr, vmax_prd)
        vmin_gtr = np.min(gt_all)
        vmin_prd = np.min(pred_all)
        vmin_out = min(vmin_gtr, vmin_prd)
        abs_vmax_out = max(np.abs(vmax_out), np.abs(vmin_out))
        
        if input_data is not None:
            vmax_inp = np.max(input_data[:, ivar])
            vmin_inp = np.min(input_data[:, ivar])
            abs_vmax_inp = max(np.abs(vmax_inp), np.abs(vmin_inp))
            
            h_inp = axes_inp[ivar].scatter(
                x=coords[:, 0],
                y=coords[:, 1],
                c=input_data[:, ivar],
                cmap=(cmap_symmetric if symmetric[ivar] else cmap_asymmetric),
                vmax=(abs_vmax_inp if symmetric[ivar] else vmax_inp),
                vmin=(-abs_vmax_inp if symmetric[ivar] else vmin_inp),
                **_SCATTER_SETTINGS,
            )
            scatter_objects['inp'].append(h_inp)
            cb_inp = plt.colorbar(h_inp, cax=axes_cb_inp[ivar], orientation='horizontal')
            cb_inp.formatter.set_powerlimits((-0, 0))
        else:
            h_inp = axes_inp[ivar].scatter([], [], **_SCATTER_SETTINGS)
            scatter_objects['inp'].append(h_inp)
        
        # Ground truth
        h_gt = axes_gt[ivar].scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            c=gt_sequence[0, :, ivar],
            cmap=(cmap_symmetric if symmetric[ivar] else cmap_asymmetric),
            vmax=(abs_vmax_out if symmetric[ivar] else vmax_out),
            vmin=(-abs_vmax_out if symmetric[ivar] else vmin_out),
            **_SCATTER_SETTINGS,
        )
        scatter_objects['gt'].append(h_gt)
        cb_gt = plt.colorbar(h_gt, cax=axes_cb_gt[ivar], orientation='horizontal')
        cb_gt.formatter.set_powerlimits((-0, 0))
        
        # Prediction
        h_pred = axes_pred[ivar].scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            c=pred_sequence[0, :, ivar],
            cmap=(cmap_symmetric if symmetric[ivar] else cmap_asymmetric),
            vmax=(abs_vmax_out if symmetric[ivar] else vmax_out),
            vmin=(-abs_vmax_out if symmetric[ivar] else vmin_out),
            **_SCATTER_SETTINGS,
        )
        scatter_objects['pred'].append(h_pred)
        
        if show_error:
            # Compute RELATIVE error as percentage (0-100%)
            abs_err_0 = np.abs(gt_sequence[0, :, ivar] - pred_sequence[0, :, ivar])
            rel_err_0 = abs_err_0 / (np.abs(gt_sequence[0, :, ivar]) + 1e-10) * 100
            rel_err_0 = np.clip(rel_err_0, 0, 100)
            
            h_err = axes_err[ivar].scatter(
                x=coords[:, 0],
                y=coords[:, 1],
                c=rel_err_0,
                cmap=cmap_asymmetric,
                vmin=0,
                vmax=100,  # FIXED scale: 0-100%
                **_SCATTER_SETTINGS,
            )
            scatter_objects['error'].append(h_err)
            cb_err = plt.colorbar(h_err, cax=axes_cb_err[ivar], orientation='horizontal')
            cb_err.set_label('Rel. Error (%)', fontsize=8)
            cb_err.ax.xaxis.set_tick_params(labelsize=8)
        else:
            scatter_objects['error'].append(None)
    
    axes_inp[0].set(title='Input')
    axes_gt[0].set(title='Ground truth')
    axes_pred[0].set(title='Prediction')
    if show_error:
        axes_err[0].set(title='Relative Error (%)')
    
    for ivar in range(n_channels):
        label = names[ivar] if names and ivar < len(names) else f'Variable {ivar:02d}'
        axes_inp[ivar].set(ylabel=label)
    
    cb_axes = [axes_cb_inp, axes_cb_gt]
    if show_error:
        cb_axes.append(axes_cb_err)
    for ax in [ax for axs in cb_axes for ax in axs if ax is not None]:
        ax: plt.Axes
        ax.xaxis.get_offset_text().set(size=8)
        ax.xaxis.set_tick_params(labelsize=8)
    
    # Add progress bar
    fig.subplots_adjust(bottom=0.15)
    progress_ax = fig.add_axes([0.15, 0.02, 0.7, 0.012])
    progress_bar = progress_ax.barh([0], [0], height=0.8, color='steelblue', alpha=0.7)
    progress_ax.set_xlim(0, n_timesteps)
    progress_ax.set_ylim(-0.5, 0.5)
    progress_ax.axis('off')
    
    progress_text = progress_ax.text(0.5, -3.0, '', ha='center', va='top',
                                     transform=progress_ax.transAxes, fontsize=10,
                                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def animate(frame):
        """Update function for animation."""
        for ivar in range(n_channels):
            scatter_objects['gt'][ivar].set_array(gt_sequence[frame, :, ivar])
            scatter_objects['pred'][ivar].set_array(pred_sequence[frame, :, ivar])
            
            if show_error and scatter_objects['error'][ivar] is not None:
                # Compute RELATIVE error as percentage
                abs_err = np.abs(gt_sequence[frame, :, ivar] - pred_sequence[frame, :, ivar])
                rel_err = abs_err / (np.abs(gt_sequence[frame, :, ivar]) + 1e-10) * 100
                rel_err = np.clip(rel_err, 0, 100)
                scatter_objects['error'][ivar].set_array(rel_err)
            
            if dynamic_colorscale:
                gt_frame = gt_sequence[frame, :, ivar]
                pred_frame = pred_sequence[frame, :, ivar]
                vmin_frame = min(gt_frame.min(), pred_frame.min())
                vmax_frame = max(gt_frame.max(), pred_frame.max())
                
                if symmetric[ivar]:
                    abs_vmax_frame = max(abs(vmin_frame), abs(vmax_frame))
                    vmin_frame = -abs_vmax_frame
                    vmax_frame = abs_vmax_frame
                
                scatter_objects['gt'][ivar].set_clim(vmin=vmin_frame, vmax=vmax_frame)
                scatter_objects['pred'][ivar].set_clim(vmin=vmin_frame, vmax=vmax_frame)
        
        progress_bar[0].set_width(frame + 1)
        
        if time_values and frame < len(time_values):
            progress_text.set_text(f'Time: {time_values[frame]:.3e} (frame {frame+1}/{n_timesteps})')
        else:
            progress_text.set_text(f'Frame: {frame+1}/{n_timesteps}')
        
        all_scatters = []
        for key in scatter_objects:
            all_scatters.extend([obj for obj in scatter_objects[key] if obj is not None])
        return all_scatters + [progress_bar[0], progress_text]
    
    anim = FuncAnimation(fig, animate, frames=n_timesteps,
                        interval=interval, blit=False, repeat=True)
    
    print(f"Saving animation to {save_path}...")
    try:
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=1000//interval, dpi=150)
        elif save_path.endswith('.mp4'):
            anim.save(save_path, writer='ffmpeg', fps=1000//interval, dpi=150)
        else:
            save_path_gif = save_path + '.gif'
            anim.save(save_path_gif, writer='pillow', fps=1000//interval, dpi=150)
            print(f"Animation saved as {save_path_gif}")
            return
        print(f"Animation saved successfully: {save_path}")
    except Exception as e:
        print(f"Failed to save animation: {e}")
        print("Try installing pillow (pip install pillow) for GIF support")
    
    plt.close(fig)



