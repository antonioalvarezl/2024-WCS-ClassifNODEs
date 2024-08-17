import os
import imageio
import matplotlib.pyplot as plt
import torch
import random
import numpy as np
from scipy.interpolate import interp1d
from matplotlib import rc    
#rc for runtime configuration, interp1d for interpolation, and Axes3D for 3D plotting

def traj_gif(model, inputs, targets, dpi=200, path='', fps=5):
    # Plotting configuration
    rc("text", usetex=True)
    plt.rcParams.update({'font.size': 18, 'xtick.labelsize': 13, 'ytick.labelsize': 13, 
                         'text.usetex': True, 'font.family': 'serif',
                         'grid.linestyle': 'dotted', 'grid.color': 'lightgray'})
    alpha, alpha_line = 0.9, 0.5

    #Options:
    dyn_lims = False # If True, the limits of the plot are dynamic.
    normalize = False # If True, the plot is normalized.
    hyp = True # If True, the hyperplanes are plotted
    paths = False  # If True, paths are plotted
        
    # Validate filename
    filename = f"trajectories.gif"
    if not filename.endswith(".gif"):
        raise RuntimeError(f"Name must end with .gif, but ends with {filename}")
    base_filename = filename[:-4]
    
    # Define colors based on targets
    color = ['C0' if t > 0.5 else '#FF5733' for t in targets]
    
    # Compute trajectories and normalize if required
    _, trajectories = model(inputs)
    trajectories = trajectories.detach().numpy()
    
    if normalize:
        trajectories = np.tanh(trajectories)
        x_min, x_max, y_min, y_max = -1.1, 1.1, -1.1, 1.1
    else:
        x_min, x_max = trajectories[:, :, 0].min(), trajectories[:, :, 0].max()
        y_min, y_max = trajectories[:, :, 1].min(), trajectories[:, :, 1].max()
        margin = 0.1
        x_range, y_range = x_max - x_min, y_max - y_min
        x_min, x_max = x_min - margin * x_range, x_max + margin * x_range
        y_min, y_max = y_min - margin * y_range, y_max + margin * y_range
    
    T, dt = model.T, model.dt
    timesteps = int(T / dt) + 1
    integration_time = torch.linspace(0.0, T, timesteps).numpy()
    
    # Interpolate trajectories for smoother plots
    interp_time = 120
    _time = torch.linspace(0.0, T, interp_time).numpy()
    interp_funcs = [interp1d(integration_time, trajectories[:, i, j], kind='cubic', fill_value='extrapolate') 
                    for i in range(inputs.shape[0]) for j in range(2)]
    
    # Generate frames for GIF
    gif_names = []
    for t in range(interp_time):
        fig, ax = plt.subplots()
        current_time = _time[t]
        title = f"$N={len(targets)}$, $t={current_time:.2f}$"
        plt.title(title, fontsize=20)
        ax.set_axisbelow(True)
        ax.grid(True)
        ax.set_facecolor('whitesmoke')
        
        # Current point coordinates
        x_coords = [func(current_time) for func in interp_funcs[::2]]
        y_coords = [func(current_time) for func in interp_funcs[1::2]]

        # Update dynamic limits if enabled
        if not normalize and dyn_lims:
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            x_range, y_range = x_max - x_min, y_max - y_min
            x_min, x_max = x_min - margin * x_range, x_max + margin * x_range
            y_min, y_max = y_min - margin * y_range, y_max + margin * y_range
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # Plot points
        ax.scatter(x_coords, y_coords, c=color, alpha=alpha, marker='o', linewidth=0.65, edgecolors='black', zorder=3)
        k = int(_time[t]*model.num_vals/T)
        # Optionally plot hyperplanes and trajectories
        if hyp and t < interp_time - 1:
            _plot_hyperplanes(ax, model, x_min, x_max, y_min, y_max, k)
        if t == interp_time - 1:
            _plot_boundary(ax, model, x_min, x_max, y_min, y_max)
        if t > 0 and paths:
            _plot_paths(ax, interp_funcs, _time, t, color, alpha_line, inputs.shape[0])

        # Save frame
        frame_filename = os.path.join(path, f"{base_filename}_{t}.png")
        gif_names.append(frame_filename)
        plt.savefig(frame_filename, format='png', dpi=dpi)
        plt.close(fig)
    
    # Create GIF and clean up temporary images
    gif_path = _create_gif(gif_names, path, filename, fps)
    return gif_path

def _plot_hyperplanes(ax, model, x_min, x_max, y_min, y_max, k):
    # Plot hyperplanes based on model architecture
    weights = model.linear_layer.weight.data.numpy()
    x_points = np.linspace(x_min, x_max, 100)
    
    for i in range(weights.shape[0]):
        if model.architecture == 'bottleneck':
            a = model.fwd_dynamics.fc1_time[k].weight.detach().numpy()
            b = model.fwd_dynamics.fc1_time[k].bias.detach().numpy()
        elif model.architecture == 'inside':
            a = model.fwd_dynamics.fc2_time[k].weight.detach().numpy()
            b = model.fwd_dynamics.fc2_time[k].bias.detach().numpy()
        else: 
            b = model.fwd_dynamics.fc2_time[k].bias.detach().numpy()
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
        zz = a[i][0] * xx + a[i][1] * yy + b[i]
        ax.contourf(xx, yy, zz, levels=[-np.inf, 0], colors='black', alpha=0.75)
        ax.contourf(xx, yy, zz, levels=[0, np.inf], colors='lightgray', alpha=0.25)
    
        if weights[i][1] == 0:
            y_points = np.linspace(y_min, y_max, 100)
            x_points = -b[i] / a[i][0]
        else:
            x_points = np.linspace(x_min, x_max, 100)
            y_points = -a[i][0] / a[i][1] * x_points - b[i] / a[i][1]
        ax.plot(x_points, y_points, 'k--', lw=2)

def _plot_boundary(ax, model, x_min, x_max, y_min, y_max):
    # Plot boundary decision based on final layer
    weights = model.linear_layer.weight.data.numpy()
    bias = model.linear_layer.bias.data.numpy()
    x_points = np.linspace(x_min, x_max, 100)

    if weights[0][1] == 0:  # Vertical line case
        x_val = - (bias[0] - 0.5) / weights[0][0]
        y_points = np.linspace(y_min, y_max, 100)
        ax.fill_betweenx(y_points, x_val, x_max, color='lightblue', alpha=0.5)  # Fill to the right of the line
        ax.fill_betweenx(y_points, x_val, x_min, color='#F0B27A', alpha=0.5)  # Fill to the left of the line
        ax.plot([x_val] * 100, y_points, 'k-', lw=2)  # Vertical line plot
    else:  # Non-vertical line case
        y_points = -weights[0][0] / weights[0][1] * x_points - (bias[0] - 0.5) / weights[0][1]
        zz = weights[0][0] * x_points + weights[0][1] * y_points + bias[0] - 0.5
        ax.fill_between(x_points, y_points, y_max, color='lightblue', where=zz >= 0, interpolate=True, alpha=0.5)
        ax.fill_between(x_points, y_points, y_min, color='#F0B27A', where=zz <= 0, interpolate=True, alpha=0.5)
        ax.plot(x_points, y_points, 'k-', lw=2)

def _plot_paths(ax, interp_funcs, _time, t, color, alpha_line, num_inputs):
    # Plot paths of the trajectories
    for i in range(num_inputs):
        ax.plot(interp_funcs[2 * i](_time)[:t + 1], 
                interp_funcs[2 * i + 1](_time)[:t + 1], 
                c=color[i], alpha=alpha_line, linewidth=0.75, zorder=1)

def _create_gif(gif_names, path, filename, fps):
    # Create GIF from saved frames
    imgs = [np.array(imageio.imread(name)) for name in gif_names]
    gif_path = os.path.join(path, filename)
    imageio.mimwrite(gif_path, imgs, fps=fps)
    
    # Cleanup temporary images
    for img_path in gif_names:
        os.remove(img_path)
    
    return gif_path

def select_random_samples(data_loader, num_samples):
    total_samples = len(data_loader.dataset)
    if num_samples > total_samples:
        raise ValueError("Requested more samples than available in the dataset")
    # Randomly select indices from the dataset
    selected_indices = random.sample(range(total_samples), num_samples)

    # Collect the selected samples and their targets
    inputs = torch.stack([data_loader.dataset[i][0] for i in selected_indices])

    targets = torch.stack([data_loader.dataset[i][1] for i in selected_indices])

    return inputs, targets


