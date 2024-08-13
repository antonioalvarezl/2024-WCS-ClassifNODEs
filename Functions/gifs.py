import os
import imageio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import random
import numpy as np
from scipy.interpolate import interp1d
from matplotlib import rc    
#rc for runtime configuration, interp1d for interpolation, and Axes3D for 3D plotting

def traj_gif(model, inputs, targets, dpi=200, alpha=0.9, alpha_line=1.0, path='', fps=5, dyn_lims=False, normalize = False, hyp = False, paths = True):
    rc("text", usetex = True)
    font = {'size'   : 18}
    rc('font', **font)
    filename = f"trajectories.gif"
    # Validates the filename to ensure it's appropriate for a GIF file
    if not filename.endswith(".gif"):
        raise RuntimeError("Name must end in with .gif, but ends with {}".
                           format(filename))
    base_filename = filename[:-4]
    
    #Determines the colors for plotting based on the targets (2/3 labels only)
    if False in (t < 2 for t in targets): 
        color = ['mediumpurple' if targets[i] == 2.0 
                  else 'gold' if targets[i] == 0.0 
                  else 'mediumseagreen' 
                  for i in range(len(targets))]
    else:
        color = ['C0' if t > 0.5 else 'C1' for t in targets]
    #Computes the trajectories using the model over specified timesteps and detaches the result from PyTorch's computation graph.
    _, trajectories = model(inputs)
    trajectories = trajectories.detach().numpy() 
    
    if normalize: 
        trajectories = np.tanh(trajectories)
        x_min, x_max = -1.1,1.1
        y_min, y_max = -1.1,1.1
        
    elif not dyn_lims:
        x_min, x_max = trajectories[:, :, 0].min(), trajectories[:, :, 0].max()
        y_min, y_max = trajectories[:, :, 1].min(), trajectories[:, :, 1].max()
        margin = 0.1
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min, x_max = x_min - margin * x_range, x_max + margin * x_range
        y_min, y_max = y_min - margin * y_range, y_max + margin * y_range
    
    T, dt = model.T, model.dt
    timesteps = int(T/dt) + 1
    integration_time = torch.linspace(0.0, T, timesteps).numpy()
    
    # Interpolates the trajectories to obtain smoother plots
    interp_time = 120
    _time = torch.linspace(0.0, T, interp_time).numpy()
    interp_funcs = [interp1d(integration_time, trajectories[:, i, j], 
                         kind='cubic', fill_value='extrapolate') 
                for i in range(inputs.shape[0]) for j in range(2)]
    # Plotting configuration
    label_size = 13
    gif_names = []
    plt.rcParams.update({'xtick.labelsize': label_size, 'ytick.labelsize': label_size,
                         'text.usetex': True, 'font.family': 'serif',
                         'grid.linestyle': 'dotted', 'grid.color': 'lightgray'})
  
    # Generation of frames for the GIF
    for t in range(interp_time):  
        fig, ax = plt.subplots()
        switch = int(np.max(_time[t]*model.num_vals/T-1,0))
        k = int(_time[t]*model.num_vals/T)
        title = f"$N={len(targets)}$, $t={_time[t]:.2f}$, switches={switch}"
        plt.title(title, fontsize=20)
        ax.set_axisbelow(True)
        ax.grid(True)
        ax.set_facecolor('whitesmoke')
          
        # Precompute scatter data
        x_coords = [func(_time)[t] for func in interp_funcs[::2]]
        y_coords = [func(_time)[t] for func in interp_funcs[1::2]]

            
        if not normalize and dyn_lims: 
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            margin = 0.1
            x_range, y_range = x_max - x_min, y_max - y_min
            x_min, x_max = x_min - margin * x_range, x_max + margin * x_range
            y_min, y_max = y_min - margin * y_range, y_max + margin * y_range
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        # Hide axes
        #ax.xaxis.set_visible(False)
        #ax.yaxis.set_visible(False)
            
        ax.scatter(x_coords, y_coords, c=color, alpha=alpha, marker='o', linewidth=0.65, edgecolors='black', zorder=3)
        if hyp == True:
            if t == interp_time - 1:
                weights = model.linear_layer.weight.data.numpy()
                bias = model.linear_layer.bias.data.numpy()
                
                x_points = np.linspace(x_min, x_max, 100)
                if weights[0][1] == 0:  # Vertical line case
                    x_val = - (bias[0] - 0.5) / weights[0][0]
                    y_points = np.linspace(y_min, y_max, 100)
                    plt.fill_betweenx(y_points, x_val, x_max, color='lightblue', alpha=0.5)  # Fill to the right of the line
                    plt.fill_betweenx(y_points, x_val, x_min, color='#F0B27A', alpha=0.5)  # Fill to the left of the line
                    plt.plot([x_val] * 100, y_points, 'k-', lw=2)  # Vertical line plot
                else:  # Non-vertical line case
                    y_points = -weights[0][0] / weights[0][1] * x_points - (bias[0] - 0.5) / weights[0][1]
                    zz = weights[0][0] * x_points + weights[0][1] * y_points + bias[0] - 0.5
                    plt.fill_between(x_points, y_points, y_max, color='lightblue', where=zz >= 0, interpolate=True, alpha=0.5)
                    plt.fill_between(x_points, y_points, y_min, color='#F0B27A', where=zz <= 0, interpolate=True, alpha=0.5)
                    plt.plot(x_points, y_points, 'k-', lw=2)

            else:
                if model.architecture == 'bottleneck':
                    w = model.fwd_dynamics.fc3_time[k].weight.detach().numpy()
                    a = model.fwd_dynamics.fc1_time[k].weight.detach().numpy()
                    b = model.fwd_dynamics.fc1_time[k].bias.detach().numpy()
                elif model.architecture == 'inside':
                    a = model.fwd_dynamics.fc2_time[k].weight.detach().numpy()
                    b = model.fwd_dynamics.fc2_time[k].bias.detach().numpy()
                else: 
                    w = model.fwd_dynamics.fc2_time[k].weight.detach().numpy()
                    b = model.fwd_dynamics.fc2_time[k].bias.detach().numpy()
                weights = model.linear_layer.weight.data.numpy()
                bias = model.linear_layer.bias.data.numpy()
                for i in range(weights.shape[0]):
                    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
                    zz = a[i][0] * xx + a[i][1] * yy + b[i]
                    plt.contourf(xx, yy, zz, levels=[-np.inf,0], colors='black', alpha=0.75)
                    plt.contourf(xx, yy, zz, levels=[0, np.inf], colors='lightgray', alpha=0.25)
                    if weights[i][1] == 0:
                        y_points = np.linspace(y_min, y_max, 100)
                        x_points = - b[i]/a[i][0]
                else:
                    x_points = np.linspace(x_min, x_max, 100)
                    y_points = -a[i][0]/a[i][1] * x_points - b[i]/a[i][1]
                plt.plot(x_points, y_points, 'k--', lw=2)   
                         

        if t > 0 and paths == True:
            for i in range(inputs.shape[0]):
                ax.plot(interp_funcs[2*i](_time)[:t+1], 
                        interp_funcs[2*i+1](_time)[:t+1], 
                        c=color[i], alpha=alpha_line, linewidth=0.75, zorder=1)        
        # Save frames
        frame_filename = os.path.join(path, f"{base_filename}_{t}.png")
        gif_names.append(frame_filename)
        plt.savefig(frame_filename, format='png', dpi=dpi)   
        plt.close(fig)    
    
    # Save last figure with transformed level sets
    # levelsets(model, bar=False, plotlim=[min(x_min, y_min), max(x_max, y_max)], step = 0.1, transformed_sets = True)
    # final_image_filename = os.path.join(path, f"{base_filename}_final.png")
    # plt.savefig(final_image_filename, format='png', dpi=dpi)
           
    # Create GIF
    imgs = [np.array(imageio.imread(name)) for name in gif_names]
    gif_path = os.path.join(path, filename)
    imageio.mimwrite(gif_path, imgs, fps=fps)
    
    # Cleanup temporary images
    for img_path in gif_names:
        os.remove(img_path)
    return gif_path 

def traj_gif_3d(model, inputs, targets, dpi=200, alpha=0.9, alpha_line=1, path='', fps=5, dyn_lims=False, normalize=True, hyp=False):
    
    plt.rc("text", usetex=True)
    font = {'size': 18}
    plt.rc('font', **font)
    filename = f"trajectories.gif"
    if not filename.endswith(".gif"):
        raise RuntimeError(f"Name must end in with .gif, but ends with {filename}")

    base_filename = filename[:-4]
    if False in (t < 3 for t in targets):
        color = ['mediumpurple' if targets[i] == 2.0 else 'gold' if targets[i] == 0.0 else 'mediumseagreen' for i in range(len(targets))]
    else:
        color = ['C0' if t > 0.5 else 'C1' for t in targets]

    _, trajectories = model(inputs)
    trajectories = trajectories.detach().numpy()

    if normalize:
        trajectories = np.tanh(trajectories)
        x_min, x_max = y_min, y_max = z_min, z_max = -1.1, 1.1
    elif not dyn_lims:
        x_min, x_max = trajectories[:, :, 0].min(), trajectories[:, :, 0].max()
        y_min, y_max = trajectories[:, :, 1].min(), trajectories[:, :, 1].max()
        z_min, z_max = trajectories[:, :, 2].min(), trajectories[:, :, 2].max()
        margin = 0.1
        x_min, x_max = x_min - margin * (x_max - x_min), x_max + margin * (x_max - x_min)
        y_min, y_max = y_min - margin * (y_max - y_min), y_max + margin * (y_max - y_min)
        z_min, z_max = z_min - margin * (z_max - z_min), z_max + margin * (z_max - z_min)

    T, dt = model.T, model.dt
    timesteps = int(T/dt) + 1
    integration_time = torch.linspace(0.0, T, timesteps).numpy()

    interp_time = 100
    _time = torch.linspace(0.0, T, interp_time).numpy()
    interp_funcs = [interp1d(integration_time, trajectories[:, i, j], kind='cubic', fill_value='extrapolate') for i in range(inputs.shape[0]) for j in range(3)]

    gif_names = []
    plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'grid.linestyle': 'dotted', 'grid.color': 'lightgray'})

    for t in range(interp_time):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        switch = np.floor(_time[t]*(model.num_vals-1)/T)
        title = f"$N={len(targets)}$, $t={_time[t]:.2f}$, switches={switch}"
        plt.title(title, fontsize=20)
        ax.grid(False)
        ax.set_facecolor('whitesmoke')

        x_coords = [func(_time)[t] for func in interp_funcs[::3]]
        y_coords = [func(_time)[t] for func in interp_funcs[1::3]]
        z_coords = [func(_time)[t] for func in interp_funcs[2::3]]

        if not normalize and dyn_lims:
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            z_min, z_max = min(z_coords), max(z_coords)
            margin = 0.1
            x_min, x_max = x_min - margin * (x_max - x_min), x_max + margin * (x_max - x_min)
            y_min, y_max = y_min - margin * (y_max - y_min), y_max + margin * (y_max - y_min)
            z_min, z_max = z_min - margin * (z_max - z_min), z_max + margin * (z_max - z_min)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        ax.scatter(x_coords, y_coords, z_coords, c=color, alpha=alpha, marker='o', linewidth=0.65, edgecolors='black', zorder=3)
        if hyp == True:
            if t == interp_time - 1:
                weights = model.linear_layer.weight.data.numpy()
                bias = model.linear_layer.bias.data.numpy()
                
                x_points = np.linspace(x_min, x_max, 100)
                y_points = np.linspace(y_min, y_max, 100)
                x_grid, y_grid = np.meshgrid(x_points, y_points)
                if weights[0][-1] == 0:  # Special case for vertical plane
                    x_val = - (bias[0] - 0.5) / weights[0][0]
                    z_grid = np.linspace(z_min, z_max, 100)
                    y_grid_2d, z_grid_2d = np.meshgrid(y_points, z_grid)
                    ax.plot_surface(np.full_like(y_grid_2d, x_val), y_grid_2d, z_grid_2d, color='lightblue', alpha=0.5)
                else:
                    z_grid = (-weights[0][0] * x_grid - weights[0][1] * y_grid - (bias[0] - 0.5)) / weights[0][2]
                    zz = weights[0][0] * x_grid + weights[0][1] * y_grid + weights[0][2] * z_grid + bias[0] - 0.5

                    # Plot the hyperplane
                    ax.plot_surface(x_grid, y_grid, z_grid, color='lightblue', alpha=0.5)

                    # To visualize the regions above and below the hyperplane
                    ax.contour3D(x_grid, y_grid, z_grid, 50, cmap='coolwarm', alpha=0.3)
        if t > 0 and len(targets)<50:
            for i in range(inputs.shape[0]):
                ax.plot(interp_funcs[3*i](_time)[:t+1], interp_funcs[3*i+1](_time)[:t+1], interp_funcs[3*i+2](_time)[:t+1], c=color[i], alpha=alpha_line, linewidth=0.75, zorder=1)

        frame_filename = os.path.join(path, f"{base_filename}_{t}.png")
        gif_names.append(frame_filename)
        plt.savefig(frame_filename, format='png', dpi=dpi)
        plt.close(fig)

    # Generate GIF from images
    # You will need to use an external tool like ImageMagick or similar to create the actual GIF from the saved frames.

    # Create GIF
    imgs = [imageio.imread(name) for name in gif_names]
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
