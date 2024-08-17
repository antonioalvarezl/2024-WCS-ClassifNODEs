import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
import numpy as np
import torch
from matplotlib.ticker import FormatStrFormatter

@torch.no_grad()
def levelsets(model, ax=None, path=None, fig_name=None, footnote=None, 
              contour=True, bar=True, plotlim=[-0.5, 1.5], step=0.01, dpi=100, 
              points=[], transformed_sets=False, return_fig=False):
    """
    Generates and plots the level sets for a given model.
    
    Parameters:
    - model (torch.nn.Module): The model whose level sets are to be plotted.
    - ax (matplotlib.axes.Axes): Axes on which to plot. If None, a new figure and axes are created.
    - path (str): Path where the plot should be saved.
    - fig_name (str): Name of the figure file.
    - footnote (str): Footnote text to be added to the figure.
    - contour (bool): Whether to plot contours.
    - bar (bool): Whether to add a color bar.
    - plotlim (list): Plot limits for the x and y axes.
    - step (float): Step size for the meshgrid.
    - dpi (int): Dots per inch for the figure.
    - points (list): List containing X0 and X1 points to be plotted.
    - transformed_sets (bool): Whether to plot transformed sets.
    - return_fig (bool): Whether to return the figure and axes.

    Returns:
    - If return_fig is True, returns the figure and axes.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X0, X1 = points

    # Initialize figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=dpi)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    if footnote:
        plt.figtext(0.5, 0, footnote, ha="center", fontsize=10)

    # Move model to the appropriate device
    model.to(device)

    # Generate a meshgrid for the model input
    x1 = torch.arange(plotlim[0], plotlim[1], step, device=device)
    x2 = torch.arange(plotlim[0], plotlim[1], step, device=device)
    xx1, xx2 = torch.meshgrid(x1, x2, indexing='xy')
    model_inputs = torch.stack([xx1, xx2], dim=-1)

    # Get model predictions
    if model.trained:
        preds, _ = model(model_inputs, 0)
    else:
        preds, _ = model(model_inputs)

    # Handle classification outputs with multiple dimensions
    if model.output_dim > 1:
        preds = torch.nn.Softmax(dim=2)(preds)

    # Adjust coordinates if using transformed sets
    if transformed_sets:
        transformed_inputs, _ = model(model_inputs, 0)
        xx1, xx2 = transformed_inputs[:, :, 0], transformed_inputs[:, :, 1]

    # Prepare for plotting
    if preds.dim() == 3:
        preds = preds[:, :, 0]  # Assuming we are interested in class 1 probability
        preds = preds.unsqueeze(2)

    # Set plot parameters
    ax.set_xlim(plotlim)
    ax.set_ylim(plotlim)
    ax.set_aspect('equal')
    ax.grid(False)
        
    # Plot the contours if required
    if contour:
        if preds.dim() == 3:
            colors = ['#FF5733', [1, 1, 1], to_rgb("C0")]  # Gradient from Orange to Blue
            cm = LinearSegmentedColormap.from_list("Custom", colors, N=40)
            z = preds.detach().cpu().numpy().reshape(xx1.shape)
            levels = np.linspace(0., 1., 8).tolist()
        elif preds.dim() == 2:
            colors = [to_rgb("C0"), [1, 1, 1], '#FF5733']
            cm = LinearSegmentedColormap.from_list("Custom", colors, N=40)
            z = preds.detach().cpu().numpy()
            z = np.clip(z, 0, 1)
            levels = np.linspace(0, 1., 8).tolist()

        xx1_np = xx1.detach().cpu().numpy()
        xx2_np = xx2.detach().cpu().numpy()
        cont = ax.contourf(xx1_np, xx2_np, z, levels, alpha=1, cmap=cm, zorder=0)
        if bar:
            cbar = fig.colorbar(cont, fraction=0.046, pad=0.04)
            cbar.ax.set_ylabel('Prediction Probability')

    # Plot points
    point_size = 16
    ax.scatter(X0[:, 0], X0[:, 1], c='C0', marker='X', edgecolor="black", linewidth=0.65, alpha=0.75, s=point_size)
    ax.scatter(X1[:, 0], X1[:, 1], c='#FF5733', marker='o', edgecolor="black", linewidth=0.65, alpha=0.75, s=point_size)
    ax.set_aspect('equal')

    # Save the figure if fig_name is provided
    if fig_name:
        full_path = os.path.join(path if path else '', fig_name + '.png')
        plt.savefig(full_path, bbox_inches='tight', dpi=400, format='png', facecolor='white')
        plt.clf()
        plt.show()
        plt.close(fig)

    if return_fig:
        return fig, ax

def loss_evolution(trainer, epoch, path, filename='', figsize=(8, 6), footnote=None, showl2norm=False):
    """
    Plots the evolution of the loss during training.

    Parameters:
    - trainer (object): The trainer object containing the history of training losses.
    - epoch (int): The current epoch number.
    - path (str): Path where the plot should be saved.
    - filename (str): Name of the file where the plot should be saved.
    - figsize (tuple): Size of the figure.
    - footnote (str): Footnote text to be added to the figure.
    - showl2norm (bool): Whether to show L2 norm of the parameters.
    - showgradnorm (bool): Whether to show gradient norm.

    Returns:
    - None
    """
    
    fig, ax = plt.subplots(dpi=100, figsize=figsize)
    labelsize = 10

    # Plot loss history
    epoch_scale = list(range(1, len(trainer.histories['loss_history']) + 1))
    ax.plot(epoch_scale, trainer.histories['loss_history'], 'k', alpha=0.5, label='Training Loss')
    ax.plot(epoch_scale[:epoch], trainer.histories['loss_history'][:epoch], color='k')
    ax.scatter(epoch + 1, trainer.histories['loss_history'][epoch], color='k', zorder=1)

    # Optionally plot L2 norm of parameters
    if showl2norm:
        ax.plot(epoch_scale, trainer.histories['l2normparam_history'], 'C3--', alpha=0.5, label='L2 Norm of Parameters')
        ax.plot(epoch_scale[:epoch], trainer.histories['l2normparam_history'][:epoch], 'C3--')
        ax.scatter(epoch + 1, trainer.histories['l2normparam_history'][epoch], color='C3', zorder=1)

    ax.autoscale_view()
    ax.grid(True, zorder=-2)
    ax.yaxis.tick_right()
    ax.set_aspect('auto')
    ax.set_axisbelow(True)
    ax.set_xlabel('Epochs', size=labelsize)
    ax.set_ylabel('Loss', size=labelsize)
    ax.legend(prop={'size': labelsize}, framealpha=1)

    if footnote:
        plt.figtext(0.5, -0.005, footnote, ha="center", fontsize=9)

    # Save the figure if filename is provided
    if filename:
        figname = os.path.join(path, filename)
        plt.savefig(figname, bbox_inches='tight', dpi=100, format='png', facecolor='white')
        plt.close(fig)
    else:
        plt.show()

def plot_data(model, inputs, targets, N, dpi=200, alpha=0.75, path='', trajs=False, init=False, final=False, rescale = False):
    """
    Plots data points or their trajectories as generated by the model.

    Parameters:
    - model (torch.nn.Module): The model used to generate trajectories.
    - inputs (Tensor): Input data points.
    - targets (Tensor): Labels of the data points.
    - N (int): Number of data points.
    - dpi (int): Resolution of the plot.
    - alpha (float): Transparency level of the plotted points.
    - path (str): Directory where the plot will be saved.
    - trajs (bool): If True, plot trajectories of the data points.
    - init (bool): If True, plot initial points.
    - final (bool): If True, plot final points.

    Returns:
    - final_image_filename (str): The path to the saved plot image.
    """

    T, dt = model.T, model.dt
    timesteps = int(T / dt) + 1

    # Plotting configuration
    plt.rcParams.update({
        'xtick.labelsize': 13, 
        'ytick.labelsize': 13,
        'text.usetex': True, 
        'font.family': 'serif',
        'grid.linestyle': 'dotted', 
        'grid.color': 'lightgray'
    })

    # Compute trajectories
    _, trajectories = model(inputs)
    trajectories = trajectories.detach().numpy()

    fig, ax = plt.subplots(figsize=(4, 4))  # Fixed image size
    ax.set_axisbelow(True)
    ax.grid(True)
    ax.set_facecolor('whitesmoke')

    def get_coords(time_index):
        x_coords = trajectories[time_index, :, 0]
        y_coords = trajectories[time_index, :, -1]
        return x_coords, y_coords

    # Set default file name
    final_image_filename = os.path.join(path, "plot.png")

    # Plot trajectories or initial/final points
    if trajs:
        final_image_filename = os.path.join(path, "trajectories.png")
        for i in range(N):
            ax.plot(
                trajectories[:, i, 0],
                trajectories[:, i, -1],
                color='C0' if targets[i] == 0 else '#FF5733',
                alpha=alpha * 0.5
            )
        x_coords, y_coords = trajectories[:, :, 0], trajectories[:, :, -1]
    else:
        time_index = 0 if init else timesteps - 1
        x_coords, y_coords = get_coords(time_index)
        final_image_filename = os.path.join(path, "initial_points.png") if init else os.path.join(path, "final_points.png")
        
        # Scatter plot for different target classes
        ax.scatter(
            x_coords[targets == 0], y_coords[targets == 0], 
            c='C0', s=int(3000 / N), alpha=alpha,
            marker='X', linewidth=0.65, edgecolors='black', zorder=3
        )
        ax.scatter(
            x_coords[targets == 1], y_coords[targets == 1], 
            c='#FF5733', s=int(3000 / N), alpha=alpha,
            marker='o', linewidth=0.65, edgecolors='black', zorder=3
        )

    # Determine plot limits and adjust to make it square
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    margin = 0.1
    max_range = max(x_max - x_min, y_max - y_min)
    x_center, y_center = (x_max + x_min) / 2, (y_max + y_min) / 2

    # Adjust limits to the nearest multiples of 0.5
    x_min = np.floor((x_center - max_range / 2 - margin * max_range) * 2) / 2
    x_max = np.ceil((x_center + max_range / 2 + margin * max_range) * 2) / 2
    y_min = np.floor((y_center - max_range / 2 - margin * max_range) * 2) / 2
    y_max = np.ceil((y_center + max_range / 2 + margin * max_range) * 2) / 2
    
    # Customize ticks and background for the initial points plot
    if init:
        if rescale==True:
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
        else:
            ax.set_xlim(-0.5,1.5)
            ax.set_ylim(-0.5,1.5)
    else:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        x_ticks = np.linspace(x_min, x_max, 5)
        y_ticks = np.linspace(y_min, y_max, 5)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    ax.set_aspect('equal')  # Ensures that the plot is square

    # Overlay decision boundary for final plot
    if final:
        weights = model.linear_layer.weight.data.numpy()
        bias = model.linear_layer.bias.data.numpy()
        x_points = np.linspace(x_min, x_max, 2)
        y_points = -weights[0][0] / weights[0][-1] * x_points - (bias[0] - 0.5) / weights[0][-1]
        zz = weights[0][0] * x_points + weights[0][-1] * y_points + bias[0] - 0.5

        plt.fill_between(x_points, y_points, y_max, color='#F7ABAB', where=zz >= 0, interpolate=True, alpha=0.5)
        plt.fill_between(x_points, y_points, y_min, color='lightblue', where=zz <= 0, interpolate=True, alpha=0.5)
        plt.plot(x_points, y_points, 'k-', lw=2)

    # Save and close plot
    plt.savefig(final_image_filename, format='png', dpi=dpi)
    plt.close(fig)
    return final_image_filename

def plot_logloss(trainer, full_path, export_fig=False):
    """
    Plots the logarithmic loss over epochs during training.

    Parameters:
    - trainer (object): The trainer object containing the history of training losses.
    - full_path (str): Directory where the plot will be saved.
    - export_fig (bool): If True, save the figure to the specified path.

    Returns:
    - None
    """
    
    log_histories = np.log10(trainer.histories['loss_history'])
    fig = plt.figure(figsize=(8, 24), dpi=500)

    plt.figure()
    if trainer.classif:
        title = 'Successful classification'
    else:
        if trainer.noimp:
            title = "Stopping criterion - No improvement for {} epochs".format(trainer.patience)
        elif trainer.relerr:
            title = "Stopping criterion - Relative error"
        elif trainer.nonconv:
            title = "Stopping criterion - Error over threshold in 20000 or 40000 epochs"
        else:
            title = "Reached maximum number of {} epochs".format(trainer.max_epochs)
            
    title = 'log Error vs Training Epochs: ' + title
    plt.title(title)
    plt.plot(log_histories, color='b', linewidth=2)
    plt.xlabel('Epochs')
    plt.xlim(0, len(trainer.histories['loss_history']) - 1)
    plt.grid()
    plt.tight_layout()
    
    if export_fig:
        pathfigslosses = os.path.join(full_path, 'LossVsEpochs.png')
        plt.savefig(pathfigslosses, bbox_inches='tight', dpi=300, format='png', facecolor='white')
    
    plt.show()
