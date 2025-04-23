from datetime import datetime
from torch import optim
import os
import torch
import torch.nn.functional as F
import matplotlib.colors as colors
from sklearn.metrics import f1_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from _models import NBVAE, VAE
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
import umap

# -----------------------
# -  Plotting defaults  -
# -----------------------
pio.templates["myname"] = go.layout.Template(
    layout=go.Layout(
    colorway=['#636EFA', '#F8766D', '#00BF7D', '#A3A500', '#E76BF3'],
    autosize=True,
    width=700,
    height=500,
    margin=dict(
        l=0,
        r=0,
        t=0,
        b=0,
        pad=0),
    font=dict(size=24,
          family="Palatino"),
    yaxis=dict(title=dict(font=dict(size=24))),
    xaxis=dict(title=dict(font=dict(size=24))),
    yaxis_tickfont_size=24,
    xaxis_tickfont_size=24,
    yaxis_titlefont_size=30,
    xaxis_titlefont_size=30,
    )
)            
pio.templates.default = 'myname+ggplot2'

def set_all_seeds(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def remove_axes(fig):
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks([])
    plt.yticks([])
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)

def set_plt_layout():
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.style.use('ggplot')
    plt.rcParams.update({'axes.labelsize': 22})
    plt.rcParams.update({'xtick.labelsize': 18})
    plt.rcParams.update({'ytick.labelsize': 18})
    plt.rcParams.update({'legend.fontsize': 18})
    

def reset_plt_layout():
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.style.use('default')

# -----------------
# -  Model utils  -
# -----------------
# def save_model(model, optimizer, loss_hist, path):
#     if not os.path.exists('checkpoints'):
#         os.makedirs('checkpoints')
        
#     full_path = f"checkpoints/{path}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
#     torch.save({
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'loss_hist': loss_hist
#     }, full_path)
#     print(f"Model saved to {full_path}.")


# def load_model(path, model_type, model_params, dev):
#     # check if path is a full path or just a filename
#     if not os.path.exists(path):
#         path = f"checkpoints/{path}"
#     checkpoint = torch.load(path, map_location=dev)
#     # create model based on model_type
#     if model_type == "gmvae":
#         model = GMVAE(**model_params).to(dev)
#     elif model_type == "vae":
#         model = VAE(**model_params).to(dev)
#     elif model_type == "nbvae":
#         model = NBVAE(**model_params).to(dev)
#     else:
#         raise ValueError("Model type not recognized.")   

#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#     loss_hist = checkpoint['loss_hist']
#     print(f"Model loaded from {path}.")
#     return model, optimizer, loss_hist

def save_model(model, optimizer, loss_hist, model_params, path):
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_hist': loss_hist, 
        'model_params': model_params
    }
    full_path = os.path.join('checkpoints', f"{path}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")
    torch.save(checkpoint, full_path)
    print(f"Model saved to {full_path}.")

def load_model(path, model_type, model_params=None, dev=None):
    if not os.path.exists(path):
        path = os.path.join("checkpoints", path)
    checkpoint = torch.load(path, map_location=dev)
    if model_params is None and 'model_params' in checkpoint:
        model_params = checkpoint['model_params']
    if model_type == "gmvae":
        model = GMVAE(**model_params).to(dev)
    elif model_type == "vae":
        model = VAE(**model_params).to(dev)
    elif model_type == "nbvae":
        model = NBVAE(**model_params).to(dev)
    else:
        raise ValueError("Model type not recognized.")
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    loss_hist = checkpoint.get('loss_hist', None)
    print(f"Model loaded from {path}.")
    return model, optimizer, loss_hist, model_params 





# ----------------------
# -  Evaluation utils  -
# ----------------------
def best_ari(mu, labels, ks=np.arange(3,16,1)):
    res = np.zeros(len(ks))
    for i, k in enumerate(ks):
        kmeans = KMeans(n_clusters=k, random_state=0)
        predicted_labels = kmeans.fit_predict(mu)
        res[i] = metrics.adjusted_rand_score(labels, predicted_labels)
    print("best k:", ks[np.argmax(res)], "ari:", np.max(res))
    return res, ks[np.argmax(res)], res


def evaluate_dataset(data, log_data, data_lib, model, flavor=None, compute_ari=False, ari_ks=np.arange(3,16,1), labels=None, mask=None, obs=None):
    dev = next(model.parameters()).device
    data = data.to(dev)
    log_data = log_data.to(dev)
    data_lib = data_lib.to(dev)

    if labels is not None: 
        labels = labels.to(dev)
        mask = mask.to(dev)

    metrics = {}
    with torch.no_grad():
        x_hat, mu, _ = model(log_data)

        if flavor == "embryotime":
            label_preds = x_hat[:,-1]
            x_hat = x_hat[:,:-1]
            metrics["l1"]     = torch.mean(torch.abs(labels[mask] - label_preds[mask])).item()
            metrics["l1_all"] = torch.mean(torch.abs(labels - label_preds)).item()
        elif flavor == "cardiotoxin":
            label_preds = F.sigmoid(x_hat[:,-1])
            x_hat = x_hat[:,:-1]
            metrics["acc"]     = torch.mean(((label_preds[mask] > 0.5) == labels[mask]).float()).item()
            metrics["acc_all"] = torch.mean(((label_preds > 0.5) == labels).float()).item()
            metrics["f1"]     = f1_score(labels.cpu().numpy(), (label_preds > 0.5).cpu().numpy(), average='binary', sample_weight=mask.cpu().numpy())
            metrics["f1_all"] = f1_score(labels.cpu().numpy(), (label_preds > 0.5).cpu().numpy(), average='binary')

        metrics["rmse"] = F.mse_loss(torch.log1p(x_hat), torch.log1p(data/data_lib), reduction='mean').item()
        metrics["mae"] = F.l1_loss(torch.log1p(x_hat), torch.log1p(data/data_lib), reduction='mean').item()

        # unused for now
        # metrics['nll'] = model.nb.log_prob(data, data_lib, x_hat).mean().item()

        x_hat = x_hat.cpu(); mu = mu.cpu()

        if compute_ari:
            subgroup = "time" if flavor == "cardiotoxin" else "cell_type"

            if mu.shape[1] > 2:
                umap_model = umap.UMAP(n_components=2, random_state=0)
                mu_2d = umap_model.fit_transform(mu)
            else:
                mu_2d = mu

            res, k, res = best_ari(mu_2d, obs[subgroup].cat.codes.values, ari_ks)
            metrics["ari"] = np.max(res)
            print(f'ari: {metrics["ari"]:.4f}')

        if flavor == "embryotime":
            print(f'rmse: {metrics["rmse"]:.4f} mae: {metrics["mae"]:.4f} Subset L1: {metrics["l1"]:.4f} All L1: {metrics["l1_all"]:.4f}')
        elif flavor == "cardiotoxin":
            print(f'rmse: {metrics["rmse"]:.4f} mae: {metrics["mae"]:.4f} Sub acc: {metrics["acc"]:.4f} All acc: {metrics["acc_all"]:.4f} Sub F1: {metrics["f1"]:.4f} All F1: {metrics["f1_all"]:.4f}')
        else:
            print(f'rmse: {metrics["rmse"]:.4f} mae: {metrics["mae"]:.4f}')

    return x_hat, mu, metrics


# --------------------
# -  Coloring utils  -
# --------------------
def truncate_colormap(name, minval=0.0, maxval=1.0, n=100):
    cmap = plt.get_cmap(name)
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def take_cmap_colors(cmap, N, cmap_range=(0, 1), return_fmt="float"):
    """
    From https://cmasher.readthedocs.io/_modules/cmasher/utils.html#take_cmap_colors
    Takes `N` equally spaced colors from the provided colormap `cmap` and
    returns them.

    Parameters
    ----------
    cmap : str or :obj:`~matplotlib.colors.Colormap` object
        The registered name of the colormap in :mod:`matplotlib.cm` or its
        corresponding :obj:`~matplotlib.colors.Colormap` object.
    N : int or None
        The number of colors to take from the provided `cmap` within the given
        `cmap_range`.
        If *None*, take all colors in `cmap` within this range.

    Optional
    --------
    cmap_range : tuple of float. Default: (0, 1)
        The normalized value range in the colormap from which colors should be
        taken.
        By default, colors are taken from the entire colormap.
    return_fmt : {'float'/'norm'; 'int'/'8bit'; 'str'/'hex'}. Default: 'float'
        The format of the requested colors.
        If 'float'/'norm', the colors are returned as normalized RGB tuples.
        If 'int'/'8bit', the colors are returned as 8-bit RGB tuples.
        If 'str'/'hex', the colors are returned using their hexadecimal string
        representations.

    Returns
    -------
    colors : list of {tuple; str}
        The colors that were taken from the provided `cmap`.
    """
    from matplotlib.colors import Colormap, ListedColormap as LC, to_hex, to_rgb
    from matplotlib import cm as mplcm

    # Convert provided fmt to lowercase
    return_fmt = return_fmt.lower()

    # Obtain the colormap
    cmap = mplcm.get_cmap(cmap)

    # Check if provided cmap_range is valid
    if not ((0 <= cmap_range[0] <= 1) and (0 <= cmap_range[1] <= 1)):
        raise ValueError("Input argument 'cmap_range' does not contain "
                         "normalized values!")

    # Extract and convert start and stop to their integer indices (inclusive)
    start = int(np.floor(cmap_range[0]*cmap.N))
    stop = int(np.ceil(cmap_range[1]*cmap.N))-1

    # Pick colors
    if N is None:
        index = np.arange(start, stop+1, dtype=int)
    else:
        index = np.array(np.rint(np.linspace(start, stop, num=N)), dtype=int)
    colors = cmap(index)

    # Convert colors to proper format
    if return_fmt in ('float', 'norm', 'int', '8bit'):
        colors = np.apply_along_axis(to_rgb, 1, colors)
        if return_fmt in ('int', '8bit'):
            colors = np.array(np.rint(colors*255), dtype=int)
        colors = list(map(tuple, colors))
    else:
        colors = list(map((lambda x: to_hex(x).upper()), colors))

    # Return colors
    return(colors)


def categorical_cmap(nc, nsc, cmap="tab10", continuous=False):
    # From https://stackoverflow.com/questions/47222585/matplotlib-generic-colormap-from-tab10
    import matplotlib.colors
    if nc > plt.get_cmap(cmap).N:
        raise ValueError("Too many categories for colormap.")
    if continuous:
        ccolors = plt.get_cmap(cmap)(np.linspace(0,1,nc))
    else:
        ccolors = plt.get_cmap(cmap)(np.arange(nc, dtype=int))
    cols = np.zeros((nc*nsc, 3))
    for i, c in enumerate(ccolors):
        chsv = matplotlib.colors.rgb_to_hsv(c[:3])
        arhsv = np.tile(chsv,nsc).reshape(nsc,3)
        arhsv[:,1] = np.linspace(chsv[1],0.25,nsc)
        arhsv[:,2] = np.linspace(chsv[2],1,nsc)
        rgb = matplotlib.colors.hsv_to_rgb(arhsv)
        cols[i*nsc:(i+1)*nsc,:] = rgb       
    cmap = matplotlib.colors.ListedColormap(cols)
    return cmap

