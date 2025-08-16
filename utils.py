import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def generate_click_heatmap(clicks, out_path, canvas_width=1200, canvas_height=2000, bins=60):
    """
    clicks: list of dicts with {'x': float, 'y': float} in pixels (relative to content canvas)
    """
    ensure_dir(out_path)
    if not clicks:
        # Make blank canvas
        fig = plt.figure(figsize=(6,10), dpi=100)
        plt.xlim([0, canvas_width])
        plt.ylim([canvas_height, 0])
        plt.title("No clicks recorded yet.")
        plt.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        return out_path

    xs = [c['x'] for c in clicks]
    ys = [c['y'] for c in clicks]

    # 2D histogram
    heatmap, xedges, yedges = np.histogram2d(xs, ys, bins=bins,
                                             range=[[0, canvas_width], [0, canvas_height]])
    heatmap = heatmap.T  # transpose so y is vertical

    fig = plt.figure(figsize=(6,10), dpi=100)
    plt.imshow(heatmap, extent=[0, canvas_width, canvas_height, 0], interpolation='nearest')
    plt.title("Click Heatmap")
    plt.xlabel("X (px)")
    plt.ylabel("Y (px)")
    plt.colorbar(label="Click density")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    return out_path
