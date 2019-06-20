import numpy as np

import matplotlib.pylab  as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_point_cloud(x, y, z, show=True, show_axis=True, in_fixed_range=False, marker='.', s=10, alpha=.8, figsize=(8, 8), elev=10, azim=240, axis=None, title=None, *args, **kwargs):
    if axis is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = axis
        fig = axis
    if title is not None:
        plt.title(title)
    sc = ax.scatter(x, y, z, marker=marker, s=s, alpha=alpha, *args, **kwargs)
    ax.view_init(elev=elev, azim=azim)
    if in_fixed_range:
        ax.set_xlim3d(-0.25, 0.25)
        ax.set_ylim3d(-0.25, 0.25)
        ax.set_zlim3d(-0.25, 0.25)
    else:
        miv = 0.7 * np.min([np.min(x), np.min(y), np.min(z)])  # Multiply with 0.7 to squeeze free-space.
        mav = 0.7 * np.max([np.max(x), np.max(y), np.max(z)])
        ax.set_xlim(miv, mav)
        ax.set_ylim(miv, mav)
        ax.set_zlim(miv, mav)
        plt.tight_layout()
    if not show_axis:
        plt.axis('off')
    if show:
        plt.show()
    else:
        fig.canvas.draw()
    return fig

if __name__ == '__main__':

    pcs = np.load("output/val_pcs.npy")

    print(pcs.shape)
    print(pcs[0].shape)

    views = []
    for idx, pc in enumerate(pcs[:1]):
        fig = plot_3d_point_cloud(pc[:, 0], pc[:, 1], pc[:, 2],
                                  show = False, in_fixed_range=True);
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        views.append(data.reshape(fig.canvas.get_width_height()[::-1] + (3,)))
        if idx % 100 == 0:
            print("done with idx: {}".format(idx))
    print(views[0].shape)
    np.save("output/val_views.npy", np.array(views))
