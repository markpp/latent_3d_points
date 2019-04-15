import os.path as osp
import numpy as np
import matplotlib.pylab  as plt
from mpl_toolkits.mplot3d import Axes3D

import argparse
from pyntcloud import PyntCloud
import pandas as pd
from latent_3d_points.src.autoencoder import Configuration as Conf
from latent_3d_points.src.point_net_ae import PointNetAutoEncoder
from latent_3d_points.src.tf_utils import reset_tf_graph


def plot_3d_point_cloud(x, y, z, show=True, show_axis=True, in_u_sphere=False,
                        marker='.', s=8, alpha=.8, figsize=(5, 5), elev=10,
                        azim=240, axis=None, title=None, *args, **kwargs):
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

    if in_u_sphere:
        ax.set_xlim3d(-0.5, 0.5)
        ax.set_ylim3d(-0.5, 0.5)
        ax.set_zlim3d(-0.0, 1.5)
    else:
        miv = 0.7 * np.min([np.min(x), np.min(y), np.min(z)])  # Multiply with 0.7 to squeeze free-space.
        mav = 0.7 * np.max([np.max(x), np.max(y), np.max(z)])
        ax.set_xlim(miv, mav)
        ax.set_ylim(miv, mav)
        ax.set_zlim(miv, mav)
        plt.tight_layout()

    if not show_axis:
        plt.axis('off')

    if 'c' in kwargs:
        plt.colorbar(sc)

    if show:
        plt.show()

    return fig

def points2file(points,filename):
    df = pd.DataFrame(points,columns=['x', 'y', 'z'])
    pc = PyntCloud(df)
    pc.to_file(filename,as_text=True)

if __name__ == '__main__':
    """
    Main function for executing the .py script.
    Command:
        -p path/<filename>.npy
    """
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-l", "--list", type=str,
                    help="list of point cloud files")
    args = vars(ap.parse_args())

    with open(args["list"]) as f:
        pc_files = f.read().splitlines()

    test_pcs = np.empty([len(pc_files), 2048, 3], dtype=np.float32)
    for idx, point_file in enumerate(pc_files[:]):
        cloud = PyntCloud.from_file(point_file)
        test_pcs[idx, :, :] = cloud.points[:2048]
    #print(test_pcs[0].shape)



    reset_tf_graph()
    ae_configuration = 'input/configuration'
    ae_conf = Conf.load(ae_configuration)
    ae_conf.encoder_args['verbose'] = False
    ae_conf.decoder_args['verbose'] = False
    ae = PointNetAutoEncoder(ae_conf.experiment_name, ae_conf)

    restore_epoch = 3000
    ae.restore_model('model/3', restore_epoch, verbose=True)

    #latent_code = ae.transform(test_pcs[:1])
    latent_codes = ae.get_latent_codes(test_pcs)


    '''
    aug_test_pcs = np.empty([len(test_pcs), 2048, 3], dtype=np.float32)
    for idx, test_pc in enumerate(test_pcs[:]):
        aug_test_pcs[idx, :, 0] = np.add(test_pc[:,0],-0.2)
        aug_test_pcs[idx, :, 1] = np.add(test_pc[:,1],0.0)
        aug_test_pcs[idx, :, 2] = np.add(test_pc[:,2],0.0)


    aug_latent_codes = ae.get_latent_codes(aug_test_pcs)
    '''
    #print(latent_codes)


    '''
    a = latent_codes[1]
    b = aug_latent_codes[1]
    diff = a-b
    steps = np.linspace(0.0, 1.0, num=4)
    interpolations = []
    for step in steps:
        interpolations.append(a-step*diff)

    reconstructions = ae.decode(interpolations)

    for id, rec in enumerate(reconstructions):
        #plot_3d_point_cloud(rec[:, 0], rec[:, 1], rec[:, 2], in_u_sphere=True)
        points2file(rec,"rec_{}-{}.ply".format(id,len(reconstructions)))
    #feed_pc, feed_model_names, _ = all_pc_data.next_batch(10)
    #reconstructions = ae.reconstruct(feed_pc)[0]
    #latent_codes = ae.transform(feed_pc)
    '''
