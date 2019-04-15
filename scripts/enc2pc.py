import os.path as osp
import numpy as np
import json
import argparse
from pyntcloud import PyntCloud
import pandas as pd
from latent_3d_points.src.autoencoder import Configuration as Conf
from latent_3d_points.src.point_net_ae import PointNetAutoEncoder
from latent_3d_points.src.tf_utils import reset_tf_graph


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
    ap.add_argument("-e", "--enc", type=str,
                    help="numpy saved array")
    args = vars(ap.parse_args())

    enc = np.load(args["enc"])



    reset_tf_graph()
    ae_configuration = 'input/configuration'
    ae_conf = Conf.load(ae_configuration)
    ae_conf.encoder_args['verbose'] = False
    ae_conf.decoder_args['verbose'] = False
    ae = PointNetAutoEncoder(ae_conf.experiment_name, ae_conf)

    restore_epoch = 3000
    ae.restore_model('model/3', restore_epoch, verbose=True)


    reconstructions = ae.decode(enc)

    id = 0
    points2file(reconstructions[id],"rec_{}.ply".format(id))
