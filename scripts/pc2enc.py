import os.path as osp
import numpy as np
import json
import argparse
from pyntcloud import PyntCloud
import pandas as pd
from latent_3d_points.src.autoencoder import Configuration as Conf
from latent_3d_points.src.point_net_ae import PointNetAutoEncoder
from latent_3d_points.src.tf_utils import reset_tf_graph



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

    np_anno = []
    for files in pc_files:
        pose_path = files[:-9]+".json"
        #print(pose_path)
        with open(pose_path) as pose_file:
            jp = json.load(pose_file)
            p = [float(jp["pos"]["x"]),float(jp["pos"]["y"]),float(jp["pos"]["z"])]
            n = [float(jp["orn"]["x"]),float(jp["orn"]["y"]),float(jp["orn"]["z"])]
            np_anno.append([p,n])
    np.save("anno",np.array(np_anno))

    test_pcs = np.empty([len(pc_files), 2048, 3], dtype=np.float32)
    for idx, point_file in enumerate(pc_files[:]):
        cloud = PyntCloud.from_file(point_file)
        test_pcs[idx, :, :] = cloud.points[:2048]
    np.save("pcs",np.array(test_pcs))

    reset_tf_graph()
    ae_configuration = 'input/configuration'
    ae_conf = Conf.load(ae_configuration)
    ae_conf.encoder_args['verbose'] = False
    ae_conf.decoder_args['verbose'] = False
    ae = PointNetAutoEncoder(ae_conf.experiment_name, ae_conf)

    restore_epoch = 3000
    ae.restore_model('model/3', restore_epoch, verbose=True)

    latent_codes = ae.get_latent_codes(test_pcs)

    print(latent_codes.shape)
    np.save("latent",np.array(latent_codes))
