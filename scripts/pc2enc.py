import os.path as osp
import numpy as np
import json
import argparse
from pyntcloud import PyntCloud
import pandas as pd
from latent_3d_points.src.autoencoder import Configuration as Conf
from latent_3d_points.src.point_net_ae import PointNetAutoEncoder
from latent_3d_points.src.tf_utils import reset_tf_graph

n_points = 1024
model_dir = 'trained_model/zeroedrot'
restore_epoch = 20000

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
        pc_files = f.read().splitlines()[:]

    names = []
    anno = []
    pcs = np.empty([len(pc_files), n_points, 3], dtype=np.float32)
    for idx,pc_path in enumerate(pc_files):
        names.append(pc_path.split('/')[-1])

        cloud = PyntCloud.from_file(pc_path)
        pcs[idx, :, :] = cloud.points[:n_points]

        # load annotation
        pose_path = pc_path[:-4]+".json"
        #print(pose_path)
        with open(pose_path) as pose_file:
            jp = json.load(pose_file)
            p = [float(jp["pos"]["x"]),float(jp["pos"]["y"]),float(jp["pos"]["z"])]
            x, y, z, w = float(jp["orn"]["x"]), float(jp["orn"]["y"]), float(jp["orn"]["z"]), float(jp["orn"]["w"])
            # forward vector
            nfx = 2 * (x*z + w*y)
            nfy = 2 * (y*z - w*x)
            nfz = 1 - 2 * (x*x + y*y)
            # up vector
            nux = 2 * (x*y - w*z)
            nuy = 1 - 2 * (x*x + z*z)
            nuz = 2 * (y*z + w*x)
            # left vector
            nlx = 1 - 2 * (y*y + z*z)
            nly = 2 * (x*y + w*z)
            nlz = 2 * (x*z - w*y)

            #n = [nlx, nly, nlz]
            pose = [p[0], p[1], p[2], nlx, nly, nlz]
            anno.append(pose)

    np.save("output/names",np.array(names))
    np.save("output/anno",np.array(anno))
    #np.save("output/pcs",pcs)

    reset_tf_graph()
    ae_configuration = model_dir+'/configuration'
    ae_conf = Conf.load(ae_configuration)
    ae_conf.encoder_args['verbose'] = False
    ae_conf.decoder_args['verbose'] = False
    ae = PointNetAutoEncoder(ae_conf.experiment_name, ae_conf)

    ae.restore_model(model_dir, restore_epoch, verbose=True)

    latent_codes = ae.get_latent_codes(pcs)

    print(latent_codes.shape)
    np.save("output/latent",np.array(latent_codes))
