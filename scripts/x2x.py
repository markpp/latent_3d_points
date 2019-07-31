import os.path as osp
import numpy as np
import json
import argparse
from pyntcloud import PyntCloud
import pandas as pd
from latent_3d_points.src.autoencoder import Configuration as Conf
from latent_3d_points.src.point_net_ae import PointNetAutoEncoder
from latent_3d_points.src.tf_utils import reset_tf_graph

model_dir = 'trained_model/hanging/ct_hanging_emd_1024_16'
restore_epoch = 1000

dataset = "val"

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
    '''
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--enc", type=str,
                    help="numpy saved array")
    args = vars(ap.parse_args())

    enc = np.load(args["enc"])
    '''
    pcs = np.load("output/{}_pcs.npy".format(dataset))
    names = np.load("output/{}_names.npy".format(dataset))


    reset_tf_graph()
    ae_configuration = model_dir+'/configuration'
    ae_conf = Conf.load(ae_configuration)
    ae_conf.encoder_args['verbose'] = False
    ae_conf.decoder_args['verbose'] = False
    ae = PointNetAutoEncoder(ae_conf.experiment_name, ae_conf)

    ae.restore_model(model_dir, restore_epoch, verbose=True)

    recs = []
    rec_losses = []
    for pc in pcs:
        pc = np.expand_dims(pc, axis=0)
        rec, l = ae.reconstruct(pc, GT=pc, compute_loss=True)
        recs.append(rec)
        rec_losses.append(l)

    np.save("output/{}_rec_loss".format(dataset),np.array(rec_losses))
    np.save("output/{}_recs".format(dataset),np.array(recs))

    #for id in range(0,3):
    #for id in range(0,len(reconstructions)):
        #print(names[id])
        #points2file(reconstructions[id],"output/rec_{}".format(names[id]))
