import os.path as osp
import numpy as np


from latent_3d_points.src.ae_templates import mlp_architecture_ala_iclr_18, default_train_params
from latent_3d_points.src.autoencoder import Configuration as Conf
from latent_3d_points.src.point_net_ae import PointNetAutoEncoder

from latent_3d_points.src.in_out import snc_category_to_synth_id, create_dir, PointCloudDataSet, \
                                        load_all_point_clouds_under_folder

from latent_3d_points.src.tf_utils import reset_tf_graph
from latent_3d_points.src.general_utils import plot_3d_point_cloud


if __name__ == '__main__':
    """
    Main function for executing the .py script.
    Command:
        -p path/<filename>.npy
    """
    # construct the argument parser and parse the arguments
    #ap = argparse.ArgumentParser()
    #ap.add_argument("-l", "--list", type=str,
    #                help="list of point cloud files")
    #args = vars(ap.parse_args())

    #top_in_dir = '../data/shape_net_core_uniform_samples_2048/' # Top-dir of where point-clouds are stored.
    #top_in_dir = '../data/shape_net_selection/' # Top-dir of where point-clouds are stored.
    top_in_dir = '/home/dmri/Documents/github/latent_3d_points/data/shape_net_selection/train/' # Top-dir of where point-clouds are stored.
    ae_configuration = '../data/shape_net_ae/configuration'

    experiment_name = 'ear_ae_test'
    n_pc_points = 2048                # Number of points per model.
    bneck_size = 64                  # Bottleneck-AE size
    restore_epoch = 500

    #syn_id = snc_category_to_synth_id()[class_name]
    #class_dir = osp.join(top_in_dir , syn_id)
    class_dir = top_in_dir
    all_pc_data = load_all_point_clouds_under_folder(class_dir, n_threads=8, file_ending='.ply', verbose=True)



    reset_tf_graph()
    ae_conf = Conf.load(ae_configuration)
    ae_conf.encoder_args['verbose'] = False
    ae_conf.decoder_args['verbose'] = False
    ae = PointNetAutoEncoder(ae_conf.experiment_name, ae_conf)
    ae.restore_model(ae_conf.train_dir, restore_epoch, verbose=True)

    latent_codes = ae.get_latent_codes(all_pc_data.point_clouds)


    reconstructions = ae.decode(latent_codes)


    i = 1
    plot_3d_point_cloud(all_pc_data.point_clouds[i][:, 0],
                        all_pc_data.point_clouds[i][:, 1],
                        all_pc_data.point_clouds[i][:, 2], in_u_sphere=True);


    plot_3d_point_cloud(reconstructions[i][:, 0],
                        reconstructions[i][:, 1],
                        reconstructions[i][:, 2], in_u_sphere=True);
