import os.path as osp
import sys
sys.path.append('../..')

from latent_3d_points.src.ae_templates import mlp_architecture_ala_iclr_18_small, default_train_params
from latent_3d_points.src.autoencoder import Configuration as Conf
from latent_3d_points.src.point_net_ae import PointNetAutoEncoder
from latent_3d_points.src.in_out import snc_category_to_synth_id, create_dir, PointCloudDataSet, \
                                        load_all_point_clouds_under_folder
from latent_3d_points.src.tf_utils import reset_tf_graph
from latent_3d_points.src.general_utils import plot_3d_point_cloud

TRAIN = True
load_pre_trained_ae = False
restore_epoch = 0

if __name__ == '__main__':
    top_out_dir = '../data/'         # Use to save Neural-Net check-points etc.
    n_pc_points = 1024               # Number of points per model.
    bneck_size = 16                  # Bottleneck-AE size
    ae_loss = 'emd'                  # Loss to optimize: 'emd' or 'chamfer'
    experiment_name = 'kin_laying_{}_{}_{}'.format(ae_loss,n_pc_points,bneck_size)
    train_pc_data = load_all_point_clouds_under_folder('/home/dmri/datasets/in_use/train_1024/', n_threads=8, file_ending='.ply', verbose=True)
    val_pc_data = load_all_point_clouds_under_folder('/home/dmri/datasets/in_use/val_1024/', n_threads=8, file_ending='.ply', verbose=True)
    print("batch size should be < {} and {}".format(train_pc_data.num_examples,val_pc_data.num_examples))

    if TRAIN:
        train_params = default_train_params(single_class=False)
        encoder, decoder, enc_args, dec_args = mlp_architecture_ala_iclr_18_small(n_pc_points, bneck_size)
        train_dir = create_dir(osp.join(top_out_dir, experiment_name))

        conf = Conf(n_input = [n_pc_points, 3],
                loss = ae_loss,
                training_epochs = train_params['training_epochs'],
                batch_size = train_params['batch_size'],
                denoising = train_params['denoising'],
                learning_rate = train_params['learning_rate'],
                train_dir = train_dir,
                loss_display_step = train_params['loss_display_step'],
                saver_step = train_params['saver_step'],
                z_rotate = train_params['z_rotate'],
                encoder = encoder,
                decoder = decoder,
                encoder_args = enc_args,
                decoder_args = dec_args
               )
        conf.experiment_name = experiment_name
        conf.held_out_step = 50   # How often to evaluate/print out loss on
        conf.save(osp.join(train_dir, 'configuration'))
        conf = Conf.load(osp.join(top_out_dir, experiment_name) + '/configuration')

        reset_tf_graph()
        ae = PointNetAutoEncoder(conf.experiment_name, conf)
    if load_pre_trained_ae:
        ae.restore_model(conf.train_dir, epoch=restore_epoch)

    if TRAIN:
        buf_size = 1 # Make 'training_stats' file to flush each output line regarding training.
        fout = open(osp.join(conf.train_dir, 'train_stats.txt'), 'a', buf_size)
        train_stats = ae.train(train_pc_data, conf, log_file=fout, held_out_data=val_pc_data)
        fout.close()
