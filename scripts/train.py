import os.path as osp

from latent_3d_points.src.ae_templates import mlp_architecture_ala_iclr_18, default_train_params
from latent_3d_points.src.autoencoder import Configuration as Conf
from latent_3d_points.src.point_net_ae import PointNetAutoEncoder

from latent_3d_points.src.in_out import snc_category_to_synth_id, create_dir, PointCloudDataSet, \
                                        load_all_point_clouds_under_folder

from latent_3d_points.src.tf_utils import reset_tf_graph
from latent_3d_points.src.general_utils import plot_3d_point_cloud

TRAIN = True

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


    top_out_dir = '../data/'          # Use to save Neural-Net check-points etc.

    experiment_name = 'ear_ae'
    n_pc_points = 2048                # Number of points per model.
    bneck_size = 64                  # Bottleneck-AE size
    ae_loss = 'emd'                   # Loss to optimize: 'emd' or 'chamfer'

    train_pc_data = load_all_point_clouds_under_folder('/home/dmri/Documents/github/latent_3d_points/data/ear_data/unordered/train/', n_threads=8, file_ending='.ply', verbose=True)
    val_pc_data = load_all_point_clouds_under_folder('/home/dmri/Documents/github/latent_3d_points/data/ear_data/unordered/val/', n_threads=8, file_ending='.ply', verbose=True)
    print("batch size must be larger than train {} and val {} set".format(train_pc_data.num_examples,val_pc_data.num_examples))

    if TRAIN:
        train_params = default_train_params(single_class=False)

        encoder, decoder, enc_args, dec_args = mlp_architecture_ala_iclr_18(n_pc_points, bneck_size)
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
                                 # held_out data (if they are provided in ae.train() ).
        conf.save(osp.join(train_dir, 'configuration'))

    load_pre_trained_ae = True
    if load_pre_trained_ae:
        conf = Conf.load(osp.join(top_out_dir, experiment_name) + '/configuration')
        reset_tf_graph()
        ae = PointNetAutoEncoder(conf.experiment_name, conf)
        restore_epoch = 30000
        ae.restore_model(conf.train_dir, epoch=restore_epoch)


    if TRAIN:
        #reset_tf_graph()
        #ae = PointNetAutoEncoder(conf.experiment_name, conf)

        buf_size = 1 # Make 'training_stats' file to flush each output line regarding training.
        fout = open(osp.join(conf.train_dir, 'train_stats.txt'), 'a', buf_size)
        train_stats = ae.train(train_pc_data, conf, log_file=fout, held_out_data=val_pc_data)
        fout.close()

    feed_pc, feed_model_names, _ = val_pc_data.next_batch(10)
    reconstructions = ae.reconstruct(feed_pc)[0]
    latent_codes = ae.transform(feed_pc)

    i = 1
    plot_3d_point_cloud(feed_pc[i][:, 0],
                        feed_pc[i][:, 1],
                        feed_pc[i][:, 2], in_u_sphere=False);


    plot_3d_point_cloud(reconstructions[i][:, 0],
                        reconstructions[i][:, 1],
                        reconstructions[i][:, 2], in_u_sphere=False);
