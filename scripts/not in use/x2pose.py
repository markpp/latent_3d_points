import numpy as np
import time 

from pyntcloud import PyntCloud
import pandas as pd

from tools import plot
from tools import convert
from tools import evaluate
from tools import simple_nn
from tools import pptk_viewer

TRAIN = False

runs = ["pca_ctkin_0","pca_ctkin_1","latent_ctkin_0","latent_ctkin_1"] # <feature>_<dataset>_<#>

if __name__ == '__main__':
    """
    Main function for executing the .py script.
    Command:
        -p path/<filename>.npy
    """

    test_y = np.load("output/val_anno.npy")
    test_names = np.load("output/val_names.npy")
    test_pcs = np.load("output/val_pcs.npy")

    print(test_pcs.shape)

    train_y = np.load("output/train_anno.npy")
    train_names = np.load("output/train_names.npy")

    for name in runs[:1]:
        feature_type = name.split('_')[0]
        test_x = np.load("output/val_{}.npy".format(feature_type))
        train_x = np.load("output/train_{}.npy".format(feature_type))

        num_train, input_dim = train_x.shape
        _, output_dim = train_y.shape

        if TRAIN:
            loss_hist = simple_nn.train(input_dim, output_dim, train_x, train_y, test_x, test_y, name)
            plot.plot_loss(loss_hist, name)
        model = simple_nn.load(name)
            
        # make predictions on the test data
        train_loss = model.evaluate(train_x, train_y, verbose=0)
        val_loss = model.evaluate(test_x, test_y, verbose=0)
        print("{} - train: {}, val {}".format(model.metrics_names, train_loss, val_loss))
        with open('logs/nn_losses.txt','a') as f:
            f.write('{} {} {} {}\n'.format(int(time.time()), name, train_loss, val_loss))

        '''
        print("[INFO] predicting...")
        preds = model.predict(test_x)

        evaluate.evaluate_point_normal(test_y, preds)

        ids = [0,50,100]
        for id in ids:
            print("testing on id {}".format(id))
            name = test_names[id][:-4]
            #pose2cloud(preds[id],"output/pred-gt_{}.ply".format(names[id]))

            convert.pose2json(test_y[id], 'output/{}_gt.json'.format(test_names[id]))
            convert.pose2json(preds[id], 'output/{}_pred.json'.format(test_names[id]))

            np.set_printoptions(precision=3, suppress=True)
            print("GT")
            print(test_y[id])
            print("prediction")
            print(preds[id])
        '''
        ids = [0,50]
        for id in ids:
            pptk_viewer.show_points_with_pose(test_pcs[id], test_y[id,:3], test_y[id,3:])
            time.sleep(3.0)
