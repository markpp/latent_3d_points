import os.path as osp
import numpy as np
import json
import argparse
from pyntcloud import PyntCloud
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from settings import *

enable_plot = 1
enable_pca = 1
enable_kde = 0
enable_pose = 0

def plot_sets(set_train, set_val, name):
    _, num_dim = set_train.shape

    if num_dim == 2:
        plt.figure(figsize=(6, 6))
        plt.scatter(set_train[:, 0], set_train[:, 1], c='r',
                    alpha=.4, s=5)
        plt.scatter(set_val[:, 0], set_val[:, 1], c='b',
                    alpha=.4, s=5)

    else:
        num_dim = 6
        fig, axs = plt.subplots(figsize=(12, 14), nrows=num_dim)
        fig.subplots_adjust(left = 0.02, right = 0.98, hspace = 0.4)
        for id in range(num_dim):
            sns.distplot(set_train[:,id], hist=False, rug=True, color="r", ax=axs[id], label = "train");
            sns.distplot(set_val[:,id], hist=False, rug=True, color="b", ax=axs[id], label = "val");
            axs[id].yaxis.set_visible(False)
            axs[id].set_title("dim. {}".format(id))

        #fig, axs = plt.subplots(figsize=(6*num_dim, 10), ncols=num_dim)
        #for i in range(num_dim):
        #sns.violinplot(x=labels, y=z[:, i], ax=axs[i])
    #plt.show()
    plt.savefig('plots/{}_dim_{}.png'.format(name,num_dim))


def points2file(points,filename):
    df = pd.DataFrame(points,columns=['x', 'y', 'z'])
    pc = PyntCloud(df)
    pc.to_file(filename,as_text=True)

if __name__ == '__main__':
    ###############
    ## Load data ##
    ###############
    #train_pcs = np.load("output/train_pcs.npy")
    val_pcs = np.load("output/{}_pcs.npy".format(DATASET))
    print(val_pcs.shape)

    #train_recs = np.load("output/train_recs.npy")
    val_recs = np.load("output/{}_recs.npy".format(DATASET))[:,0,:,:]
    print(val_recs.shape)
    #train_names = np.load("output/train_names.npy")
    #vals_names = np.load("output/vals_names.npy")

    train_latent = np.load("output/train_latent.npy")
    val_latent = np.load("output/{}_latent.npy".format(DATASET))

    #'''
    train_rec_loss = np.load("output/train_rec_loss.npy")
    val_rec_loss = np.load("output/{}_rec_loss.npy".format(DATASET))

    if enable_pose:
        val_pos_error = np.mean(np.absolute(np.load("output/{}_pos_error.npy".format(DATASET))), axis = 1)
        val_norm_error = np.mean(np.absolute(np.load("output/{}_norm_error.npy".format(DATASET))), axis = 1)

    n_train = train_latent.shape[0]
    n_val = val_latent.shape[0]
    print("Train sample: {}, Val samples {}".format(n_train, n_val))


    ##################
    ## Process data ##
    ##################

    if enable_pca:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        train_pca = pca.fit_transform(train_latent)
        print("explained_variance_ratio: {}".format(pca.explained_variance_ratio_))
        val_pca = pca.transform(val_latent)

    # measure distance from each test sample to nearest known sample
    val_dist = []
    for v in val_latent:
        val_dist.append(np.array([np.linalg.norm(v-t) for t in train_latent]).min()*25)
    val_dist = np.array(val_dist)
    # find best and worst matches
    idx_b2w = np.argsort(val_dist)

    print(val_dist[idx_b2w[:3]])
    print(val_dist[idx_b2w[-3:]])

    for b_id in idx_b2w[:3]:
        points2file(val_pcs[b_id],"output/pcs/val_org_best_{}.ply".format(b_id))
        points2file(val_recs[b_id],"output/pcs/val_rec_best_{}.ply".format(b_id))

    for w_id in idx_b2w[-3:]:
        points2file(val_pcs[w_id],"output/pcs/val_org_worst_{}.ply".format(w_id))
        points2file(val_recs[w_id],"output/pcs/val_rec_worst_{}.ply".format(w_id))

    if enable_kde:
        # create density map using train samples and use that
        from sklearn.neighbors import KernelDensity
        x, y = train_pca[:,0], train_pca[:,1]

        # create grid of sample locations (default: 100x100)
        xx, yy = np.mgrid[x.min():x.max():100j,
                          y.min():y.max():100j]

        xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
        xy_train  = np.vstack([y, x]).T

        #d = xy_train.shape[0]
        #n = xy_train.shape[1]
        #bw = (n * (d + 2) / 4.)**(-1. / (d + 4)) # silverman

        #from sklearn.model_selection import GridSearchCV
        #grid = GridSearchCV(KernelDensity(),{'bandwidth': np.linspace(0.01, 1.0, 30)},cv=20) # 20-fold cross-validation
        #grid.fit(xy_train)
        #print(grid.best_params_)

        kde = KernelDensity(bandwidth=0.11,metric='euclidean',kernel='gaussian', algorithm='ball_tree')
        kde.fit(xy_train)

        # score_samples() returns the log-likelihood of the samples
        z = np.exp(kde.score_samples(xy_sample))
        zz = np.reshape(z, xx.shape)



    ###########
    ## Plots ##
    ###########
    if enable_plot:
        # plot distribution of train and test sets
        plot_sets(train_latent[:], val_latent[:], "train_test_latent")

        if enable_pca:
            plot_sets(train_pca[:,:2], val_pca[:,:2], "train_test_pca")

        df = pd.DataFrame(np.concatenate((train_pca,val_pca)),columns=['pc0', 'pc1'])
        df['Error'] = np.concatenate((train_rec_loss,val_rec_loss))
        df['Dataset'] = np.concatenate((['train'] * n_train, ['val'] * n_val)) #np.concatenate((np.zeros(n_train),np.ones(n_val)))
        df['Distance'] = np.concatenate((np.ones(n_train)*5,np.array(val_dist)))
        if enable_pose:
            df['PosError'] = np.concatenate((np.ones(n_train)*5,np.array(val_pos_error)*600))
            df['NormError'] = np.concatenate((np.ones(n_train)*5,np.array(val_norm_error)*60))

        sns.scatterplot(x="pc0", y="pc1", data=df,
                        hue="Dataset", palette=["blue","green"], size="Error",
                        sizes=(df['Error'].min(), df['Error'].max()))
        plt.savefig('plots/{}.png'.format("train_test_pca_rec_error"))

        plt.clf()
        sns.scatterplot(x="pc0", y="pc1", data=df,
                        hue="Dataset", palette=["blue","green"], size="Distance",
                        sizes=(df['Distance'].min(), df['Distance'].max()))
        plt.savefig('plots/{}.png'.format("train_test_pca_latent_dist"))

        if enable_pose:
            plt.clf()
            sns.scatterplot(x="pc0", y="pc1", data=df,
                            hue="Dataset", palette=["blue","green"], size="PosError",
                            sizes=(df['PosError'].min(), df['PosError'].max()))
            plt.savefig('plots/{}.png'.format("train_test_pca_pos_error"))

            plt.clf()
            sns.scatterplot(x="pc0", y="pc1", data=df,
                            hue="Dataset", palette=["blue","green"], size="NormError",
                            sizes=(df['NormError'].min(), df['NormError'].max()))
            plt.savefig('plots/{}.png'.format("train_test_pca_norm_error"))

        cmap = sns.cubehelix_palette(start=0, light=1, as_cmap=True)
        sns.jointplot(x='pc0', y='pc1', data=df[(df['Dataset']=='train')], kind="kde", height=7, space=0)
        plt.savefig('plots/{}.png'.format("train_pca_density"))

        cmap = sns.cubehelix_palette(start=0, rot=-.4, light=1, as_cmap=True)
        sns.jointplot(x='pc0', y='pc1', data=df[(df['Dataset']=='val')], kind="kde", height=7, space=0)
        plt.savefig('plots/{}.png'.format("test_pca_density"))

        '''

        plt.pcolormesh(xx, yy, zz)
        plt.scatter(x, y, s=2, facecolor='white')


        logprob_train = kde.score_samples(train_pca)
        print(logprob_train)
        logprob_val = kde.score_samples(val_pca)
        print(logprob_val)


        '''
        #plt.show()




    #for id in range(0,3):
    #for id in range(0,len(reconstructions)):
        #print(names[id])
        #points2file(reconstructions[id],"output/rec_{}".format(names[id]))
