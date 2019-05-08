# USAGE
# python cnn_regression.py --dataset Houses-dataset/Houses\ Dataset/

# import the necessary packages
from keras.optimizers import Adam
from keras.models import model_from_json

from sklearn.preprocessing import normalize


from NN import models
import numpy as np
import math
from pyntcloud import PyntCloud
import pandas as pd
import json

TRAIN = True

def pose2cloud(pose,filename):
    anno_points = []
    anno_points.append(pose[:3]) # TODO: add color
    anno_points.append([pose[0]+pose[3],pose[1]+pose[4],pose[2]+pose[5]])
    df = pd.DataFrame(np.array(anno_points),columns=['x', 'y', 'z'])
    pc = PyntCloud(df)
    pc.to_file(filename,as_text=True)

def pose2json(pose,filename):
    # make unit vector
    length = math.sqrt(pose[3]*pose[3]+pose[4]*pose[4]+pose[5]*pose[5])
    pose[3] /= length
    pose[4] /= length
    pose[5] /= length

    nz = [pose[3],pose[4],pose[5]]
    nx = np.cross(nz,[0,1,0])
    ny = np.cross(nz,nx)
    orn = vecs2quad(ny,nx,nz) # why does x,y need to be switched? theres is a problem here

    with open(filename, 'w') as outfile:
        data = {'pos':{'x': float(pose[0]),
                       'y': float(pose[1]),
                       'z': float(pose[2])},
                'orn':{'x': float(orn[0]),
                       'y': float(orn[1]),
                       'z': float(orn[2]),
                       'w': float(orn[3])}
                }
        json.dump(data, outfile, indent=4)

def vecs2quad(nf,nu,nl):
    m00, m01, m02 = nl[0], nl[1], nl[2]
    m10, m11, m12 = nu[0], nu[1], nu[2]
    m20, m21, m22 = nf[0], nf[1], nf[2]

    num8 = (m00 + m11) + m22;
    if (num8 > 0.0):
      num = math.sqrt(num8 + 1.0);
      w = num * 0.5;
      num = 0.5 / num;
      x = (m12 - m21) * num;
      y = (m20 - m02) * num;
      z = (m01 - m10) * num;
      return [x,y,z,w];

    elif ((m00 >= m11) and (m00 >= m22)):
      num7 = math.sqrt(((1.0 + m00) - m11) - m22);
      num4 = 0.5 / num7;
      x = 0.5 * num7;
      y = (m01 + m10) * num4;
      z = (m02 + m20) * num4;
      w = (m12 - m21) * num4;
      return [x,y,z,w];

    elif (m11 > m22):
      num6 = math.sqrt(((1.0 + m11) - m00) - m22);
      num3 = 0.5 / num6;
      x = (m10 + m01) * num3;
      y = 0.5 * num6;
      z = (m21 + m12) * num3;
      w = (m20 - m02) * num3;
      return [x,y,z,w];

    num5 = math.sqrt(((1.0 + m22) - m00) - m11);
    num2 = 0.5 / num5;
    x = (m20 + m02) * num2;
    y = (m21 + m12) * num2;
    z = 0.5 * num5;
    w = (m01 - m10) * num2;
    return [x,y,z,w];

if __name__ == '__main__':
    """
    Main function for executing the .py script.
    Command:
        -p path/<filename>.npy
    """
    x = np.load("output/latent.npy")
    y = np.load("output/anno.npy")
    names = np.load("output/names.npy")

    testset_end = 2000
    test_x = x[:testset_end]
    #test_y = y[:100,0]
    test_y = y[:testset_end,:]
    print(test_y[0])

    test_y = test_y.reshape(len(test_y),6)
    train_x = x[testset_end:]
    #train_y = y[100:,0]
    train_y = y[testset_end:,:]
    train_y = train_y.reshape(len(train_y),6)

    #do k-fold https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/
    if TRAIN:
        model = models.create_mlp(16, 6, regress=True)
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=1e-3, decay=1e-3 / 100)) # mean_squared_error

        # train the model
        print("[INFO] training model...")
        print("train_x {}, train_y {}, test_x {}, test_y {}".format(train_x.shape,train_y.shape,test_x.shape,test_y.shape))
        model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=200, batch_size=128)

        # serialize model to JSON
        model_json = model.to_json()
        with open("trained_model/nn_model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("trained_model/nn_weights.h5")
    else:
        # load json and create model
        json_file = open('trained_model/nn_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights("trained_model/nn_weights.h5")
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=1e-3, decay=1e-3 / 200)) # mean_squared_error

    #test_x = train_x
    #test_y = train_y
    # make predictions on the testing data

    score = model.evaluate(test_x, test_y, verbose=0)
    print("{}: {}".format(model.metrics_names, score))


    print("[INFO] predicting...")
    preds = model.predict(x)


    ids = [0,5000]
    for id in ids:
        name = names[id][:-4]
        print(name)
        pose2cloud(preds[id],"output/pred-gt_{}.ply".format(names[id]))

        pose2json(y[id], 'output/gt_{}.json'.format(names[id]))
        pose2json(preds[id], 'output/pred_{}.json'.format(names[id]))

    np.set_printoptions(precision=3)
    print(preds[id])
    #print(test_y[id])
    print(y[id])
