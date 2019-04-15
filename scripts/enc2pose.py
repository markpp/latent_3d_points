# USAGE
# python cnn_regression.py --dataset Houses-dataset/Houses\ Dataset/

# import the necessary packages
from keras.optimizers import Adam
from keras.models import model_from_json

from NN import models
import numpy as np
from pyntcloud import PyntCloud
import pandas as pd

TRAIN = True

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
    x = np.load("latent.npy")
    y = np.load("anno.npy")
    print(x.shape)

    test_x = x[100:]
    test_y = y[100:,0]
    print(test_y.shape)
    train_x = x[:100]
    train_y = y[:100,0]

    #model = models.create_cnn(64, 64, 1, regress=True)
    model = models.create_mlp(64, regress=True)
    opt = Adam(lr=1e-3, decay=1e-3 / 200)

    if TRAIN:
        model.compile(loss="mean_squared_error", optimizer=opt) # mean_squared_error

        # train the model
        print("[INFO] training model...")
        print("train_x {}, train_y {}, test_x {}, test_y {}".format(train_x.shape,train_y.shape,test_x.shape,test_y.shape))
        model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=500, batch_size=64)

        # serialize model to JSON
        model_json = model.to_json()
        with open("nn_model.json", "w") as json_file:
            json_file.write(model_json)

    else:
        # load json and create model
        json_file = open('nn_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.compile(loss="mean_squared_error", optimizer=opt) # mean_squared_error


    # make predictions on the testing data
    print("[INFO] predicting...")
    preds = model.predict(test_x)

    #
    id = 1

    anno_points = []
    anno_points.append(preds[id]) # TODO: add color
    anno_points.append(test_y[id])


    points2file(np.array(anno_points),"pred-gt_{}.ply".format(id))

    print(preds[id])
    print(test_y[id])
