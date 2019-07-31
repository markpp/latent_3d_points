import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import vg

import pandas as pd


def evaluate_point_normal(gts,preds):
    #diff = np.abs(gts-preds)
    pos = []
    ang = []
    for i in range(len(gts)):
        diff = gts[i]-preds[i]
        pos.append(diff[:3])
        #angles = [np.arctan2(math.sqrt(diff[4]*diff[4]+diff[5]*diff[5]), diff[3]),
        #          np.arctan2(math.sqrt(diff[3]*diff[3]+diff[5]*diff[5]), diff[4]),
        #          np.arctan2(math.sqrt(diff[4]*diff[4]+diff[3]*diff[3]), diff[5])]
        angles = [vg.signed_angle(np.array([diff[3],diff[4],diff[5]]),np.array([1,0,0]), look=vg.basis.y, units="deg"),
                  vg.signed_angle(np.array([diff[3],diff[4],diff[5]]),np.array([0,1,0]), look=vg.basis.z, units="deg"),
                  vg.signed_angle(np.array([diff[3],diff[4],diff[5]]),np.array([0,0,1]), look=vg.basis.x, units="deg")]
        ang.append(angles)
    # Creates two subplots and unpacks the output array immediately
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))

    ax1 = sns.boxplot(data=pd.DataFrame(np.array(pos),columns=['dx', 'dy', 'dz']), ax = ax1)
    ax1.set_title('Absolute error compared to demo')
    ax1.set_ylabel('[m]')

    ax2 = sns.boxplot(data=pd.DataFrame(np.array(ang),columns=['ax', 'ay', 'az']), ax = ax2)
    ax2.set_title('Absolute error compared to demo')
    ax2.set_ylabel('[deg]')

    plt.tight_layout()
    plt.savefig('plots/pose_error.png')
    #plt.show()