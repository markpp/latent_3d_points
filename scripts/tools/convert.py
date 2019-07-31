import numpy as np
import math

from pyntcloud import PyntCloud
import pandas as pd

import json

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
        out = []
        out.append(data)
        json.dump(out, outfile, indent=4)

def vecs2quad(nf,nu,nl):
    m00, m01, m02 = nl[0], nl[1], nl[2]
    m10, m11, m12 = nu[0], nu[1], nu[2]
    m20, m21, m22 = nf[0], nf[1], nf[2]

    num8 = (m00 + m11) + m22
    if (num8 > 0.0):
      num = math.sqrt(num8 + 1.0)
      w = num * 0.5
      num = 0.5 / num
      x = (m12 - m21) * num
      y = (m20 - m02) * num
      z = (m01 - m10) * num
      return [x,y,z,w]

    elif ((m00 >= m11) and (m00 >= m22)):
      num7 = math.sqrt(((1.0 + m00) - m11) - m22)
      num4 = 0.5 / num7
      x = 0.5 * num7
      y = (m01 + m10) * num4
      z = (m02 + m20) * num4
      w = (m12 - m21) * num4
      return [x,y,z,w]

    elif (m11 > m22):
      num6 = math.sqrt(((1.0 + m11) - m00) - m22)
      num3 = 0.5 / num6
      x = (m10 + m01) * num3
      y = 0.5 * num6
      z = (m21 + m12) * num3
      w = (m20 - m02) * num3
      return [x,y,z,w]

    num5 = math.sqrt(((1.0 + m22) - m00) - m11)
    num2 = 0.5 / num5
    x = (m20 + m02) * num2
    y = (m21 + m12) * num2
    z = 0.5 * num5
    w = (m01 - m10) * num2
    return [x,y,z,w]