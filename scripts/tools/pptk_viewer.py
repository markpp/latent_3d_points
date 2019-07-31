import pptk
import numpy as np

v = False

def show_points(points,p0):
    global v
    if v is False:
        v = pptk.viewer(points)
        v.set(point_size=0.001, lookat = p0, r = 0.5, show_grid = False, show_info = False, show_axis = True)
    else:
        v.clear()
        v.load(points)
        v.set(point_size=0.001, lookat = p0, r = 0.5, show_grid = False, show_info = False, show_axis = True)

def show_points_with_point(points,p0):
    p_cloud = points[:,:3]
    c_cloud = np.full((len(p_cloud),3),[0.4, 0.4, 0.4])

    points = np.concatenate((p_cloud, [p0]), axis=0)
    colors = np.concatenate((c_cloud, [[1, 0, 0]]), axis=0)

    global v
    if v is False:
        v = pptk.viewer(points)
        v.attributes(colors)
        v.set(bg_color = [1,1,1,1], point_size=0.002, lookat = p0, r = 0.5, show_grid = False, show_info = False, show_axis = True)
    else:
        v.clear()
        v.load(points)
        v.attributes(colors)
        v.set(bg_color = [1,1,1,1], point_size=0.002, lookat = p0, r = 0.5, show_grid = False, show_info = False, show_axis = True)

def show_points_with_pose(points,p0,n):
    p_cloud = points[:,:3]
    c_cloud = np.full((len(p_cloud),3),[0.4, 0.4, 0.4])

    num_points = 40
    lin_lenght = 0.15
    p_norm = p0+np.outer(np.linspace(0,lin_lenght,num_points),n)
    c_norm = np.full((len(p_norm),3),[1, 0, 0])

    points = np.concatenate((p_norm, p_cloud), axis=0)
    colors = np.concatenate((c_norm, c_cloud), axis=0)

    global v
    if v is False:
        v = pptk.viewer(points)
        v.attributes(colors)
        v.set(bg_color = [1,1,1,1], point_size=0.002, lookat = p0, r = 0.5, show_grid = False, show_info = False, show_axis = True)
    else:
        v.clear()
        v.load(points)
        v.attributes(colors)
        v.set(bg_color = [1,1,1,1], point_size=0.002, lookat = p0, r = 0.5, show_grid = False, show_info = False, show_axis = True)
