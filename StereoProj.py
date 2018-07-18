# from read_cfg import *
from structure import *
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import os

# os.chdir("E:\Dropbox (MIT)\Desktop\NanoParticle\hP2_6.50_0.70")
os.chdir("E:\Dropbox (MIT)\Desktop\NanoParticle\I1_7.50_0.53")

# cfgfile = "a{0}.cfg".format(0)
cfgfile = "b{0:09d}.cfg".format(3450000)

type_dict = {'F':1}
cfg = cfg_file_class()
cfg.read_cfg(cfgfile, type_dict)

r_cut = 1.2
cfg.create_neigh_list(r_cut)

# data = pair_correlation(cfg, 0.1, r_cut, 1000)
# center = data[0]
# g = data[1]
# plt.plot(center, g)
# plt.show()

phi_bin = 100
theta_bin = 100
hist = bood(cfg, 5, phi_bin, theta_bin)

myfile = open("I1", 'w')
np.savetxt(myfile, hist)
