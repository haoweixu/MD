from read_cfg import *
from structure import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
# cfgfile = "b{0}.cfg".format(1)
cfgfile = "b{0:09d}.cfg".format(1000000)

type_dict = {'F':1}

cfg = cfg_file_class()
cfg.read_cfg(cfgfile, type_dict)

r_cut = 1.4
cfg.create_neigh_list(r_cut)

phi_bin = 50
theta_bin = 50
hist = stereo_proj(cfg, 2, phi_bin, theta_bin)

x = np.zeros((phi_bin,theta_bin))
y = np.zeros((phi_bin,theta_bin))
z = np.zeros((phi_bin,theta_bin))

for i in range(phi_bin):
  for j in range(theta_bin):
    theta = math.pi * j / theta_bin
    phi = 2*math.pi * i / phi_bin
    x[i,j] = math.sin(theta) * math.cos(phi)
    y[i,j] = math.sin(theta) * math.sin(phi)
    # x[i,j] = math.cos(phi)
    # y[i,j] = math.sin(phi)
    z[i,j] = hist[i, j]

fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()




plt.imshow()
plt.colorbar()
plt.show()


# data = pair_correlation(cfg, 0.1, r_cut, 1000)

# center = data[0]
# g = data[1]

# plt.plot(center, g)
# plt.show()

# atoms = cfg.atoms
#
# f_write = open("neigh_num", 'w')
#
# write_buf = ""
# for atom in atoms:
#   write_buf += str(atom.neigh_num()) + "\n"
# f_write.write(write_buf)
