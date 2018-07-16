from read_cfg import *
import matplotlib.pyplot as plt

cfgfile = "b{0:09d}.cfg".format(1000000)

type_dict = {'F':1}

cfg = cfg_file_class()
cfg.read_cfg(cfgfile, type_dict)

r_cut = 10.0

cfg.create_neigh_list(r_cut)

data = cfg.pair_correlation(0.1, 10.0, 1000)

center = data[0]
g = data[1]

plt.plot(center, g)
plt.show()

atoms = cfg.atoms

f_write = open("neigh_num", 'w')

write_buf = ""
# for atom in atoms:
#   write_buf += str(atom.neigh_num()) + "\n"
# f_write.write(write_buf)

