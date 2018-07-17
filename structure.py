# This file includes some methods to analyze the structure
# of atoms contained in a cfg_class object

from read_cfg import *

def pair_correlation(cfg, low, high, nbins):
  atoms = cfg.atoms
  natoms = cfg.natoms
  hist = np.zeros(nbins)
  for i in range(natoms):
    atomi = atoms[i]
    neigh_dist = [neighii.dist for neighii in atomi.neigh]
    ihist, edges = np.histogram(neigh_dist, bins=nbins, range=(low, high))
    hist = [hist[ii] + ihist[ii] for ii in range(len(hist))]

  center = [(edges[ii] + edges[ii + 1]) / 2 for ii in range(0, len(edges) - 1)]
  deltar = center[1] - center[0]

  g = [hist[ii] / (4 * math.pi * (center[ii]) ** 2 * deltar * natoms ** 1) \
       for ii in range(len(hist))]

  return [center, g]

def stereo_proj(cfg, min_neigh, phi_bin=50, theta_bin=50):
  atoms = cfg.atoms
  hist = np.zeros( (phi_bin, theta_bin) )
  theta_incre = math.pi / theta_bin
  phi_incre = 2*math.pi / phi_bin

  for atomi in atoms:
    neigh_num = atomi.neigh_num()
    if neigh_num < min_neigh:
      continue
    ri = atomi.r
    for neighj in atomi.neigh:
      rj = neighj.r
      theta_ij = int( math.floor(theta(ri, rj)/theta_incre) )
      phi_ij = int( math.floor(phi(ri, rj)/phi_incre) )
      if phi_ij >= phi_bin:
        print(phi_ij, phi(ri, rj))
        phi_ij =  phi_bin - 1
      hist[phi_ij, theta_ij] += 1

  return hist