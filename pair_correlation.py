# The Albenze parameter of Si is simply:
# A = sum[ (cos(theta_i)+1/3)^2 ]
# where the sum is over the angles formed by neighbors of an atom
# if N_neight == 4

from __future__ import print_function
# from scipy import special
import numpy as np
# import pypar as pp
import cmath
import itertools
import math
import matplotlib.pyplot as plt
import copy
import os

# find k smallest elements in an array, with an algorithm similar to quick-sort 
# return the index of these k elements in the array
# adapted from Numerical Recipes
def select3(array, k):
  n = len(array)
  l = 0
  r = n-1
  id = [i for i in range(0, n)]
  while True:
    if r<=l+1:
      if r==l+1 and array[r]<array[l]:
        array[r], array[l] = array[l], array[r]
        id[r], id[l] = id[l], id[r]
      break
    else:
      mid = (l+r) >> 1
      array[mid], array[l+1] = array[l+1], array[mid]
      id[mid], id[l+1] = id[l+1], id[mid]
      
      if array[l] > array[r]:
        array[r], array[l] = array[l], array[r]
        id[r], id[l] = id[l], id[r]
        
      if array[l+1] > array[r]:
        array[r], array[l+1] = array[l+1], array[r]
        id[r], id[l+1] = id[l+1], id[r]
        
      if array[l] > array[l+1]:
        array[l+1], array[l] = array[l], array[l+1]
        id[l+1], id[l] = id[l], id[l+1]
        
      i = l+1
      j = r
      a = array[l+1]
      a_id = id[l+1]
      while True: 
        i += 1
        while array[i] < a:
          i += 1
        j -= 1
        while array[j] > a:
          j -= 1
        if j < i:
          break
        array[i], array[j] = array[j], array[i]
        id[i], id[j] = id[j], id[i]
      
      id[l+1] = id[j]
      array[l+1] = array[j]
      id[j] = a_id
      array[j] = a
      if j >= k:
        r = j-1
      if j <= k:
        l = i
  return id[0:k]

class neighbor:
  def __init__(self, r, pbc, id, dist):
    self.r = r
    self.pbc = pbc
    self.id = id
    self.dist = dist
    
# atom class. more atom properties shall be added here.
class atom:
  def __init__(self, mass, type, r, data):
    self.mass = mass
    self.type = type
    self.r    = r
    self.data  = data
    self.neigh = []
  def neigh_num(self):
    return len(self.neigh)
    
# class for cfg file, containing:
# header, an array of strings
# atoms, an array of atoms
# box, size of the simulation box
# natoms, number of real atoms
# nghost, number of ghost atoms
class cfg_file_class:
  def __init__(self):
    self.nghost = 0
    
  # read cfg file, note that neighbor list for atoms is not created
  def read_cfg(self, cfg_filename, type_dict):
    header = []
    atoms = []
    aux_dict = {}
    aux_count = 0
    
    f_read = open(cfg_filename, 'r')
    cur_line = f_read.readline()
    line_split = cur_line.split()
    
    dim = np.zeros(3)
    # read header
    while True:
      try:
        float(line_split[0])
        break
      except:
        header.append(cur_line)
        if line_split[0] == "H0(1,1)":
          dim[0] = float(line_split[2])
        elif line_split[0] == "H0(2,2)":
          dim[1] = float(line_split[2])
        elif line_split[0] == "H0(3,3)":
          dim[2] = float(line_split[2])
        elif line_split[0][0:9] == "auxiliary":
          aux_dict.update({line_split[-1]: aux_count})
          aux_count += 1
          
        cur_line = f_read.readline()
        line_split = cur_line.split()
    
    # read first atom
    mass_string = cur_line
    type_string = f_read.readline()
    data_string = f_read.readline()
    mass = float(mass_string)
    type = type_string.rstrip('\n')
    data = [float(i) for i in data_string.split()]
    new_atom = atom(mass, type, [data[i]*dim[i] for i in range(3)], \
                [ii for ii in data])
    new_atom.pbc = [0, 0, 0]
    if 'id' in aux_dict:
      new_atom.id = int( data[ 3+aux_dict['id'] ] )
    if ('ix' in aux_dict) and ('iy' in aux_dict) and ('iz' in aux_dict):
      new_atom.image = [ data[ 3+aux_dict['ix'] ],  data[ 3+aux_dict['iy'] ], \
                         data[ 3+aux_dict['iz'] ]  ]
      
    atoms.append(new_atom)
    
    # read remaining atoms
    while True:
      try:
        mass_string = f_read.readline()
        type_string = f_read.readline() # type_string is "O\n", note a '\n' is appended
        data_string = f_read.readline()
        mass = float(mass_string)
        type = type_string.rstrip('\n')       
        data = [float(i) for i in data_string.split()]
        new_atom = atom(mass, type, [data[i]*dim[i] for i in range(3)], \
                [ii for ii in data])
        if 'id' in aux_dict:
          new_atom.id = int( data[ 3+aux_dict['id'] ] )
        if ('ix' in aux_dict) and ('iy' in aux_dict) and ('iz' in aux_dict):
          new_atom.image = [ data[ 3+aux_dict['ix'] ],  data[ 3+aux_dict['iy'] ], \
                         data[ 3+aux_dict['iz'] ]  ]
        new_atom.pbc = [0, 0, 0]
        atoms.append(new_atom)
      except:
        break
    
    self.header = header
    self.atoms = atoms
    self.natoms = len(atoms)
    self.box = dim
    self.filename = cfg_filename
    self.type_dict = type_dict
    self.aux_dict = aux_dict
    self.entry_count = aux_count + 3
  
  def create_ghost_atoms(self, r_cut):
    self.nghost = 0
    natoms = self.natoms
    box = self.box
    for i in range(0, natoms):
      images = []
      atomi = self.atoms[i]
      ri = atomi.r
      nx_p = int( math.floor( (r_cut-ri[0])/box[0]+1 ) )
      nx_m = int( math.floor( (r_cut+ri[0])/box[0] ) )
      ny_p = int( math.floor( (r_cut-ri[1])/box[1]+1 ) )
      ny_m = int( math.floor( (r_cut+ri[1])/box[1] ) )
      nz_p = int( math.floor( (r_cut-ri[2])/box[2]+1 ) )
      nz_m = int( math.floor( (r_cut+ri[2])/box[2] ) )
      for ii in range(-nx_m, nx_p+1):
        for jj in range(-ny_m, ny_p+1):
          for kk in range(-nz_m, nz_p+1):
            if [ii,jj,kk] == [0,0,0]:
              continue
            r_shift = [ri[0]+ii*box[0], ri[1]+jj*box[1], ri[2]+kk*box[2]]
            new_atom = copy.copy(atomi)
            new_atom.r = r_shift
            new_atom.pbc = [ii, jj, kk]
            self.atoms.append(new_atom)
            self.nghost += 1
   
  def create_neigh_list(self, r_cut, nnn=None):
    self.create_ghost_atoms(r_cut)
    atoms = self.atoms
    natoms = self.natoms
    nghost = self.nghost
    nall = natoms + nghost

    # find out atoms (including ghost atoms) that are within 
    # the r_cut cutoff, add them to th neighbor list
    for i in range(0, natoms):
      ri = atoms[i].r
      for j in itertools.chain(range(0,i), range(i+1,nall)):
        rj = atoms[j].r
        distij = dist(ri, rj)
        if distij < r_cut:
          new_neigh = neighbor(rj, atoms[j].pbc, atoms[j].id, distij)
          self.atoms[i].neigh.append(new_neigh)
      
      # if nnn is not None, nnn nearest neighbors are retained
      # while others are discarded
      if nnn != None:
        atomi = atoms[i]
        n_now = atomi.neigh_num()
        if n_now > nnn:
          neigh_dist = [atomi.neigh[ii].dist for ii in range(n_now)]
          nnn_id = select3(neigh_dist, nnn)
          new_neigh = [atomi.neigh[ii] for ii in nnn_id]
          atomi.neigh = new_neigh
  
  def neigh4(self):
    enum = [ [0,1], [0,2], [0,3], [1,2], [1,3], [2,3] ]
    
    natoms = self.natoms
    id = np.zeros(natoms)
    id_in_array = np.zeros(natoms)
    neigh_num = np.zeros(natoms)
    op = np.zeros(natoms)
    
    atoms = self.atoms
    for i in range(natoms):
#			print(len(atoms[i].neigh))
      op_i = 0
      atomi = atoms[i]
      idi = int(atomi.id)
      id_in_array[idi-1] = i
      id[idi-1] = idi
      neigh_numi = len(atomi.neigh)
      neigh_num[idi-1] = neigh_numi
      if neigh_numi == 4:
        r = atomi.r
        neigh = atomi.neigh
        for eee in enum:
          r1 = neigh[eee[0]].r
          ra = [r1[jj]-r[jj] for jj in range(3)]
          r2 = neigh[eee[1]].r
          rb = [r2[jj]-r[jj] for jj in range(3)]
          cos_theta_ab = Cos_theta(ra, rb)
          op_i += (cos_theta_ab + 1.0/3.0)**2
        op[idi-1] = op_i
      else:
        op[idi-1] = None
    return [id, neigh_num, op, id_in_array]
  
  def write_crystal_to_cfg_Silicon(self, op, id_in_array, thres, costheta):
    f = open("cry_"+self.filename, 'w')
    atoms = self.atoms
    box = self.box
    crystal_num = 0
    
    # determine whether an atom is crystallized or not
    # write to cfg when it is
    write_buffer = ""
    count = 0
    for i in range(self.natoms): 
      op_i = op[i]
      if op_i == None:
        continue
      if op_i <= thres:
        j = int(id_in_array[i])
        atomi = atoms[int(j)]
        crystal_num += 1
        write_buffer += \
            "{0}\n".format(atomi.mass) \
            + "{0}\n".format(atomi.type) \
            + '\t'.join(str(ii) for ii in atomi.data) + '\t' \
            + "{0}\n".format(costheta[count])
        count += 1
        
    # write header
    header = self.header
    header_wbuffer = "Number of particles = {0}\n".format(crystal_num)
    for i in range(1, 12):
      header_wbuffer += header[i]
    header_wbuffer += "entry_count = {}\n".format(self.entry_count+1)
    for i in range(13, len(header)):
      header_wbuffer += header[i]
    header_wbuffer += "auxiliary[{0}] = ctheta\n".format(self.entry_count-3)
    
    # finally, write to file
    write_buffer = header_wbuffer + write_buffer
    f.write(write_buffer)
   
  def grain_orientation(self, op, id_in_array, thres):
#    f = open("cry_"+self.filename, 'w')
    atoms = self.atoms
    box = self.box
    crystal_num = 0
    
    cos_theta = []
    theta = np.zeros(4)
    
    # determine whether an atom is crystallized or not
    # write to cfg when it is
    write_buffer = ""
    for i in range(self.natoms):
      op_i = op[i]
      if op_i == None:
        continue
      if op_i <= thres:
        j = id_in_array[i]
        atomi = atoms[int(j)]
        nnn = atomi.neigh_num()
        ri = atomi.r
        for j in range(nnn):
          rj = atomi.neigh[j].r
          dist_ij = dist(ri, rj)
          theta[j] = (rj[2]-ri[2])/dist_ij
        cos_theta.append(np.amax(theta))
    return cos_theta
    
  def pair_correlation(self, low, high, nbins):
    atoms = self.atoms
    natoms = self.natoms
    hist = np.zeros(nbins)
    for i in range(natoms):
      atomi = atoms[i]
      neigh_dist = [neighii.dist for neighii in atomi.neigh]
      ihist, edges = np.histogram(neigh_dist, bins=nbins, range=(low, high) )
      hist = [hist[ii]+ihist[ii] for ii in range(len(hist))]
    
    center = [(edges[ii]+edges[ii+1])/2 for ii in range(0, len(edges)-1)]
    deltar = center[1]-center[0]
    
    g = [hist[ii]/(4*math.pi * (center[ii])**2 * deltar * natoms**1) \
        for ii in range(len(hist))]
    
    return [center, g]

# calculate the distance between two atoms    
def dist(r1, r2):
  if len(r1)!=3 or len(r2)!=3:
    raise Exception("A position vector must have length 3.")
  dist2 = (r1[0]-r2[0])**2 + (r1[1]-r2[1])**2 + (r1[2]-r2[2])**2
  return math.sqrt(dist2)
 
# calculate the azimuthal angle of r2 relative to r1
def phi(r1, r2):
  if len(r1)!=3 or len(r2)!=3:
    raise Exception("A position vector must have length 3.")
  x = r1[0] - r2[0]
  y = r1[1] - r2[1]
  distxy = math.sqrt(x**2 + y**2)
  if y>0:
    return math.acos(x/distxy)
  else:
    return 2*math.pi-math.acos(x/distxy) 

# calculate the polar angle of r2 relative to r1 
def theta(r1, r2):
  if len(r1)!=3 or len(r2)!=3:
    raise Exception("A position vector must have length 3.")
  dist12 = dist(r1, r2)
  z = r2[2] - r1[2]
  return math.acos(z/dist12)  
 
# calculate cos_theta between two vectors 
def Cos_theta(r1, r2):
  if len(r1)!=3 or len(r2)!=3:
    raise Exception("A position vector must have length 3.")
  dot12 = r1[0]*r2[0] + r1[1]*r2[1] + r1[2]*r2[2]
  r1norm = math.sqrt(r1[0]*r1[0] + r1[1]*r1[1] + r1[2]*r1[2])
  r2norm = math.sqrt(r2[0]*r2[0] + r2[1]*r2[1] + r2[2]*r2[2])
  cos_ = dot12 / (r1norm*r2norm)
  return cos_

    
  
  
##---------------------------main---------------------------##                        

# file_num = range(1, 2100, 100) + range(2100, 2800, 5) + range(2800, 4400, 20) + range(4400, 5200, 5) + range(5200, 7374, 20)

# file_num = range(1, 2200, 5)

file_num = range(1501, 2004, 100)

# file_num = range(2000,2500,10)

file_local = file_num

r_cut = 10.0

write_buffer = ""

f_write = open("pair_corr", 'w')

for num in file_local:
  filename = "b{0:05d}.cfg".format(num)
  cfg = cfg_file_class()
  type_dict = {'Si':1}
  cfg.read_cfg(filename, type_dict)
  cfg.create_neigh_list(r_cut)
  
  data = cfg.pair_correlation(0.1, 10.0, 200)
  
  centers = data[0]
  g = data[1]
  
#  plt.plot(centers, g)
#  plt.show()
 
  
  write_buffer += \
		str(num)+'\n'
  write_buffer += '\t'.join(str(i) for i in centers) + '\n'
  write_buffer += '\t'.join(str(i) for i in g) + '\n'
  
f_write.write(write_buffer)
  
	  

'''
  write_buffer += \
		str(num)+'\n' 
  
  for i in range(len(id)):
    write_buffer += str(id[i]) + '\t' \
    + str(neigh_num[i]) + '\t' + str(op[i]) + '\t'
  write_buffer += '\n'
'''
'''
if rank != 0:
  pp.send(write_buffer, 0)
else:
  for i in range(1, nprocs):
    receive_buffer = pp.receive(i)
    write_buffer += receive_buffer
    
  f_write = open("tetra_order", 'w')
  f_write.write(write_buffer)    

pp.finalize()  
'''
