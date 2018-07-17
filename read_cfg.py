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

class bin:
  def __init__(self, cord_index, is_real, atom_list):
    self.is_real = is_real
    self.index = cord_index
    self.atom_list = atom_list
    
# atom class. more atom properties shall be added here.
class atom:
  def __init__(self, mass, type, r, data):
    self.mass = mass
    self.type = type
    self.r    = r
    self.data  = data
    self.neigh = []
    self.pbc = [0, 0, 0]
    self.id = -1
    self.reduced_cord = np.zeros(3)
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
    
    H0 = np.zeros( (3,3) )
    # read header
    while True:
      try:
        float(line_split[0])
        break
      except:
        header.append(cur_line)
        for i in range(1,4):
          for j in range(1,4):
            if line_split[0] == "H0({0},{1})".format(i,j):
              H0[j-1][i-1] = float( line_split[2] )
        if line_split[0][0:9] == "auxiliary":
          aux_dict.update({line_split[-1]: aux_count})
          aux_count += 1      
        cur_line = f_read.readline()
        line_split = cur_line.split()
     
    a = H0[0]
    b = H0[1]
    c = H0[2]
    
    acubic = ( a[1]==0 and a[2]==0 )
    bcubic = ( b[0]==0 and b[2]==0 )
    ccubic = ( c[0]==0 and c[1]==0 )
    
    triclinic = not (acubic and bcubic and ccubic)
    
    # read first atom
    mass_string = cur_line
    type_string = f_read.readline()
    data_string = f_read.readline()
    mass = float(mass_string)
    type = type_string.split()[0]
    data = [float(i) for i in data_string.split()]
    
    reduced_cord = [data[i] for i in range(3)]
    r = np.matmul(H0, reduced_cord)
    new_atom = atom(mass, type, r, [ii for ii in data])
    new_atom.reduced_cord = reduced_cord
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
        type = type_string.split()[0]
#        type = type_string.rstrip('\n')       
        data = [float(i) for i in data_string.split()]
        new_atom = atom(mass, type, r, [ii for ii in data])
        new_atom.reduced_cord = [data[i] for i in range(3)]
        new_atom.r = np.matmul(H0, new_atom.reduced_cord)

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
    self.H0 = H0
    self.filename = cfg_filename
    self.type_dict = type_dict
    self.aux_dict = aux_dict
    self.entry_count = aux_count + 3
    self.triclinic = triclinic
  
  def create_ghost_atoms(self, r_cut):
    self.nghost = 0
    natoms = self.natoms
    H0 = self.H0
    
    padd = np.zeros(3)
    madd = np.zeros(3)
    if self.triclinic == True:
      for i in range(0,3):
        # inter-rotation between three directions
        if i==0:
          b = H0[:,0]
          a = H0[:,1]
          c = H0[:,2]
        elif i==1:
          b = H0[:,1]
          a = H0[:,0]
          c = H0[:,2]
        elif i==2:
          b = H0[:,2]
          a = H0[:,0]
          c = H0[:,1]
        
        ac = np.dot(a,c)
        ab = np.dot(a,b)
        bc = np.dot(b,c)
        a2 = np.dot(a,a)
        b2 = np.dot(b,b)
        c2 = np.dot(c,c)
        
        A = (bc*ac - c2*ab) / (a2*c2 - ac**2)
        C = (ab*ac - a2*bc) / (a2*c2 - ac**2)
      
        q0 = A**2 *a2 + C**2 *c2 + 2*A*C*ac - r_cut**2
        q1 = 2 * (A*ab + C*bc)
        q2 = b2
        
        sqrt_factor = q1**2 - 4*q0*q2
        if sqrt_factor > 0:
          padd[i] = (-q1 + math.sqrt(q1**2 - 4*q0*q2)) / (2*q2)
          madd[i] = ( q1 + math.sqrt(q1**2 - 4*q0*q2)) / (2*q2)
        else:
          padd[i] = 0
          madd[i] = 0
    
    # no triclinic
    else:
      for i in range(3):
        padd[i] = r_cut / H0[i][i]
        madd[i] = r_cut / H0[i][i]
    
    
    for i in range(0, natoms):
      images = []
      atomi = self.atoms[i]
      x = atomi.reduced_cord
      ri = atomi.r
      
      nx_p = int( math.floor( 1 - x[0] + padd[0] ) )
      nx_m = int( math.floor(     x[0] + madd[0] ) )
      ny_p = int( math.floor( 1 - x[1] + padd[1] ) )
      ny_m = int( math.floor(     x[1] + madd[1] ) )
      nz_p = int( math.floor( 1 - x[2] + padd[2] ) )
      nz_m = int( math.floor(     x[2] + madd[2] ) )
      
      for ii in range(-nx_m, nx_p+1):
        for jj in range(-ny_m, ny_p+1):
          for kk in range(-nz_m, nz_p+1):
            if [ii,jj,kk] == [0,0,0]:
              continue
            
            r_shift = ri
            r_shift = np.add(ii*H0[0], r_shift)
            r_shift = np.add(jj*H0[1], r_shift)
            r_shift = np.add(kk*H0[2], r_shift)
            
            new_atom = copy.deepcopy(atomi)
            new_atom.r = r_shift
            old_rc = atomi.reduced_cord
            new_atom.reduced_cord = [old_rc[0]+ii, old_rc[1]+jj, \
                                     old_rc[2]+kk]
            new_atom.pbc = [ii, jj, kk]
            self.atoms.append(new_atom)
            self.nghost += 1

  def create_neigh_list(self, r_cut, box_to_bin = 3, nnn=None):
    self.create_ghost_atoms(r_cut)
    atoms = self.atoms
    natoms = self.natoms
    nghost = self.nghost
    nall = natoms + nghost
    H0 = self.H0
    box_size_x = np.linalg.norm(H0[0][0])
    box_size_y = np.linalg.norm(H0[1][1])
    box_size_z = np.linalg.norm(H0[2][2])

    nbin_x = math.floor(box_size_x / r_cut)
    nbin_y = math.floor(box_size_y / r_cut)
    nbin_z = math.floor(box_size_z / r_cut)

    use_bin = nbin_x >= box_to_bin and \
              nbin_y >= box_to_bin and \
              nbin_z >= box_to_bin

    if not use_bin:
      for i in range(0, natoms):
        ri = atoms[i].r
        for j in itertools.chain(range(0,i), range(i+1,nall)):
          rj = atoms[j].r
          distij = dist(ri, rj)
          if distij < r_cut:
            new_neigh = neighbor(rj, atoms[j].pbc, atoms[j].id, distij)
            self.atoms[i].neigh.append(new_neigh)
    else:
      # create bins and allocate atoms into bins
      bin_list = []
      bin_size_x = 1 / nbin_x
      bin_size_y = 1 / nbin_y
      bin_size_z = 1 / nbin_z
      # include ghost bins
      nbin_x += 2
      nbin_y += 2
      nbin_z += 2

      # create bin lists
      for i in range( int(nbin_x*nbin_y*nbin_z) ):
        z_index = math.floor(i/(nbin_x*nbin_y))
        residue = i - z_index*nbin_x*nbin_y
        y_index = math.floor( residue/nbin_x )
        x_index = residue - y_index*nbin_x

        is_real = x_index>0 and x_index<(nbin_x-1) and \
                  y_index>0 and y_index<(nbin_y-1) and \
                  z_index>0 and z_index<(nbin_z-1)

        bin_list.append(bin([x_index, y_index, z_index], is_real, []))

      # allocate atoms into bins
      for i in range(nall):
        atomi = atoms[i]
        rc_i = atomi.reduced_cord
        x_bin_cord = int( math.ceil(rc_i[0] / bin_size_x) )
        y_bin_cord = int( math.ceil(rc_i[1] / bin_size_y) )
        z_bin_cord = int( math.ceil(rc_i[2] / bin_size_z) )
        bin_index = int(z_bin_cord * nbin_x * nbin_y + \
                        y_bin_cord * nbin_x + \
                        x_bin_cord )
        bin_list[bin_index].atom_list.append(i)

      # finally, create neighbor list
      for bin_id in range(len(bin_list)):
        bin_i = bin_list[bin_id]
        if not bin_i.is_real:
          continue

        # find out neighbor bins, add their atom_list together
        all_atoms = []
        for ii in [-1, 0, 1]:
          for jj in [-1, 0, 1]:
            for kk in [-1, 0, 1]:
              neigh_bin = int( bin_id + kk * nbin_x * nbin_y + \
                                        jj * nbin_x + \
                                        ii )
              all_atoms += bin_list[neigh_bin].atom_list

        for i in bin_i.atom_list:
          atomi = atoms[i]
          r_i = atomi.r
          for j in all_atoms:
            if i == j:
              continue
            atomj = atoms[j]
            r_j = atomj.r
            distij = dist(r_i, r_j)
            if distij < r_cut:
              new_neigh = neighbor(r_j, atomj.pbc, atomj.id, distij)
              atomi.neigh.append(new_neigh)

    #if nnn is not None, nnn nearest neighbors are retained
    # while others are discarded
    if nnn != None:
      for i in range(natoms):
        atomi = atoms[i]
        n_now = atomi.neigh_num()
        if n_now > nnn:
          neigh_dist = [atomi.neigh[ii].dist for ii in range(n_now)]
          nnn_id = select3(neigh_dist, nnn)
          new_neigh = [atomi.neigh[ii] for ii in nnn_id]
          atomi.neigh = new_neigh

  # write cfg file data to lammps read_data file
  # read_cfg shall be called beforehand
  def cfg2lmpin(self, lmpin_file, dict, map2cubic = False, write2tric = False, charge = None):
    f_write = open(lmpin_file, 'w')
    atoms = self.atoms
    H0 = self.H0
    # write header
    write_buffer = \
    "# Data file\n\n" \
    + "{0} atoms\n".format(self.natoms) \
    + "{0} atom types\n\n".format(len(dict)) \
    + "\t{0}\t\t{1} xlo xhi\n".format(0, H0[0][0]) \
    + "\t{0}\t\t{1} ylo yhi\n".format(0, H0[1][1]) \
    + "\t{0}\t\t{1} zlo zhi\n".format(0, H0[2][2]) 
    
    if self.triclinic and (not map2cubic) or write2tric:
      write_buffer += "\t{0}\t{1}\t{2} xy xz yz\n".format(H0[0][1], H0[0][2], H0[1][2])
    
    f_write.write(write_buffer)
    
    # write Atoms section
    write_buffer = "\nAtoms\n\n"
    for i in range(self.natoms):
      atomi = atoms[i]
      type_ID = dict[atomi.type]
      r = atomi.r
      if (self.triclinic and map2cubic):
        r = self.Map2cubic(r)
      if (charge != None):
        qi = charge[atomi.type] 
        write_buffer += "{index}\t{typeID}\t{q}\t{x}\t{y}\t{z}\n".\
          format(index=(i+1), typeID=type_ID, q=qi, x=r[0], y=r[1], z=r[2])
      else:
        write_buffer += "{index}\t{typeID}\t{x}\t{y}\t{z}\n".\
          format(index=(i+1), typeID=type_ID, x=r[0], y=r[1], z=r[2])
    f_write.write(write_buffer)
  
  def cfg2voro(self, voro_file, map2cubic=False):
    f_write = open(voro_file, 'w')
    atoms = self.atoms
    write_buffer = ""
    for i in range(self.natoms):
      atomi = atoms[i]
      r = atomi.r
      if map2cubic:
        r = self.Map2cubic(r)
      write_buffer += \
      "{0}\t{1}\t{2}\t{3}\n".format(int(atomi.id), r[0], r[1], r[2])
    f_write.write(write_buffer)
    
  def write2cfg(self, newcfgfile, watoms, add_aux=None):
    f = open(newcfgfile, 'w')
    header_buffer = ""
    if add_aux != None:
      total_aux = self.entry_count + len(add_aux)
    else:
      total_aux = self.entry_count
    
    total_atoms = len(watoms)
    header = self.header
    header_buffer = "Number of particles = {0}\n".format(total_atoms)
    for i in range(1, 12):
      header_buffer += header[i]
    header_buffer += "entry_count = {}\n".format(total_aux)
    for i in range(13, len(header)):
      header_buffer += header[i]
    if add_aux != None:
      for i in range(0, len(add_aux)):
        header_buffer += "auxiliary[{0}] = {1}\n".format(self.entry_count-3+i, add_aux[i])
    
    atoms_buffer = ""
    for i in range(total_atoms):
      atomi = watoms[i]
      atoms_buffer += \
          "{0}\n".format(atomi.mass) \
          + "{0}\n".format(atomi.type) \
          + '\t'.join(str(ii) for ii in atomi.data) + '\n'
          
    write_buffer = header_buffer + atoms_buffer
    f.write(write_buffer)
         
  def Map2cubic(self, r):
    a = self.H0[:,0]
    b = self.H0[:,1]
    c = self.H0[:,2]
    
    if b[1]>0:
      while r[1]>b[1]:
        r = np.add(r, -b)
      while r[1]<0:
        r = np.add(r, b)
    else:
      while r[1]>0:
        r = np.add(r, b)
      while r[1]<b[1]:
        r = np.add(r, -b)
        
    if a[0]>0:
      while r[0]>a[0]:
        r = np.add(r, -a)
      while r[0]<0:
        r = np.add(r, a)
    else:
      while r[0]>0:
        r = np.add(r, a)
      while r[0]<a[0]:
        r = np.add(r, -a)
    
    return r

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

