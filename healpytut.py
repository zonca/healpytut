#! Python/Healpy Tutorial
#!=======================

#!Preparation
#!-----------

#!Carver::
#!
#!   module load python
#!   module load cmbdev

#!Your laptop

#!install Entough Python Distribution for Win/Mac
#!http://www.enthought.com/products/epd_free.php

#!Components
#!----------

#!* Ipython: Interactive shell
#!* Numpy: Array math
#!* Matplotlib: Plotting
#!* Scipy: Advanced scientific tools [FFT, spline, signal processing]
#!* Healpy: Healpix for python

#!Overview
#!--------

#!* Basic types
#!* Arrays
#!* Plotting
#!* Code organization: modules,packages
#!* Healpy

#!Setup environment
#!-----------------

#! .ipython/ipythonrc
#! pdb 1, autocall 2

#! get help on function by calling:
#! healpy.nside2npix?

#!Basic types
#!-----------

#!Lists
#!~~~~~

test_list = []
test_list.append(9)
print(test_list)

test_list.append("quite a long string")
test_list.append([1, 3, 4])
test_list.append(10)
print(test_list)

#Replace
test_list[2] = 1
print(test_list)

#Slicing
print(test_list[:2])
print(test_list[-1:])

#!first python **WARNING**
#!Last element is excluded!!!
print(test_list[1:2])

#!this is C

for i in range(len(test_list)):
    print(test_list[i])

#!this is Python

for element in test_list:
    print(element)

#!Tuple
#!~~~~~

#!Like lists but not mutable, used for string interpolation, return of functions
test_tuple = (3, 4)
print(test_tuple[0])

#test_tuple[0] = 2

#!Dictionary
#!~~~~~~~~~~

test_dict = {}

test_dict["LFI28M"] = 127.
test_dict["LFI28S"] = 12.

print(test_dict)

print(test_dict["LFI28M"])

for k,v in test_dict.iteritems(): #Dictionary is **NOT ORDERED**
    print("Channel %s has value %.2f" % (k,v)) #C-style string formatting

#!Strings
#!~~~~~~~

# type of quotes does not matter
test_string = "a quite long string"
test_string = 'a quite long string'

# multiline strings
test_string = """
This is a multiline
string,
keeps formatting"""

print(test_string)

#strings interpolation

print("either using " + str(1.0) + " concatenation or interpolation for int %04d, float %.2f, exp %.1e" % (3, 1/3., 2.3))

#!Functions
#!~~~~~~~~~~~~~~~~

def sum_diff(a, b, take_abs=False):
    if take_abs:
        return abs(a+b), abs(a-b)
    else:
        return a+b, a-b


a=2; b=3
absum, abdiff = sum_diff(a, b)
ab_sumdiff = sum_diff(a, b)

print(absum)
print(ab_sumdiff)

#!Integer division
#!~~~~~~~~~~~~~~~~

#! second python **WARNING**
# 1/2 = 0 because they are integers
# 1./2 = .5 because 1. is float
# to avoid do at beginning of software
# from __future__ import division


#!Arrays
#!------

import numpy as np
a = np.array([1, 4, 5])
print(a.dtype)
a[0] = .9
print(a)

#!**Warning** type is integer

a = np.array([1, 4, 5], dtype=np.double)
a[0] = .9
print(a)

#same slicing as lists
a = np.arange(20)
print(a[10:18:2]) #2 is the step

#2D same as IDL, shape is always a **tuple**
a = np.zeros((3, 4)) 
a[1, 3] = 2
print(a)

#array itself is an object, so it has methods associated
print(a.mean())
print(a.std())
print(a.flatten())

#!Plotting
#!--------

#!Interactively with ipython -pylab
#!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from pylab import *
plot(arange(10), arange(10)**2, label='First')
errorbar(arange(10), arange(10)**3, 50., None, 'r.', markersize=10, label='Second')
annotate('I like this', xy=(5, 125), xytext=(3.5, 350),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
text(0.5, 200,'some text')
grid()
legend(loc=0)
xlabel('X axis'); ylabel('Y axis')
xlim([0, 8])
title('Test plot')
savefig('plot.png')
show()

#!In software
#!~~~~~~~~~~~~~

#! **USE NAMESPACES**

import matplotlib.pyplot as plt
import numpy as np
plt.figure()
plt.plot(np.arange(10), np.arange(10)**2, label='First')
plt.errorbar(np.arange(10), np.arange(10)**3, 50., None, 'r.', markersize=10, label='Second')
plt.annotate('I like this', xy=(5, 125), xytext=(3.5, 350),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
plt.grid()
plt.legend(loc=0)
plt.xlabel('X axis'); plt.ylabel('Y axis')
plt.xlim([0, 8])
plt.title('Test plot')
plt.savefig('plot.png')
show()

#! no namespaces?
title = "My title"
title("Other title")

#!Code organization
#!-----------------

#!Modules
#!~~~~~~~

# Modules are just .py files containing functions, simplest library 
# usually they can be imported in other scripts or executed

if __name__ == '__main__':
    print('Executing this just if directly called as python this_script.py')

#example
import healpy
print(healpy.pixelfunc)
print(healpy.pixelfunc.nside2npix)

#!Packages
#!~~~~~~~~

#! collection of modules in a folder with an __init__.py file which defines what
#! is imported on the main level

# for example:

import healpy
print(healpy)
print(healpy.nside2npix)

#!Best practice
#! Start with a single module and then split into several modules importing in __init__.py the most important functions and classes, *NOT* internal functions.

#!Healpy
#!------
#!in background calling C++ Healpix for most transforms

#!healpy by default works in RING

import healpy

m = np.arange(healpy.nside2npix(256))
healpy.mollview(m, min=0, max=m.max(), title='Mollview RING', nest=False)
show()

#! http://lambda.gsfc.nasa.gov/data/map/dr4/skymaps/7yr/raw/wmap_band_imap_r9_7yr_W_v4.fits

filename = 'wmap_band_imap_r9_7yr_W_v4.fits'
m = healpy.read_map(filename) #by default converts to RING!!
healpy.mollview(m, title='Histogram equalized', nest=False, norm='hist')
show()
m = healpy.read_map(filename, nest=True) #keeps nested
healpy.mollview(m, coord=['G','E'], title='Linear scale', unit='mK', nest=True, min=-1,max=1, xsize=2000) #xsize increases resolution
healpy.graticule()
show()

healpy.gnomview(m, rot=[0,0.3], title='Linear scale', unit='mK', format='%.2g', nest=True)
show()

print(healpy.fit_dipole(m, gal_cut=20)) # degrees

#!Smoothing
#!~~~~~~~~~

m_smoothed = healpy.smoothing(m, fwhm=60, arcmin=True)
healpy.mollview(m_smoothed, min=-1, max=1, title='Map smoothed 1 deg')

#!Rotator
#!~~~~~~~

rot = healpy.Rotator(coord=['G','E'])
theta_gal, phi_gal = np.pi/2., 0.
theta_ecl, phi_ecl = rot(theta_gal, phi_gal)
print(theta_ecl, phi_ecl)

#!Masking
#!~~~~~~~
#! http://lambda.gsfc.nasa.gov/data/map/dr4/ancillary/masks/wmap_temperature_analysis_mask_r9_7yr_v4.fits

mask = healpy.read_map('wmap_temperature_analysis_mask_r9_7yr_v4.fits').astype(np.bool)

m = healpy.read_map(filename)

#method 1: multiply arrays 
m_masked = m.copy()
m_masked[np.logical_not(mask)] = healpy.UNSEEN
healpy.mollview(m_masked, min=-1, max=1)
show()

#method 2: numpy masked arrays
m_masked = healpy.ma(m)
print(m_masked)
m_masked.mask = np.logical_not(mask)
healpy.mollview(m_masked.filled(), min=-1, max=1)
show()
figure()
plot(m_masked[:10000].compressed())
show()

healpy.write_map('wmap_masked.fits', m_masked.filled(), coord='G')

#!Spectra
#!~~~~~~~

cl = healpy.anafast(m_masked.filled(), lmax=1024)
ell = np.arange(len(cl))
plt.figure()
plt.plot(ell, ell * (ell+1) * cl)
plt.xlabel('ell'); plt.ylabel('ell(ell+1)cl'); plt.grid()
show()

healpy.write_cl('cl.fits', cl)

from glob import glob #bash like file pattern matching
print(glob('*.fits'))
