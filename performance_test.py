from math import *
import numpy as np
import copy
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import sys
import os
from joblib import dump, load

# Path adjustments
dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_path, 'dataset'))
sys.path.append(os.path.join(dir_path, 'trained_model'))
sys.path.append(os.path.join(dir_path, 'model'))

from MLP import MLP
from MesoCell import MesoCell
from MLPCell import MLPCell
from Environment import MacREV, Loading

global kn, kt, phig, alpOmega, belOmega, alpha0, ntheta, radius, confining
"""
Constitutive paramters:
- Normal contact stiffness (kn)
- Contact stiffness ratio kn/kt (ratioK)
- Inter-granular friction angle (phig)
- Anisotropy of the distribution (alphaOmega)
- Principal direction of anisotropy (beltaOmega)
- Initial opening angle (alpha0)
"""
kn = 2.0e8; kt = 0.5 * kn; phig = 30*pi/180; alpOmega = 0; belOmega = 0; alpha0 = 50* pi / 180; confining = 200000 # paraThesis Guillaume in th


"""
Computation constant:
- Radius of particles (radius)
- Number of the directions of cells (ntheta)
- The limits of the directions of cells (thetaMin, thetaMax)
- localization hypothesis (locV): 'Vhexagon', 'Vboxin', 'Vboxout'
"""
radius = 1
ntheta = 360
thetaMin = 0
thetaMax = pi
locV = 'Vhexagon'

def omegaSum(alphaOme, betaOme, n = 100):
    sum = 0
    dtheta = pi/n
    for i in range(n):
        theta = dtheta*i
        omei = 1/pi*(1+alphaOme*cos(2*(theta-betaOme)))
        sum += omei*dtheta
    return sum

def display(example, ball_filling = False):
    px,py = example.node_position()
    n = len(px)

    r = radius
    fig, ax = plt.subplots(figsize=(4,4))
    
    color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for i in range(n):
        circle = plt.Circle((px[i], py[i]), r, color=color_list[i], fill=ball_filling)
        plt.gcf().gca().add_artist(circle)

    plt.axis('equal')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.show()
    
    print("number_contact: ", example.nContact)
    print("alpha", example.alpha / np.pi * 180)
    print('d1 = ', example.d1)
    print('d2 = ', example.d2)    
    print('l1 = ', example.l1)    
    print('l2 = ', example.l2)    
    print('N1 = ', example.Fn1)
    print('N2 = ', example.Fn2)    
    print('T1 = ', example.Ft1)  
    print('open12 = ',example.Opening12)
    print('open23 = ',example.Opening23)

def isotest(Cell, skip, *args, **kwargs):
    REV = MacREV(Cell, *args, **kwargs)
    load = Loading(REV, skip)
    load.isoTest()
    res = load.res
    return res


def plot_MesoCell(res1, res2=None ):
    plt.plot(res1['eps11'], res1['sig11'], marker='o', label='MesoCell_result')
    if res2 is not None:
        plt.plot(res2['eps11'], res2['sig11'], marker='o', label='MLPCell_result')
    plt.xlabel('eps11')
    plt.ylabel('sig11')
    plt.title('Comparison of eps11 and sig11')
    plt.legend()
    plt.grid(True)
    plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# load model
trained_model_dir = os.path.join(dir_path, 'trained_model')
model_path = os.path.join(trained_model_dir, 'model_complete.pth')
model = torch.load(model_path)
model.to(device)
model.eval()

#load scaler_x
trained_model_dir = os.path.join(dir_path, 'trained_model')
scaler_path = os.path.join(trained_model_dir, 'scaler_X.joblib')
scaler_x = load(scaler_path)

#load scaler_y
dataset_dir = os.path.join(dir_path, 'dataset')
scaler_path = os.path.join(dataset_dir, 'scaler_10000.joblib')
scaler_y = load(scaler_path)

# check MLPCell 
# cell = MLPCell(1, 0, model, scaler_x, scaler_y, device)
# cell.Substrain(0.01/2**0.5,0.01/2**0.5,0)
# display(cell)

# cell = MesoCell(1,0)
# for i in range(100):
#     cell.Substrain(0.0001/2**0.5,0.0001/2**0.5,0)
# display(cell)

# skip = 0
# res1 = isotest(MesoCell,skip)
# skip = 1000
# res2 = isotest(MLPCell, skip, model, scaler_x, scaler_y, device)
# # res2 = None
# plot_MesoCell(res1,res2)