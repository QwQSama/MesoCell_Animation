from math import *
import numpy as np
import copy
import torch

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

# Labels of grains in the hexagon
#     O1
#   O2  O6
#   O3  O5
#     O4

class MLPCell:
    """
    Define a class of cell object using machine learning.
    model: trained model
    scaler_x: scaler for input
    scaler_y: scaler for output
    """

    def __init__(self, id, theta, model, scaler_x, scaler_y, device):
        self.id = id
        self.alpha = alpha0
        self.radius = radius
        self.d1 = 2 * self.radius
        self.d2 = 2 * self.radius
        self.Fn1 = 0.0
        self.Ft1 = 0.0
        self.Fn2 = 0.0
        self.nContact = 6

        self.l1 = self.d2 + 2 * self.d1 * cos(self.alpha)
        self.l2 = 2 * self.d1 * sin(self.alpha)
        
        self.l10 = self.l1
        self.l20 = self.l2

        if locV == 'Vhexagon':
            Vr1 = 0; Vr2 = -cos(self.alpha)
        if locV == 'Vboxin':
            Vr1 = 0; Vr2 = 0
        if locV == 'Vboxout':
            Vr1 = 1; Vr2 = 1
        self.Vcell0 = (self.l1 + Vr1 * 2 * self.radius) * (self.l2 + Vr2 * 2 * self.radius)

        self.theta = theta
        self.omega = 1 / pi * (1 + alpOmega * cos(2 * (theta - belOmega))) * pi / ntheta # indeed it is omega*dtheta

        self.vsignn = 0.0
        self.vsigtt = 0.0

        self.vsig11 = 0.0
        self.vsig22 = 0.0
        self.vsig12 = 0.0

        self.Sliding1 = False
        self.Opening12 = False # the contact between grains1 and 2
        self.Opening23 = False # the contact between grains2 and 3
        self.IsCont14 = False
        self.IsCont26 = False

        self.model = model
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.device = device
        
        
    def node_position(self):
        r = self.radius
        l1 = self.l1
        l2 = self.l2
        cosa =np.cos(self.alpha)
        sina =np.sin(self.alpha)
        d1 = self.d1
        d2 = self.d2
        
        x = [0,-d1*sina,-d1*sina,0,d1*sina,d1*sina]
        y = [d2/2+d1*cosa,d2/2,-d2/2,-d2/2-d1*cosa,-d2/2,d2/2]
        
        cos_theta =np.cos(self.theta)
        sin_theta =np.sin(self.theta)
            
        n = len(x)
        for i in range(n):
            x0,y0 = x[i],y[i]
            x[i] =cos_theta*x0 + sin_theta*y0
            y[i] =sin_theta*x0 + cos_theta*y0
        
        return (x,y)            

    def Substrain(self, deps11, deps22, deps12):
        """
        Update cell information for given macroscopic incremental strains (soil mechanics sign convention)
        """
        G2 = 0  # 1: considering G2 in the new version of H (2021); 0: without G2 in the old version (Nicot et al 2011)

        # select the localization hypothesis
        if locV == 'Vhexagon':
            Vr1 = 0; Vr2 = -cos(self.alpha)
        if locV == 'Vboxin':
            Vr1 = 0; Vr2 = 0
        if locV == 'Vboxout':
            Vr1 = 1; Vr2 = 1

        dl1 = -self.l1 * (deps11 * pow(cos(self.theta), 2.0) + deps22 * pow(sin(self.theta), 2.0) + 2 * deps12 * cos(self.theta) * sin(self.theta))
        dl2 = -self.l2 * (deps11 * pow(sin(self.theta), 2.0) + deps22 * pow(cos(self.theta), 2.0) - 2 * deps12 * cos(self.theta) * sin(self.theta))
        
        edl1 = dl1/self.l10 
        edl2 = dl2/self.l20 

        data = np.array([[edl1, edl2, self.alpha, self.l1, self.l2, self.Fn1, self.Fn2, self.Ft1]])
        data_scaled_x = self.scaler_x.transform(data)
        data_scaled_x = torch.tensor(data_scaled_x, dtype=torch.float32).to(self.device)

        self.model.eval()
        prediction = self.model(data_scaled_x)
        prediction = prediction.detach()
        if prediction.is_cuda:
            prediction = prediction.cpu()
        prediction = prediction.numpy()
        prediction = self.scaler_y.inverse_transform(prediction)

        N1_p, N2_p, T1_p, alpha_p = prediction[-1]
        self.alpha = alpha_p 
        self.l1 += dl1
        self.l2 += dl2
        self.Fn1 = N1_p 
        self.Ft1 = T1_p 
        self.Fn2 = N2_p 

        self.d1 = self.l2/ (2 * sin(self.alpha))
        self.d2 = self.l1 - 2 * self.d1 * cos(self.alpha)
        self.Vcell = (self.l1 + Vr2 * 2 * self.radius) * (self.l2 + Vr1 * 2 * self.radius)

        # Check for pathological cases
        vcorsignn, vcorsigtt = 0, 0

        if self.l1 - 2* self.radius <= 0:
            self.IsCont14 = True
            self.nContact = 7
            vcorsignn = kn * (2 * self.radius - self.l1) * self.l1
        else:
            self.IsCont14 = False


        if self.l2 - 2 * self.radius <= 0:
            self.IsCont26 = True
            self.nContact = 8
            vcorsigtt = 2 * kn * (2 * self.radius - self.l2) * self.l2
        else:
            self.IsCont26 = False


        if self.d1 > 2.0:
            self.Fn2 = 0
            self.Fn1, self.Ft1 = 0, 0
            self.Opening12 = True
            #self.Vcell = 0
        else:
            self.Opening12 = False

        if self.d2 > 2.0: #self.Fn2 < 0:
            self.Fn2 = 0
            self.Fn1, self.Ft1 = 0, 0
            self.Opening23 = True
            #self.Vcell = 0
        else:
            self.Opening23 = False

        # Update the stress (it includes pathological corrections)
        self.vsignn = 4.0 * self.Fn1 * self.d1 * pow(cos(self.alpha), 2.0) + 4.0 * self.Ft1 * self.d1 * cos(
            self.alpha) * sin(self.alpha) + 2.0 * self.Fn2 * self.d2 + vcorsignn
        self.vsigtt = 4.0 * self.Fn1 * self.d1 * pow(sin(self.alpha), 2.0) - 4.0 * self.Ft1 * self.d1 * cos(
            self.alpha) * sin(self.alpha) + vcorsigtt

        self.vsig11 = self.vsignn * cos(self.theta) ** 2 + self.vsigtt * sin(self.theta) ** 2
        self.vsig22 = self.vsignn * sin(self.theta) ** 2 + self.vsigtt * cos(self.theta) ** 2
        self.vsig12 = (self.vsignn - self.vsigtt) * sin(self.theta) * cos(self.theta)
    