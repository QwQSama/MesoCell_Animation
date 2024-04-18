from math import *
import numpy as np
import copy
import matplotlib.pyplot as plt

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


class MesoCell:
    """
    Define a class of meso cell object. Each hexagon has the following attributes:
    - its id which related to the direction of the cell(id)
    - its branch lenghts (d1, d2)
    - its opening angle (alpha)
    - the normal and tangential forces of contant 1 (Fn1, Ft1) (see Nicot et al 2011)
    - the normal forces of contact 2 (Fn2)
    - its hexagon lengths(l1,l2)
    - three options of the volume (Vboxout, Vboxin, Vhexagon)
    - its direction (theta)
    - the weight of this direction (ome)
    - the Love-Weber stress multiplied by volume in the local frame (n, t) (vsignn, vsigtt)
    - the Love-Weber stress multiplied by volume in the global frame (e1, e2) (vsig11, vsig22)
    - the fabric tensor multiplied by the number of contacts in the local frame (n, t) (cfabnn, cfabtt)
    - the fabric tensor  multiplied by the number of contacts in the global frame (e1, e2) (cfab11, cfab22)
    - whether sliding exsits at the contact 1 (Sliding1)
    - whether the contact between grain1 and 2 (grains 2 and 3) is open (opening12, opening23)
    - whether the particle 2 and 4 or particle 2 and 6 are contacted (IsCont14, IsCont26)
    """

    def __init__(self, id, theta):
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

        # geometrical relationships
        K11 = 2.0 * cos(self.alpha)
        K12 = 1.0
        K13 = -2.0 * self.d1 * sin(self.alpha)
        K21 = 2.0 * sin(self.alpha)
        K22 = 0.0
        K23 = 2.0 * self.d1 * cos(self.alpha)

        # closure : static equilibrium, assume no sliding
        K31 = cos(self.alpha)
        K32 = -1.0
        K33 = (self.Fn1 * sin(self.alpha) - self.Ft1 * cos(self.alpha) - kt * self.d1 * (G2 + sin(self.alpha))) / kn

        # right hand side of the compatibility equation, assume no sliding
        L1 = dl1
        L2 = dl2
        L3 = 0

        K = np.array([[K11, K12, K13], [K21, K22, K23], [K31, K32, K33]])
        L = np.array([L1, L2, L3])
        solve = np.linalg.solve(K, L)  # check whether it solves K.X=L with X unknown

        dd1 = solve[0]
        dd2 = solve[1]
        dalpha = solve[2]

        dFn1 = -kn * dd1
        dFt1 = kt * self.d1 * dalpha
        dFn2 = -kn * dd2

        if abs(self.Ft1 + dFt1) < abs(tan(phig) * (self.Fn1 - kn * dd1)):
            self.Sliding1 = False
        else:
            self.Sliding1 = True
            if self.Ft1 > 0:
                xi = 1
            else:
                xi = -1

            K31 = cos(self.alpha) + xi * tan(phig) * (G2 + sin(self.alpha))
            K33 = (self.Fn1 * sin(self.alpha) - self.Ft1 * cos(self.alpha)) / kn
            L3 = (xi * (G2 + sin(self.alpha)) * (tan(phig) * self.Fn1 - self.Ft1)) / kn

            K = np.array([[K11, K12, K13], [K21, K22, K23], [K31, K32, K33]])
            L = np.array([L1, L2, L3])
            solve = np.linalg.solve(K, L)

            dd1 = solve[0]
            dd2 = solve[1]
            dalpha = solve[2]

            dFn1 = -kn * dd1
            dFn2 = -kn * dd2
            dFt1 = xi*tan(phig)*(self.Fn1+dFn1)-self.Ft1

        self.d1 += dd1
        self.d2 += dd2
        self.alpha += dalpha

        self.l1 += dl1
        self.l2 += dl2

        self.Vcell = (self.l1 + Vr2 * 2 * self.radius) * (self.l2 + Vr1 * 2 * self.radius)
        self.Fn1 += dFn1
        self.Ft1 += dFt1
        self.Fn2 += dFn2

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
