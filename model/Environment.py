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

class MacREV:
    """
    Define a class of MacREV objects. Each texture has the following
    attributes:
    - stresses (sig11, sig22, sig12)
    - strains (eps11, eps22, eps12)
    - the global volume (vol)
    - the dictionary for initialized cells with their ids as keys (mesoCell)
    """

    def __init__(self, MesoCell, *args, **kwargs):
        self.sig11 = 0.0
        self.sig22 = 0.0
        self.sig12 = 0.0

        self.eps11 = 0.0
        self.eps22 = 0.0
        self.eps12 = 0.0

        self.mesoCell = {}
        self.specialMeso = {} #save all special cells
        vol0 = 0.0
        for i in range(ntheta):
            id = i
            thetai = thetaMin + i * (thetaMax - thetaMin) / ntheta
            celli = MesoCell(id, thetai, *args, **kwargs)
            vol0 += celli.Vcell0 * celli.omega
            self.mesoCell[id] = celli
        self.volO = vol0
        self.vol = self.volO

    def Step(self, deps11, deps22, deps12, t = 0):
        self.specialMeso["Open12"] = []
        self.specialMeso["Open23"] = []
        self.specialMeso["Contact14"] = []
        self.specialMeso["Contact26"] = []
        vsig11, vsig22, vsig12 = 0.0, 0.0, 0.0
        vol = 0.0
        for id, celli in self.mesoCell.items():
            celli.Substrain(deps11, deps22, deps12)
            vsig11 += celli.vsig11 * celli.omega
            vsig22 += celli.vsig22 * celli.omega
            vsig12 += celli.vsig12 * celli.omega
            if t == 1:
                if celli.Opening12 == True: self.specialMeso["Open12"] += [id]
                if celli.Opening23 == True: self.specialMeso["Open23"] += [id]
                if celli.IsCont14 == True: self.specialMeso["Contact14"] += [id]
                if celli.IsCont26 == True: self.specialMeso["Contact26"] += [id]

        self.vol = self.volO * (1 - self.eps11) * (1 - self.eps22)
        self.sig11 = vsig11 / self.vol
        self.sig22 = vsig22 / self.vol
        self.sig12 = vsig12 / self.vol

        self.eps11 += deps11
        self.eps22 += deps22
        self.eps12 += deps12

class Loading:
    def __init__(self, MacREV, skip=0):
        self.REV = MacREV
        self.Nstep = 0 # number of computation steps
        self.adjE = 0.0  # lateral adjustment of the strain
        self.res = {"test":[], "step":[], "eps11": [], "eps22": [], "eps12": [], "sig11": [], "sig22": [], "sig12": []}  # result storage
        self.skip = 0

    def isoTest(self, vit=1.0e3/kn, sigConfining=confining):  # vit is the strain increment between two steps
        if self.skip != 0:
            vit = vit*self.skip
        while 0.5 * (self.REV.sig11 + self.REV.sig22) <= sigConfining:
            # cn = 0  # internal number of steps to find the lateral adjustment strain
            # self.adjE = 0
            # copyREV = self._copyREV()
            # copyREV.Step(vit, 0, 0)  # first guess : oedometer compression
            # while abs(copyREV.sig22 - copyREV.sig11) / sigConfining > 10 ** -4:
            #     cn += 1
            #     if copyREV.sig22 > self.REV.sig11:
            #         self.adjE -= vit/20# reduce lateral strain
            #     else:
            #         self.adjE += vit/20# increase lateral strain
            #     copyREV = self._copyREV()
            #     copyREV.Step(vit, self.adjE, 0)

            # impose the "real" incremental strain step
            self.REV.Step(vit, vit, 0, t = 1)
            self.Nstep += 1

            # self.REV.Step(vit, vit, 0) # 0.6 50degree 0.584  'vit, 0.591*vit' para2
            # self.Nstep += 1
            if self.Nstep % 1 == 0:
                # self._output(test="iso")
                self._save("iso")
#                 self._printSpecialMeso()
#                 print("iso:", "eps11=", self.REV.eps11, "eps22=", self.REV.eps22, "S11=", self.REV.sig11,
#                       "S22=", self.REV.sig22, "alpha=",self.REV.mesoCell[90].alpha*180/pi,
#                       "theta = ", self.REV.mesoCell[90].theta*180/pi)

    def _copyREV(self):
        # copy the REV to find appropriate lateral strain in biax loading
        copyREV = copy.deepcopy(self.REV)
        return copyREV


    def biaTest(self, vit=5.0e3/kn, sigConfining=confining, Eps1=0.01):
        if self.skip != 0:
            vit = vit*self.skip
        while self.REV.eps11 < Eps1:
            # loop to find the lateral strain increment to impose
            cn = 0  # internal number of steps to find the lateral adjustment strain
            self.adjE = 0
            copyREV = self._copyREV()
            copyREV.Step(vit, 0, 0)  # first guess : oedometer compression
            while abs(copyREV.sig22 - sigConfining) / sigConfining > 10e-3:
                cn += 1
                if copyREV.sig22 > sigConfining:
                    self.adjE -= (abs(copyREV.sig22-sigConfining)*10e-2)/kn# reduce lateral strain
                else:
                    self.adjE += (abs(copyREV.sig22-sigConfining)*10e-2)/kn# increase lateral strain
                copyREV = self._copyREV()
                copyREV.Step(vit, self.adjE, 0)

            # impose the "real" incremental strain step
            self.REV.Step(vit, self.adjE, 0, t = 1)
            self.Nstep += 1

            if self.Nstep % 1 == 0:
                # self._output(test="bia")
                self._save("bia")
#                 self._printSpecialMeso()
#                 print("bia:", "eps11=", self.REV.eps11, "eps22=", self.REV.eps22, "S11=", self.REV.sig11,
#                       "S22=", self.REV.sig22, "alpha=", self.REV.mesoCell[90].alpha * 180 / pi,
#                       "vertical_theta = ", self.REV.mesoCell[90].theta * 180 / pi)
#                 print("bia:", 'deps11:', vit, 'deps22:', self.adjE)


    def proTest(self, vit=1.0e-7, ratio = -1, Eps1=0.01): # deps22 = deps11 * ratio
        if self.skip != 0:
            vit = vit*self.skip
        while self.REV.eps11 < Eps1:

            self.REV.Step(vit, vit*ratio, 0, t = 1)
            self.Nstep += 1
            if self.Nstep % 1 == 0:
                # self._output(test="pro")
                self._save("pro")
#                 self._printSpecialMeso()
#                 print("pro:", "eps11=", self.REV.eps11, "eps22=", self.REV.eps22, "S11=", self.REV.sig11,
#                       "S22=", self.REV.sig22, "alpha=",self.REV.mesoCell[0].alpha*180/pi)


    # def _output(self, name=+locV+"0223savepath", test=""):
    #     name='Measocelltest'
    #     f = open(name, 'a')
    #     print(test, self.Nstep, self.REV.eps11, self.REV.eps22, self.REV.eps12, self.REV.sig11, self.REV.sig22,
    #           self.REV.sig12, self.REV.mesoCell[90].alpha*180/pi, self.REV.mesoCell[90].theta*180/pi,
    #           self.REV.mesoCell[90].d1, self.REV.mesoCell[90].d2, file=f)
    #     f.close

    def _save(self, test="iso"):
        self.res["test"] += [test]
        self.res["step"] += [self.Nstep]
        self.res["eps11"] += [self.REV.eps11]
        self.res["eps22"] += [self.REV.eps22]
        self.res["eps12"] += [self.REV.eps12]
        self.res["sig11"] += [self.REV.sig11]
        self.res["sig22"] += [self.REV.sig22]
        self.res["sig12"] += [self.REV.sig12]

    def _printSpecialMeso(self):
        #print the id list of special cells
        print("Open12", self.REV.specialMeso["Open12"])
        print("Open23", self.REV.specialMeso["Open23"])
        print("Contact14", self.REV.specialMeso["Contact14"])
        print("Contact26", self.REV.specialMeso["Contact26"])

    def draw(self, test="all"):
        if test == "sig":
            plt.figure(1)
            plt.plot(self.res["eps11"], self.res["sig11"], "r", label="$sig11$")
            plt.plot(self.res["eps11"], self.res["sig22"], "b", label="$sig22$")
            plt.xlabel('eps11')
            plt.ylabel('sig22')
            plt.legend()
            ax2 = plt.twinx()
            ax2.plot(self.res["eps11"], self.res["eps22"], "k", label="$\u03B5$22")
            ax2.set_ylabel('$\u03B5$22', fontsize=12)
            plt.gca().invert_yaxis()
            plt.legend()
            plt.show()
        if test == "pq":
            dic = self.res
            plt.figure(3)
            plt.plot([(dic["sig11"][i] + dic["sig22"][i])/2 for i in range(len(dic["eps11"]))], [dic["sig11"][i] - dic["sig22"][i] for i in range(len(dic["eps11"]))], "r", label="$q$")
            plt.xlabel('$p$ (Pa)' )
            plt.ylabel('$q$ (Pa)')
            plt.legend()
            plt.show()

