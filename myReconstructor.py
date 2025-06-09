

import numpy as np
from skimage.transform import radon, iradon, resize
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
from skimage.data import shepp_logan_phantom

class myReconstructor():
    def __init__(self,refImm,sinA,noiseL,randV,nTheta=230):
        self.noiseL = noiseL
        self.sinA = sinA
        self.randV = randV
        self.nTheta = nTheta
        self.thetaV = np.linspace(0,180,nTheta,endpoint=False)
        fpRef = sinA*self.fwdP__(refImm)        
        sinM = fpRef+randV
        rng = np.random.default_rng(seed=42)
        sinM = rng.poisson(noiseL*sinM)
        self.sinM = sinM
        self.currRec = np.zeros_like(refImm)
    def fwdP__ (self,inpI_):
        inpI_ = ndi.gaussian_filter(inpI_,.7)
     #   inpI_ *= mask__
        fp_ = radon(inpI_,theta=self.thetaV)
        return fp_
    def bkwP__ (self,bpS_):
        bpI_ = iradon(bpS_,theta=self.thetaV,filter_name=None)
        bpI_ = ndi.gaussian_filter(bpI_,.7)
        return bpI_
        
    def MLEM_init (self,nIt=50):
        iSens = self.bkwP__(self.sinA)
        recMask = iSens>1e-5
        recMask = ndi.binary_erosion(recMask)
        iRec = np.ones_like(iSens)*recMask
        for itIdx in range(nIt):
            fwdP = self.noiseL*(self.sinA*self.fwdP__(iRec)+self.randV)
            sinRat = self.sinA*(self.sinM/fwdP)
            iBP = self.bkwP__(sinRat)
            iRec *= (iBP/(iSens+1e-5))
        self.currRec = iRec.copy()
    def rdp_grad (self,inpImm_,eps_,beta_):
        rdpG_ = np.zeros_like(self.currRec)
        kappa_ = self.kappa
        for xs in range(-1,2):
            for ys in range (-1,2):
                    if (xs == 0) and (ys==0): 
              #          print('continuing')
                        continue
                    shiftImm_ = np.roll(inpImm_,(xs,ys),axis=(0,1))
                    sk_ = np.roll(kappa_,(xs,ys),axis=(0,1))
    
                    tempW = kappa_*sk_ / np.sqrt(xs**2+ys**2)   
                    rdpG_ += tempW*(inpImm_ - shiftImm_)*(2*eps_**2 + 6* shiftImm_**2 -7*shiftImm_*inpImm_ + 5*inpImm_**2) \
                    /(5*shiftImm_**2 -8*shiftImm_*inpImm_+5*inpImm_**2 +eps_**2 )**(3/2)
    
        rdpG_ *= beta_
        return rdpG_

    def rdp_val (self,inpImm_,eps_,beta_):
        rdpSum_ = np.zeros_like(self.currRec)
        kappa_ = self.kappa
        for xs in range(-1,2):
            for ys in range (-1,2):
                    if (xs == 0) and (ys==0): 
                        continue
                    shiftImm_ = np.roll(inpImm_,(xs,ys),axis=(0,1))
                    sk_ = np.roll(kappa_,(xs,ys),axis=(0,1))
                    tempW = kappa_*sk_ / np.sqrt(xs**2+ys**2)   
                    rdpSum_ += tempW*(inpImm_ - shiftImm_)**2/np.sqrt((inpImm_**2)+(shiftImm_**2)+4*(inpImm_-shiftImm_)**2+eps_**2)
        rdpV = beta_*np.sum(np.sum(rdpSum_,axis=1),axis=0)
        return rdpV

    def rdp_hess_diag (self,eps_,beta_):
        rdpI_ = np.zeros_like(self.currRec)
        inpImm_ = self.currRec
        kappa_ = self.kappa
        for xs in range(-1,2):
            for ys in range (-1,2):
                    if (xs == 0) and (ys==0): 
                        continue
                    shiftImm_ = np.roll(inpImm_,(xs,ys),axis=(0,1))
                    sk_ = np.roll(kappa_,(xs,ys),axis=(0,1))
                    tempW = kappa_*sk_ / np.sqrt(xs**2+ys**2)   
                    tI = (
                        -7*(shiftImm_*inpImm_)**2
                        -5*eps_**2 * inpImm_**2
                        + 22*inpImm_*shiftImm_**3
                        +14*eps_**2 * inpImm_*shiftImm_
                        -7*shiftImm_**4
                        -eps_**2*shiftImm_**2
                        +2*eps_**4
                    )
                        
                    rdpI_ += tempW*tI/(5*inpImm_**2-8*inpImm_*shiftImm_+5*shiftImm_**2+eps_**2)**(5/2)
               
        rdpI_ *= beta_
        return rdpI_
    
    def rdp_den_2 (self,inpImm_,sDir_,eps_,beta_,alpha_=0):
        ssDen = 0
        kappa_ = self.kappa
        inpImm_ +=alpha_*sDir_
        for xs in range(-1,2):
            for ys in range (-1,2):
                    if (xs==0) and (ys==0):
                        continue
                    eDist = 1/ np.sqrt(xs**2+ys**2)
                    shiftImm_ = np.roll(inpImm_,(xs,ys),axis=(0,1))                         
                    sk_ = np.roll(kappa_,(xs,ys),axis=(0,1))
                    shiftSI_ = np.roll(sDir_,(xs,ys),axis=(0,1))                
                    tW = (kappa_*sk_)*eDist
                    wI = tW/(5*inpImm_**2+5*shiftImm_**2-8*inpImm_*shiftImm_+eps_**2)**(5/2)
                    diagT = sDir_**2 * ( 2*eps_**4-eps_**2*(5*inpImm_**2-14*inpImm_*shiftImm_+shiftImm_**2)+shiftImm_**2*(22*inpImm_*shiftImm_-7*inpImm_**2-7*shiftImm_**2))
                    offDiagT = sDir_*shiftSI_ * ( -2*eps_**4+2*eps_**2*(inpImm_**2-6*inpImm_*shiftImm_+shiftImm_**2)-shiftImm_*inpImm_*(22*inpImm_*shiftImm_-7*inpImm_**2-7*shiftImm_**2))
                    wI *= (diagT+offDiagT)
                    ssDen += np.sum(np.sum(wI,axis=-1),axis=-1)
        ssDen *= (beta_)
        return ssDen 

    def makeKappa (self): #,inSmSigma=0):
        self.kappa =self.bkwP__(self.sinA)        
        
    def makePrec (self,inSmSigma=0,addRDP=False,Expectation=True,fpOne=False):
        if (inSmSigma>.1):
            iRec = ndi.gaussian_filter(self.currRec,inSmSigma)
        else:
            iRec = self.currRec
        fwdP = self.noiseL*(self.sinA*self.fwdP__ (iRec)+self.randV)
        if fpOne:
            fp1 = self.fwdP__(np.ones_like(iRec))*self.sinA*(self.noiseL**2)
        else:
            fp1 = self.sinA*(self.noiseL**2)
        
        if Expectation:
            self.myPrecTomo = self.bkwP__(self.sinA*fp1/(fwdP))
        else:
            self.myPrecTomo = self.bkwP__(fp1*self.sinA*self.sinM/(fwdP**2))
        

    def PGDloop (self,nIt=100,inSmSigma=0,Conj = True,PL=False,itSS=False,beta=1e-3,eps=.01,Neg=False,inImm=None,recVect=False, truncSDIR = False,negCap = .5):
        if inImm is not None:
            self.currRec = inImm
        if inSmSigma>.1:
            iRec = ndi.gaussian_filter(self.currRec,inSmSigma)
        else:
            iRec = self.currRec
  #     self.makePrecKappa()

        self.currRec = iRec
        rdpH = self.rdp_hess_diag(eps,beta)
        curP = self.myPrecTomo+rdpH
        self.currPrec = curP
        mask = self.myPrecTomo>0
        maskI = mask.copy()
        mask = ndi.binary_erosion(mask)
        mask = ndi.binary_erosion(mask)
        mask = ndi.binary_erosion(mask)
        mask = ndi.binary_erosion(mask)
        mask = ndi.binary_erosion(mask)
        # mask = ndi.binary_erosion(mask)
        # mask = ndi.binary_erosion(mask)
        iRec *=mask
        fwdProj = self.noiseL*(self.sinA*self.fwdP__(iRec)+self.randV)
        if recVect:
            self.recVect = np.zeros(iRec.shape+(nIt,))
        for i in range(nIt):
            den = np.maximum(fwdProj,negCap*self.noiseL*self.randV)
            bpS = (self.sinM-fwdProj)/den 
            grad = self.noiseL*self.bkwP__(self.sinA*bpS)
            grad *=mask
            grad -= self.rdp_grad(iRec,eps,beta)
            sDir = grad/(curP+1e-10)
            #sDir *=maskI
            # sDir = grad/(np.sqrt(self.myPrecTomo+1e-10))
            # sDir *=(mask/np.sqrt(self.myPrecTomo+1e-10))
            if (i>0):
                
                betaCG = np.dot(grad.flat,(sDir-sDirP).flat)/np.dot(sDirP.flat,gradP.flat)   
                beta0 = np.dot(grad.flat,sDir.flat)/np.dot(sDirP.flat,gradP.flat)
            
                betaCG = max(betaCG,0)
                # if not(i%5):
                #     beta0=0
                #     betaCG=0
                if Conj:
                    if PL:
                        sDir += betaCG*sDirP
                    else:
                        sDir += (beta0*sDirP)
                    print(f'betaCG {betaCG:.2f} beta0 {beta0:.2f}')
        
            fpsd = self.noiseL*self.sinA*self.fwdP__(sDir*mask)
            sNum = np.dot(grad.flat,sDir.flat)
            #sDen = np.dot(np.pi/(self.nTheta*2)*(fpsd*self.sinM/fwdProj).flatten(),((fpsd/fwdProj).flat)) 
            sDen = np.dot((np.pi/(self.nTheta*2)*fpsd/den).flatten(),fpsd.flat)

            denP = self.rdp_den_2(iRec,sDir,eps,beta)
       #     print(f'num: {sNum:.2e} denT: {sDen:.2e} denP: {denP:.2e}',end='\t')
            
            ss = sNum/(sDen+denP)
        #    print(f'initialSS = {ss:.2e}\n') 
            sDirP = sDir.copy()
            gradP = grad.copy()
            denIt = (sDen+denP).copy()
            newNum=sNum.copy()
            minLL = np.sum(-fwdProj+self.sinM*np.log(fwdProj))
            iLL = minLL
            def ssFEval(iss_):
                nI = iRec.copy()+iss_*sDir.copy()
             #   nI[nI<0]=0
                
                cfp_ = self.noiseL*(self.fwdP__(nI)*self.sinA + self.randV)  
                flt_ = np.sum(-cfp_+self.sinM*np.log(cfp_))                
                frdp_ = -self.rdp_val(nI,eps,beta)
              #  cgV = np.dot((self.sinM/cfp_-1).flat,fpsd.flat)
              #  print(f'cgProd: {cgV:.2e}, nNeg:{nI[nI<0].size:d}')
                return(flt_+frdp_)
            if itSS:
                xv = []
                yv = []
                a = 0
                xv.append(0)
                yv.append(iLL-self.rdp_val(iRec,eps,beta))
                b = ss*10
                resphi = (np.sqrt(5)-1) / 2        
                c = b - resphi * (b - a)
                d = a + resphi * (b - a)
                fc = ssFEval(c)
                fd = ssFEval(d) 
                xv.append(c),yv.append(fc),xv.append(d),yv.append(fd)
                print(f'Init, a={a:.1e}, c={c:.1e}, d={d:.1e}, b={b:.1e},fc ={(fc-iLL):.3e} fd={(fd-iLL):.3e}')                
                for ssIt in range(100):
                    if fc < fd:
                        a = c
                        c = d
                        fc = fd
                        d = a + resphi * (b - a)
                        fd = ssFEval(d)   
                        xv.append(d), yv.append(fd)
                        print(f'fc<fd, a={a:.1e}, c={c:.1e}, d={d:.1e}, b={b:.1e},fc ={(fc-iLL):.3e} fd={(fd-iLL):.3e}')

                    else:
                        b = d
                        d = c
                        fd = fc
                        c = b - resphi * (b - a)
                        fc = ssFEval(c)
                        xv.append(c), yv.append(fc)
                        print(f'fc>fd, a={a:.1e}, c={c:.1e}, d={d:.1e}, b={b:.1e},fc ={(fc-iLL):.3e} fd={(fd-iLL):.3e}')
                    ss = (b+a)/2
                    if ((b/a)<1.3):
                        
                        plt.figure()
                        plt.plot(xv,yv,'o ')
                        break

   #         print(f'\nfinal ss: {ss:.2e}')
            fwdProj += (ss*fpsd)

            iRecP = iRec.copy()
            iRec+= ss*sDir
            if not Neg:
                
                iRec[iRec<0]=0
                if truncSDIR:
                    sDirP = (iRec-iRecP)/ss
                fwdProj = self.noiseL*(self.fwdP__(iRec)*self.sinA + self.randV)
                
            self.currRec = iRec    
            if recVect:
                self.recVect[:,:,i] = iRec