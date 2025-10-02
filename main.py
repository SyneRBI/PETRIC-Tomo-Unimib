from cil.optimisation.algorithms import Algorithm
from cil.optimisation.utilities import callbacks
import sirf.STIR as STIR
import numpy as np
import scipy.ndimage as ndi
import re
import time

class MaxIteration(callbacks.Callback):
    """
    The organisers try to `Submission(data).run(inf)` i.e. for infinite iterations (until timeout).
    This callback forces stopping after `max_iteration` instead.
    """
    def __init__(self, max_iteration: int, verbose: int = 1):
        super().__init__(verbose)
        self.max_iteration = max_iteration

    def __call__(self, algorithm: Algorithm):
        if algorithm.iteration >= self.max_iteration:
            raise StopIteration

class Submission (Algorithm):
    ''' Main implementation of the preconditioned conjugate gradient descent algorithm 
    
    
    '''
    def __init__(self, data, circulant=True,Conjugate=True, **kwargs):

        self.ConjFlag = Conjugate #flag to store whether to apply conjugate gradient
        self.CirculantFlag = circulant #flat to store whether to use _also_ the circulant part of the preconditioner
        self.x = data.OSEM_image
        self.ssL = [] # List to store computed step sizes, debugging purposes
        tImmArr = self.x.as_array()
        tImmArrSm = ndi.gaussian_filter(tImmArr,0.7)
        self.x.fill(tImmArrSm)
        self.immArr = self.x.as_array()
        epsCorr = data.additive_term.max()*1e-6
        self.epsCorrSino = epsCorr
        data.additive_term+=epsCorr
        self.data = data
        acq_model = STIR.AcquisitionModelUsingParallelproj()
        acq_model.set_acquisition_sensitivity(STIR.AcquisitionSensitivityModel(data.mult_factors))
        acq_model.set_additive_term(data.additive_term)
        acq_model.set_up(data.acquired_data, self.x)
        self.full_model = acq_model
        self.lin_model = acq_model.get_linear_acquisition_model()
        self.addCorrThr = data.additive_term*data.mult_factors
        self.addCorrThr +=epsCorr
        
        ybar = acq_model.forward(self.x)
        self.prec = self.x.get_uniform_copy(0)
        
        self.ybar = ybar
        self.precTomo = acq_model.backward(self.data.mult_factors/self.ybar.maximum(self.addCorrThr)).as_array()
        newKappa = acq_model.backward(ybar.get_uniform_copy(1))
        newKarr = newKappa.as_array()
        newKprof = np.mean(np.mean(newKarr,axis=2),axis=1)
        nz = self.x.dimensions()[0]
        zv = np.arange(-(nz-1)/2,(nz+1)/2)
        zv = np.abs(zv)
        zv[zv<0.5]=0.5
        zv = 0.5 + 0.5*( 1-zv/((nz-1)/2))
        zv = zv.reshape((nz,1,1))
        newKprof = newKprof.reshape((nz,1,1))
        zv = zv/newKprof
        newKarr = newKarr*zv
        kernel = np.ones((3,))/3
        newKarr = ndi.convolve1d(newKarr,kernel,axis=0,mode='constant')
        newKarr *=4.4
        newKarr *=100
        newKarr *=700
        newKarr = np.sqrt(newKarr)
        kappa = self.data.prior.get_kappa()
        self.kappaArr = newKarr
        kappa.fill(newKarr)
        self.data.prior.set_kappa(kappa)
        self.data.prior.set_up(self.x)
       
        precArr = self.precTomo
        mask = (precArr>1)
        rdpPrec = self.rdp_hess_diag()
        rdpPrec[rdpPrec<0]=0
        rdpPrec = ndi.gaussian_filter(rdpPrec,(0,1,1))
        precArr +=  rdpPrec
        
        structuring_element = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]]).astype(bool)
        structuring_element = structuring_element.reshape((1,3,3))  
        inMask = ndi.binary_erosion(mask,structure=structuring_element)

        precArr += 1e-5
        mask = ndi.binary_erosion(mask,structure=structuring_element,iterations=2)
       
        self.mask = mask
        np.save ('mask.npy',self.mask)
        self.prec.fill(mask/(precArr+1e-6))
        self.precArr = np.sqrt(1/precArr)
        self.sDirSTIR = self.x.get_uniform_copy(0)
        self.prevGrad =self.x.get_uniform_copy(0)
        self.prevSDir = self.x.get_uniform_copy(0)
        self.makeFFT_2D_filter()
        super().__init__()
        self.configured = True       
    
    def rdp_grad (self,alpha_=0,sDir_=0):
        inpImm_ = self.x.as_array()+alpha_*sDir_
        kappa_ = self.kappaArr
        rdpG_ = np.zeros_like(inpImm_)
        eps_ = self.data.prior.get_epsilon()
        beta_ = self.data.prior.get_penalisation_factor()
        pixS_ = self.x.voxel_sizes()        
        for xs in range(-1,2):
            for ys in range (-1,2):
                for zs in range(-1,2):
                    if (xs == 0) and (ys==0) and (zs==0): 
              #          print('continuing')
                        continue
 
                    shiftImm_ = np.roll(inpImm_,(zs,xs,ys),axis=(0,1,2))
                    sk_ = np.roll(kappa_,(zs,xs,ys),axis=(0,1,2))
                    if zs==-1:
                        shiftImm_[-1,:,:]= inpImm_[-1,:,:]
                    if zs==1:
                        shiftImm_[0,:,:] = inpImm_[0,:,:]
    
                    tempW = pixS_[1]*kappa_*sk_ / np.sqrt((zs*pixS_[0])**2+(xs*pixS_[1])**2+(ys*pixS_[2])**2)             
                    rdpG_ += tempW*(inpImm_ - shiftImm_)*(2*eps_**2 + 6* shiftImm_**2 -7*shiftImm_*inpImm_ + 5*inpImm_**2) \
                    /(5*shiftImm_**2 -8*shiftImm_*inpImm_+5*inpImm_**2 +eps_**2 )**(3/2)
    
        rdpG_ *= beta_
        return rdpG_                    
    
    def makeFFT_2D_filter (self):
        d_ = 1.01
        imShape_ = self.x.shape
        tRes_ = 900
        # find TOF res
        dataInfo = self.data.acquired_data.get_info().splitlines()
        tofLine = [line for line in dataInfo if 'TOF timing' in line]
        if len(tofLine)>0:
            regExpMatch = re.search(r':=\s*(\d+)', tofLine[0])
            tRes_ = float(regExpMatch.group(1))
        pixS_ = self.x.voxel_sizes()[1]
        
        order = 2*np.power(2,np.ceil(np.log2(imShape_[1]))).astype(np.uint32)
        self.filtOrd = order
        freqN = order//2
        nFreq = np.arange(0,freqN +1)
        filtImpResp = np.zeros((len(nFreq),))
        filtImpResp[0]=1/4
        filtImpResp[1::2]=-1/((np.pi*nFreq[1::2])**2)

        #TOF part
        if (tRes_ > 0):
            xV_ = nFreq*pixS_
            tRes_ = tRes_*0.15/2.35 # 300 mm /ns --> .3 mm/ps --> /2 because 2 photons 
            tKern_ = np.exp(-(xV_**2/(4*tRes_**2)))
            filtImpResp *=tKern_

        # Once the filter has been defined in image space, convert it to Fourier space
        filtImpResp = np.concatenate([filtImpResp,filtImpResp[-2:0:-1]])
        ftFilt = 2 * np.real(np.fft.fft(filtImpResp)) # check! when implemented correctly the imag part is zero within numerical precision
        ftFilt = ftFilt[:(freqN+1)]
        
        # with oder 2x I have many more frequencies than I need.
        
        # Apply the shepp-logan window
        fV = 2*np.pi*(np.arange(1,freqN+1))/order
#        ftFilt[1:] *= (np.sin(fV/(2*d_)) / (fV/(2*d_)))
        ftFilt[1:] *= ((.54 + .46*np.cos(fV/d_))*(fV<(np.pi*d_)))
        
        ftFilt[ftFilt<0]=0

        # interpolate to 2D
        xf = np.arange(0,order//2+1).reshape((1,order//2+1))
        yf = xf.transpose()
        freqR = np.sqrt(xf**2+yf**2)
        interpF = np.interp(freqR,nFreq,ftFilt,right=ftFilt[-1])
        interpF = np.concatenate([interpF,interpF[-2:0:-1,:]],axis=0)
        interpF = np.concatenate([interpF,interpF[:,-2:0:-1]],axis=1)
        interpF = interpF.reshape((1,order,order))     
        self.FFTFilter = interpF

 
    
    def rdp_hess_diag (self):
        kappa_ = self.kappaArr
        inpImm_ = self.x.as_array()
        rdpG_ = np.zeros_like(kappa_)
        eps_ = self.data.prior.get_epsilon()
        beta_ = self.data.prior.get_penalisation_factor()
        pixS_ = self.x.voxel_sizes()        
        for xs in range(-1,2):
            for ys in range (-1,2):
                for zs in range(-1,2):
                    if (xs == 0) and (ys==0) and (zs==0): 
              #          print('continuing')
                        continue
                    eDist = pixS_[1]/ np.sqrt((zs*pixS_[0])**2+(xs*pixS_[1])**2+(ys*pixS_[2])**2)
                    shiftImm_ = np.roll(inpImm_,(zs,xs,ys),axis=(0,1,2))
                    sk_ = np.roll(kappa_,(zs,xs,ys),axis=(0,1,2))
                    if zs==-1:
                        shiftImm_[-1,:,:]= inpImm_[-1,:,:]
                    if zs==1:
                        shiftImm_[0,:,:] = inpImm_[0,:,:]    
                        
                    tempW = kappa_*sk_ / np.sqrt(xs**2+ys**2+zs**2)   
                    tI = (
                        -7*(shiftImm_*inpImm_)**2
                        -5*eps_**2 * inpImm_**2
                        + 22*inpImm_*shiftImm_**3
                        +14*eps_**2 * inpImm_*shiftImm_
                        -7*shiftImm_**4
                        -eps_**2*shiftImm_**2
                        +2*eps_**4
                    )
                        
                    rdpG_ += tempW*tI/(5*inpImm_**2-8*inpImm_*shiftImm_+5*shiftImm_**2+eps_**2)**(5/2)
               
        rdpG_ *= beta_
        return rdpG_
        
     
    
    def rdp_den_exact (self,sDir_,alpha_=0):
              
        ssDen = 0
        inpImm_ = self.immArr+alpha_*sDir_
        kappa_ = self.kappaArr

        eps_ = self.data.prior.get_epsilon()
        beta_ = self.data.prior.get_penalisation_factor()
        pixS_ = self.x.voxel_sizes()

        for xs in range(-1,2):
            for ys in range (-1,2):
                for zs in range(-1,2):
                    if (xs == 0) and (ys==0) and (zs==0): 
                        continue
 
                    eDist = pixS_[1]/ np.sqrt((zs*pixS_[0])**2+(xs*pixS_[1])**2+(ys*pixS_[2])**2)
                    shiftImm_ = np.roll(inpImm_,(zs,xs,ys),axis=(0,1,2))                         
                    sk_ = np.roll(kappa_,(zs,xs,ys),axis=(0,1,2))
                    shiftSI_ = np.roll(sDir_,(zs,xs,ys),axis=(0,1,2))                
                    tW = (kappa_*sk_)*eDist
                    wI = tW/(5*inpImm_**2+5*shiftImm_**2-8*inpImm_*shiftImm_+eps_**2)**(5/2)
                    diagT = sDir_**2 * ( 2*eps_**4-eps_**2*(5*inpImm_**2-14*inpImm_*shiftImm_+shiftImm_**2)+shiftImm_**2*(22*inpImm_*shiftImm_-7*inpImm_**2-7*shiftImm_**2))
                    offDiagT = sDir_*shiftSI_ * ( -2*eps_**4+2*eps_**2*(inpImm_**2-6*inpImm_*shiftImm_+shiftImm_**2)-shiftImm_*inpImm_*(22*inpImm_*shiftImm_-7*inpImm_**2-7*shiftImm_**2))
                    wI *= (diagT+offDiagT)
                    wI[wI<0]=0
                    ssDen += np.sum(np.sum(np.sum(wI,axis=-1),axis=-1),axis=-1)
        ssDen *= (beta_)
        return ssDen       
  
    
    def update(self):

        capFact = 1 
        gradNum = self.data.acquired_data-self.ybar
        gradDen = self.ybar.maximum(self.addCorrThr)
        
        grad = self.full_model.backward(gradNum/gradDen)
        gradPrior = self.rdp_grad()
        gradArr = grad.as_array()-gradPrior
        grad.fill(gradArr)
        sDir = gradArr*self.precArr 
        
        if self.CirculantFlag:
            sDir = np.fft.fft2(sDir,s=(self.filtOrd,self.filtOrd),axes=(1,2))
            sDir *= self.FFTFilter
            sDir = np.real(np.fft.ifft2(sDir,s=(self.filtOrd,self.filtOrd),axes=(1,2)))
            sDir = sDir[:,:self.immArr.shape[1],:self.immArr.shape[2]]
            sDir = ndi.gaussian_filter(sDir,(0.6,0,0))
        sDir *= self.precArr
        sDir *= self.mask
        self.sDirSTIR.fill(sDir)
        
        if self.ConjFlag:
            if (self.iteration>0):
              beta = (self.sDirSTIR.dot(grad)-self.sDirSTIR.dot(self.prevGrad))/(self.prevSDir.dot(self.prevGrad))
              beta = max(0,beta)
              self.sDirSTIR.sapyb(1,self.prevSDir,beta,out=self.sDirSTIR)
        self.prevSDir = self.sDirSTIR.clone()
        self.prevGrad = grad.clone()
        sDir = self.sDirSTIR.as_array()
        
        fpSD = self.lin_model.forward(self.sDirSTIR)
        tomoDenC = fpSD.dot(fpSD/gradDen)
        numNew = self.sDirSTIR.dot(grad)
        newDenRDP = self.rdp_den_exact(self.sDirSTIR.as_array())
        
        inSS = numNew/(tomoDenC+newDenRDP)
       
 #       print('\n numNew{:.2e} tomoDen{:.2e} newRDP {:.2e} inSS {:.2e}'.format(numNew,tomoDenC,newDenRDP,inSS))
                   
        self.x.sapyb(1,self.sDirSTIR,inSS,out=self.x) 
        self.ybar.sapyb(1,fpSD,inSS,out=self.ybar)
        self.ssL.append(inSS)
        self.immArr = self.x.as_array()

        
    def update_objective(self):
        return 0


submission_callbacks = [MaxIteration(6600)]
