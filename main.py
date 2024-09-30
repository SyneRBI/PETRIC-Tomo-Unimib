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
    def __init__(self, data, **kwargs):

        self.x = data.OSEM_image
        # tImmArr = self.x.as_array()
        # tImmArrSm = ndi.gaussian_filter(tImmArr,1.2)
        # tempEps = tImmArrSm.max()*5e-3
        # mask1 = tImmArrSm>(0.5*tempEps)
        # mask1 = ndi.binary_dilation(mask1,iterations=3)
        # self.mask1 = ndi.gaussian_filter(mask1,2)
        # tImmArr = ndi.gaussian_filter(tImmArr,.3)
        # tImmArr[tImmArr<tempEps]=0
        # tImmArr = ndi.gaussian_filter(tImmArr,.3)
        
        # self.x.fill(tImmArr)
        self.immArr = 0
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
        
        self.ll = STIR.make_Poisson_loglikelihood(data.acquired_data,acq_model=acq_model)
        self.ll.set_prior(self.data.prior)
        self.ll.set_up(self.x)
        
        nAngles = data.acquired_data.dimensions()[2]
        nSSAng = 4
        self.subFactor = nAngles/nSSAng
        usedAngles = [(x*nAngles)//nAngles for x in range(nSSAng)] #, nAngles//3, (nAngles)//2, (2*nAngles)//3, (5*nAngles)//6 ]
        acqModSS = STIR.AcquisitionModelUsingParallelproj()
        acqModSS.set_acquisition_sensitivity(STIR.AcquisitionSensitivityModel(data.mult_factors.get_subset(usedAngles)))
        acqModSS.set_additive_term(data.additive_term.get_subset(usedAngles))
        acqModSS.set_up(data.acquired_data.get_subset(usedAngles),self.x)
        self.acqModSS = acqModSS
  #      self.llTomo = STIR.make_Poisson_loglikelihood(data.acquired_data.get_subset(usedAngles),acq_model = acqModSS)
   #     self.redData = data.acquired_data.get_subset(usedAngles).as_array()
   #     self.redAdd = data.additive_term.get_subset(usedAngles).as_array()
   #     self.llTomo.set_up(self.x)
        
  #      self.subTomoFull = self.llTomo.clone()
      
        ybar = acq_model.forward(self.x)
        fp1 = acq_model.forward(self.x.get_uniform_copy(1))
        self.prec = self.x.get_uniform_copy(0)
        
        self.precTomo = acq_model.backward(fp1/ybar).as_array()
        
        self.kappaArr = self.data.prior.get_kappa().as_array()
       
        precArr = self.precTomo
#        self.kappaArr = np.sqrt(precArr) #*np.sqrt(precArr.shape[1])
        #precArr = self.kappaArr.copy()
        mask = (precArr>1)
        precArr += self.rdp_hess_diag()
        
        structuring_element = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]]).astype(bool)
        structuring_element = structuring_element.reshape((1,3,3))  
        precDil = precArr.copy()                        
        for _ in range(22):
            precDil = ndi.grey_dilation(precDil,structure=structuring_element)
            precDil[mask] = precArr[mask]
        

        precDil += 1e-5
        self.prec.fill(precDil)
       # self.prec.write('prec.hv')
       #
        self.prec.fill((mask/precDil))
 #       self.prec.write('prec.hv')
       # self.prec.

        # mask = 1 - (precArr<1)
        # self.mask = mask
        # self.maskStir = self.x.get_uniform_copy(0).fill(mask)

        # self.x.fill(self.x.as_array()*mask)
        
        self.immArr = self.x.as_array()
        self.sDirSTIR = self.x.get_uniform_copy(0)
        self.prevGrad =self.x.get_uniform_copy(0)
        self.prevSDir = self.x.get_uniform_copy(0)
        super().__init__()
        self.configured = True       
    
    def makeFFT_2D_filter (self):
        d_ = .95
        imShape_ = self.x.shape
        tRes_ = 0
        # find TOF res
        dataInfo = self.data.acquired_data.get_info().splitlines()
        tofLine = [line for line in dataInfo if 'TOF timing' in line]
        if len(tofLine)>0:
            regExpMatch = re.search(r':=\s*(\d+)', tofLine[0])
            tRes_ = float(regExpMatch.group(1))
        pixS_ = self.x.voxel_sizes()[1]
        
        order = 2*np.power(2,np.ceil(np.log2(imShape_[1]))).astype(np.uint32)
       # freqN = np.power(2,np.ceil(np.log2(imShape_[1]//2))).astype(np.uint32)
      #  print (order)
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
        
        # Apply the shepp-logan window
        fV = 2*np.pi*(np.arange(1,freqN+1))/imShape_[1]
        ftFilt[1:] *= (np.sin(fV/(2*d_)) / (fV/(2*d_)))
        ftFilt[ftFilt<0]=0

        # interpolate to 2D
        xf = np.arange(0,imShape_[1]//2+1).reshape((1,imShape_[1]//2+1))
        yf = xf.transpose()
        freqR = np.sqrt(xf**2+yf**2)
        interpF = np.interp(freqR,nFreq,ftFilt,right=0)
        if (imShape_[1]%2):
            interpF = np.concatenate([interpF,interpF[-1:0:-1,:]],axis=0)
            interpF = np.concatenate([interpF,interpF[:,-1:0:-1]],axis=1)
            interpF = interpF.reshape((1,)+imShape_[1:])            
        else:
            interpF = np.concatenate([interpF,interpF[-2:0:-1,:]],axis=0)
            interpF = np.concatenate([interpF,interpF[:,-2:0:-1]],axis=1)
            interpF = interpF.reshape((1,)+imShape_[1:])
        self.FFTFilter = interpF
        self.invFilt = 1/interpF
        self.invFilt /= self.invFilt[0,0,0]
 
    
    def rdp_hess_diag (self):
        inpImm_ = ndi.gaussian_filter(self.x.as_array(),1.3)
        kappa_ = ndi.gaussian_filter(self.kappaArr,1.1)
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
                    eDist = pixS_[1]/ np.sqrt((zs*pixS_[0])**2+(xs*pixS_[1])**2+(ys*pixS_[2])**2)
                    shiftImm_ = np.roll(inpImm_,(zs,xs,ys),axis=(0,1,2))
                    sk_ = np.roll(kappa_,(zs,xs,ys),axis=(0,1,2))
                    if zs==-1:
                        shiftImm_[-1,:,:]= inpImm_[-1,:,:]
                    if zs==1:
                        shiftImm_[0,:,:] = inpImm_[0,:,:]     
                    rdpG_ += (eDist*2)*(kappa_*sk_)*(eps_ +2 * shiftImm_)**2 /(inpImm_+ shiftImm_ + 2*np.abs(inpImm_-shiftImm_ )+eps_)** 3 
                    
        rdpG_ *= beta_
        rdpG_ = ndi.gaussian_filter(rdpG_,0.6)
        return rdpG_

    
    def rdp_den_exact (self,sDir_,alpha_=0):
              
        ssDen = 0
        inpImm_ = self.immArr+alpha_*sDir_
        inpImm_[inpImm_<0]=0
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

                    if zs==-1:
                        shiftImm_[-1,:,:]= inpImm_[-1,:,:]
                        shiftSI_[-1,:,:] = sDir_[-1,:,:]
                    if zs==1:
                        shiftImm_[0,:,:] = inpImm_[0,:,:]
                        shiftSI_[0,:,:] = sDir_[0,:,:]
                    wI = 1/(inpImm_+ shiftImm_ + 2 * np.abs(inpImm_-shiftImm_) + eps_)**3
                    wI *= (kappa_*sk_ )
                    wI *= ((2*shiftImm_+eps_)**2 *  sDir_**2 -(2*inpImm_+eps_)*(2*shiftImm_+eps_)*sDir_*shiftSI_)
                    
                    ssDen += np.sum(np.sum(np.sum(wI,axis=-1),axis=-1),axis=-1)
        ssDen *= (2*beta_)
        return ssDen       

    
    def update(self):

## Mettere la log likelihood al posto che gradienti manuali
## Calcolare denominatore tomografico approssimato con filtro?  

        grad = self.ll.gradient(self.x)
    #    grad.write('grad.hv')
        self.sDirSTIR = (grad*self.prec)
   #     self.sDirSTIR.write('sDir.hv')
        if (self.iteration>0):
            if (self.iteration>3):
            #    print('doing beta')
                beta = (self.sDirSTIR.dot(grad)-self.sDirSTIR.dot(self.prevGrad))/(self.prevSDir.dot(self.prevGrad))
                beta = max(0,beta)
            else:
            #    print ('doing beta quad')
                beta = self.sDirSTIR.dot(grad)/(self.prevSDir.dot(self.prevGrad))
            
            
            self.sDirSTIR.sapyb(1,self.prevSDir,beta)
        self.prevSDir = self.sDirSTIR.clone()
        self.prevGrad = grad.clone()
    
       
        ybar2 = self.acqModSS.forward(self.x)
        fpSD2 = self.acqModSS.get_linear_acquisition_model().forward(self.sDirSTIR)
        tomoDenC = self.subFactor* fpSD2.dot(fpSD2/ybar2)
        
        numNew = self.sDirSTIR.dot(grad)
        newDenRDP = self.rdp_den_exact(self.sDirSTIR.as_array())

    #    print('numNew{:.2e} tomoDen{:.2e} newRDP {:.2e}'.format(numNew,tomoDenC,newDenRDP))
        inSS = numNew/(tomoDenC+newDenRDP)
 
        self.x.sapyb(1,self.sDirSTIR,inSS,out=self.x) 
        self.x.maximum(0, out=self.x)
        self.immArr = self.x.as_array()

        
    def update_objective(self):
        return 0
        
         #   ssTomo = ssNum/ssDen
submission_callbacks = [MaxIteration(660)]
