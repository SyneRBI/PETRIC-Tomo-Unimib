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
        tImmArr = self.x.as_array()
        tImmArr = ndi.gaussian_filter(tImmArr,(0.2,0.2,0.2))
        tempEps = tImmArr.max()*1e-2
        tImmArr[tImmArr<tempEps]=0
        # self.x.fill(tImmArr)
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
        self.subFactor = nAngles/3
        usedAngles = [0, nAngles//3, (2*nAngles)//3] #, nAngles//3, (nAngles)//2, (2*nAngles)//3, (5*nAngles)//6 ]
        acqModSS = STIR.AcquisitionModelUsingParallelproj()
        acqModSS.set_acquisition_sensitivity(STIR.AcquisitionSensitivityModel(data.mult_factors.get_subset(usedAngles)))
        acqModSS.set_additive_term(data.additive_term.get_subset(usedAngles))
        acqModSS.set_up(data.acquired_data.get_subset(usedAngles),self.x)
        self.acqModSS = acqModSS
        self.llTomo = STIR.make_Poisson_loglikelihood(data.acquired_data.get_subset(usedAngles),acq_model = acqModSS)
        self.llTomo.set_up(self.x)
        
        self.makeFFT_2D_filter()
        ybar = acq_model.forward(self.x)
        fp1 = acq_model.forward(self.x.get_uniform_copy(0))
        self.precTomo = acq_model.backward(fp1/ybar).as_array()
        
        self.kappaArr = self.data.prior.get_kappa().as_array()
       
        precArr = self.precTomo
#        self.kappaArr = np.sqrt(precArr) #*np.sqrt(precArr.shape[1])
        #precArr = self.kappaArr.copy()
        precArr += self.rdp_hess_diag()
        mask = (precArr>1)
        structuring_element = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]]).astype(bool)
        structuring_element = structuring_element.reshape((1,3,3))  
        precDil = precArr.copy()                        
        for _ in range(22):
            precDil = ndi.grey_dilation(precDil,structure=structuring_element)
            precDil[mask] = precArr[mask]
        

        precDil += 1e-5
        self.prec = np.sqrt(1/precDil)

        mask = 1 - ndi.binary_dilation(precArr<1,iterations=1)
        self.mask = mask

        self.x.fill(self.x.as_array()*mask)
        
        self.immArr = self.x.as_array()
        self.sDirSTIR = self.x.get_uniform_copy(0)
        self.sDirArr = self.immArr.copy()
        self.prevGrad = np.zeros_like(self.immArr)
        self.prevSDir = self.immArr.copy()
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
                    eDist = pixS_[1]/ np.sqrt((zs*pixS_[0])**2+(xs*pixS_[1])**2+(ys*pixS_[2])**2)
                    shiftImm_ = np.roll(inpImm_,(zs,xs,ys),axis=(0,1,2))
                    sk_ = np.roll(kappa_,(zs,xs,ys),axis=(0,1,2))
                    if zs==-1:
                        shiftImm_[-1,:,:]= inpImm_[-1,:,:]
                    if zs==1:
                        shiftImm_[0,:,:] = inpImm_[0,:,:]     
                    rdpG_ += (eDist*2)*(kappa_*sk_)*(eps_ +2 * shiftImm_)**2 /(inpImm_+ shiftImm_ + 2*np.abs(inpImm_-shiftImm_ )+eps_)** 3 
                    
        rdpG_ *= beta_
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
        eps_ = self.data.prior.get_epsilon()

## Mettere la log likelihood al posto che gradienti manuali
## Calcolare denominatore tomografico approssimato con filtro?  

        grad = self.ll.gradient(self.x)
        gradArr = grad.as_array()
        # Search direction is gradient divived by preconditioner

        self.sDirArr= gradArr*self.prec
  #      sDir = np.fft.fft2(self.sDirArr,axes=(1,2))
 #       sDir *= self.FFTFilter
 #       self.sDirArr= np.real(np.fft.ifft2(sDir,axes=(1,2)))
        self.sDirArr *= (self.prec*self.mask)
        self.sDirSTIR.fill(self.sDirArr)
        print('applied prec')
        if (self.prevGrad.max()>0):
            beta = np.dot((gradArr-self.prevGrad).flat,self.sDirArr.flat)/np.dot(self.prevGrad.flat,self.prevSDir.flat)
            beta2 =  np.dot((gradArr).flat,self.sDirArr.flat)/np.dot(self.prevGrad.flat,self.prevSDir.flat)
            beta = max(0,beta)
            if (beta==0):
                print ('resetting')
            #print(beta2/beta)
            if ((beta/beta2)>1.3):
                beta=0
                print('reset')
            print ('beta P-R = {:.2f} beta quadratic = {:.2f}'.format(beta,beta2))
            self.sDirArr += beta*self.prevSDir
        self.prevSDir = self.sDirArr.copy()
        self.prevGrad = gradArr.copy()
        print('computed conjugate')

        # Compute step size
        # tD = (self.sDirArr * np.sqrt(self.precTomo))
        # tD = np.fft.fft2(tD,axes=(1,2))
        # tD *= self.invFilt
        # tD = np.real(np.fft.ifft2(tD,axes=(1,2)))
        # tD *= np.sqrt(self.precTomo)
        # tomoDen = np.dot(tD.flat,self.sDirArr.flat)
         ## compare with "better" hessian
        # ybar = self.full_model.forward(self.x)
        # fpSD = self.lin_model.forward(self.sDirSTIR)
        # tomoDenTrue = fpSD.dot(fpSD/ybar)
        
        ybar2 = self.acqModSS.forward(self.x)
        fpSD2 = self.acqModSS.get_linear_acquisition_model().forward(self.sDirSTIR)
        tomoDen = self.subFactor* fpSD2.dot(fpSD2/ybar2)
        
    #    tomoDenBis = self.sDirSTIR.dot(self.ll.multiply_with_Hessian(self.x,self.sDirSTIR))
     #   print(tomoDenBis)
        
        numNew = np.dot(self.sDirArr.flat,gradArr.flat)
        newDenRDP = self.rdp_den_exact(self.sDirArr)
      #  ssString = 'New step size: num = {:.1e} rdpDen = {:.1e} tomoDen = {:.1e} trueTomo = {:.1e} tomoBis = {:.1e} stepSize = {:.1e}'
        
        stepSize = (numNew)/(tomoDen+newDenRDP)
        ssString = 'New step size: num = {:.1e} rdpDen = {:.1e} tomoDen = {:.1e} stepSize = {:.1e}'
        print(ssString.format(numNew,newDenRDP,tomoDen,stepSize))
        #print(ssString.format(numNew,newDenRDP,tomoDen,tomoDenTrue,tomoDenBis,stepSize))
        if (stepSize<0):
            print('neg step size')
            stepSize = abs(stepSize*.1) 

       # if ((newDenRDP/tomoDen)>20):
            ## compute step size only varying numRDP
        rdpNum = -self.sDirSTIR.dot(self.data.prior.gradient(self.x))
        tomoNum = numNew-rdpNum
        print('rdpNum = {:.1e} TomoNum = {:.1e}'.format(rdpNum,tomoNum))
        inSS = stepSize #*2
        xc = inSS
        xa = 0
        # search teh maximum
        for dummy in range(4):
            rdpNum = -self.sDirSTIR.dot(self.data.prior.gradient(self.x.sapyb(1,self.sDirSTIR,inSS).maximum(0)))
            tomoNum = self.subFactor*(self.llTomo.gradient(self.x.sapyb(1,self.sDirSTIR,inSS).maximum(0)).dot(self.sDirSTIR))
         #   print('rdp num={:.1e}'.format(rdpNum))
            if (rdpNum>(-tomoNum)):
                xc = (inSS*2)
                xa = (inSS*0.5)
                inSS*=2
                print('doubled SS')
            else:
           #     xc = inSS
                break
        inSS = (xa+xc)/2
#           print('xc = {:.1e} xa = {:.1e}'.format(xc,xa))
        for bisIt in range(10):
            rdpNum = -self.sDirSTIR.dot(self.data.prior.gradient(self.x.sapyb(1,self.sDirSTIR,inSS).maximum(0)))
            tomoNum = self.subFactor*(self.llTomo.gradient(self.x.sapyb(1,self.sDirSTIR,inSS).maximum(0)).dot(self.sDirSTIR))
            if (rdpNum>(-tomoNum)):
                xa = inSS
                inSS = (xc+inSS)/2
            else:
                xc = inSS
                inSS = (xa+inSS)/2
            ssString = '\t Loop ss: tomoNum = {:.1e},  rdpNum = {:.1e}, nextStepSize = {:.1e}'
            print(ssString.format(tomoNum,rdpNum,inSS)) #, end='\t')     
#            print('xc = {:.1e} xa = {:.1e}'.format(xc,xa))
            if ((xc/xa-1)<.05):
                break
            
        # else:
            # inSS = stepSize
            # oldSS = 0
            # oldNum = numNew
            # newNum = oldNum
            # for newtIt in range(10):

                # newGrad = self.llTomo.gradient(self.x.sapyb(1,self.sDirSTIR,inSS).maximum(0))
                # newNum = self.sDirSTIR.dot(newGrad)*self.subFactor #+ self.data.prior.grad
                # newNum -= self.sDirSTIR.dot(self.data.prior.gradient(self.x.sapyb(1,self.sDirSTIR,inSS).maximum(0)))
                # delta = - newNum*(inSS-oldSS)/(newNum-oldNum)
                # oldSS = inSS
                # inSS +=delta
                # if (inSS<0):
                    # inSS = (stepSize/2)
                # if (inSS/stepSize)>100:
                    # inSS = stepSize
                    # self.prevSDir[:] = 0
                    # print('too high ratio')
                    # break
                # oldNum = newNum                
                # ssString = '\t Loop ss: newNum = {:.1e} delta = {:.1e}, nextStepSize = {:.1e}'
                # print(ssString.format(newNum,delta,inSS)) #, end='\t')    
         #       print(abs(delta/inSS))
                # if (abs(delta/inSS)<0.1):
                    # break
                    
                
#            print ('passing no bueno!')
#            pass

            

        px = self.x.copy()
        self.x.sapyb(1,self.sDirSTIR,inSS,out=self.x) #    += (sDir) #*self.mask)
        self.x.maximum(0, out=self.x)
      #  self.prevSDir = (self.x - px).as_array()
        self.immArr = self.x.as_array()

        
    def update_objective(self):
        return 0
        
         #   ssTomo = ssNum/ssDen
submission_callbacks = [MaxIteration(660)]