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
        data.additive_term+=epsCorr
        self.data = data
        acq_model = STIR.AcquisitionModelUsingParallelproj()
        acq_model.set_acquisition_sensitivity(STIR.AcquisitionSensitivityModel(data.mult_factors))
        acq_model.set_additive_term(data.additive_term)
        acq_model.set_up(data.acquired_data, self.x)
        self.full_model = acq_model
        self.lin_model = acq_model.get_linear_acquisition_model()
        self.kappaArr = self.data.prior.get_kappa().as_array()
        self.ybar = acq_model.forward(self.x)
        self.prec = self.x.get_uniform_copy(0)
        fp1 = self.lin_model.forward(self.x.get_uniform_copy(1))
        #self.prec = acq_model.backward(data.mult_factors/self.ybar)
        self.prec = acq_model.backward(fp1/self.ybar)
        
        precArr = self.prec.as_array()
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
        self.prec.fill(np.sqrt(1/precDil))
       # self.prec.fill((1/precDil))
        self.prec.write('prec.hv')
        #self.prec
     #   print('\n\n there are ' + str(np.max(np.isnan(self.prec.as_array()))) + ' NaNs in the prec')
        self.mask = self.x.get_uniform_copy(0)
        self.mask.fill(mask)
        
        self.prevGrad = self.x.get_uniform_copy(0)
        self.prevSDir = self.x.get_uniform_copy(0)
        self.makeFFT_2D_filter()
        self.immArr = self.x.as_array()
        super().__init__()
        self.configured = True       
    
    def rdp_grad (self):
        inpImm_ = self.immArr
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
                    rdpG_ += tempW*(inpImm_ - shiftImm_)*(inpImm_ + 3 * shiftImm_ + 2* eps_ + 2* np.abs(inpImm_-shiftImm_)) /(np.abs(inpImm_)+ np.abs(shiftImm_) + 2*np.abs(inpImm_-shiftImm_ )+eps_)** 2 
                    
        rdpG_ *= beta_
        return rdpG_        

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
        
        order = np.power(2,np.ceil(np.log2(imShape_[1]))).astype(np.uint32)
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
 
    
    def rdp_hess_diag (self):
        inpImm_ = self.x.as_array()
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
                    rdpG_ += 2*tempW*(eps_ +2 * shiftImm_)**2 /(inpImm_+ shiftImm_ + 2*np.abs(inpImm_-shiftImm_ )+eps_)** 3 
                    
        rdpG_ *= beta_
        return rdpG_

    
    def rdp_step_size (self,sDir_,alpha_=0):
        
        ssNum = 0
        ssDen = 0
        inpImm_ = self.x.as_array()
        kappa_ = self.kappaArr

        eps_ = self.data.prior.get_epsilon()
        beta_ = self.data.prior.get_penalisation_factor()
        pixS_ = self.x.voxel_sizes()
   #     alpha_ = 0

        for xs in range(-1,2):
            for ys in range (-1,2):
                for zs in range(-1,2):
                    if (xs == 0) and (ys==0) and (zs==0): 
                        continue

                    shiftImm_ = np.roll(inpImm_,(zs,xs,ys),axis=(0,1,2))                         
                    sk_ = np.roll(kappa_,(zs,xs,ys),axis=(0,1,2))
                    shiftSI_ = np.roll(sDir_,(zs,xs,ys),axis=(0,1,2))                

                    if zs==-1:
                        shiftImm_[-1,:,:]= inpImm_[-1,:,:]
                        shiftSI_[-1,:,:] = sDir_[-1,:,:]
                    if zs==1:
                        shiftImm_[0,:,:] = inpImm_[0,:,:]
                        shiftSI_[0,:,:] = sDir_[0,:,:]
                    wI = 1/(inpImm_+ shiftImm_ + alpha_ * (sDir_ + shiftSI_) + 2 * np.abs(inpImm_-shiftImm_+ alpha_ * (sDir_ - shiftSI_)) + eps_)**2
                    wI *= (inpImm_ + 3*shiftImm_ + alpha_ * (sDir_ + 3* shiftSI_) + 2 * np.abs(inpImm_-shiftImm_+ alpha_ * (sDir_ - shiftSI_)) + 2*eps_)
                    wI *= pixS_[1]*kappa_*sk_ / np.sqrt((zs*pixS_[0])**2+(xs*pixS_[1])**2+(ys*pixS_[2])**2)
                    ssNum -= np.matmul((inpImm_-shiftImm_).flatten().T,((sDir_-shiftSI_)*wI).flat)
                    ssDen += np.matmul((shiftSI_-sDir_).flatten().T,((shiftSI_-sDir_)*wI).flat)
        ssNum *= (beta_)
        ssDen *= (beta_)
        return ssNum,ssDen

    
    def rdp_step_size_old (self,sDir_):
        
        ssNum = 0
        ssDen = 0
        inpImm_ = self.x.as_array()
        kappa_ = self.kappaArr

        eps_ = self.data.prior.get_epsilon()
        beta_ = self.data.prior.get_penalisation_factor()
        pixS_ = self.x.voxel_sizes()
        alpha_ = 0

        for xs in range(-1,2):
            for ys in range (-1,2):
                for zs in range(-1,2):
                    if (xs == 0) and (ys==0) and (zs==0): 
                        continue

                    shiftImm_ = np.roll(inpImm_,(zs,xs,ys),axis=(0,1,2))                         
                    sk_ = np.roll(kappa_,(zs,xs,ys),axis=(0,1,2))
                    shiftSI_ = np.roll(sDir_,(zs,xs,ys),axis=(0,1,2))                

                    if zs==-1:
                        shiftImm_[-1,:,:]= inpImm_[-1,:,:]
                        shiftSI_[-1,:,:] = sDir_[-1,:,:]
                    if zs==1:
                        shiftImm_[0,:,:] = inpImm_[0,:,:]
                        shiftSI_[0,:,:] = sDir_[0,:,:]
                    wI = 1/(np.abs(inpImm_)+ np.abs(shiftImm_) + alpha_ * (sDir_ + shiftSI_) + 2 * np.abs(inpImm_-shiftImm_+ alpha_ * (sDir_ - shiftSI_)) + eps_)
                    wI *= pixS_[1]*kappa_*sk_ / np.sqrt((zs*pixS_[0])**2+(xs*pixS_[1])**2+(ys*pixS_[2])**2)
                    ssNum -= np.matmul((inpImm_-shiftImm_).flatten().T,((sDir_-shiftSI_)*wI).flat)
                    ssDen += np.matmul((shiftSI_-sDir_).flatten().T,((shiftSI_-sDir_)*wI).flat)
        ssNum *= (beta_)
        ssDen *= (beta_)
        return ssNum,ssDen
        
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
                    wI *= pixS_[1]*kappa_*sk_ / np.sqrt((zs*pixS_[0])**2+(xs*pixS_[1])**2+(ys*pixS_[2])**2)
                    wI *= ((2*shiftImm_+eps_)**2 * sDir_**2 -(2*inpImm_+eps_)*(2*shiftImm_+eps_)*sDir_*shiftSI_)
                    
                    ssDen += np.sum(np.sum(np.sum(wI,axis=-1),axis=-1),axis=-1)
        ssDen *= (2*beta_)
        return ssDen       

    
    def update(self):
        eps_ = self.data.prior.get_epsilon()

   #     gradSino = (self.data.acquired_data-self.ybar)/yDen 
        gradSino = self.data.acquired_data/self.ybar - 1
     #   ts = time.time()
        gradI = self.full_model.backward(gradSino) 

        # Compute gradient of penalty
        pGrad = self.data.prior.gradient(self.x)
        #pGrad = gradI.get_uniform_copy(0)
       # pGrad.fill(self.rdp_grad())
        grad = gradI - pGrad
        
        # Search direction is gradient divived by preconditioner

        sDir = grad*self.prec

        ftS = np.fft.fft2(sDir.as_array(),axes=(1,2))
        ftS *= self.FFTFilter
        ftS = np.real(np.fft.ifft2(ftS,axes=(1,2)))
        ftS = ndi.gaussian_filter(ftS,(0.4,0,0))
        sDir.fill(ftS)
        sDir *= self.prec
        
        sDir *= self.mask
        #sDir = sDir/(self.prec.sqrt())
        sDirArr = sDir.as_array()
        
        if (self.prevGrad.max()>0):
            beta = (grad-self.prevGrad).dot(sDir)/self.prevGrad.dot(self.prevSDir)
            beta = max(0,beta)
            sDir += beta*self.prevSDir
        self.prevSDir = sDir.clone()
        self.prevGrad = grad.clone()

        ## compute step size
      #  ts = time.time()
        fpSD = self.lin_model.forward(sDir) #,subset_num=0,num_subsets=42) #*multCorr
       # print ('FPSD took' + str(time.time()-ts))
        ssNum = sDir.dot(gradI)

    #    ssDen = fpSD.dot((fpSD/yDen)) #*42
        ssDen = fpSD.dot(fpSD/self.ybar) #*42
        # tSS = 0
        
        
  #      for _ in range(10):
        ssNP, ssDP = self.rdp_step_size_old(sDirArr)

        ssString = 'Old step size: tomoNum = {:.1e} tomoDen = {:.1e} rdpNum = {:.1e} rdpDen = {:.1e}, stepSize = {:.1e}'
        stepSize = (ssNum+ssNP)/(ssDen+ssDP)
        print(ssString.format(ssNum,ssDen,ssNP,ssDP,stepSize))

        numNew = sDir.dot(grad)
        newDenRDP = self.rdp_den_exact(sDirArr)
        ssString = 'New step size: num = {:.1e} rdpDen = {:.1e} stepSize = {:.1e}'
        stepSize = (numNew)/(ssDen+newDenRDP)
        print(ssString.format(numNew,newDenRDP,stepSize))

        # Yet another step size:
  
        inSS = stepSize
        resetFlag = False
        tomoDenOld = ssDen
        tomoDenA0 = ssDen
        rdpDenOld = newDenRDP
        newSS = 0
        for ssIt in range(2):
            ts = time.time()
            ybarNit = self.ybar.sapyb(1,fpSD,inSS)
            ybarNit = ybarNit.maximum(self.data.additive_term*.2)
            tomoNum = fpSD.dot(self.data.acquired_data/ybarNit -1 )
            if np.isnan(tomoNum):
                print('nanNum at {:.2e}'.format(inSS))
                inSS*=0.9
                continue
            tTomoNum = time.time()-ts
           # print (tTomoNum)
            
            if ssIt>0:
                if np.isinf(tomoNum):
                    print('reset num tomo too high')
                    resetFlag = True
            tomoNumOld = tomoNum
           # tomoDen = (fpSD/(self.ybar.sapyb(1,fpSD,inSS))).dot(self.data.acquired_data/(self.ybar.sapyb(1,fpSD,inSS)))
            ts = time.time()
            # tomoDen = fpSD.dot((fpSD/(self.ybar.sapyb(1,fpSD,inSS)))*(self.data.acquired_data/(self.ybar.sapyb(1,fpSD,inSS))))
            tomoDen = fpSD.dot((fpSD/ybarNit)*(self.data.acquired_data/ybarNit))
            tTomoDen = time.time()-ts
            if (tomoDen/tomoDenA0) > 3:
                print ('testing neg in ybar')
               # if (self.ybar.sapyb(1,fpSD,inSS).sapyb(1,self.data.additive_term,-.25).min()<0):
                if (ybarNit.sapyb(1,self.data.additive_term,-.25).min()<0):
                    print('neg in sino den')
                    inSS *=.75
                    continue
                else:
                    if (tomoDen/tomoDenA0)>10:
                        print ('break tomoDen 10x tomoDen Or')
                        inSS *=.5
                        break
                    else:
                        print('all good')
                # continue
            ts = time.time()
            rdpNum = -sDir.dot(self.data.prior.gradient(self.x.sapyb(1,sDir,inSS).maximum(0)))
            tRdpNum = time.time()-ts
            if ssIt>0:
                if np.isinf(rdpNum):
                    resetFlag = True
            rdpNumOld = rdpNum
            ts = time.time()
            rdpDen = self.rdp_den_exact(sDirArr,alpha_=inSS)
            tRdpDen = time.time()-ts
           # timeStr = 'tomoNum: {:.1e} s tomoDen: {:.1e} s rdpNum {:.1e} s rdpDen {:.1e} s'
           # print (timeStr.format(tTomoNum,tTomoDen,tRdpNum,tRdpDen))
            oldSS = newSS
            newSS = (tomoNum+rdpNum)/(rdpDen+tomoDen)
            ssString = '\t Loop ss: tomoNum = {:.1e} tomoDen = {:.1e} rdpNum = {:.1e} rdpDen = {:.1e}, stepSize = {:.1e}'
            print(ssString.format(tomoNum,tomoDen,rdpNum,rdpDen,newSS), end='\t')
                       
            if resetFlag:
                print ('resetting')
                inSS = inSS/2
            else:
                if ((oldSS*newSS)<0) & (ssIt>2):
                    inSS += newSS/3
                else:
                    inSS += ((newSS)*1/1.5)
                    if ((newSS/oldSS) > .66) & (ssIt > 2):
                        print('thinking about ultra step')
                        if ((tomoDen/tomoDenOld) < 1.4) & ((rdpDen/rdpDenOld)<1.4) :
                            inSS += (1.2*newSS)
                            print ('ultra step')
            
            if np.abs((newSS/inSS))<1e-3:
                print ('breaking')
                break
            print ('new tot step={:.2f}'.format(inSS))
            tomoDenOld = tomoDen
            rdpDenOld = rdpDen
            resetFlag = False
            
            

        px = self.x.copy()
        self.x.sapyb(1,sDir,inSS,out=self.x) #    += (sDir) #*self.mask)
        
        
        self.x.maximum(0, out=self.x)
        self.prevSDir = self.x - px
        self.full_model.forward(self.x,out=self.ybar)
        
    def update_objective(self):
        return 0
        
         #   ssTomo = ssNum/ssDen
submission_callbacks = [MaxIteration(660)]