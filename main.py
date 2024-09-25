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
        
        self.ybar = acq_model.forward(self.x)
        self.prec = self.x.get_uniform_copy(0)
        self.kappa = self.data.prior.get_kappa()
        
        fp1 = self.lin_model.forward(self.x.get_uniform_copy(1))
        #self.prec = acq_model.backward(data.mult_factors/self.ybar)
        self.prec = acq_model.backward(fp1/self.ybar)
        
        self.kappaArr = np.sqrt(self.prec.as_array())
       #self.data.prior.set_kappa(self.kappa.fill(self.kappaArr))
       # self.data.prior.set_up(self.x)
        
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
 #       mask = 1 - ndi.binary_dilation(precArr<1,iterations=2)
        self.mask = self.x.get_uniform_copy(0)
        self.mask.fill(mask)
        
        self.prevGrad = self.x.get_uniform_copy(0)
        self.prevSDir = self.x.get_uniform_copy(0)
        self.makeFFT_2D_filter()
        self.immArr = self.x.as_array()
        self.ybarNit = self.ybar.get_uniform_copy(0)
        self._ybarNit = self.ybar.get_uniform_copy(0)
        self.fpSD = self.ybar.get_uniform_copy(0)
        self.Sino1 = self.ybar.get_uniform_copy(1)
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

## Mettere la log likelihood al posto che gradienti manuali
## Calcolare denominatore tomografico approssimato con filtro?  

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

   #     ftS = np.fft.fft2(sDir.as_array(),axes=(1,2))
  #      ftS *= self.FFTFilter
  #      ftS = np.real(np.fft.ifft2(ftS,axes=(1,2)))
  #      ftS = ndi.gaussian_filter(ftS,(0.4,0,0))
 #       sDir.fill(ftS)
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
        self.lin_model.forward(sDir,out=self.fpSD) #,subset_num=0,num_subsets=42) #*multCorr
        ssNum = sDir.dot(gradI)
        
        # fpsdArr = fpSD.as_array()
        # yBarArr = self.ybar.as_array()
        # yArr = self.data.acquired_data.as_array()
        self.ybar.power(-1,out=self._ybarNit)
        ssDen = self.fpSD.dot(self.fpSD*self._ybarNit)
        
        #ssDen = np.dot(fpsdArr.flat,(fpsdArr/yBarArr).flat)
        
    #    ssDen = fpSD.dot((fpSD/yDen)) #*42
        #ssDen = fpSD.dot(fpSD/self.ybar) #*42



        numNew = sDir.dot(grad)
        newDenRDP = self.rdp_den_exact(sDirArr)
        ssString = 'New step size: num = {:.1e} rdpDen = {:.1e} tomoDen = {:.1e} stepSize = {:.1e}'
        stepSize = (numNew)/(ssDen+newDenRDP)
        print(ssString.format(numNew,newDenRDP,ssDen,stepSize))

        # sde = sDirArr.max()
        # sDirRat = -(self.immArr)/(sDirArr-(sde*1e-6))
        # maxSS = sDirRat[sDirRat>0].min()
        # inMaxSS = maxSS
        # contFlag = True
        # while (contFlag):
            # maxSS *=2
            # tempImm = self.immArr + maxSS * sDirArr
            # if (tempImm.min() < -(2*eps_)) or (tempImm[tempImm<(-.3*eps_)].size>10):
                # contFlag = False
                # maxSS /=2
        # print ('inmSS = {:.2e}, maxSS ={:.2e}'.format(inMaxSS,maxSS))
        # Yet another step size:
  
        inSS = stepSize
        resetFlag = False
        tomoDenOld = ssDen
        tomoDenA0 = ssDen
        rdpDenOld = newDenRDP
        newSS = 0
        f0 = numNew
        ssMin = 0
        fa = 1
        fc = -1
        xa = 0
        xc = stepSize*2
        if (inSS<0):
            inSS = abs(inSS)*.5
            xc = inSS*1.5
        newNum = 1
    
        while (newNum>0):
           ts = time.time()
           self.ybar.sapyb(1,self.fpSD,inSS,out=self.ybarNit)
           print('adding took' + str(time.time()-ts))
           if self.ybarNit.min()<(-1e-20):
               inSS*=.5
               break
           
           self.ybarNit.power(-1,out=self._ybarNit)
           self.data.acquired_data.sapyb(self._ybarNit,self.Sino1,-1,out=self.ybarNit)
           #tomoNum = np.dot(fpsdArr.flat,(yArr/yBarNit-1).flat)
           #tomoNum = fpSD.dot(self.data.acquired_data/ybarNit -1 ) 
           tomoNum = self.ybarNit.dot(self.fpSD)
           rdpNum = -sDir.dot(self.data.prior.gradient(self.x.sapyb(1,sDir,inSS).maximum(0)))
           newNum = tomoNum + rdpNum
        #   print ('curSS{:.1e} tomoN={:.1e} rdpN={:.1e} newNum={:.1e}'.format(inSS,tomoNum,rdpNum,newNum))
           if (newNum>0):
            inSS*=2
            xc = inSS
           else:
            xc = inSS
            inSS *=.5
        
        
        
        for ssIt in range(2):
            ts = time.time()
            yBarArr = self.ybar.as_array()
            fpsdArr = self.fpSD.as_array()
            yArr = self.data.acquired_data.as_array()
            t1 = time.time()
            print ('as array took ' + str(t1-ts))
            
            yBarNit = yBarArr + inSS*fpsdArr
   #             yBarNit[yBarNit<0] = addCorrArr[yBarNit<addCorrArr]
            tomoNum = np.dot(fpsdArr.flat,(yArr/yBarNit-1).flat)            
            print ('numpy dot and add took ' + str(time.time()-t1))
            ts = time.time()            
            self.ybar.sapyb(1,self.fpSD,inSS,out=self.ybarNit)
            t2 = time.time()
            print('adding took ' + str(t2-ts))
            if self.ybarNit.min()<(-1e-10):
               xc = inSS
               inSS= (inSS+xa)/2   
               print (self.ybarNit.min())
               print (self.epsCorrSino)
               print('red continuining')
               continue
            t3 = time.time()
            print ('if took' + str(t3-ts))
     
            self.ybarNit.power(-1,out=self._ybarNit)
            t4 = time.time()            
            print ('inversion took' + str(t4-t3))
            self.data.acquired_data.sapyb(self._ybarNit,self.Sino1,-1,out=self.ybarNit)
            t5 = time.time()
            print('full sapyb took ' + str(t5-t4))
            tomoNum = self.ybarNit.dot(self.fpSD)
            print('dot took ' + str(time.time()-t5))
           #tomoNum = np.dot(fpsdArr.flat,(yArr/yBarNit-1).flat)            
         #   tomoDen = fpSD.dot((fpSD/ybarNit)*(self.data.acquired_data/ybarNit))
            rdpNum = -sDir.dot(self.data.prior.gradient(self.x.sapyb(1,sDir,inSS).maximum(0)))

         #   rdpDen = self.rdp_den_exact(sDirArr,alpha_=inSS)
         #   oldSS = newSS
            newNum = tomoNum + rdpNum
            if ((newNum)>0):
                xa = inSS
                inSS=(inSS+xc)/2
            else:
                xc = inSS
                inSS = (inSS+xa)/2
            if (xa>xc):
                print ('breakin wtf?!?')
                break
            if (abs(xc/xa)-1)<1e-1:
                break

            tempImm = self.immArr + inSS * sDirArr
      #      tStr = 'Glob Min: {:.2f}, N< -eps/3: {:d}, N<-eps: {:d}, N<-2eps: {:d}, N<-4eps {:d}'
     #       print(tStr.format(tempImm.min()/(-eps_),tempImm[tempImm<-(eps_/3)].size,tempImm[tempImm<-(eps_)].size,tempImm[tempImm<-(eps_*2)].size,tempImm[tempImm<-(eps_*4)].size))
 
            #newSS = (tomoNum+rdpNum)/(rdpDen+tomoDen)
            # ssString = '\t Loop ss: tomoNum = {:.1e} tomoDen = {:.1e} rdpNum = {:.1e} rdpDen = {:.1e}, stepSize = {:.1e}'
            # print(ssString.format(tomoNum,tomoDen,rdpNum,rdpDen,newSS), end='\t')
            ssString = '\t Loop ss: tomoNum = {:.1e} rdpNum = {:.1e}, stepSize = {:.1e}'
            print(ssString.format(tomoNum,rdpNum,inSS)) #, end='\t')            
            
            # inSS +=newSS
            # print('curSS = {:.2f}'.format(inSS))
            

        px = self.x.copy()
        self.x.sapyb(1,sDir,inSS,out=self.x) #    += (sDir) #*self.mask)
        
        if self.x.min()<0:
            self.x.maximum(0, out=self.x)
            self.prevSDir = self.x - px
        #if (tempImm[tempImm<(-3*eps_)].size)>10:
            self.full_model.forward(self.x,out=self.ybar)
        else:
            
   #     else:
            print ('adding fpSD')
            self.ybar.sapyb(1,self.fpSD,inSS,out = self.ybar)
        self.immArr = self.x.as_array()
        
    def update_objective(self):
        return 0
        
         #   ssTomo = ssNum/ssDen
submission_callbacks = [MaxIteration(660)]