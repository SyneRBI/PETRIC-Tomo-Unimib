from cil.optimisation.algorithms import Algorithm
from cil.optimisation.utilities import callbacks
import sirf.STIR as STIR
import numpy as np
import scipy.ndimage as ndi
import re

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
        self.x.fill(tImmArr)
        epsCorr = data.additive_term.max()*1e-6
 #       data.additive_term += epsCorr
        self.data = data
        acq_model = STIR.AcquisitionModelUsingParallelproj()
        acq_model.set_acquisition_sensitivity(STIR.AcquisitionSensitivityModel(data.mult_factors))
        acq_model.set_additive_term(data.additive_term+epsCorr)
        acq_model.set_up(data.acquired_data, self.x)
        self.full_model = acq_model
        self.lin_model = acq_model.get_linear_acquisition_model()
        self.ybar = acq_model.forward(self.x)
        self.prec = acq_model.backward(data.mult_factors/self.ybar)
        
        mask = (self.prec.as_array()<1)
        masko = (1- mask.copy())
        mask = ndi.binary_dilation(mask,iterations=2)
        mask = 1-mask
        maskSmooth = ndi.gaussian_filter(mask.astype(np.float32),(0,1.1,1.1))
        self.prec += 1e-10
        self.prec.fill(1/np.sqrt(self.prec.as_array()))
     #   print('\n\n there are ' + str(np.max(np.isnan(self.prec.as_array()))) + ' NaNs in the prec')
        self.mask = self.x.get_uniform_copy(0)
        self.mask.fill(masko)
        
        self.kappaArr = self.data.prior.get_kappa().as_array()
        
        
        self.prevGrad = self.x.get_uniform_copy(0)
        self.prevSDir = self.x.get_uniform_copy(0)
        self.makeFFT_2D_filter()
        super().__init__()
        self.configured = True        

    def makeFFT_2D_filter (self):
        d_ = .65
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
        print (order)
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
 
    
   
    
    def rdp_step_size (self,sDir_):
        
  #      print('\n RDP Step size' + str(sDir_.shape))
        ssNum = 0
        ssDen = 0
        inpImm_ = self.x.as_array()
        kappa_ = self.kappaArr

        
  #      print(kappa_.shape)
        eps_ = self.data.prior.get_epsilon()
        beta_ = self.data.prior.get_penalisation_factor()
        pixS_ = self.x.voxel_sizes()
        alpha_ = 0

 #       a2 = np.zeros_like(inpImm_)
 #       np.roll(a2,(-1,1,1),axis=(0,1,2))
 #       print('rolled test imm')
       # denImm_ = inpImm_ + alpha * sDir_
        for xs in range(-1,2):
            for ys in range (-1,2):
                for zs in range(-1,2):
                    if (xs == 0) and (ys==0) and (zs==0): 
        #                print('continuing')
                        continue
            #        print ('try rolling')
          #          print('image' , end='\t')
                    shiftImm_ = np.roll(inpImm_,(zs,xs,ys),axis=(0,1,2))                         
         #           print ('image', end='\t')
                    sk_ = np.roll(kappa_,(zs,xs,ys),axis=(0,1,2))
            #        print ('kappa')
           #         print(type(sDir_))
           #         print(sDir_.shape)
                    shiftSI_ = np.roll(sDir_,(zs,xs,ys),axis=(0,1,2))                
                    

           #         print(inpImm_.shape)
               
           #         print('done')
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
        ssNum *= (beta_*2)
        ssDen *= (beta_*2)
     #   print('done RDP ss')
        return ssNum,ssDen

    
    def update(self):
   #     self.test_step()
        gradSino = self.data.acquired_data/self.ybar - 1
        gradI = self.full_model.backward(gradSino) 
        # Compute gradient of penalty
        pGrad = self.data.prior.gradient(self.x)
        grad = gradI - pGrad
        grad.write('grad.hv')

        # Search direction is gradient divived by preconditioner
        #sDir = grad / (self.prec) # 
        #sDir = grad/(self.prec.sqrt())
        self.prec.write('prec.hv')
        sDir = grad*self.prec
        sDir.write('first_mult.hv')
     #   sDir *= self.mask
        ftS = np.fft.fft2(sDir.as_array(),axes=(1,2))
        ftS *= self.FFTFilter
        ftS = np.real(np.fft.ifft2(ftS,axes=(1,2)))
        ftS = ndi.gaussian_filter(ftS,(0.5,0,0))
        sDir.fill(ftS)
        sDir *= self.prec
        
        #sDir *= self.mask
        #sDir = sDir/(self.prec.sqrt())
        
        if (self.prevGrad.max()>0):
            beta = (grad-self.prevGrad).dot(sDir)/self.prevGrad.dot(self.prevSDir)
            sDir += beta*self.prevSDir
        self.prevSDir = sDir.clone()
        self.prevGrad = grad.clone()

        ## compute step size
        fpSD = self.lin_model.forward(sDir) #,subset_num=0,num_subsets=42) #*multCorr
        ssNum = sDir.dot(gradI)
        ssDen = fpSD.dot((fpSD/self.ybar)) #*42
        ssNP, ssDP = self.rdp_step_size(sDir.as_array())

        stepSize = (ssNum+ssNP)/(ssDen+ssDP)
        sDir *= stepSize
        sDir.write('sDir.hv')
     #   print('stepSize=' + str(stepSize))

        self.x += (sDir) #*self.mask)

        self.x.maximum(0, out=self.x)
        self.full_model.forward(self.x,out=self.ybar)
        
    def update_objective(self):
        return 0
        
         #   ssTomo = ssNum/ssDen
submission_callbacks = [MaxIteration(660)]
