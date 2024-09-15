from cil.optimisation.algorithms import Algorithm
from cil.optimisation.utilities import callbacks
import sirf.STIR as STIR
import numpy as np
import scipy.ndimage as ndi
import scipy.optimize as sci
import re

def L_grad(alfa,y,y_bar,fp_sdir):


    #y_bar_next =np.abs(y_bar+alfa*fp_sdir)+0.0001
    
    #NOT WORKING!!!
    y_bar_next               =  y_bar+alfa*fp_sdir
    y_bar_next[y_bar_next<=0]=0.01      
    
    return np.sum( (y*fp_sdir)/(y_bar_next)-fp_sdir)



def R_grad (alfa,x_old,kappa_,eps_,pixS_,sdir):
    R=0
    x_next=  x_old+alfa*sdir
    x_next[x_next<=0]=0
   
    for xs in range(-1,2):
        for ys in range (-1,2):
            for zs in range(-1,2):
                
                if (xs == 0) and (ys==0) and (zs==0): 
                    continue
                    
                x_shift    = np.roll(x_next,(zs,xs,ys),axis=(0,1,2)) 
                s_shift    = np.roll(sdir,(zs,xs,ys),axis=(0,1,2)) 
                sk_        = np.roll(kappa_,(zs,xs,ys),axis=(0,1,2))
                
                if zs==-1:
                    x_shift[-1,:,:]   =x_next[-1,:,:]
          
                    
                if zs==1:
                    x_shift[0,:,:] = x_next[0,:,:]


                f     = x_next - x_shift
                g     = x_next + x_shift
                diff_s= sdir - s_shift
                sum_s = sdir + s_shift
                
                #wI       = 2*f*diff_s*(g+2*f)-f**2*(sum_s+2*np.where(f*diff_s > 0, 1, -1))
                wI       = 2*f*diff_s*(g + np.abs(f))-f**2*(sum_s+eps_)
                
                wI      /= ( g + 2 * np.abs(f) + eps_)**2
          
                
                wI      *= pixS_[1]*kappa_*sk_ / np.sqrt((zs*pixS_[0])**2+(xs*pixS_[1])**2+(ys*pixS_[2])**2)
                R       += np.sum(np.sum(np.sum(wI,axis=-1),axis=-1),axis=-1)
        return R

def Obj_grad (alfa,y,y_bar,fp_sdir, x_old,kappa_,eps_,pixS_,sdir):
      
        return L_grad(alfa,y,y_bar,fp_sdir)-\
                (1/700)*R_grad(alfa,x_old,kappa_,eps_,pixS_,sdir)











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
        self.prec += 1e-10
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

    def step_size(self,a,b,sDir_,FP_sDir):
        return sci.root_scalar(Obj_grad,args=(self.data.acquired_data.as_array().flatten(),\
                                        self.ybar.as_array().flatten(),\
                                        FP_sDir.as_array().flatten(),\
                                        self.x.as_array(),\
                                        self.data.prior.get_kappa().as_array(),self.data.prior.get_epsilon(),\
                                        self.x.voxel_sizes(),\
                                        sDir_.as_array()),\
                                        bracket=[a,b],method='brentq',\
                                        xtol=0.01, maxiter=20)


    
    def update(self):
        #self.test_step()
        gradSino = self.data.acquired_data/self.ybar - 1
        gradI = self.full_model.backward(gradSino) 
        # Compute gradient of penalty
        pGrad = self.data.prior.gradient(self.x)
        grad = gradI - pGrad

        # Search direction is gradient divived by preconditioner
        #sDir = grad / (self.prec) # 
        sDir = grad/(self.prec.sqrt())
        ftS = np.fft.fft2(sDir.as_array(),axes=(1,2))
        ftS *= self.FFTFilter
        ftS = np.real(np.fft.ifft2(ftS,axes=(1,2)))
        ftS = ndi.gaussian_filter(ftS,(0.5,0,0))
        sDir.fill(ftS)
        sDir /= self.prec.sqrt()
        
        if (self.prevGrad.max()>0):
            beta = (grad-self.prevGrad).dot(sDir)/self.prevGrad.dot(self.prevSDir)
            sDir += beta*self.prevSDir
        self.prevSDir = sDir.clone()
        self.prevGrad = grad.clone()

        ## compute step size
        fpSD = self.lin_model.forward(sDir) 
        #ssNum = sDir.dot(gradI)
        #ssDen = fpSD.dot((fpSD/self.ybar))
        #ssNP, ssDP=self.rdp_step_size(sDir.as_array())

        stepSize = self.step_size(0.1,2,sDir,fpSD)
        print(stepSize.root)

        self.x += (stepSize.root*sDir)

        self.x.maximum(0, out=self.x)
        self.full_model.forward(self.x,out=self.ybar)
        
    def update_objective(self):
        return 0
        
         #   ssTomo = ssNum/ssDen
submission_callbacks = [MaxIteration(30)]
