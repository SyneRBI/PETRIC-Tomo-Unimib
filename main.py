from cil.optimisation.algorithms import Algorithm
from cil.optimisation.utilities import callbacks
#assert issubclass(ISTA, Algorithm)
import sirf.STIR as STIR

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
    def __init__(self, data, initial, **kwargs):
        self.x = initial.copy()
        data.additive_term += data.additive_term.max()*1e-6
        self.data = data
        
        acq_model = STIR.AcquisitionModelUsingParallelproj()
        acq_model.set_acquisition_sensitivity(data.mult_factors)
        acq_model.set_additive_term(data.additive_term)
        acq_model.set_up(data.acquired_data, self.x)
        self.full_model = acq_model
        self.lin_model = acq_model.get_linear_acquisition_model()
        self.configured = True
        self.ybar = acq_model.forward(self.x)
        self.prec = acq_model.backward(data.mult_factors/self.ybar)
        self.prec += 1e-10

    def rdf_step_size (self,sDir_):
        ssNum = 0
        ssDen = 0
        inpImm_ = self.x.as_array()
        kappa_ = self.data.prior.get_kappa().as_array()
        penEps_ = self.data.prior.get_epsilon()
        beta_ = self.data.prior.get_prior_strength()
        voxS_ = self.x.voxel_sizes()
       # denImm_ = inpImm_ + alpha * sDir_
        for xs in range(-1,2):
            for ys in range (-1,2):
                for zs in range(-1,2):
                    if (xs == 0) and (ys==0) and (zs==0): 
        #                print('continuing')
                        continue
                    shiftImm_ = np.roll(inpImm_,(zs,xs,ys),axis=(0,1,2))
                    shiftSI_ = np.roll(sDir_,(zs,xs,ys),axis=(0,1,2))                
                    sk_ = np.roll(kappa_,(zs,xs,ys),axis=(0,1,2))
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
        return ssNum,ssDen

    
    def update(self):
        gradSino = self.data.acquired_data/self.ybar - 1
        gradI = self.full_model.backward(gradSino) 
        # Compute gradient of penalty
        pGrad = self.data.prior.gradient(image)
        grad = gradI - pGrad

        # Search direction is gradient divived by preconditioner
        sDir = grad / (self.prec) # 

        ## compute step size
        fpSD = self.lin_model.forward(sDir) #,subset_num=0,num_subsets=42) #*multCorr
        ssNum = sDir.dot(gradI)
        ssDen = fpSD.dot((fpSD/self.ybar)) #*42
        ssNP, ssDP = self.rdp_step_size(sDir)

        stepSize = (ssNum+ssNP)/(ssDen+ssDP)

        self.x += (stepSize*sDir)

        self.x.maximum(0, out=self.x)
        self.full_model.forward(self.x,out=self.ybar)
        
         #   ssTomo = ssNum/ssDen
