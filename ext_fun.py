import numpy as np 
import scipy.optimize as sci


L_eps=0.001


#rdp gradient 
def rdp_grad (inpImm_,kappa_,eps_,pixS_):
    rdpG_ = np.zeros_like(inpImm_)
    for xs in range(-1,2):
        for ys in range (-1,2):
            for zs in range(-1,2):
                if (xs == 0) and (ys==0) and (zs==0): 
                    print('continuing')
                    continue
                shiftImm_ = np.roll(inpImm_,(zs,xs,ys),axis=(0,1,2))
                sk_ = np.roll(kappa_,(zs,xs,ys),axis=(0,1,2))
                if zs==-1:
                    shiftImm_[-1,:,:]= inpImm_[-1,:,:]
                if zs==1:
                    shiftImm_[0,:,:] = inpImm_[0,:,:]

                tempW = pixS_[1]*kappa_*sk_ / np.sqrt((zs*pixS_[0])**2+(xs*pixS_[1])**2+(ys*pixS_[2])**2)             
                rdpG_ += tempW*(inpImm_ - shiftImm_)*(inpImm_ + 3 * shiftImm_ + 2* eps_ + 2* np.abs(inpImm_-shiftImm_)) /(np.abs(inpImm_)+ np.abs(shiftImm_) + 2*np.abs(inpImm_-shiftImm_ )+eps_)** 2 
    return rdpG_


def rdp_step_size (inpImm_,sDir_,kappa_,eps_,pixS_,alpha_=0):
    ssNum = 0
    ssDen = 0
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
    return ssNum,ssDen


def rdp_value (inpImm_,kappa_,eps_,pixS_):
    val=0
    for xs in range(-1,2):
        for ys in range (-1,2):
            for zs in range(-1,2):
                if (xs == 0) and (ys==0) and (zs==0): 
    #                print('continuing')
                    continue
                shiftImm_ = np.roll(inpImm_,(zs,xs,ys),axis=(0,1,2))    
                sk_ = np.roll(kappa_,(zs,xs,ys),axis=(0,1,2))
                if zs==-1:
                    shiftImm_[-1,:,:]= inpImm_[-1,:,:]     
                if zs==1:
                    shiftImm_[0,:,:] = inpImm_[0,:,:]
                wI = 1/(np.abs(inpImm_) + np.abs(shiftImm_)  + 2 * np.abs(inpImm_-shiftImm_) + eps_)
                wI *= pixS_[1]*kappa_*sk_ / np.sqrt((zs*pixS_[0])**2+(xs*pixS_[1])**2+(ys*pixS_[2])**2)
                val += np.sum(np.sum(np.sum( (inpImm_-shiftImm_)**2 * wI ,axis=-1),axis=-1),axis=-1)
        return val



def L(alfa,y,y_bar,fp_sdir):
    
    
    y_bar_next               =  y_bar+alfa*fp_sdir
    #y_bar_next=np.abs(y_bar_next) +0.0001
    y_bar_next[y_bar_next<=0]=L_eps
    
    #b=addCorr.as_array().flatten()
    #y_bar_next[y_bar_next<b]=b

    
    
    return np.sum(y*np.log(y_bar_next) - y_bar_next)
    
def R(alfa,x_old,kappa_,eps_,pixS_,sdir):
    
    R=0
    x_next             =  x_old+alfa*sdir
    x_next[x_next<=0]=0
    for xs in range(-1,2):
        for ys in range (-1,2):
            for zs in range(-1,2):
                
                if (xs == 0) and (ys==0) and (zs==0): 
                    continue
                    
                x_shift    = np.roll(x_next,(zs,xs,ys),axis=(0,1,2)) 
                sk_        = np.roll(kappa_,(zs,xs,ys),axis=(0,1,2))
                
                if zs==-1:
                    x_shift[-1,:,:]   =x_next[-1,:,:]
          
                    
                if zs==1:
                    x_shift[0,:,:] = x_next[0,:,:]

                f     = x_next - x_shift
                g     = x_next + x_shift
                
                #wI       = 1/(np.abs(x) + np.abs(x_shift)  + 2 * np.abs(x-x_shift) + eps_)
                wI       = f**2
                wI      /= g+2*np.abs(f)+eps_
                wI      *= pixS_[1]*kappa_*sk_ / np.sqrt((zs*pixS_[0])**2+(xs*pixS_[1])**2+(ys*pixS_[2])**2)
                R       += np.sum(np.sum(np.sum( wI ,axis=-1),axis=-1),axis=-1)
    return R



def L_grad(alfa,y,y_bar,fp_sdir):


    y_bar_next               =  y_bar+alfa*fp_sdir
    #y_bar_next=np.abs(y_bar_next) +0.0001
    y_bar_next[y_bar_next<=0]=L_eps
    
    #b=addCorr.as_array().flatten()
    #y_bar_next[y_bar_next<b]=b     
    
    return np.sum( (y*fp_sdir)/(y_bar_next)-fp_sdir)


def R_grad (alfa,x_old,kappa_,eps_,pixS_,sdir):
    R_prime          =0
    #R_second         =0
    x_next           =  x_old+alfa*sdir
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
                
        
                N        = 2*f*diff_s*(g + np.abs(f))-f**2*(sum_s+eps_)
                D        = ( g + 2 * np.abs(f) + eps_)
                molt_fac = pixS_[1]*kappa_*sk_ / np.sqrt((zs*pixS_[0])**2+(xs*pixS_[1])**2+(ys*pixS_[2])**2)
                R_prime       += np.sum(np.sum(np.sum(molt_fac*N/D**2,axis=-1),axis=-1),axis=-1)
                #R_second      += np.sum(np.sum(np.sum(molt_fac*\
                                #(2*diff_s**2*(g+3*np.abs(f))*D**2 +\
    return R_prime


def Fun(alfa,inpImm_,kappa_,eps_,pixS_,sdir,y,y_bar,fp_sdir):
    
    return  L(alfa,y,y_bar,fp_sdir)- \
            (1/700)*R(alfa,inpImm_,kappa_,eps_,pixS_,sdir)

def Obj_grad (alfa,y,y_bar,fp_sdir, x_old,kappa_,eps_,pixS_,sdir):
      
        return L_grad(alfa,y,y_bar,fp_sdir)-\
                (1/700)*R_grad(alfa,x_old,kappa_,eps_,pixS_,sdir)


























