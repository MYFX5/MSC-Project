
# Requirements for algorithms
import numpy as np
import scipy.special as sp
from functools import partial
from scipy.spatial.distance import cdist
from sklearn import neighbors

# Requirements for simulations/monitoring algorithms
import time
from IPython.display import clear_output
import inspect
from sklearn.metrics import roc_auc_score

# Requirements For plots and diagnostics
import matplotlib.pyplot as plt
import scipy.stats as sps
from scipy.stats import norm
from mpl_toolkits import mplot3d

class kernel_funcs:
    
    # Gaussian kernel
    def gaussian(d,s):
        return s*np.exp(-0.5*d**2) 
    
    # Cauchy kernel
    def cauchy(d,s):
        return s*1/(1+0.5*d**2)
        
    # Matern kernel
    def matern0(d,s):
        return s*np.exp(-d) 
    
    # Gaussian kernel gradient
    def grad_gaussian(K,X,l,s):
        Xd=(X-X.T)**2
        return -(K*Xd)*l 
    
    # Cauchy kernel gradient
    def grad_cauchy(K,X,l,s):
        Xd=(X-X.T)**2
        return -(K**2*Xd)*l
    
    # Gradient of K wrt scale s
    def gradK_s(K,s):
        return K/s
    
    def ARD_lm(l,s,X,kern):
        return kern(cdist(X,X, metric = "seuclidean", V = 1/l**2),s)
    
    def ARD_lmtest(l,s,X,Xtest,kern):
        return kern(cdist(X,Xtest, metric = "seuclidean", V = 1/l**2),s).T

class model_funcs:

    # Gradient of marginal likelihood trace term interior (Kinvy @ Kinvy.T - Kinv)
    def grad_logL_A(y,Ktild):
        Kinv = np.linalg.inv(Ktild)
        Kinvy=Kinv @ y
        return  Kinvy @ Kinvy.T - Kinv

    # Gradient of marginal likelihood (takes as input Kinvy @ Kinvy.T - Kinv) and dK/dparam
    def grad_log_L(A,G):
        return 0.5*np.sum(A * G.T)
    
    # Marginal likelihood
    def logL(y,K):
        L = np.linalg.cholesky(K)
        logL=-1/2*y.T @ np.linalg.solve(K,y)-np.sum(np.log(np.diag(L)))
        return logL
    # Getting ELBO 
    
    def ELBO(m,L,s,sigma,reg,y,X,lmbda,logpi,log1_pi,v_l0,v_l1,subsample,svi_subsample,a,ahat,b,bhat,kern, select, samples = 1):
        """
        Function to get Evidence lower bound in GP-SS model. 
        Uses stochastic subsampling to estimate <logp(y|theta)>.

        Parameters
        ----------
        mu : mean inverse length-scales
        L : L @ L.T = covariance of inverse length-dcales
        s : scale
        sigma : noise variance
        reg : nugget regularisation
        y : output vector (N x 1)
        X : input matrix X (N x D)
        lmbda : <binary inclusion variables>
        logpi : <logpi>
        log1_pi : <log(1-pi)>
        v_l0 : spike precision
        v_l1 : slab precision
        subsample : # SGD samples to estimate <logp(y|theta)>
        svi_subsample : # SGD samples to estimate <logp(y|theta)>
        a : prior hyperparameter in p(pi)
        b : prior hyperparameter in p(pi)
        ahat : variational parameter in q(pi)
        bhat : variational parameter in q(pi)
        kern : kernel function
        """
        # Setting dimensions and common objects
        n,p=np.shape(X)
        p,k = np.shape(L)
        R=np.diag(np.ones(subsample))*(reg+sigma**2)      

        # Regularising probabilities
        lmbda[lmbda<1e-10]=1e-10
        lmbda[lmbda>(1-1e-10)]=1-1e-10

        # Getting <logp(y,theta,v_l0)>

        # SGD sampling
        scale_ratio = n/subsample
        for j in range(samples):
            shuffle=np.random.choice(n,subsample,False)
            y_sample=y[shuffle]
            X_sample=X[shuffle]
            Elogl = 0        
            for i in range(svi_subsample):
                l = L @ np.random.normal(0,1,k)+m
                K=kernel_funcs.ARD_lm(l,s,X_sample,kern)
                Ktild=K+R
                Elogl+=model_funcs.logL(y_sample,Ktild)/svi_subsample
        
        # Getting -KL[q(theta)||e^<logp(theta|gamma)>_q(gamma)], -KL[q(gamma)||e^<logp(gamma|pi)>_q(pi)] and -KL[q(pi)||p(pi)]
        S = L @ L.T+ np.diag(np.ones(p))*1e-10
        neg_kl_theta = -0.5*np.sum((v_l1*lmbda[select]+v_l0*(1-lmbda[select]))*(m[select]**2+np.diag(S)[select]))+0.5*np.sum(1-lmbda)*np.log(v_l0)+0.5*np.sum(lmbda)*np.log(v_l1)
        neg_kl_gamma = np.sum(lmbda*(logpi-np.log(lmbda))+(1-lmbda)*(log1_pi-np.log(1-lmbda)))
        neg_kl_pi = (a-ahat)*logpi + (b-bhat)*log1_pi+sp.gammaln(ahat)+sp.gammaln(bhat)-sp.gammaln(ahat+bhat)
        if np.any(L!=0):
            C = np.linalg.cholesky(S)
            neg_kl_theta += 1/2*np.sum(np.log(np.diag(C)))
        
        return Elogl+neg_kl_theta+neg_kl_gamma+neg_kl_pi

class draw_GP:
    def draw_GP_ARD(n,p,q,sigma,corr,s,l,plot_YX,cop,r2=[]):
        """
        Draws GP inputs and outputs for SE kernel
        - selects random subset (length q) of length-scales to be active
        """
        
        # Drawing selector vector and updating length-scales
        select=np.random.choice(p,q,False)
        lselect=np.zeros(p)
        lselect[select]=l[select]
        
        # Drawing inputs X and computing L2 distance matrix
        Sigma=np.ones((p,p))*corr+np.diag(np.ones(p)*(1-corr))
        X=np.random.multivariate_normal(np.zeros(p),Sigma,n)
        if cop:
            X = sps.norm.cdf(X)
            X *= 1/np.sqrt(np.var(X))
        Xd=kernel_funcs.Xdist(X)
        
        # Computing cov-kernel matrix and drawing y,f,e
        K=kernel_funcs.ARD(l=lselect,s=s,Xd=Xd)
        F=np.transpose(np.random.multivariate_normal(mean=np.zeros(n),cov=K,size=1))
        if r2!=[]:
            sigma = np.sqrt((1-r2)/r2*np.var(F))
        e=np.reshape(np.random.normal(0,sigma,n),(n,1))
        print("R2=",np.var(F)/(np.var(F)+np.var(e)))
        Y=F+e
        
        if plot_YX and q==1:
            plt.subplots(figsize=(20,10))
            plt.scatter(X[:,select],Y, label="Y vs. relevant X")
            plt.scatter(X[:,select],F, label="F vs. relevant X")
            plt.xlabel("X")
            plt.ylabel("Y,F")
            plt.title("Plot of Y,F vs. single relevant X")
            plt.legend()
            plt.show()
        
        return Y,F,X,Xd,Xd.sum(2),e,lselect,s,sigma,select
    
    def draw_GP_ARD_lm(n,ntest,p,q,sigma2,corr,s,l,plot_YX,kern,cop,r2=[]):
        """
        LOW MEMORY VERSION Draws GP inputs and outputs for various kernel functions
        - selects random subset (length q) of length-scales to be active
        """
        
        # Drawing selector vector and updating length-scales
        select=np.linspace(0,p-1,p) < q
        lselect=np.zeros(p)
        lselect[select]=l[select]
        
        # Drawing inputs X
        Sigma=np.ones((p,p))*corr+np.diag(np.ones(p)*(1-corr))
        X=np.random.multivariate_normal(np.zeros(p),Sigma*sigma2,n+ntest)
        if cop:
            X = sps.norm.cdf(X)
            X *= 1/np.sqrt(np.var(X))
    
        # Computing cov-kernel matrix and drawing y,f,e
        K=kernel_funcs.ARD_lm(l=lselect[select],s=s,X=X[:,select],kern=kern)
        F=np.transpose(np.random.multivariate_normal(mean=np.zeros(n+ntest),cov=K,size=1))
        if r2!=[]:
            sigma = np.sqrt((1-r2)/r2*np.var(F))
        e=np.reshape(np.random.normal(0,sigma,n+ntest),(n+ntest,1))
        print("R2=",np.var(F)/(np.var(F)+np.var(e)))
        Y=F+e
        
        if plot_YX and q==1:
            plt.subplots(figsize=(20,10))
            plt.scatter(X[:,select],Y, label="Y vs. relevant X")
            plt.scatter(X[:,select],F, label="F vs. relevant X")
            plt.xlabel("X")
            plt.ylabel("Y,F")
            plt.title("Plot of Y,F vs. single relevant X")
            plt.legend()
            plt.show()
            
        return Y,F,X,e,lselect,s,sigma,select
    
    def draw_additive_parametric(n,ntest,p,q,sigma2,coefs_outer,coefs_inner,plot_YX):
    
        select = np.concatenate((np.ones(q),np.zeros(p-q)))==1
        X = np.random.random((n+ntest)*p).reshape(n+ntest,p)
        Z = X[:,:q] * coefs_inner
        f = np.sin(Z) * coefs_outer
        F = f.sum(1)
        e = np.random.normal(0,sigma2**0.5, n+ntest).reshape(n+ntest,)
        Y = F+e

        if plot_YX:
            fig,axs = plt.subplots(q, figsize = (10,20))
            for i in range(q):
                axs[i].scatter(X[:,i].reshape(n+ntest,1),f[:,i])
            plt.show()

        return Y,F,X,e,sigma2**0.5,select

    def draw_additive_parametric_vehtari(n,ntest,p,q,sigma2,plot_YX,scale_var):

        select = np.concatenate((np.ones(q),np.zeros(p-q)))==1
        X = np.random.random((n+ntest)*p).reshape(n+ntest,p)
        Z = X[:,:q] * np.linspace(np.pi/10, 2*np.pi, q)
        f = np.sin(Z)
        if scale_var:
            f = f/(np.var(f,0)**0.5).reshape(1,len(f.T))
        F = f.sum(1)
        e = np.random.normal(0,sigma2**0.5, n+ntest).reshape(n+ntest,)
        Y = F+e

        if plot_YX:
            fig,axs = plt.subplots(q, figsize = (10,20))
            for i in range(q):
                axs[i].scatter(X[:,i].reshape(n+ntest,1),f[:,i])
            plt.show()

        return Y,F,X,e,sigma2**0.5,select
    
    def draw_parametric_savitsky(n,ntest,p,q,correlation):

        if not correlation:
            q = 6
            select = np.concatenate((np.ones(q),np.zeros(p-q)))==1
            coefs_inner = np.concatenate((np.repeat(1,4), [3],[5]))
            X = np.random.random((n+ntest)*p).reshape(n+ntest,p)
            Z = X[:,:q] * coefs_inner
            F = (np.column_stack((Z[:,:(q-2)],(np.sin(Z[:,q-2:])).reshape(n+ntest,2)))).sum(1)

        else:
            q = 8
            select = np.concatenate((np.ones(q),np.zeros(p-q)))==1
            coefs_inner = np.concatenate((np.repeat(1,2), np.repeat(1.5,2), np.repeat(3,2), np.repeat(1.5,2)))
            X = np.random.random((n+ntest)*(p-21)).reshape(n+ntest,p-21)
            Sigma = np.diag(np.ones(21)*0.3)+0.7
            Xcorr = sps.norm.cdf(np.random.multivariate_normal(np.zeros(21),Sigma,n+ntest))
            X = np.column_stack((X[:,:5],Xcorr[:,0].reshape(n+ntest,1), X[:,5:], Xcorr[:,1:]))
            Z = X[:,:q] * coefs_inner
            F = Z[:,0]+Z[:,1]+np.sin(Z[:,2])*np.sin(Z[:,3])+np.sin(Z[:,4])+np.sin(Z[:,5])+Z[:,6]*Z[:,7]

        e = np.random.normal(0, 0.05,n+ntest).reshape(n+ntest,)
        Y = F+e
        
        return Y,F,X,e,0.05,select
    
class fit:  

    def get_NN(y,X,xtest,l,s,kern,NN,fraction):
    
        if fraction<1:
            shuffled_index = np.random.choice(len(X),int(len(X)*fraction),False)
            Xsubset = X[shuffled_index]
            ysubset = y[shuffled_index]
            
            dists = ((xtest-Xsubset)**2*l**2).sum(1)
            nn = shuffled_index[np.argsort(dists)[:NN]]
        else:
            dists = ((xtest-X)**2*l**2).sum(1)
            nn =np.argsort(dists)[:NN]
        dists = sorted(dists)[:NN]   

        return dists, nn
    
    def get_subsample(y,X,l,s,subsample,kern, sampling_strat, nn_fraction,lmbda):
        n,p= np.shape(X)
        select = lmbda>0.5

        if sampling_strat !="unif":
            # Getting NN set
            i = int(np.random.choice(n,1,False))
            dist, samples = fit.get_NN(y,X[:,select],X[i,select], l[select], s, kern, subsample, nn_fraction)
        else:    
            # Getting subsample of data
            samples=np.random.choice(n,subsample,False)
        y_sample=y[samples]
        X_sample=X[samples]

        return y_sample, X_sample

    def get_svi_subsample(y,X,m,L,s,sigma,reg,kern):

        # getting length-scale samples
        p,k = np.shape(L)
        xi = np.random.normal(0,1,k)
        l = L @ xi + m
        K=kernel_funcs.ARD_lm(l,s,X,kern)
        R=np.diag(np.ones(len(K)))*(reg+sigma**2)      
        Ktild=K+R

        return K, Ktild, l, xi

    def get_gradients_gp_ss(y_sample,X_sample,K,Ktild,l,xi,m,L,s,sigma,v_l1,v_l0,v_g,lmbda,subsample,grad_kern):
            p = len(l)

            # Getting common term in Marginal log likelihood gradient (Kinv_y)(Kinv_y)^T-K_inv
            A=model_funcs.grad_logL_A(y_sample,Ktild)

            # Gradient wrt m,L
            grad_logL_m=np.zeros(p)
            grad_logL_logsd = np.zeros(np.shape(L))
            for i in range(p):
                Xsample_i=X_sample[:,i].reshape(subsample,1)
                g=grad_kern(K=K,X=Xsample_i,l=l[i],s=s)
                grad_logL_m[i]=(model_funcs.grad_log_L(A,g))-m[i]*(v_l1*lmbda[i]+v_l0*(1-lmbda[i]))
                grad_logL_logsd[i]=(model_funcs.grad_log_L(A,g))*L[i]*xi-L[i]**2*(v_l1*lmbda[i]+v_l0*(1-lmbda[i]))
            M = L @ L.T + np.diag(np.ones(p))*1e-10
            grad_logL_logsd += L*np.linalg.solve(M, L)    


            # Gradient wrt s
            g=kernel_funcs.gradK_s(K,s)  
            grad_logL_s=model_funcs.grad_log_L(A,g)-v_g

            # Gradient wrt sigma
            g=np.diag(np.ones(subsample))*sigma*2
            grad_logL_sig=model_funcs.grad_log_L(A,g)-sigma*v_g

            return grad_logL_m, grad_logL_logsd, grad_logL_s, grad_logL_sig

        
    def get_step_size(new_gradient, sum_sq_grads, sum_grads, beta, beta2, eps, learn_rate, optimisation, subsample, t, n):
            
        if optimisation == "adam":
            sum_sq_grads = (1-beta2)*new_gradient**2+beta2*sum_sq_grads
            sum_grads = (1-beta)*new_gradient+beta*sum_grads
            step_size = learn_rate*sum_grads/(sum_sq_grads**0.5+eps)
       
        elif optimisation == "amsgrad":
            sum_sq_grads = np.maximum((1-beta2)*new_gradient**2+beta2*sum_sq_grads,sum_sq_grads)
            sum_grads = (1-beta)*new_gradient+beta*sum_grads
            step_size = learn_rate*sum_grads/(sum_sq_grads**0.5+eps)
        
        elif subsample<n:
            step_size = learn_rate/t
        
        else:
            step_size = learn_rate
            
        return step_size, sum_sq_grads, sum_grads



    """
    Defining function to fit GP with Spike and Slab prior on hyperparameters (using low/full rank posterior)
    """
    def GP_fit_SS_lm(y,X,m0,L0,s0,sig0,reg,subsample,svi_subsample,sampling_strat,nn_fraction,learn_noise,
                           tol,optimisation,learn_rate,beta,beta2,eps,v_l0,v_l1,v_g,lmbda,maxiter,print_,kern,grad_kern,
                            sum_sq_grads_m, sum_grads_m,sum_sq_grads_L, sum_grads_L,sum_sq_grads_s, sum_grads_s, sum_sq_grads_sig, 
                             sum_grads_sig, newsumgrads, store_ls, M, q):
        """
        Parameters
        ----------
        y : output vector (N x 1)
        X : input matrix (N x D)
        m0 : initial inverse-lengthscale mean
        L0 : initial inverse-lengthscale variance
        s0 : initial scale
        sig0 : initial noise
        reg : initial nugget regularisation
        subsample : #samples to draw in SGD (without replacement)
        svi_subsample : # SVI samples to draw
        sampling_strat : "unmif" or "nn" for SGD subsmpling
        nn_fraction : % data to subsample for nn search
        learn_noise : True = learn sigma^2
        tol : convergence tolerance in ||param-param_old|| over VI iterations
        optimisation : "adam", "amsgrad", "gd" (sgd version included)
        learn_rate : learning rate
        beta : ADAM retention factor (first moments)
        beta2 : ADAM retention factor (second moments)
        eps : ADAM epsilon imprecision
        v_l0 : spike precision on inverse lengthscales (default = 1e+3)
        v_l1 : slab precision on inverse lengthscales (default = 1e-3)
        v_g :  prior precision on scale and sigma
        lmbda : < binary inclusion variables > 
        maxiter : maximum # SGD iterations
        print_ : True = print
        kern :  Kernel function
        grad_kern : Kernel funtion gradient wrt. i^th inverse lengthscale
        XXX : to fill in sum grads
        newsumgrads : True = Initialise sum_sq_grads and sum_grads in optimisation
        store_ls : True = store LS mean/sd at each step
        M : previously stored LS mean/sd

        Returns
        -------
        m,L,s,sigma,logl,sumgrads_objs
        """
        n,p = np.shape(X)

        # Initialising hyperparameters
        m=np.ones(p)*m0
        L=L0
        lmbda=np.ones(p)*lmbda
        s=s0
        sigma=sig0
        Lmin = np.min(L)
        if Lmin==0:
            S = np.zeros((p,p))
        
        # Initialising optimisation algorithm terms
        if newsumgrads:
            sum_sq_grads_m=np.zeros(p)
            sum_sq_grads_L = np.zeros(np.shape(L))
            sum_sq_grads_s=0 
            sum_sq_grads_sig=0 
            sum_grads_m=0 
            sum_grads_L=np.shape(L) 
            sum_grads_s=0
            sum_grads_sig=0    

        # Initialising logl
        loglnew=-100
        logl=-200

        # Getting initial subsample
        y_sample, X_sample = fit.get_subsample(y,X,m,s,subsample,kern, sampling_strat, nn_fraction,lmbda)

        # Commencing SGD
        t=0
        while abs(loglnew-logl)>tol and t<maxiter:

            t+=1

            # Getting gradients
            grad_logL_m,grad_logL_logsd,grad_logL_s,grad_logL_sig = 0,0,0,0
            for i in range(svi_subsample):
                K, Ktild, l, xi = fit.get_svi_subsample(y_sample,X_sample,m,L,s,sigma,reg,kern)
                gradm,gradlogsd,grads,gradsig = fit.get_gradients_gp_ss(y_sample,X_sample,K,Ktild,l,xi,m,L,s,sigma,v_l1,v_l0,v_g,lmbda,subsample,grad_kern)
                grad_logL_m+=gradm/svi_subsample
                if Lmin>0:
                    grad_logL_logsd+=gradlogsd/svi_subsample
                grad_logL_s+=grads/svi_subsample
                grad_logL_sig+=gradsig/svi_subsample

            # Getting step sizes
            step_size_m, sum_sq_grads_m, sum_grads_m = fit.get_step_size(grad_logL_m, sum_sq_grads_m, sum_grads_m, beta, beta2, eps, learn_rate, optimisation, subsample, t, n)
            if Lmin>0:
                step_size_L, sum_sq_grads_L, sum_grads_L = fit.get_step_size(grad_logL_logsd, sum_sq_grads_L, sum_grads_L, beta, beta2, eps, learn_rate, optimisation, subsample, t, n)
            step_size_s, sum_sq_grads_s, sum_grads_s = fit.get_step_size(grad_logL_s, sum_sq_grads_s, sum_grads_s, beta, beta2, eps, learn_rate, optimisation, subsample, t, n)
            step_size_sig, sum_sq_grads_sig, sum_grads_sig = fit.get_step_size(grad_logL_sig, sum_sq_grads_sig, sum_grads_sig, beta, beta2, eps, learn_rate, optimisation, subsample, t, n)

            # Taking SGD step
            m+=step_size_m
            if Lmin>0:
                L*= np.exp(step_size_L)
                S = L @ L.T+1e-10*np.diag(np.ones(p))
            s+=step_size_s
            if learn_noise:
                sigma+=step_size_sig

            # Subsampling for next step
            y_sample, X_sample = fit.get_subsample(y,X,m,s,subsample,kern, sampling_strat, nn_fraction,lmbda)

            # Updating and printing logL
            logl=loglnew
            loglnew= -0.5*np.sum((v_l1*lmbda+v_l0*(1-lmbda))*(m**2+np.diag(S))-(1-lmbda)*np.log(v_l0)-lmbda*np.log(v_l1))
            if Lmin>0:
                C = np.linalg.cholesky(S)
                loglnew += np.sum(np.log(np.diag(C)))
            for i in range(svi_subsample):
                K, Ktild, l, xi = fit.get_svi_subsample(y_sample,X_sample,m,L,s,sigma,reg,kern)
                loglnew+=model_funcs.logL(y_sample,Ktild)/svi_subsample
            if print_:
                print(loglnew)

            if print_:
                if store_ls:
                    if t==1:
                        if q>=p:
                            q=p-1
                        colors = np.concatenate((np.repeat("red", q),np.repeat("orange", p-q)))
                    if not t % 10:
                        M = np.append(M,np.concatenate((np.abs(m),np.sqrt(np.diag(S)))).reshape(1,2*p), axis=0)
                        clear_output(wait=True)        
                        fig,axs = plt.subplots(2,figsize = (15,15))
                        fig.set_facecolor('black')
                        axs[0].set_facecolor('black')
                        axs[1].set_facecolor('black')
                        for i in np.arange(p)[::-1]:
                            lines0 = axs[0].plot(np.abs(M[:,i]), color = colors[i], linewidth = 0.5+1.5*(i<q))
                        for i in np.arange(p,2*p)[::-1]:
                            lines1 = axs[1].plot(np.abs(M[:,i]), color = colors[i-p], linewidth = 0.5+1.5*(i<(p+q)))
                        plt.show()
                print(loglnew)

            # Printing if maxiter reahed
            #if t==maxiter:
            #    print("maximum iterations reached, terminating")

        # returning final params and logL 
        return m,L,s,sigma,loglnew,M, sum_sq_grads_m, sum_grads_m,sum_sq_grads_L, sum_grads_L,sum_sq_grads_s, sum_grads_s, sum_sq_grads_sig, sum_grads_sig


    # Function to update lambda in VI iterations
    def get_lmbda(m,L,vl0,vl1,logpi,log1_pi):

        lmbda1=vl1**0.5*np.exp(-0.5*(m**2+np.diag(L @ L.T+1e-10))*vl1)*np.exp(logpi)
        lmbda0=vl0**0.5*np.exp(-0.5*(m**2+np.diag(L @ L.T+1e-10))*vl0)*np.exp(log1_pi)

        return lmbda1/(lmbda1+lmbda0)

    def get_pi(lmbda,alpha,beta):
        a=(np.sum(lmbda)+alpha)
        b=(len(lmbda)-np.sum(lmbda)+beta)
        digamma_a=sp.digamma(a)-sp.digamma(a+b)
        digamma_b=sp.digamma(b)-sp.digamma(a+b)
        return digamma_a, digamma_b, a, b
    

    """
    Defining VB_EM function for Gaussian process with spike and slab priors on length-scale - zero temperature variational posterior
    """


    def VB_EM_GP_SS(y,X,m0=0.01,L0=1e-2,k = 10, s0=1,sig0=1,lmbda0=1,logpi0=0,log1_pi0=0,v0=1e+4,v1=1e-4,vg=1e-4,a=1e-3,b=1e-3,reg=0.01,learn_spike=False,learn_noise=True,min_VBEM_iter=1,
                            max_VBEM_iter=10,max_GP_fit_iter=100,VBEM_tol=1e-4,GP_fit_tol=1e-5,subsample=100,svi_subsample = 1, ELBO_sample=1000, ELBO_svi_sample = 10, sampling_strat="nn", nn_fraction=1,
                            optimisation="amsgrad",learn_rate=0.025,Beta=0.9,Beta2=0.99,eps=1e-8,print_GP_fit=False,print_VBEM=True,
                            timer=True,kern=kernel_funcs.gaussian,grad_kern = kernel_funcs.grad_gaussian,ltrue=[],dampen_lmbda_update=[], newsumgrads=False, store_ls = False,  ZT_init_iter = 100,
                           init_PIP_update=True, learn_rate_mult = 0.1,iter_remove = False, q=[], final_prune = False, seed=[]):
        """
        Parameters
        ----------
        y : output vector (N x 1)
        X : input matrix (N x P)
        m0 : initial value of inverse length-scale means
        L0 : L0 @ L0.T + eps = initial value of inverse length-scale cov matrix
        k : Covariance matrix rank for inverse length-scales
        s0 : initial value of global scale
        sig0 : initial noise variance
        lmbda0 : initial binary inclusion probabilities   
        logpi0 : initial < log pi >
        log1_pi0 : initial < log 1-pi >
        v0 : initial spike precision
        v1 : slab precision
        vg : prior precision on scale and noise variance
        a : prior hyperparameter on p(pi)
        b : prior hyperparameter on p(pi)
        reg : nugget regularisation
        learn_spike : True = learn spike precision v0
        learn_noise : True = learn sigma^2
        min_VBEM_iter : minimum # VBEM iterations
        max_VBEM_iter : maximum # VBEM iterations
        max_GP_fit_iter : maximum # iterations for innter GP_fit procedure to maximise q(l)
        VBEM_tol : convergence tolerance for VBEM iterations (|param-param_old|)
        GP_fit_tol : convergence tolerance for inner GP_fit procedure to maximise q(l) (<logp>)
        subsample : # SGD samples to draw
        sampling_strat : "nn" for nearest neighbour (NA for low memory), else uniform
        ELBO_sample : # samples to use in ELBO update
        optimisation : "adam", "amsgrad", or "sgd"
        learn_rate : SGD learning rate
        Beta : adam/amdgrad E(g) retention factor
        Beta2 : adam/amsgrad E(g2) retention factor
        eps : adam/amsgrad epsilon imprecision
        print_GP_fit : True = print ELBO at each VB-EM iteration
        print_VBEM : True = print <logp(y,l)> for inner ML-SGD procedure to maximise q(l)
        timer : True = Time
        kern : Kernel function
        grad_kern : Kernel function gradient
        ltrue : If not empty, specifies true lengthscales if running simuations
        dampen_lmbda_update : If not empty, specifies how much to dampen lmbda updates
        newsumgrads : True = Re-initialise optimisation SSG and SG at each step
        store_ls : True = Store and plot inverse-lengthscales at each iterations
        init_PIP_update : Boolean, True = update lmbda, pi during initialisation
        iter_remove : Boolean, True = iteratively remove variables with low PIP (<0.01)

        Returns
        -------
        m,L,s,sig,lmbda,v0,elbo
        """

        # Setting dimensions and timer (and computing distance matrix if required)
        t=time.time()
        n,p = np.shape(X)

        # Transforming initial parameter settings to correct dimensionality
        m=m0*np.ones(p)
        L = np.zeros((p,k))
        M = np.column_stack((m.reshape(1,p),np.diag(L @ L.T + np.diag(np.ones(p))*1e-10).reshape(1,p)))
        s=s0
        sig=sig0
        lmbda=lmbda0*np.ones(p)
        logpi=logpi0
        log1_pi=log1_pi0
        v0=v0

        # Default fill ins        
        ELBO_sample = int(min(n,ELBO_sample))
        if not dampen_lmbda_update:
            damp = 0
        else:
            damp = dampen_lmbda_update
        if not max_GP_fit_iter:
            max_GP_fit_iter=np.max(100,n/subsample)                
        if not q:
            q = p
        if seed:
            np.random.seed(seed)

        # Initialisating convergence criteria
        i=0
        Param=np.concatenate((m,L.flatten(),lmbda))
        Param_diff_min=10    
        Elbo_new=-1e+7
        Elbo_diff_min = 100

        # Initialising SSGs for ADAM/AMSgrad
        sum_sq_grads_m=np.zeros(p)
        sum_sq_grads_L = np.zeros(np.shape(L))
        sum_sq_grads_s=0 
        sum_sq_grads_sig=0 
        sum_grads_m=np.zeros(p)
        sum_grads_L=np.zeros((np.shape(L)))
        sum_grads_s=0
        sum_grads_sig=0 

        # Initialising selections
        select = np.repeat(True,p)
        
        # Initialising params using ZT
        if ZT_init_iter>0:
            m,L_,s,sig,loglnew,M,sum_sq_grads_m, sum_grads_m,sum_sq_grads_L_, sum_grads_L_,sum_sq_grads_s, sum_grads_s, sum_sq_grads_sig, sum_grads_sig=fit.GP_fit_SS_lm(
                        y,X,m0=m,L0=L,s0=s,sig0=sig,reg=reg,subsample=subsample,svi_subsample=svi_subsample,sampling_strat=sampling_strat,nn_fraction=nn_fraction,
                        learn_noise = learn_noise,tol=GP_fit_tol,optimisation=optimisation,learn_rate=learn_rate,beta=Beta,beta2=Beta2,eps=eps,
                        v_l1=v1,v_l0=v0,v_g=vg,lmbda=lmbda,maxiter=ZT_init_iter,print_=print_GP_fit,kern=kern,grad_kern=grad_kern,
                        sum_sq_grads_m=sum_sq_grads_m, sum_grads_m=sum_grads_m,sum_sq_grads_L=sum_sq_grads_L, sum_grads_L=sum_grads_L,
                        sum_sq_grads_s=sum_sq_grads_s, sum_grads_s=sum_grads_s, sum_sq_grads_sig=sum_sq_grads_sig, sum_grads_sig=sum_grads_sig, 
                        newsumgrads = newsumgrads, store_ls = store_ls, M=M, q=q)
            
            if init_PIP_update:
                # Getting update to lmbda
                lmbda_new=fit.get_lmbda(m,L,v0,v1,logpi,log1_pi)
                lmbda = (1-damp)*lmbda_new+damp*lmbda

                # Getting update to pi
                logpi,log1_pi,ahat,bhat=fit.get_pi(lmbda,alpha=a,beta=b)
                
            # Updating active set
            if iter_remove:
                selectnew = lmbda>=0.01
                M = M[:,np.tile(selectnew,2)[np.tile(select,2)]]
                select = selectnew
                
        learn_rate*=learn_rate_mult
        L = np.ones((p*k)).reshape(p,k)
        L *= 1/np.sqrt(np.diag(L @ L.T))[0]*L0

        # Running VBEM iterations
        while (i<max_VBEM_iter and (Param_diff_min>VBEM_tol or Elbo_diff_min > 1)) or i<min_VBEM_iter:

            # Running GP_fit algorithm
            m[select],L[select],s,sig,logl,M,sum_sq_grads_m[select], sum_grads_m[select],sum_sq_grads_L[select], sum_grads_L[select],sum_sq_grads_s, sum_grads_s, sum_sq_grads_sig, sum_grads_sig=fit.GP_fit_SS_lm(
                        y,X[:,select],m0=m[select],L0=L[select],s0=s,sig0=sig,reg=reg,subsample=subsample,svi_subsample=svi_subsample,sampling_strat=sampling_strat,nn_fraction=nn_fraction,
                        learn_noise = learn_noise,tol=GP_fit_tol,optimisation=optimisation,learn_rate=learn_rate,beta=Beta,beta2=Beta2,eps=eps,
                        v_l1=v1,v_l0=v0,v_g=vg,lmbda=lmbda[select],maxiter=max_GP_fit_iter,print_=print_GP_fit,kern=kern,grad_kern=grad_kern,
                        sum_sq_grads_m=sum_sq_grads_m[select], sum_grads_m=sum_grads_m[select],sum_sq_grads_L=sum_sq_grads_L[select], sum_grads_L=sum_grads_L[select],
                        sum_sq_grads_s=sum_sq_grads_s, sum_grads_s=sum_grads_s, sum_sq_grads_sig=sum_sq_grads_sig, sum_grads_sig=sum_grads_sig, 
                        newsumgrads = newsumgrads, store_ls = store_ls, M=M, q=q)

            # Getting update to lmbda
            lmbda_new=fit.get_lmbda(m[select],L[select],v0,v1,logpi,log1_pi)
            lmbda[select] = (1-damp)*lmbda_new+damp*lmbda[select]

            # Getting update to pi
            logpi,log1_pi,ahat,bhat=fit.get_pi(lmbda,alpha=a,beta=b)
            
            # Updating active set
            if iter_remove:
                selectnew = lmbda>=0.01
                M = M[:,np.tile(selectnew,2)[np.tile(select,2)]]
                select = selectnew

            # Optional update to spike (hyperparameter optimisation)
            if learn_spike:
                v0 = np.sum((1-lmbda)+1e-10)/np.sum((1-lmbda)*l**2+1e-10)

            # Getting ELBO estimate
            Elbo_old=Elbo_new
            Elbo_new = model_funcs.ELBO(m,L,s,sig,reg,y,X,lmbda,logpi,log1_pi,v0,v1,ELBO_sample,ELBO_svi_sample,a,ahat,b,bhat,kern, select)
            
            # Tracking convergence criterion
            i+=1
            Param_old = Param
            Param = np.concatenate((m,L.flatten(),lmbda))
            Param_diff = np.sum(np.abs(Param-Param_old))/len(Param)
            if Param_diff < Param_diff_min:
                Param_diff_min = Param_diff
            if Elbo_new - Elbo_old < Elbo_diff_min:
                Elbo_diff_min =  Elbo_new - Elbo_old

            # Printing updates
            if print_VBEM:
                print("run {0}, diff = {1}, elbo = {2}".format(i,Param_diff,Elbo_new))
                print("sigma is ", sig)
                if np.any(ltrue):
                    print("length-scale norm is :", np.linalg.norm((m**2)**0.5-(ltrue**2)**0.5))
            
        # Setting variational parameters of unselected dimensions to zero
        if iter_remove:
            m[select==False]=0
            L[select==False]=0
        
        # Pruning at last stage if final_prune
        if final_prune:
            m[lmbda<0.01]=0
            L[lmbda<0.01]=0

        if timer:
            print("run time is :", time.time()-t)

        return m,L,s,sig,lmbda,v0,[logpi,log1_pi,ahat,bhat],Elbo_new
    
    
    def hyper_opt_SSGP(y, X, training_algorithm, testing_algorithm, hyper_arg, hyper_vals, method =  "CV", folds = 3, metric = "elbo", training_args=[], training_arg_vals=[]):
        """
        Parameters
        ----------

        y : response vector
        X : input matrix
        training algorithm : SSGP algorithm for training
        testing algorithm : algorithm used for getting posterior predictive moments
        hyper_arg : hyperparameter argument name in training algorithm
        hyper_vals : list of hyperparameter values to be iterated over
        method : CV or ML
        folds : number of CV folds
        metric : "elbo", "ml", "mse", "predictive"
        training_args : list of arguments for training algorithm to be changed from defaults
        training_arg_vals : list of argument values for training algorithm to be changed from defaults
        """

        n,p = np.shape(X)

        # Setting up storage object
        Loss = np.zeros(len(hyper_vals[0]))

        # Getting master arguments of algorithms, and updating based on non-default values provided
        training_master_args = inspect.getfullargspec(training_algorithm)[0]
        training_master_arg_defaults = inspect.getfullargspec(training_algorithm)[3]
        testing_master_args= inspect.getfullargspec(testing_algorithm)[0]
        testing_master_arg_defaults = inspect.getfullargspec(testing_algorithm)[3]

        # Updating arguments of training algorithm with non-defaults specified
        function_args = list(training_master_arg_defaults)
        if training_args:
            for j in range(len(training_args)):
                index = np.where(training_args[j]==np.array(training_master_args))[0][0]-2
                function_args[index] = training_arg_vals[j]
        # Determining method
        if method == "CV":

            # Shuffling the data and splitting into folds
            shuffled_indexes = np.random.choice(n,n,False)
            y_shuffle = y[shuffled_indexes]
            X_shuffle = X[shuffled_indexes]

            n_per_fold = int(n/folds)

            # Looping over spike values
            for i in range(len(hyper_vals[0])):

                # Doing CV over folds
                for f in range(folds):

                    # Getting CVtraining and CVtest set
                    y_test = y_shuffle[(f*n_per_fold):min((f+1)*n_per_fold, n)]
                    X_test = X_shuffle[(f*n_per_fold):min((f+1)*n_per_fold, n)]

                    if f>0:
                        y_train = y_shuffle[:(f*n_per_fold)]
                        X_train = X_shuffle[:(f*n_per_fold)]
                    if f<(folds-1):
                        if f>0:
                            y_train = np.append(y_train,y_shuffle[(f+1)*n_per_fold:],0)
                            X_train = np.append(X_train,X_shuffle[(f+1)*n_per_fold:],0)
                        else:
                            y_train = y_shuffle[(f+1)*n_per_fold:]
                            X_train = X_shuffle[(f+1)*n_per_fold:]
                    print(np.shape(y_test),np.shape(X_test))

                    # Updating training args based on spike value
                    current_training_args = [y_train,X_train]+function_args
                    for j in range(len(hyper_arg)):
                        index_spike =  np.where(hyper_arg[j]==np.array(training_master_args))[0][0]
                        current_training_args[index_spike] = hyper_vals[j][i]
                    current_training_args = tuple(current_training_args)

                    # Running training algorithm
                    results = training_algorithm(*current_training_args)

                    # Running testing algorithm
                    select = results[3]>0.5
                    fm, fv, ym, yv = testing_algorithm(y=y_train, X=X_train, Xtest = X_test, l=results[0], s=results[1], sig=results[2])

                    # Compute predictive distribution or MSE
                    if metric == "mse":
                        Loss[i] += np.mean((y_test - ym)**2)
                    else:
                        if np.sum(yv!=0)==len(yv):
                            Loss[i] += -0.5*np.sum(np.log(np.diag(yv)))-0.5* (y_test-ym).reshape(1,len(yv)) @ np.linalg.solve(yv,(y_test-ym).reshape(len(yv),1))
                        else:
                            L = np.linalg.cholesky(yv)
                            Loss[i] += -np.sum(np.log(np.diag(L)))-0.5* (y_test-ym).reshape(1,len(yv)) @ np.linalg.solve(yv,(y_test-ym).reshape(len(yv),1))

        else:

            Results = []
            Selections = np.zeros(len(hyper_vals[0]))
            selections = 1
            # Looping over spike values
            for i in range(len(hyper_vals[0])):
                
                if selections>0: # MAKE SURE TO FEED IN VALUES SO THAT LEAST SELECTIONS HAPPEN LAST

                    # Updating training args based on hyperopt value
                    current_training_args = [y,X]+function_args
                    for j in range(len(hyper_arg)):
                        index_spike =  np.where(hyper_arg[j]==np.array(training_master_args))[0][0]
                        current_training_args[index_spike] = hyper_vals[j][i]
                    current_training_args = tuple(current_training_args)

                    # Running training algorithm
                    Results.append(training_algorithm(*current_training_args))
                    
                    # Determine active selections
                    selections = np.sum(Results[i][0]!=0)
                    Selections[i] = selections

                    # Compute predictive distribution or MSE
                    Loss[i] = -Results[i][len(Results[i])-1]
                    
                else:
                    Loss[i] = np.max(Loss)
                    Results.append(Results[i-1])
                    Selections[i] = Selections[i-1]
               
        # Computing best values
        best_loss = np.min(Loss)
        best_val = []
        for j in range(len(hyper_vals)):
            best_val.append(hyper_vals[j][np.where(best_loss == Loss)[0][0]])
        if len(hyper_vals)==1:
            best_val = best_val[0]

        return [best_loss, best_val], Selections, Loss, Results

class diagnostics:
    
    def plot_length_scales_pip(ltrue,l,s,kern,lmbda,plotsize,width=[]):
        
        # Setting up objects
        p=len(l)
        if kern==kernel_funcs.lin:
            l=l*s**0.5
        if width==[]:
            width=2*p/1000
        True_select = (ltrue!=0)*1
        
        # Plotting PIP results
        plt.subplots(figsize=plotsize)
        plt.title( "Estimated vs. true PIP")
        plt.bar(np.arange(len(lmbda))+width,lmbda,width=width, label = "PIP")
        plt.bar(np.arange(len(l)),True_select,width=width, label = "True")
        plt.legend()
    
        # Plotting length-scale results
        plt.subplots(figsize=plotsize)
        plt.title("Estimated vs. true inverse length-scales")
        plt.bar(np.arange(len(ltrue)),(ltrue**2)**0.5, width=width, label = "True")
        plt.bar(np.arange(len(l))+width,(l**2)**0.5,width=width, label = "GP")
        plt.legend()
    
    def get_ls(mu,L,lmbda):
        p = len(mu)
        gamma = np.random.random(p)<lmbda
        S = L @ L.T
        if np.max(np.abs(L))>0:
            S+=np.diag(np.ones(p))*1e-10
        l = np.random.multivariate_normal(mu, S, 1)
        return l*gamma
        
    def get_pred_posterior_GP(y,X,Xtest,l,s,sig,reg,kern, post_var=False):
        
        K=kernel_funcs.ARD_lm(l=l,s=s,X=X,kern=kern)
        Ktest=kernel_funcs.ARD_lmtest(l=l,s=s,X=X,Xtest=Xtest,kern=kern)
        if post_var:
            Ktesttest=kernel_funcs.ARD_lm(l=l,s=s,X=Xtest,kern=kern)
        Ktild=K+np.diag(np.ones(len(K)))*(reg+sig**2) 
        fpost_mean=Ktest @ np.linalg.solve(K+np.diag(np.ones(len(K)))*0.01,y)
        ypost_mean=Ktest @ np.linalg.solve(Ktild,y)
        if post_var:
            fpost_var=Ktesttest-Ktest @ np.linalg.solve(K+np.diag(np.ones(len(K)))*0.01,Ktest.T)
            ypost_var=Ktesttest-Ktest @ np.linalg.solve(Ktild,Ktest.T)
        else:
            fpost_var,ypost_var = 0,0
        
        return fpost_mean, fpost_var, ypost_mean, ypost_var
    
    def get_pred_posterior_GP_NN(y,X,Xtest,l,s,sig,reg,kern,grad_kern,select,NN=100,pred_selected = False, fraction=1, grad_steps=0, learn_rate=0.1, tol=1e-4,print_grad=False):
    
        # Setting up storage objects
        m,p = np.shape(Xtest)
        fpost_mean=np.zeros((m,1))
        ypost_mean=np.zeros((m,1))
        fpost_var=np.zeros((m,m))
        ypost_var=np.zeros((m,m))  

        Xselect = X[:,select]
        Xtestselect = Xtest[:,select]
        lselect = l[select]
        
        
        # Getting predictions           
        for i in range(m):

            # Getting NN set using selected dimensions
            dist, nn = fit.get_NN(y,Xselect,Xtestselect[i], lselect, s, kern, NN, fraction)

            # Doing gradient update to local expert with select dimensions
            if grad_steps>0:

                # Restricting dataset
                Xnn = Xselect[nn,:]
                ynn=y[nn]  
                lnn = lselect[:]
                q = len(lnn)

                # Initialising convergence criteria
                k = 0
                diff=1
                lnn_old = lnn+1
                while  (k < grad_steps and np.abs(lnn_old-lnn).mean()>tol):

                    # Getting kernel matrix
                    K=kernel_funcs.ARD_lm(lnn,s,Xnn,kern)
                    R=np.diag(np.ones(len(K)))*(reg+sig**2)      
                    Ktild=K+R

                    # Getting common term in Marginal log likelihood gradient (Kinv_y)(Kinv_y)^T-K_inv
                    A=model_funcs.grad_logL_A(ynn,Ktild)

                    # Gradient wrt l
                    gradl=np.zeros(q)
                    for j in range(q):
                        Xi=Xnn[:,j].reshape(NN,1)
                        g=grad_kern(K=K,X=Xi,l=lnn[j],s=s)
                        gradl[j]=model_funcs.grad_log_L(A,g)

                    # Gradient wrt s
                    g=kernel_funcs.gradK_s(K,s)  
                    grads=model_funcs.grad_log_L(A,g)

                    # Gradient wrt sigma
                    g=np.diag(np.ones(len(K)))*sig*2
                    gradsig=model_funcs.grad_log_L(A,g)

                    # Taking gradient step
                    lnn_old = lnn+1e-20
                    lnn+=learn_rate*gradl
                    s+=learn_rate*grads
                    sig+=learn_rate*gradsig

                    # Updating convergence criteria
                    k+=1

                    if print_grad:
                        print(lnn,np.abs(lnn-lnn_old).mean())

                # Updating lengthscales and data to predict if gradsteps taken
                if pred_selected:
                    lpred = lnn
                    Xi=Xnn
                    Xtesti = Xtestselect[i]
                else:
                    lpred = np.zeros(p)
                    lpred[select] = lnn
                    lpred[select==False] = l[select==False]
                    Xi = X[nn,:]
                    Xtesti = Xtest[i]
                yi=ynn
                    
            # Updating lengthscales and data to predict with if no gradsteps taken
            else:
                if pred_selected:
                    lpred = lselect
                    Xi = Xselect[nn,:]
                    Xtesti = Xtestselect[i]
                else:
                    lpred = l
                    Xi = X[nn,:]
                    Xtesti=Xtest[i]
                yi=y[nn]

            # Making predictions 
            K=kernel_funcs.ARD_lm(l=lpred,s=s,X=Xi,kern=kern)
            Ktest=kernel_funcs.ARD_lmtest(l=lpred,s=s,X=Xi,Xtest=Xtesti.reshape(1,len(Xtesti)),kern=kern)
            Ktesttest=kernel_funcs.ARD_lm(l=lpred,s=s,X=Xtesti.reshape(1,len(Xtesti)),kern=kern)
            Ktild=K+np.diag(np.ones(len(K)))*(reg+sig**2) 

            KtestK = np.linalg.solve(K+np.diag(np.ones(len(K)))*reg, Ktest.T).T
            KtestKtild = np.linalg.solve(Ktild, Ktest.T).T
            fpost_mean[i] = KtestK @ yi
            ypost_mean[i] = KtestKtild @ yi
            fpost_var[i,i] = Ktesttest-KtestK @ Ktest.T
            ypost_var[i,i] = Ktesttest- KtestKtild @ Ktest.T
            
            if not round(i/m*100,2) % 10:
                print(i/m*100, "% complete")

        return fpost_mean, fpost_var, ypost_mean, ypost_var
    
    def get_pred_posterior_GP_NN_CV(y,X,results,reg,kern,NN=100, fraction=1,post_var=False, print_=True, use_tree=False, leaf_size=100, seed=[], MC_iters=10):
    
        if seed:
            np.random.seed(seed)
            
        # Set up
        n,p = np.shape(X) 
        logpredictive = 0
        
        # Getting relevant dimensions and model parameters
        select = results[0]!=0
        print(np.sum(select))
        if np.max(select)==0:
            select[np.random.choice(p,1,False)]=True
            results[0][select]=1e-3
        q = np.sum(select)
        Xselect = X[:,select].reshape(n,q)
        s = results[2]
        sig = results[3]
        
        # Getting average LOO-LPD
        for mc in range(MC_iters):
            lselect = diagnostics.get_ls(results[0],results[1],select).flatten()[select]
            
            # Building tree
            if use_tree:
                tree = neighbors.BallTree(Xselect,leaf_size,"seuclidean",V=1/lselect**2)

            # Getting predictions           
            for i in range(n):

                ypostmean,ypostvar = 0,1

                Xi = Xselect[i].reshape(1,q)
                yi = y[i]

                # Getting NN set using selected dimensions
                if use_tree:
                    nn = tree.query(Xi, k=NN+1)[1][0][1:]
                else:
                    nn = fit.get_NN(y,Xselect,Xi, lselect, s, kern, NN+1, fraction)[1][1:]

                # Making predictions 
                K=kernel_funcs.ARD_lm(l=lselect,s=s,X=Xselect[nn],kern=kern)
                Ktest=kernel_funcs.ARD_lmtest(l=lselect,s=s,X=Xselect[nn],Xtest=Xi,kern=kern)
                if post_var:
                    Ktesttest=kernel_funcs.ARD_lm(l=lselect,s=s,X=Xi,kern=kern)
                Ktild=K+np.diag(np.ones(len(K)))*(reg+sig**2) 
                KtestKtild = np.linalg.solve(Ktild, Ktest.T).T
                ypostmean += KtestKtild @ y[nn]
                if post_var:
                    ypostvar += Ktesttest- KtestKtild @ Ktest.T

                logpredictive +=  -0.5*np.log(ypostvar)-0.5*(ypostmean-yi)**2/ypostvar

        return logpredictive/MC_iters-n/2*np.log(2*np.pi)
    
    def get_BMA_predictions(y,X,Xtest,testing_algorithm, Results,weights, MC_samples, MC_iters):
        
        # Getting MC_samples to discretise weights
        sampled_weights = np.random.multinomial(MC_samples,weights, 1)[0]/MC_samples
        
        # Getting preditions per model
        fmean,ymean = 0,0
        
        for mc in range(MC_iters):
        
            for i in range(len(weights)):

                if sampled_weights[i]>0:

                    l = diagnostics.get_ls(Results[i][0],Results[i][1],Results[i][0]!=0).flatten()
                    s = Results[i][2]
                    sig = Results[i][3]
                    select = l!=0

                    fm, fv, ym, yv = testing_algorithm(y=y, X=X[:,select], Xtest = Xtest[:,select], l=l[select], s=s,sig=sig, post_var=False)

                    fmean += sampled_weights[i]*fm
                    ymean += sampled_weights[i]*ym
        
        return fmean/MC_iters, ymean/MC_iters
    
    def plot_GP_post_fit(ftest,ytest,Xtest,fpost_mean,fpost_var,ypost_mean,ypost_var,q,select,plot_var,plotsize):
    
        ntest = len(ftest)
        
        if q==1:
            plt.subplots(figsize=plotsize)
            plt.scatter(Xtest[:,select],ytest)
            plt.scatter(Xtest[:,select],ftest)
            plt.scatter(Xtest[:,select],fpost_mean)
            if plot_var:
                plt.scatter(Xtest[:,select],fpost_mean+2.65*np.sqrt(np.diag(ypost_var).reshape(ntest,1)))
                plt.scatter(Xtest[:,select],fpost_mean-2.65*np.sqrt(np.diag(ypost_var).reshape(ntest,1)))
                plt.legend(('Y','F','GP f(x)', 'Upper 99%', 'Lower 99%'))
            else:
                plt.legend(('Y','F','GP f(x)'))
            plt.title("Posterior mean GP predictions vs. Y vs. F")
            plt.show()
            
        if q==2:
            # Creating figure 
            fig,axs= plt.subplots(figsize=plotsize)
            ax = plt.axes(projection='3d')
    
    
            # Creating color map 
            my_cmap = plt.get_cmap('hot') 
    
            # Creating plot 
            trisurf = ax.plot_trisurf(Xtest[:,select[0]].reshape(ntest,), Xtest[:,select[1]].reshape(ntest,), fpost_mean.reshape(ntest,), 
                                     cmap = my_cmap, 
                                     linewidth = 0.2,  
                                     antialiased = True, 
                                     edgecolor = 'grey')   
            fig.colorbar(trisurf, ax = ax, shrink = 0.5, aspect = 10) 
            ax.set_title('Plot of F vs. relevant X') 
    
            # Adding labels 
            ax.set_xlabel('Input 1', fontweight ='bold')  
            ax.set_ylabel('Input 2', fontweight ='bold')  
            ax.set_zlabel('Output', fontweight ='bold') 
    
            plt.show()
    
        if q==3:
            # Creating figure 
            fig,axs= plt.subplots(figsize=plotsize)
            ax = plt.axes(projection='3d')
    
    
            # Creating color map 
            my_cmap = plt.get_cmap('hot') 
    
            # Creating plot 
            trisurf = ax.scatter(Xtest[:,select[0]].reshape(ntest,), Xtest[:,select[1]].reshape(ntest,), Xtest[:,select[1]].reshape(ntest,), 
                                     c = fpost_mean.reshape(ntest,), cmap = my_cmap)   
            fig.colorbar(trisurf, ax = ax, shrink = 0.5, aspect = 10) 
            ax.set_title('Plot of F vs. relevant X') 
    
            # Adding labels 
            ax.set_xlabel('Input 1', fontweight ='bold')  
            ax.set_ylabel('Input 2', fontweight ='bold')  
            ax.set_zlabel('Output', fontweight ='bold') 
    
            plt.show()
    
        plt.subplots(figsize=plotsize)
        plt.plot(ftest,ftest,color='black')
        plt.scatter(ypost_mean,ftest, color='red')
        plt.legend(('Latent function','GP f(x)','Reg f(x)=Xb'))
        plt.title("Posterior mean GP predictions vs. latent ftest")
        plt.show()

class simulations:

    def MSE_pc(x,y):
        assert np.all(np.shape(x) == np.shape(y))
        return np.sum((x-y)**2)/np.sum((y-y.mean())**2)


    def do_simulation_VBEMSSGP(y, X, ftest, ytest, Xtest, q, algorithm_training, algorithm_testing, nmodels, args, arg_vals, SS_GP, order_relevant_vars, order_irrelevant_vars, VS_threshs, predict_selected,select, hyper_opt=[], hyper_arg=[], hyper_vals=[], ltrue=[],MC_pred = False, MC_pred_draws = 10, model_select = [], post_fit_subsample=[], train=[], model_weighting=[], post_var = []):
        """
        Parameters
        ----------
        y : output training vector (n x 1)
        X : input training matrix (n x p)
        ftest : latent test vector (ntest x 1)
        ytest : observed test vector (ntest x 1)
        Xtest : inpute test matrix (ntest x p)
        q : # relevant variables
        algorithm_training : GP training algorithm to be used in simulations (returns l,s,sigma,lambda, v0)
        algorithm_testing : GP testing algorithms (list) to be used in simulations (returns f_mean, f_var, y_mean, y_var)
                            (will most likely need to pass in a function f = partial(f, XXX))
        nmodels : Int - number of models to run using the algorithm
        args : List of lists (strings) of arguments to be changed from default for 'm' models (m -> a)
        arg_vals : List of lists (strings) of arguments to be changed from default for 'm' models (m -> a)
        ltrue: Float64 lengthscales if true GP is used to draw data (d)
        post_fit_GP : Boolean vector , True = refit model m post VS (m)
        SS_GP :  Boolean vector , True =  model is a Spike-Slab GP
        order_relevant_vars : Boolean , True = order lenthscales and PIP for relevant variables
        order_irrrelevant_vars : Boolean , True = order lenthscales and PIP for irrelevant variables
        VS_threshs : List of Float64 vector of variable selection thresholds to use for each model
        select : Boolean vector , True = relevant generating variable
        predict_selected : Boolean, True = predict with selected variables only (lowest threshold)
        MC_pred : Boolean, True = draw from posterior to predict
        MC_pred_draws : # samples to draw from posterior to predict if MC_test = True
        
        Returns
        -------
        Runtime, Lambda, M, V, L1norm, L2norm, MSE_F, MSE_Y, Acc, Weighted_Acc, TPR, TNR, PPV, NPV, AUC
        """
        n,p = np.shape(X)
        m = nmodels
        t = len(VS_threshs[0])
        k = arg_vals[0][np.where(np.array(args[0])=="k")[0][0]]
        
        if not hyper_opt:
            hyper_opt = np.repeat(False, m)
        if not post_fit_subsample:
            post_fit_subsample = n
        if not train:
            train = np.repeat(True, m)
        if not model_select:
            model_select = np.repeat(True,m)
        if not model_weighting:
            model_weighting = np.repeat("elbo", m)
        if not post_var:
            post_var = np.repeat("True",m)

        # Storage objects
        Runtime=np.zeros(m)
        Lambda = np.zeros((m, p))
        M = np.zeros((m,p))
        V = np.zeros((m,p,k))
        L1norm=np.zeros(m)
        L2norm=np.zeros(m)
        MSE_F=np.zeros(m)
        MSE_Y=np.zeros(m)
        Acc=np.zeros((m,t))
        Weighted_Acc=np.zeros((m,t))
        TPR=np.zeros((m,t))
        TNR=np.zeros((m,t))
        PPV=np.zeros((m,t))
        NPV=np.zeros((m,t))
        MCC=np.zeros((m,t))
        AUC=np.zeros(m)

        # Getting master arguments for training algorithm
        master_args = inspect.getfullargspec(algorithm_training)[0]
        master_arg_defaults = inspect.getfullargspec(algorithm_training)[3]
        
        true_select = np.concatenate((np.repeat(True,q), np.repeat(False,p-q)))

        # Looping over models
        for j in range(m):
            
            # Only train if need to
            if train[j]:
                
                # Hyperopt (run sequence of models)
                if hyper_opt[j]:
                    start = time.time()                
                    best_pair,selection_path,losses,Results =  fit.hyper_opt_SSGP(y, X, algorithm_training, algorithm_testing, hyper_arg, hyper_vals, 
                                                                              method =  "ML", training_args=args[j],training_arg_vals=arg_vals[j])
                    train_time = time.time()-start
                    Runtime[j] += train_time

                # No hyperopt        
                else:
                    # Getting list of arguments to use for each model
                    function_args = [y,X]+list(master_arg_defaults)
                    for i in range(len(args[j])):
                                   index = np.where(args[j][i]==np.array(master_args))[0][0]
                                   function_args[index] = arg_vals[j][i]
                    function_args = tuple(function_args)

                    # Training model with specified arguments
                    start = time.time()
                    l, s, sig, lmbda, v0,qpi,elbo =  algorithm_training(*function_args)
                    train_time = time.time()-start
                    Runtime[j] += train_time  
                    
            # No training
            else:     
                # Updating performance metrics with previous value
                Runtime[j] += train_time
                l = [l]

            # Model selection or model averaging in ensemble case
            if hyper_opt[j]:   
                
                # Model averaging case (note prediction is MC over models!)
                start = time.time()
                PIP,Ls,Ss,Sigs = np.zeros((len(Results), p)),np.zeros((len(Results), p)),np.zeros((len(Results), )),np.zeros((len(Results), ))
                weights = np.zeros(len(Results))
                logevidences = np.zeros(len(Results))

                # Get log predictives if required
                if model_weighting[j]=="elbo":
                    for i in range(len(Results)):
                        logevidences[i] = Results[i][len(Results[i])-1]
                else:
                    logevidences = np.zeros(len(Results))
                    for i in range(len(Results)):
                        logevidences[i] =  diagnostics.get_pred_posterior_GP_NN_CV(y,X,Results[i],0.01,kern=kernel_funcs.gaussian,NN=n, fraction=1,post_var=True, print_=False, use_tree=False, leaf_size=100, seed=0, MC_iters = MC_pred_draws)
                max_logevidence = np.max(logevidences)

                # Do weighting
                for i in range(len(Results)):
                    logevidence = logevidences[i]
                    if logevidence >= max_logevidence-500:
                        if model_select[j]:
                            weights[i] = (logevidence==max_logevidence)*1
                        else:
                            weights[i] = np.exp(logevidence-max_logevidence)
                    PIP[i] = Results[i][4]
                    Ls[i] = np.abs(Results[i][0])
                    Ss[i] = Results[i][2]
                    Sigs[i] = Results[i][3]
                weights = weights/weights.sum()
                l = [Ls.T @ weights]
                s = np.sum(Ss*weights)
                sig = np.sum(Sigs*weights)
                lmbda = PIP.T @ weights
                Runtime[j] += time.time()-start
                
                
            # Storing length-scales and PIP:
            if order_relevant_vars:
                lmbda_q = sorted(lmbda[:q], reverse = True)
                l_q = sorted(np.abs(l)[:q], reverse = True)
            else:
                lmbda_q = lmbda[:q]
                l_q = np.abs(l)[:q]
            if order_irrelevant_vars:
                lmbda_pq = sorted(lmbda[q:], reverse = True)
                l_pq = sorted(np.abs(m)[q:], reverse = True)
            else:
                lmbda_pq = lmbda[q:]
                l_pq = np.abs(l)[q:]
            Lambda[j,:] =  np.concatenate((lmbda_q,lmbda_pq))
            M[j,:] =  np.concatenate((l_q, l_pq))

            # Computing MSE
            start = time.time()
            fpost_mean, ypost_mean = diagnostics.get_BMA_predictions(y,X,Xtest,algorithm_testing, Results,weights, MC_samples=1000, MC_iters = MC_pred_draws)        
            MSE_F[j]=simulations.MSE_pc(fpost_mean,ftest)
            MSE_Y[j]=simulations.MSE_pc(ypost_mean,ytest)
            Runtime[j]+=time.time()-start

            # Computing length-scale norm
            if len(ltrue)>0:
                L2norm[j] = np.sqrt(np.sum((np.abs(l)-np.abs(ltrue))**2))
                L1norm[j] = np.sum(np.abs(np.abs(l)-np.abs(ltrue)))
            else:
                L1norm[j], L2norm[j] = 0,0

            # Computing accuracy and VS metrics
            if SS_GP[j]:
                thresh_var = lmbda
            else:
                thresh_var = l

            if len(np.unique(thresh_var))>1:
                AUC[j] = roc_auc_score(select,thresh_var)

            for i in range(len(VS_threshs[j])):
                GP_select=(thresh_var>VS_threshs[j][i])*1
                Acc_vec=select*1-GP_select 
                PPV[j,i]=np.mean(select[GP_select>0])
                NPV[j,i]=np.mean((1-select[GP_select==0]))
                TPR[j,i]=np.mean(GP_select[select>0])
                TNR[j,i]=np.mean((1-GP_select[select==0]))
                Acc[j,i]=1-np.abs(Acc_vec).mean()
                Weighted_Acc[j,i] = (TPR[j,i]+TNR[j,i])/2
                
                TP = np.sum(GP_select[select>0])
                TN = np.sum(1-GP_select[select==0])
                FP = np.sum(1-select[GP_select>0])
                FN = np.sum(select[GP_select==0])
                
                MCC[j,i]=(TP*TN-FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
                
                if (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)==0:
                    MCC[j,i]=0

        # Returning results
        return Runtime, Lambda, M, V, L1norm, L2norm, MSE_F, MSE_Y, Acc, Weighted_Acc, TPR, TNR, PPV, NPV, AUC, MCC
        
    def simulator(nruns, runstart,n,ntest, p, q, corr, r2, sigma2, sigma2x, ltrue, strue, kern_draw, parametric, scale_var_parametric, plot_YX, dupe,
                 algorithm_training, algorithm_testing, nmodels, args, arg_vals, post_fit_GP, SS_GP, order_relevant_vars, order_irrelevant_vars, VS_threshs, print_=True):

        m = nmodels
        t = len(VS_threshs[0])
        
        # Storage objects
        Runtime=np.zeros((nruns, m))
        Lambda = np.zeros((nruns, m, p))
        M = np.zeros((nruns, m, p))
        V = np.zeros((nruns, m, p, k))
        L1norm=np.zeros((nruns, m))
        L2norm=np.zeros((nruns, m))
        MSE_F=np.zeros((nruns, m))
        MSE_Y=np.zeros((nruns,m))
        Acc=np.zeros((nruns,m,t))
        Weighted_Acc=np.zeros((nruns,m,t))
        TPR=np.zeros((nruns,m,t))
        TNR=np.zeros((nruns,m,t))
        PPV=np.zeros((nruns,m,t))
        NPV=np.zeros((nruns,m,t))
        AUC=np.zeros((nruns,m))
        
        for run in range(nruns):

            """
            Generating data and scaling data
            """
            np.random.seed(run)
            t=time.time()
            if parametric=="vehtari":
                Y,F,X,e,sigma,select=draw_GP.draw_additive_parametric_vehtari(n,ntest,p,q,sigma2,plot_YX,scale_var=scale_var_parametric)
                lselect = []
            if parametric=="savitsky":
                Y,F,X,e,sigma,select=draw_GP.draw_parametric_savitsky(n,ntest,p,q)
                lselect=[]
            else:
                Y,F,X,e,lselect,strue,sigma,select=draw_GP.draw_GP_ARD_lm(n,ntest,p,q,sigma2x,corr,strue,ltrue,plot_YX,kern=kern_draw,cop=False,r2=r2)

            Y = Y.reshape(n+ntest,1)
            F = F.reshape(n+ntest,1)

            Y = (Y-Y.mean())/Y.var()**0.5
            X = (X-X.mean(0))/X.var(0)**0.5
            F = (F-F.mean())/F.var()**0.5
            #np.random.seed()

            # Adding dupe feature
            if dupe:
                X[:,select+5]=X[:,select]+np.random.normal(0,np.sqrt(dupe_noise*np.var(X[:,select])),(n+ntest)*q).reshape(n+ntest,q)
                print ("Dupe correlation is ", np.corrcoef(X[:,select].T,X[:,select+5].T)[0,1])

            # Getting training and test set
            ytest=Y[n:]
            Xtest=X[n:]
            ftest=F[n:]
            y=Y[:n]
            X=X[:n]
            f=F[:n]
            print("data generated")
            if parametric=="GP":
                print("Length-scales are: ",lselect[select])
            print("Noise variance is: ",sigma**2)
            print("Average data variance is: ", np.mean(np.var(X,0)))
            print("Time taken to draw data is : ", time.time()-t, "\n")

            """
            Running algorithm
            """


            Runtime[run], Lambda[run], M[run], V[run], L1norm[run], L2norm[run], MSE_F[run], MSE_Y[run], Acc[run], Weighted_Acc[run], TPR[run], TNR[run], PPV[run], NPV[run], AUC[run] = simulations.do_simulation_VBEMSSGP(
                                       y, X, ftest, ytest, Xtest, q, algorithm_training = algorithm_training, algorithm_testing = algorithm_testing, 
                                       nmodels = nmodels, args = args, arg_vals = arg_vals, post_fit_GP = post_fit_GP, SS_GP = SS_GP, 
                                       order_relevant_vars = order_relevant_vars, order_irrelevant_vars = order_irrelevant_vars, VS_threshs = VS_threshs, select = select, ltrue=lselect)
            if print_:
                print("RUN {0}".format(run))
                print("Runtime is:", Runtime[run])
                print("Weighted accuracy is:", Weighted_Acc[run])
                print("TPR is:", TPR[run])
                print("PPV is:", PPV[run])
                print("L1norm is:", L1norm[run])
                print("L2norm is:", L2norm[run])
                print("MSE_F is:", MSE_F[run])
                print("MSE_Y is:", MSE_Y[run], "\n")
            
            Names = ["Runtime", "Lambda", "M", "V", "L1norm", "L2norm", "MSE_F", "MSE_Y", "Acc", "Weighted_Acc", "TPR", "TNR", "PPV", "NPV", "AUC"]
        
        return [Names, Runtime, Lambda, M, V, L1norm, L2norm, MSE_F, MSE_Y, Acc, Weighted_Acc, TPR, TNR, PPV, NPV, AUC]
