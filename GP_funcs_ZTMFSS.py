
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
    
    # Wave kernel
    def periodic(d,s):
        return np.exp(-2*np.sin(d*np.pi)**2)
    
    # Gaussian kernel gradient
    def grad_gaussian(K,X,l,s):
        Xd=(X-X.T)**2
        return -(K*Xd)*l 
    
    # Cauchy kernel gradient
    def grad_cauchy(K,X,l,s):
        Xd=(X-X.T)**2
        return -(K**2*Xd)*l
    
    def grad_lin(K,X,l,s):
        return 2*l*s*X @ X.T

    # Gradient of K wrt scale s
    def gradK_s(K,s):
        return K/s

    def ARD_lm(l,s,X,kern):
        if kern!="lin":
            return kern(cdist(X,X, metric = "seuclidean", V = 1/l**2),s)
        else:
            Z = np.diag(l) @ X.T
            return (Z.T @ Z)*s

    def ARD_lmtest(l,s,X,Xtest,kern):
        if kern!="lin":
            return kern(cdist(X,Xtest, metric = "seuclidean", V = 1/l**2),s).T
        else:
            Z = np.diag(l) @ X.T
            Ztest = np.diag(l) @ Xtest.T
            return (Ztest.T @ Z)*s

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
    def ELBO(l,s,sigma,reg,y,X,lmbda,logpi,log1_pi,v_l0,v_l1,subsample,a,ahat,b,bhat,kern,temp,select,samples=1):
        """
        Function to get Evidence lower bound in GP-SS model. 
        Uses stochastic subsampling to estimate <logp(y|theta)>.
        
        Parameters
        ----------
        l : length-scales (zero-temperature variational posterior mean)
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
        subsample : # samples to estimate <logp(y|theta)>
        a : prior hyperparameter in p(pi)
        b : prior hyperparameter in p(pi)
        ahat : variational parameter in q(pi)
        bhat : variational parameter in q(pi)
        kern : kernel function
        temp : temperature of prior
        select : indicators for current selections
        samples : # sets of size subsample to use in ELBO approximation
        """
        n,p=np.shape(X)
    
        # Regularising probabilities
        lmbda[lmbda<1e-10]=1e-10
        lmbda[lmbda>(1-1e-10)]=1-1e-10
    
        # Getting <logp(y,theta,v_l0)>
        Elogl=0
        if subsample==n:
            K=kernel_funcs.ARD_lm(l[select],s,X[:,select],kern)
            R=np.diag(np.ones(len(K)))*(reg+sigma**2)      
            Ktild=K+R
            Elogl+=model_funcs.logL(y,Ktild)
        else:
            scale_ratio = n/subsample
            for j in range(samples):
                shuffle=np.random.choice(n,subsample,False)
                y_sample=y[shuffle]
                X_sample=X[shuffle]
                K=kernel_funcs.ARD_lm(l[select],s,X_sample[:,select],kern)
                R=np.diag(np.ones(len(K)))*(reg+sigma**2)      
                Ktild=K+R
                Elogl+=model_funcs.logL(y_sample,Ktild)
            Elogl *= scale_ratio/samples
            
        Elogpyth = Elogl-0.5*np.sum((v_l1*lmbda[select]+v_l0*(1-lmbda[select]))*l[select]**2)/temp+0.5*np.sum(1-lmbda)*np.log(v_l0)/temp+0.5*np.sum(lmbda)*np.log(v_l1)/temp
    
        # Getting -KL[q(gamma)||e^<logp(gamma|pi)>_q(pi)] and -KL[q(pi)||p(pi)]
        neg_kl_gamma = np.sum(lmbda*(logpi-np.log(lmbda))+(1-lmbda)*(log1_pi-np.log(1-lmbda)))
        neg_kl_pi = (a-ahat)*logpi + (b-bhat)*log1_pi+sp.gammaln(ahat)+sp.gammaln(bhat)-sp.gammaln(ahat+bhat)
            
        return Elogpyth+neg_kl_gamma+neg_kl_pi

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
 
    def draw_parametric_savitsky(n,ntest,p,q,correlation, corr=0.7, sd_corr = 0.28):

        if not correlation:
            q = 6
            select = np.concatenate((np.ones(q),np.zeros(p-q)))==1
            coefs_inner = np.concatenate((np.repeat(1,4), [3],[5]))
            X = np.random.random((n+ntest)*p).reshape(n+ntest,p)
            Z = X[:,:q] * coefs_inner
            F = (np.column_stack((Z[:,:(q-2)],(np.sin(Z[:,q-2:])).reshape(n+ntest,2)))).sum(1)
            e = np.random.normal(0, 0.05,n+ntest).reshape(n+ntest,)
            sigma=0.05

        else:
            q = 8
            select = np.concatenate((np.ones(q),np.zeros(p-q)))==1
            coefs_inner = np.concatenate((np.repeat(1,2), np.repeat(1.5,2), np.repeat(3,2), np.repeat(1.5,2)))
            X = np.random.random((n+ntest)*(p-21)).reshape(n+ntest,p-21)
            Sigma = np.diag(np.ones(21)*(1-corr))+corr
            Xcorr = sps.norm.cdf(np.random.multivariate_normal(np.zeros(21),Sigma,n+ntest))
            X = np.column_stack((X[:,:5],Xcorr[:,0].reshape(n+ntest,1), X[:,5:], Xcorr[:,1:]))
            Z = X[:,:q] * coefs_inner
            F = Z[:,0]+Z[:,1]+np.sin(Z[:,2])*np.sin(Z[:,3])+np.sin(Z[:,4])+np.sin(Z[:,5])+Z[:,6]*Z[:,7]
            e = np.random.normal(0, sd_corr,n+ntest).reshape(n+ntest,)
            sigma = sd_corr

        Y = F+e
        
        return Y,F,X,e,sigma,select
    
    def draw_parametric_sin_2d(n, ntest, p, start, end, corrxz, r2,lin = False, data_shift = False, corrxztest=1e-3):
        """
        Function to draw latent function from 2d sinusoidal function

        n = # samples to train on (may change to ensure grid dimensions work)
        ntest = # samples to test on (may change to ensure grid dimensions work)
        p = # noise variables
        start = linspace start for generator variables
        end = linspace end for generator variables
        corrxz = correlation between noise variables and generator variables
        r2 = r-squared (noise vs. signal)
        lin = True/False for whether X2 is linear generating dimension
        """

        # Determining relative noise variance based on corrxz
        alpha = np.max(1/corrxz**2-2,0)
        alphatest = np.max(1/corrxztest**2-2,0)

        # Drawing generating variables
        xtrain = np.linspace(start, end, int(np.sqrt(n)))
        ytrain = np.linspace(start, end, int(np.sqrt(n)))
        xtest = np.linspace(start, end, int(np.sqrt(ntest)))+(n==ntest)*np.mean(np.abs(xtrain[1:]-xtrain[:(len(xtrain)-1)]))/2
        ytest = np.linspace(start, end, int(np.sqrt(ntest)))+(n==ntest)*np.mean(np.abs(ytrain[1:]-ytrain[:(len(ytrain)-1)]))/2
        ntrain = len(ytrain)**2
        ntest = len(ytest)**2

        # Creating grid of values
        x1,x2= np.meshgrid(xtrain,ytrain)
        if lin:
            z = 2*x2+np.sin(10*np.pi*x1)
        else:
            z =  +(np.tan(x1)+np.tan(x2)+np.sin(2*np.pi*x1)+np.sin(2*np.pi*x2)
              +np.cos(4*np.pi**2*x1*x2))

      # Creating grid of values and displaying latent test function
        x1test,x2test= np.meshgrid(xtest,ytest)
        if lin:
            ztest = 2*x2test+np.sin(10*np.pi*x1test)
        else:
            ztest =  +(np.tan(x1test)+np.tan(x2test)+np.sin(2*np.pi*x1test)+np.sin(2*np.pi*x2test)
              +np.cos(4*np.pi**2*x1test*x2test))
        z_min, z_max = ztest.min(), ztest.max()
        fig, ax = plt.subplots(figsize =(15,10))
        c = ax.pcolormesh(x1test, x2test, ztest, cmap='RdBu', vmin=z_min, vmax=z_max)
        ax.set_title('Test surface')
        ax.axis([x1test.min(), x1test.max(), x2test.min(), x2test.max()])
        fig.colorbar(c, ax=ax)
        plt.show()

        # Creating dataset to return
        F = np.row_stack((z.reshape(ntrain,1),ztest.reshape(ntest,1)))
        sigma = np.sqrt((1-r2)/r2*np.var(F))
        e = np.random.normal(0,sigma, ntrain+ntest).reshape(ntrain+ntest,1)
        Y = F+e
        
        z_min, z_max = Y[:ntrain].min(), Y[:ntrain].max()
        fig, ax = plt.subplots(figsize =(15,10))
        c = ax.pcolormesh(x1, x2, Y[:ntrain].reshape(len(ytrain),len(ytrain)), cmap='RdBu', vmin=z_min, vmax=z_max)
        ax.set_title('Training surface')
        ax.axis([x1.min(), x1.max(), x2.min(), x2.max()])
        fig.colorbar(c, ax=ax)
        plt.show()
        
        X = np.row_stack((np.column_stack((x1.reshape(ntrain,), x2.reshape(ntrain,))),
                          np.column_stack((x1test.reshape(ntest,), x2test.reshape(ntest,)))))
        if data_shift:
            Ztrain = (X[:ntrain] @ np.ones(2)).reshape(ntrain,1) + np.random.normal(0,alpha**0.5*np.var(X[:ntrain,0])**0.5,(ntrain)*p).reshape(ntrain,p)
            Ztest = (X[ntrain:] @ np.ones(2)).reshape(ntest,1) + np.random.normal(0,alphatest**0.5*np.var(X[ntrain:,0])**0.5,(ntest)*p).reshape(ntest,p)
            Z = np.row_stack((Ztrain,Ztest))
        else:
            Z = (X @ np.ones(2)).reshape(ntrain+ntest,1) + np.random.normal(0,alpha**0.5*np.var(X[:,0])**0.5,(ntrain+ntest)*p).reshape(ntrain+ntest,p)
        X = np.column_stack((X,Z))
        select = (np.linspace(0,p+1,p+2)<2)

        # Returning dataset
        return Y,F,X,e,sigma,select,ntrain,ntest
    
        
    def draw_parametric_sin_2d_new(n, ntest, p, start, end, corrxz,corrzz, r2,block_corr=False,lin = False):
        """
        Function to draw latent function from 2d sinusoidal function

        n = # samples to train on (may change to ensure grid dimensions work)
        ntest = # samples to test on (may change to ensure grid dimensions work)
        p = # noise variables
        start = linspace start for generator variables
        end = linspace end for generator variables
        corrxz = correlation between noise variables and generator variables
        r2 = r-squared (noise vs. signal)
        lin = True/False for whether X2 is linear generating dimension
        """

        # Drawing generating dimensions (2d)
        if block_corr:
            xtrain = np.random.random(n)
            ytrain = np.random.random(n)
            xtest = np.linspace(start, end, int(np.sqrt(ntest)))
            ytest = np.linspace(start, end, int(np.sqrt(ntest)))
            ntrain = n 
            ntest = len(ytest)**2
            x1,x2= xtrain,ytrain
        else:
            xtrain = np.linspace(start, end, int(np.sqrt(n)))
            ytrain = np.linspace(start, end, int(np.sqrt(n)))
            diff = np.mean(np.abs(xtrain[1:]-xtrain[:(len(xtrain)-1)]))/2
            xtest = np.linspace(start, end, int(np.sqrt(ntest)))+(n==ntest)*diff
            ytest = np.linspace(start, end, int(np.sqrt(ntest)))+(n==ntest)*diff
            ntrain = len(ytrain)**2
            ntest = len(ytest)**2
            x1,x2= np.meshgrid(xtrain,ytrain)

        # Creating latent and observed response
        if lin:
            z = 2*x2+np.sin(10*np.pi*x1)
        else:
            z =  +(np.tan(x1)+np.tan(x2)+np.sin(2*np.pi*x1)+np.sin(2*np.pi*x2)
              +np.cos(4*np.pi**2*x1*x2))

        x1test,x2test= np.meshgrid(xtest,ytest)
        if lin:
            ztest = 2*x2test+np.sin(10*np.pi*x1test)
        else:
            ztest =  +(np.tan(x1test)+np.tan(x2test)+np.sin(2*np.pi*x1test)+np.sin(2*np.pi*x2test)
              +np.cos(4*np.pi**2*x1test*x2test))

        F = np.row_stack((z.reshape(ntrain,1),ztest.reshape(ntest,1)))
        sigma = np.sqrt((1-r2)/r2*np.var(F))
        e = np.random.normal(0,sigma, ntrain+ntest).reshape(ntrain+ntest,1)
        Y = F+e
        
        # Plotting training and test surface
        if not block_corr:
            z_min, z_max = Y[:ntrain].min(), Y[:ntrain].max()
            fig, ax = plt.subplots(figsize =(15,10))
            c = ax.pcolormesh(x1, x2, Y[:ntrain].reshape(len(ytrain),len(ytrain)), cmap='RdBu', vmin=z_min, vmax=z_max)
            ax.set_title('Training surface')
            ax.axis([x1.min(), x1.max(), x2.min(), x2.max()])
            fig.colorbar(c, ax=ax)
            plt.show()      
        z_min, z_max = ztest.min(), ztest.max()
        fig, ax = plt.subplots(figsize =(15,10))
        c = ax.pcolormesh(x1test, x2test, ztest, cmap='RdBu', vmin=z_min, vmax=z_max)
        ax.set_title('Test surface')
        ax.axis([x1test.min(), x1test.max(), x2test.min(), x2test.max()])
        fig.colorbar(c, ax=ax)
        plt.show()

        # Transforming generating dimensions to draw noise dimensions
        x1 = x1.reshape(ntrain,)
        x2 = x2.reshape(ntrain,)
        x1test = x1test.reshape(ntest,)
        x2test = x2test.reshape(ntest,)
        X1 = norm.ppf(x1/(1+1e-4)+1e-8/(1+1e-4))
        X2 = norm.ppf(x2/(1+1e-4)+1e-8/(1+1e-4))
        if not block_corr:
            X1test = norm.ppf((x1test-diff*(ntrain==ntest))/(1+1e-4)+1e-8/(1+1e-4))
            X2test = norm.ppf((x2test-diff*(ntrain==ntest))/(1+1e-4)+1e-8/(1+1e-4))
        else:
            X1test = norm.ppf(x1test/(1+1e-4)+1e-8/(1+1e-4))
            X2test = norm.ppf(x2test/(1+1e-4)+1e-8/(1+1e-4))  
        X = np.column_stack((X1,X2))
        Xtest = np.column_stack((X1test,X2test))
        
        # Drawing noise dimensions using Gaussian conditioning and copulas
        P = np.ones((2,p))*corrxz
        if block_corr:
            P[1,int((p)/2):]=0
            P[0,:int((p)/2)]=0
            length_group1 = len(P[0,:int((p)/2)])
            length_group2 = len(P[0,int((p)/2):])
            Sigma1 = np.diag(np.ones(length_group1))*(1-corrzz)+corrzz
            Sigma2 = np.diag(np.ones(length_group1))*(1-corrzz)+corrzz
            Sigma = np.block([[Sigma1, np.zeros((length_group1, length_group2))],
                             [np.zeros((length_group2, length_group1)), Sigma2]])
        else:
            Sigma = np.diag(np.ones(p))*(1-corrzz)+corrzz
        Covmat = Sigma - P.T @ P
        var_x = np.var(X1)
        var_xtest = np.var(X1test)
        Z = np.random.multivariate_normal(np.zeros(p), var_x*Covmat, ntrain)+X @ P    
        Z = norm.cdf(Z)
        Ztest = np.random.multivariate_normal(np.zeros(p), var_xtest*Covmat, ntest)+Xtest @ P    
        Ztest = norm.cdf(Ztest)
        
        # Joining and returning dataset
        X = np.row_stack((np.column_stack((x1,x2,Z)),np.column_stack((x1test,x2test,Ztest))))
        select = (np.linspace(0,p+1,p+2)<2)
        return Y,F,X,e,sigma,select,ntrain,ntest
    
    def draw_parametric_sin_2d_new2(n, ntest, p, start, end, corrxz,corrzz, r2,block_corr=False,lin = False):
        """
        Function to draw latent function from 2d sinusoidal function - DRAWS DATA RANDOMLY

        n = # samples to train on (may change to ensure grid dimensions work)
        ntest = # samples to test on (may change to ensure grid dimensions work)
        p = # noise variables
        start = linspace start for generator variables
        end = linspace end for generator variables
        corrxz = correlation between noise variables and generator variables
        r2 = r-squared (noise vs. signal)
        lin = True/False for whether X2 is linear generating dimension
        """

        # Drawing generating dimensions (2d)
        xtrain = np.random.random(n)
        ytrain = np.random.random(n)
        xtest = np.linspace(start, end, int(np.sqrt(ntest)))
        ytest = np.linspace(start, end, int(np.sqrt(ntest)))
        ntrain = n 
        ntest = len(ytest)**2
        x1,x2= xtrain,ytrain

        # Creating latent and observed response
        if lin:
            z = 2*x2+np.sin(10*np.pi*x1)
        else:
            z =  +(np.tan(x1)+np.tan(x2)+np.sin(2*np.pi*x1)+np.sin(2*np.pi*x2)
              +np.cos(4*np.pi**2*x1*x2))

        x1test,x2test= np.meshgrid(xtest,ytest)
        if lin:
            ztest = 2*x2test+np.sin(10*np.pi*x1test)
        else:
            ztest =  +(np.tan(x1test)+np.tan(x2test)+np.sin(2*np.pi*x1test)+np.sin(2*np.pi*x2test)
              +np.cos(4*np.pi**2*x1test*x2test))

        F = np.row_stack((z.reshape(ntrain,1),ztest.reshape(ntest,1)))
        sigma = np.sqrt((1-r2)/r2*np.var(F))
        e = np.random.normal(0,sigma, ntrain+ntest).reshape(ntrain+ntest,1)
        Y = F+e
        
        # Plotting training and test surface      
        z_min, z_max = ztest.min(), ztest.max()
        fig, ax = plt.subplots(figsize =(15,10))
        c = ax.pcolormesh(x1test, x2test, ztest, cmap='RdBu', vmin=z_min, vmax=z_max)
        ax.set_title('Test surface')
        ax.axis([x1test.min(), x1test.max(), x2test.min(), x2test.max()])
        fig.colorbar(c, ax=ax)
        plt.show()

        # Transforming generating dimensions to draw noise dimensions
        x1 = x1.reshape(ntrain,)
        x2 = x2.reshape(ntrain,)
        x1test = x1test.reshape(ntest,)
        x2test = x2test.reshape(ntest,)
        X1 = norm.ppf(x1/(1+1e-4)+1e-8/(1+1e-4))
        X2 = norm.ppf(x2/(1+1e-4)+1e-8/(1+1e-4))
        X1test = norm.ppf(x1test/(1+1e-4)+1e-8/(1+1e-4))
        X2test = norm.ppf(x2test/(1+1e-4)+1e-8/(1+1e-4))  
        X = np.column_stack((X1,X2))
        Xtest = np.column_stack((X1test,X2test))
        
        # Drawing noise dimensions using Gaussian conditioning and copulas
        P = np.ones((2,p))*corrxz
        if block_corr:
            P[1,int((p)/2):]=0
            P[0,:int((p)/2)]=0
            length_group1 = len(P[0,:int((p)/2)])
            length_group2 = len(P[0,int((p)/2):])
            Sigma1 = np.diag(np.ones(length_group1))*(1-corrzz)+corrzz
            Sigma2 = np.diag(np.ones(length_group1))*(1-corrzz)+corrzz
            Sigma = np.block([[Sigma1, np.zeros((length_group1, length_group2))],
                             [np.zeros((length_group2, length_group1)), Sigma2]])
        else:
            Sigma = np.diag(np.ones(p))*(1-corrzz)+corrzz
        Covmat = Sigma - P.T @ P
        var_x = np.var(X1)
        var_xtest = np.var(X1test)
        Z = np.random.multivariate_normal(np.zeros(p), var_x*Covmat, ntrain)+X @ P    
        Z = norm.cdf(Z)
        Ztest = np.random.multivariate_normal(np.zeros(p), var_xtest*Covmat, ntest)+Xtest @ P    
        Ztest = norm.cdf(Ztest)
        
        # Joining and returning dataset
        X = np.row_stack((np.column_stack((x1,x2,Z)),np.column_stack((x1test,x2test,Ztest))))
        select = (np.linspace(0,p+1,p+2)<2)
        return Y,F,X,e,sigma,select,ntrain,ntest
    
    
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
    
    def get_subsample(y,X,l,s,sigma,reg,subsample,kern, sampling_strat, nn_fraction,lmbda):
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

        # Computing kernel matrices
        K=kernel_funcs.ARD_lm(l,s,X_sample,kern)
        R=np.diag(np.ones(len(K)))*(reg+sigma**2)      
        Ktild=K+R

        return y_sample, X_sample, K, Ktild
    
    def get_gradients_gp_ss(y_sample,X_sample,K,Ktild,l,s,sigma,v_l1,v_l0,v_g,lmbda,subsample,grad_kern,temp,n):
            p = len(l)
            
            # Getting common term in Marginal log likelihood gradient (Kinv_y)(Kinv_y)^T-K_inv
            A=model_funcs.grad_logL_A(y_sample,Ktild)
    
            # Gradient wrt l
            grad_logL_l=np.zeros(p)
            for i in range(p):
                Xsample_i=X_sample[:,i].reshape(subsample,1)
                g=grad_kern(K=K,X=Xsample_i,l=l[i],s=s)
                grad_logL_l[i]=model_funcs.grad_log_L(A,g)-l[i]*(v_l1*lmbda[i]+v_l0*(1-lmbda[i]))/temp*subsample/n
                    
            # Gradient wrt s
            g=kernel_funcs.gradK_s(K,s)  
            grad_logL_s=model_funcs.grad_log_L(A,g)-v_g*subsample/n
                    
            # Gradient wrt sigma
            g=np.diag(np.ones(subsample))*sigma*2
            grad_logL_sig=model_funcs.grad_log_L(A,g)-sigma*v_g*subsample/n
            
            return grad_logL_l , grad_logL_s, grad_logL_sig
            
        
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
    
    def get_subsample_fullpost(y,X,subsample):
    
        # Getting subsample of data
        n = len(y)
        samples=np.random.choice(n,subsample,False)
        y_sample=y[samples]
        X_sample=X[samples]

        return y_sample, X_sample
    
    def get_svi_subsample_fullpost(y,X,m,L,s,sigma,reg,kern):
        
        # getting length-scale samples
        p,k = np.shape(L)
        xi = np.random.normal(0,1,k)
        l = L @ xi + m
        K=kernel_funcs.ARD_lm(l,s,X,kern)
        R=np.diag(np.ones(len(K)))*(reg+sigma**2)      
        Ktild=K+R

        return K, Ktild, l, xi

    def get_gradients_gp_ss_fullpost(y_sample,X_sample,K,Ktild,l,xi,m,L,s,sigma,v_1,v_g,subsample,grad_kern):
            p = len(l)

            # Getting common term in Marginal log likelihood gradient (Kinv_y)(Kinv_y)^T-K_inv
            A=model_funcs.grad_logL_A(y_sample,Ktild)

            # Gradient wrt m,L
            grad_logL_m=np.zeros(p)
            grad_logL_logsd = np.zeros(np.shape(L))
            for i in range(p):
                Xsample_i=X_sample[:,i].reshape(subsample,1)
                g=grad_kern(K=K,X=Xsample_i,l=l[i],s=s)
                grad_logL_m[i]=(model_funcs.grad_log_L(A,g))-m[i]*v_1
                grad_logL_logsd[i]=(model_funcs.grad_log_L(A,g))*L[i]*xi-L[i]**2*v_1
            M = L @ L.T + np.diag(np.ones(p))*1e-10
            grad_logL_logsd += L*np.linalg.solve(M, L)    

            # Gradient wrt s
            g=kernel_funcs.gradK_s(K,s)  
            grad_logL_s=model_funcs.grad_log_L(A,g)-v_g

            # Gradient wrt sigma
            g=np.diag(np.ones(subsample))*sigma*2
            grad_logL_sig=model_funcs.grad_log_L(A,g)-sigma*v_g

            return grad_logL_m, grad_logL_logsd, grad_logL_s, grad_logL_sig


    def GP_fit_SS_lm(y,X,l0,s0,sig0,reg,subsample,sampling_strat,nn_fraction,learn_noise,
                        tol,optimisation,learn_rate,beta,beta2,eps,v_l0,v_l1,v_g,lmbda,maxiter,print_,kern,grad_kern,
                        sum_sq_grads_l, sum_grads_l,sum_sq_grads_s, sum_grads_s, sum_sq_grads_sig, sum_grads_sig, newsumgrads, store_ls, L,temp, q=[], printmaxiter=False, X_mult=1):
        """
        Parameters
        ----------
        y : output vector (N x 1)
        X : input matrix (N x D)
        l0 : initial inverse-lengthscale
        s0 : initial scale
        sig0 : initial noise
        reg : initial nugget regularisation
        subsample : #samples to draw in SGD (without replacement)
        sampling_strat : "unif" for uniform, "nn" (or any word) for NN
        nn_fraction : Float \in (0,1], fraction of data for stochastic NN search 
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
        store_ls : True = store LS's at each step
        L : previously stored LS's mean/sd
        temp : temperature of p(theta,gamma,pi)
        
        Returns
        -------
        l,s,sigma,logl
        """
        
        plt.rcParams.update({'text.color' : "white",
                      'xtick.color' : "white",
                      'ytick.color' : "white",
                     'axes.labelcolor' : "white"})
        
        n,p = np.shape(X)
    
        # Initialising hyperparameters
        l=np.ones(p)*l0
        lmbda=np.ones(p)*lmbda
        s=s0
        sigma=sig0
        
        # Initialising optimisation algorithm terms
        if newsumgrads:
            sum_sq_grads_l=np.zeros(p)
            sum_sq_grads_s=0 
            sum_sq_grads_sig=0 
            sum_grads_l=0 
            sum_grads_L=0
            sum_grads_sig=0  
        
        # Initialising logl
        loglnew=-100
        logl=-200
        
        # Getting initial subsample
        subsample = int(subsample)
        y_sample, X_sample, K, Ktild = fit.get_subsample(y,X,l,s,sigma,reg, subsample,kern, sampling_strat, nn_fraction, lmbda)
              
        # Commencing SGD
        t=0
        while abs(loglnew-logl)>tol and t<maxiter:
            
            t+=1
            
            # Getting gradients
            grad_logL_l,grad_logL_s,grad_logL_sig = fit.get_gradients_gp_ss(y_sample,X_sample, K, Ktild, l, s, sigma, v_l1, v_l0, v_g, lmbda, subsample,grad_kern=grad_kern,temp=temp, n=n)
            
            # Getting step sizes
            step_size_l, sum_sq_grads_l, sum_grads_l = fit.get_step_size(grad_logL_l, sum_sq_grads_l, sum_grads_l, beta, beta2, eps, learn_rate/X_mult, optimisation, subsample, t, n)
            step_size_s, sum_sq_grads_s, sum_grads_s = fit.get_step_size(grad_logL_s, sum_sq_grads_s, sum_grads_s, beta, beta2, eps, learn_rate, optimisation, subsample, t, n)
            step_size_sig, sum_sq_grads_sig, sum_grads_sig = fit.get_step_size(grad_logL_sig, sum_sq_grads_sig, sum_grads_sig, beta, beta2, eps, learn_rate, optimisation, subsample, t, n)
            
            # Taking SGD step
            l+=step_size_l
            if kern!="lin":
                s+=step_size_s
            if learn_noise:
                sigma+=step_size_sig
            s=np.abs(s)
            
            # Subsampling for next step
            y_sample, X_sample, K, Ktild = fit.get_subsample(y,X,l,s,sigma,reg, subsample, kern, sampling_strat, nn_fraction, lmbda)
            
            # Updating and printing logL
            logl=loglnew
            loglnew=model_funcs.logL(y_sample,Ktild)*n/subsample-0.5*np.sum((v_l1*lmbda+v_l0*(1-lmbda))*l**2)/temp+np.sum(1-lmbda)/2*np.log(v_l0)/temp+np.sum(lmbda)/2*np.log(v_l1)/temp
            if print_:
                if store_ls:
                    if t==1:
                        if not q:
                            colors = np.repeat("orange", p)
                        else: 
                            colors = np.concatenate((np.repeat("red",q), np.repeat("orange",p-q)))
                    if not t % 10:
                        L = np.append(L,l.reshape(1,p), axis=0)
                        clear_output(wait=True)        
                        fig,axs = plt.subplots(figsize = (15,15))
                        fig.set_facecolor('black')
                        axs.set_facecolor('black')
                        for i in np.arange(p)[::-1]:
                            lines0 = axs.plot(np.abs(L[:,i]), color = colors[i], linewidth = 1)
                        plt.show()
                print(loglnew)
            
            # Printing if maxiter reahed
            if printmaxiter:
                if t==maxiter:
                    print("maximum iterations reached, terminating")
            
        # returning final params and logL 
        return l,s,sigma,loglnew,L,sum_sq_grads_l, sum_grads_l,sum_sq_grads_s, sum_grads_s, sum_sq_grads_sig, sum_grads_sig
    
    # Function to update lambda in VI iterations
    def get_lmbda(l,vl0,vl1,logpi,log1_pi,temp):
        
        lmbda1=(vl1**0.5*np.exp(-0.5*l**2*vl1)*np.exp(logpi))**(1/temp)
        lmbda0=(vl0**0.5*np.exp(-0.5*l**2*vl0)*np.exp(log1_pi))**(1/temp)
        
        return lmbda1/(lmbda1+lmbda0)
    
    # Function to update < logpi > and < log1-pi > in VI iterations
    def get_pi(lmbda,alpha,beta,temp):
        a=(np.sum(lmbda)+alpha)**(1/temp)
        b=(len(lmbda)-np.sum(lmbda)+beta)**(1/temp)
        digamma_a=sp.digamma(a)-sp.digamma(a+b)
        digamma_b=sp.digamma(b)-sp.digamma(a+b)
        return digamma_a, digamma_b, a, b
    
    
    def VB_EM_GP_SS(y,X,l0=0.01,s0=[],sig0=[],lmbda0=1,logpi0=0,log1_pi0=0,v0=1e+4,v1=1e-4,vg=1e-4,a=1e-3,b=1e-3,reg=0.01,learn_spike=False,learn_noise=True,min_VBEM_iter=5,
                            max_VBEM_iter=20,max_GP_fit_iter=100,init_GP_iter = 100, VBEM_tol=[],GP_fit_tol=1e-6,subsample=100,sampling_strat = "nn", nn_fraction = 1, ELBO_sample=1000,
                            optimisation="amsgrad",learn_rate=0.025,Beta=0.9,Beta2=0.99,eps=1e-8,print_GP_fit=False,print_VBEM=True,
                            timer=True,kern=kernel_funcs.gaussian,grad_kern = kernel_funcs.grad_gaussian,ltrue=[],dampen_lmbda_update=0,newsumgrads=False, store_ls = False, 
                            temp = 1, post_fit=False, learn_rate_mult_post=0.1, iter_remove = False, iter_remove_freq = 1, q=[], seed = [], learn_rate_mult = 1, X_mult = 1, force_include = [], final_ELBO_sample = 1, final_prune = False):
        """
        Parameters
        ----------
        y : output vector (N x 1)
        X : input matrix (N x P)
        l0 : initial value of inverse length-scales
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
        init_GP_iter : max_GP_fit_iter for first iter
        VBEM_tol : convergence tolerance for VBEM iterations (|param-param_old|)
        GP_fit_tol : convergence tolerance for inner GP_fit procedure to maximise q(l) (<logp>)
        subsample : # SGD samples to draw
        sampling_strat : "unif" for uniform, "nn" (or any word) for NN
        nn_fraction : Float \in (0,1], fraction of data for stochastic NN search 
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
        temp : temperature of p(theta,gamma,pi)
        post_fit : Boolean, True = fit full posterior covariance in post run
        learn_rate_mult : Multiplier on learning rate for post-fit run
        iter_remove : Boolean, True = iteratively remove variables with low PIP (<0.01) during VB iterations
        X_mult : scaling factor for inputs
        force_include : vector of forced inclusion indicators per variable

        Returns
        -------
        param,s,sig,lmbda,logl,v0
        """
        
        # Setting dimensions and timer (and computing distance matrix if required)
        t=time.time()
        n,p = np.shape(X)
        
        # Rescaling initial inverse lengthscales and X
        Xscale  = X*X_mult
        l0*=1/X_mult
        
        if not q:
            q=p
        # Transforming initial parameter settings to correct dimensionality
        l=l0*np.ones(p)
        L = l.reshape(1,p)
        lmbda=lmbda0*np.ones(p)
        logpi=logpi0
        log1_pi=log1_pi0
        if not s0:
            s = np.var(y)
        else:
            s = s0
        if not sig0:
            sig = np.var(y)**0.5
        else:
            sig=sig0
            
        # Filling in missing parameter settings
        damp = dampen_lmbda_update
        if not VBEM_tol:
            VBEM_tol = 0.1/p
        if n<subsample:
            subsample = n
        if n<ELBO_sample:
            ELBO_sample = n
        if seed:
            np.random.seed(seed)
        
        # Initialising SSGs for ADAM/AMSgrad
        sum_sq_grads_l=np.zeros(p)
        sum_sq_grads_s=0 
        sum_sq_grads_sig=0 
        sum_grads_l=np.zeros(p)
        sum_grads_s=0 
        sum_grads_sig=0 
        
        # Initialisating convergence criteria
        i=0
        Param=np.concatenate((l,lmbda))
        Param_diff_min=10    
        Elbo_new=-1e+7
        Elbo_diff_min = 100
        if max_GP_fit_iter==[]:
            max_GP_fit_iter=np.max(100,n/subsample)
            
        # Initialising selections
        select = np.repeat(True,p)
        latest_select_out = np.repeat(True,p)
        if np.any(force_include):
            lmbda[force_include]=1
        else:
            force_include = np.repeat(False, p)
        
        # Running VBEM iterations
        while (i<max_VBEM_iter and (Param_diff_min>VBEM_tol or Elbo_diff_min > 1)) or i<min_VBEM_iter:
            
            if i==0:
                max_iter = init_GP_iter
            if i==1:
                max_iter = max_GP_fit_iter
                learn_rate *= learn_rate_mult
                
            # Running GP_fit algorithm
            l[select],s,sig,logl,L,sum_sq_grads_l[select], sum_grads_l[select],sum_sq_grads_s, sum_grads_s, sum_sq_grads_sig, sum_grads_sig=fit.GP_fit_SS_lm(
                        y,Xscale[:,select],l0=l[select],s0=s,sig0=sig,reg=reg,subsample=subsample,sampling_strat = sampling_strat, nn_fraction = nn_fraction,learn_noise = learn_noise,
                        tol=GP_fit_tol,optimisation=optimisation,learn_rate=learn_rate,beta=Beta,beta2=Beta2,eps=eps,
                        v_l1=v1,v_l0=v0,v_g=vg,lmbda=lmbda[select],maxiter=max_iter,print_=print_GP_fit,kern=kern,grad_kern=grad_kern,
                        sum_sq_grads_l=sum_sq_grads_l[select], sum_grads_l=sum_grads_l[select],sum_sq_grads_s=sum_sq_grads_s, sum_grads_s=sum_grads_s, 
                        sum_sq_grads_sig=sum_sq_grads_sig, sum_grads_sig=sum_grads_sig, newsumgrads=newsumgrads, store_ls=store_ls, L=L, temp=temp, q=min(q,np.sum(select)), X_mult=X_mult)
    
            # Getting update to lmbda
            lmbda_new=fit.get_lmbda(l[(select+latest_select_out) * (force_include!=True)],v0,v1,logpi,log1_pi, temp)
            lmbda[(select+latest_select_out) * (force_include!=True)] = (1-damp)*lmbda_new+damp*lmbda[(select+latest_select_out) * (force_include!=True)]
    
            # Getting update to pi
            logpi,log1_pi,ahat,bhat=fit.get_pi(lmbda,alpha=a,beta=b, temp=temp)
            
            # Updating active set
            if iter_remove and not i % iter_remove_freq:
                selectnew = (lmbda>=0.01)+force_include
               # L = L[:,selectnew[select]]
                l[selectnew==False]=0
                latest_select_out = select!=selectnew
                select = selectnew
            
            # Optional update to spike (hyperparameter optimisation)
            if learn_spike:
                v0 = np.sum((1-lmbda)+1e-10)/np.sum((1-lmbda)*l**2+1e-10)
    
            # Getting ELBO estimate
            Elbo_old=Elbo_new
            Elbo_new = model_funcs.ELBO(l,s,sig,reg,y,Xscale,lmbda,logpi,log1_pi,v0,v1,ELBO_sample,a,ahat,b,bhat,kern,temp,select,1)
    
            # Tracking convergence criterion
            i+=1
            Param_old = Param
            Param = np.concatenate((l,lmbda))
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
                    print("length-scale norm is :", np.linalg.norm((l**2)**0.5-(ltrue**2)**0.5))
        
        # Getting final elbo
        Elbo_new = 0
        np.random.seed(0)
        for i in range(final_ELBO_sample):
            Elbo_new += model_funcs.ELBO(l,s,sig,reg,y,Xscale,lmbda,logpi,log1_pi,v0,v1,ELBO_sample,a,ahat,b,bhat,kern,temp,select, 1)/final_ELBO_sample
            
        # Pruning at last stage if final_prune
        if final_prune:
            l[lmbda<0.01]=0

        if post_fit:
            # Reducing dataset to selected dimensions and initialising covariance
            select = lmbda>=0.5
            #print(select*1)
            X = X[:,select]
            m = l[select]*X_mult
            assert len(m)==np.sum(select)
            q = np.sum(select)
            print(q)
            L = np.ones((q**2)).reshape(q,q)
            L *= 1/np.sqrt(np.diag(L @ L.T))[0]*1e-2
            M = np.concatenate((np.abs(m),np.sqrt(np.diag(L @ L.T + 1e-10)))).reshape(1,2*q)

            # Initialising SSGs for ADAM/AMSgrad and updating optimisation settings
            sum_sq_grads_m=np.zeros(q)
            sum_sq_grads_L = np.zeros(np.shape(L))
            sum_sq_grads_s=0 
            sum_sq_grads_sig=0 
            sum_grads_m=0 
            sum_grads_s=0 
            sum_grads_L=0
            sum_grads_sig=0
            maxiter=1e+3
            GP_fit_tol=1e-3
            learn_rate *=learn_rate_mult_post/learn_rate_mult

            # Running model
            m,L,s,sig,loglnew,M,sum_sq_grads_m_, sum_grads_m_,sum_sq_grads_L_, sum_grads_L_,sum_sq_grads_s_, sum_grads_s_, sum_sq_grads_sig_, sum_grads_sig_=fit.GP_fit_SS_fullpost_lm(
                    y,Xscale,m0=m,L0=L,s0=s,sig0=sig,reg=reg,subsample=subsample,svi_subsample=1,learn_noise = learn_noise,
                    tol=GP_fit_tol,optimisation=optimisation,learn_rate=learn_rate,beta=Beta,beta2=Beta2,eps=eps,
                    v_l1=v1,v_g=vg,maxiter=maxiter,print_=print_GP_fit,kern=kern,grad_kern=grad_kern,
                    sum_sq_grads_m=sum_sq_grads_m, sum_grads_m=sum_grads_m,sum_sq_grads_L=sum_sq_grads_L, sum_grads_L=sum_grads_L,
                    sum_sq_grads_s=sum_sq_grads_s, sum_grads_s=sum_grads_s, sum_sq_grads_sig=sum_sq_grads_sig, sum_grads_sig=sum_grads_sig, 
                    newsumgrads = newsumgrads, store_ls = store_ls, M=M)   

            # Creating list of local parameters to return
            l_select = np.zeros(p)
            l_select[select] = m
            param = [l_select,L]
        
        else:
            param = [l*X_mult]

        if timer:
            print("run time is :", time.time()-t)

        return param,s,sig,lmbda,v0,[logpi,log1_pi,ahat,bhat],Elbo_new
   
    
    def GP_fit_SS_fullpost_lm(y,X,m0,L0,s0,sig0,reg,subsample,svi_subsample,learn_noise,
                           tol,optimisation,learn_rate,beta,beta2,eps,v_l1,v_g,maxiter,print_,kern,grad_kern,
                            sum_sq_grads_m, sum_grads_m,sum_sq_grads_L, sum_grads_L,sum_sq_grads_s, sum_grads_s, sum_sq_grads_sig, sum_grads_sig, newsumgrads, store_ls, M):
        """
        Parameters
        ----------
        y : output vector (N x 1)
        X : input matrix (N x D)
        m0 : initial inverse-lengthscale mean
        L0 : initial inverse-lengthscale variance (L @ L.T = S)
        s0 : initial scale
        sig0 : initial noise
        reg : initial nugget regularisation
        subsample : #samples to draw in SGD (without replacement)
        svi_subsample : # SVI samples to draw
        learn_noise : True = learn sigma^2
        tol : convergence tolerance in ||param-param_old|| over VI iterations
        optimisation : "adam", "amsgrad", "gd" (sgd version included)
        learn_rate : learning rate
        beta : ADAM retention factor (first moments)
        beta2 : ADAM retention factor (second moments)
        eps : ADAM epsilon imprecision
        v_l1 : slab precision on inverse lengthscales (default = 1e-3)
        v_g :  prior precision on scale and sigma
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
        m,L,s,sigma,logl,subgrads_objs
        """
        n,p = np.shape(X)

        # Initialising hyperparameters
        m=np.ones(p)*m0
        L=L0
        s=s0
        sigma=sig0
        Lmin = np.min(L)
        S = L @ L.T+1e-10*np.diag(np.ones(p))
        step_base = np.diag(S)**0.5/5

        # Initialising optimisation algorithm terms
        if newsumgrads:
            sum_sq_grads_m=np.zeros(p)
            sum_sq_grads_L = np.zeros(np.shape(L))
            sum_sq_grads_s=0 
            sum_sq_grads_sig=0 
            sum_grads_m=0 
            sum_grads_s=0 
            sum_grads_L=0
            sum_grads_sig=0    

        # Initialising logl
        loglnew=-100
        logl=-200

        # Getting initial subsample
        y_sample, X_sample = fit.get_subsample_fullpost(y,X,subsample)

        # Commencing SGD
        t=0
        while abs(loglnew-logl)>tol and t<maxiter:

            t+=1

            # Getting gradients
            grad_logL_m,grad_logL_logsd,grad_logL_s,grad_logL_sig = 0,0,0,0
            for i in range(svi_subsample):
                K, Ktild, l, xi = fit.get_svi_subsample_fullpost(y_sample,X_sample,m,L,s,sigma,reg,kern)
                gradm,gradlogsd,grads,gradsig = fit.get_gradients_gp_ss_fullpost(y_sample,X_sample,K,Ktild,l,xi,m,L,s,sigma,v_l1,v_g,subsample,grad_kern)
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
            m+=step_size_m/np.diag(S)**0.5*step_base
            if Lmin>0:
                L*= np.exp(step_size_L)
            s+=step_size_s
            if learn_noise:
                sigma+=step_size_sig

            # Subsampling for next step
            y_sample, X_sample = fit.get_subsample_fullpost(y,X,subsample)

            # Updating and printing logL
            logl=loglnew
            S = L @ L.T+1e-10*np.diag(np.ones(p))
            C = np.linalg.cholesky(S)
            loglnew=model_funcs.logL(y_sample,Ktild)-0.5*np.sum((v_l1)*(m**2+np.diag(S)))+np.sum(np.log(np.diag(C)))

            if print_:
                if store_ls:
                    if t==1:
                            colors =np.repeat("red", p)
                    if not t % 10:
                        M = np.append(M,np.concatenate((np.abs(m),np.sqrt(np.diag(S)))).reshape(1,2*p), axis=0)
                        clear_output(wait=True)        
                        fig,axs = plt.subplots(2,figsize = (15,15))
                        fig.set_facecolor('black')
                        axs[0].set_facecolor('black')
                        axs[1].set_facecolor('black')
                        for i in np.arange(p)[::-1]:
                            lines0 = axs[0].plot(np.abs(M[:,i]), color = colors[i], linewidth = 1)
                        for i in np.arange(p,2*p)[::-1]:
                            lines1 = axs[1].plot(np.abs(M[:,i]), color = colors[i-p], linewidth = 1)
                        plt.show()
                print(loglnew)

            # Printing if maxiter reahed
            if t==maxiter:
                print("maximum iterations reached, terminating")

        # returning final params and logL 
        return m,L,s,sigma,loglnew,M, sum_sq_grads_m, sum_grads_m,sum_sq_grads_L, sum_grads_L,sum_sq_grads_s, sum_grads_s, sum_sq_grads_sig, sum_grads_sig
    
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
                    fm, fv, ym, yv = testing_algorithm(y=y_train, X=X_train, Xtest = X_test, l=results[0][0], s=results[1], sig=results[2])

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
                    selections = np.sum(Results[i][0][0]!=0)
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
 
    
    def cv_hyperopt(y,X,reg,testing_algorithm,Results,folds, train_largest = True,post_var=True, post_cov=True,seed=[]):
        
        
        """
        Function to do prediction only cross-validation over learned hyperparameter solutions for SS-GP
        Parameters
        __________
        
        y : training outputs
        X : training inputs
        reg : regularisation for kernel matrix (jitter)
        testing_algorithm : algorithm to produce testing predictions
        Results : list of outputs from VBEMGPSS or hyperopt
        folds : number of CV folds to use in testing
        largest_set : "Train" = n_cv_test = n/folds, else n_cv_train = n/folds
        post_cov : True = score using NLPD with covariance
        post_var : True = score using NLPD with diagonal covariance (else MSE)
        Returns
        ______
        Loss : vector of losses as measured by NLPD or MSE (summed_)
        """
        n,p = np.shape(X)
        Loss = np.zeros(len(Results))
        
        # Shuffling the data and splitting into folds
        if seed:
            np.random.seed(seed)
        shuffled_indexes = np.random.choice(n,n,False)
        y_shuffle = y[shuffled_indexes]
        X_shuffle = X[shuffled_indexes]
        n_per_fold = int(n/folds)

        # Looping over spike values
        for i in range(len(Results)):
            select = np.where(Results[i][0][0]!=0)[0].astype(int)
            l = Results[i][0][0][select]
            s = Results[i][1]
            sigma = Results[i][2]

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
                
                if train_largest:
                    X_tr = X_train
                    y_tr = y_train
                    X_te = X_test
                    y_te = y_test
                else:
                    X_tr = X_test
                    y_tr = y_test
                    X_te = X_train
                    y_te = y_train
                
                # Running testing algorithm
                fm, fv, ym, yv = testing_algorithm(y=y_tr, X=X_tr[:,select], Xtest = X_te[:,select], l=l, s=s,sig=sigma, post_var=post_var)

                # Compute predictive distribution or MSE
                if not (post_var or post_cov):
                    Loss[i] += 0.5*np.sum((y_te - ym)**2)
                if post_var and not post_cov:
                    yv = np.diag(yv)
                    Loss[i] += 0.5**np.sum((y_te -ym)**2/yv)+0.5*np.sum(np.log(yv))
                if post_cov:
                    Loss[i] += 0.5**(y_te -ym).T @ np.linalg.solve(yv,y_te-ym)+np.sum(np.log(np.diag(np.linalg.cholesky(yv))))
            
            # Correcting loss for number of times each point appears in test set
            if not train_largest:
                Loss *= 1/(folds-1)
                    
        return Loss

class diagnostics:
    
    def plot_length_scales_pip(ltrue,l,lmbda,plotsize=(20,10),width=[]):
        
        # Setting up objects
        p=len(l)
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
        
    def get_ls(mu,lmbda):
        p = len(mu)
        gamma = np.random.random(p)<lmbda
        return (mu*gamma).reshape(p,)
        
    def get_pred_posterior_GP(y,X,Xtest,l,s,sig,reg,kern, post_var=False, latents = True):
        fpost_mean, fpost_var, ypost_mean, ypost_var = 0,0,0,0
        K=kernel_funcs.ARD_lm(l=l,s=s,X=X,kern=kern)
        Ktest=kernel_funcs.ARD_lmtest(l=l,s=s,X=X,Xtest=Xtest,kern=kern)
        if post_var:
            Ktesttest=kernel_funcs.ARD_lm(l=l,s=s,X=Xtest,kern=kern)
        Ktild=K+np.diag(np.ones(len(K)))*(reg+sig**2) 
        ypost_mean=Ktest @ np.linalg.solve(Ktild,y)
        if latents:
            fpost_mean=Ktest @ np.linalg.solve(K+np.diag(np.ones(len(K)))*reg,y)
        if post_var:
            ypost_var=Ktesttest-Ktest @ np.linalg.solve(Ktild,Ktest.T)+np.diag(np.ones(len(Ktesttest)))*(reg+sig**2)
            if latents:
                fpost_var=Ktesttest-Ktest @ np.linalg.solve(K+np.diag(np.ones(len(K)))*reg,Ktest.T)+np.diag(np.ones(len(Ktesttest)))*reg
        
        return fpost_mean, fpost_var, ypost_mean, ypost_var
    
    def get_pred_posterior_GP_NN(y,X,Xtest,l,s,sig,reg,kern,grad_kern,select=[],NN=100,pred_selected = False, fraction=1, grad_steps=0, learn_rate=0.1, tol=1e-4,
                                 print_grad=False,post_var=False, latents = True, print_=True):
    
        # Setting up storage objects
        m,p = np.shape(Xtest)
        
        if not np.any(select):
            select = np.repeat(True,p)
            
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
            if post_var:
                Ktesttest=kernel_funcs.ARD_lm(l=lpred,s=s,X=Xtesti.reshape(1,len(Xtesti)),kern=kern)
            Ktild=K+np.diag(np.ones(len(K)))*(reg+sig**2) 

            KtestK = np.linalg.solve(K+np.diag(np.ones(len(K)))*reg, Ktest.T).T
            KtestKtild = np.linalg.solve(Ktild, Ktest.T).T
            if latents:
                fpost_mean[i] = KtestK @ yi
            ypost_mean[i] = KtestKtild @ yi
            if post_var:
                if latents:
                    fpost_var[i,i] = Ktesttest-KtestK @ Ktest.T
                ypost_var[i,i] = Ktesttest- KtestKtild @ Ktest.T
            if print_:
                if not round(i/m*100,2) % 10:
                    print(i/m*100, "% complete")


        return fpost_mean, fpost_var, ypost_mean, ypost_var
    
    def get_pred_posterior_GP_NN_CV(y,X,results,reg,kern,NN=100, fraction=1,post_var=False, print_=True, use_tree=False, leaf_size=100, seed=[]):
    
        if seed:
            np.random.seed(seed)
            
        # Set up
        n,p = np.shape(X) 
        logpredictive = 0
        
        # Getting relevant dimensions and model parameters
        select = results[0][0]!=0
        if np.max(select)==0:
            select[np.random.choice(p,1,False)]=True
            results[0][0][select]=1e-3
        q = np.sum(select)
        Xselect = X[:,select].reshape(n,q)
        lselect = results[0][0][select]
        s = results[1]
        sig = results[2]
        
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

        return logpredictive-n/2*np.log(2*np.pi)
    
    def get_KL(results,v0,v1,a,b):
        
        # getting objects
        l = results[0][0]
        lmbda = results[3]
        qpi= results[len(results)-2]
        logpi,log1_pi,ahat,bhat = qpi[0],qpi[1],qpi[2],qpi[3]
        
        # Regularising probabilities
        lmbda[lmbda<1e-10]=1e-10
        lmbda[lmbda>(1-1e-10)]=1-1e-10
        
        neg_kl_theta = -0.5*np.sum((v1*lmbda+v0*(1-lmbda))*l**2)+0.5*np.sum(1-lmbda)*np.log(v0)+0.5*np.sum(lmbda)*np.log(v1)
        neg_kl_gamma = np.sum(lmbda*(logpi-np.log(lmbda))+(1-lmbda)*(log1_pi-np.log(1-lmbda)))
        neg_kl_pi = (a-ahat)*logpi + (b-bhat)*log1_pi+sp.gammaln(ahat)+sp.gammaln(bhat)-sp.gammaln(ahat+bhat)
        
        return neg_kl_theta+neg_kl_gamma+neg_kl_pi
    
    def get_BMA_predictions(y,X,Xtest,testing_algorithm, Results,weights, MC_samples):
        
        # Getting MC_samples to discretise weights
        sampled_weights = np.random.multinomial(MC_samples,weights, 1)[0]/MC_samples
        
        # Getting preditions per model
        fmean,ymean = 0,0
        
        for i in range(len(weights)):
            
            if sampled_weights[i]>0:
            
                l = Results[i][0][0]
                s = Results[i][1]
                sig = Results[i][2]
                select = l!=0

                fm, fv, ym, yv = testing_algorithm(y=y, X=X[:,select], Xtest = Xtest[:,select], l=l[select], s=s,sig=sig, post_var=False)

                fmean += sampled_weights[i]*fm
                ymean += sampled_weights[i]*ym
        
        return fmean, ymean

            
    
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

    def do_simulation_VBEMSSGP(y, X, ftest, ytest, Xtest, q, algorithm_training, algorithm_testing, nmodels, args, arg_vals, SS_GP, order_relevant_vars, order_irrelevant_vars, VS_threshs, predict_selected, select, hyper_opt=[], hyper_arg=[], hyper_vals=[],ltrue=[], MC_pred = False, MC_pred_draws = 10, post_fit=False, model_select = [], post_fit_subsample=[], train=[], model_weighting=[], post_var = []):
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
        hyper_opt : Boolean, True = do hyperparameter optimisation using fit.hyperopt()
        hyper_arg : argument to be tuned as hyperparameter
        hyper_vals : list of hyperparameter vals
        ltrue: Float64 lengthscales if true GP is used to draw data (d)
        SS_GP :  Boolean vector , True =  model is a Spike-Slab GP
        order_relevant_vars : Boolean , True = order lenthscales and PIP for relevant variables
        order_irrrelevant_vars : Boolean , True = order lenthscales and PIP for irrelevant variables
        VS_threshs : List of Float64 vector of variable selection thresholds to use for each model
        select : Boolean vector , True = relevant generating variable
        predict_selected : Boolean list, True = predict with selected variables only (lowest threshold)
        MC_pred : Boolean list , True = draw from posterior to predict
        MC_pred_draws : # samples to draw from posterior to predict if MC_test = True
        post_fit : Boolean, True = fit full posterior over ils for selected dimensions
        model_select : Boolean, True = select best model out of hyperargs, else do model averaging
        Returns
        -------
        Runtime, Lambda, L, L1norm, L2norm, MSE_F, MSE_Y, Acc, Weighted_Acc, TPR, TNR, PPV, NPV, AUC
        """
        n,p = np.shape(X)
        m = nmodels
        t = len(VS_threshs[0])
        
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
        L = np.zeros((m, p))
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
                
                # Hyperopt
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
                        logevidences[i] =  diagnostics.get_pred_posterior_GP_NN_CV(y,X,Results[i],0.01,kern=kernel_funcs.gaussian,NN=n, fraction=1,post_var=True, print_=False, use_tree=False, leaf_size=100, seed=0)
                max_logevidence = np.max(logevidences)

                # Do weighting
                for i in range(len(Results)):
                    logevidence = logevidences[i]
                    if logevidence >= max_logevidence-500:
                        if model_select[j]:
                            weights[i] = (logevidence==max_logevidence)*1
                        else:
                            weights[i] = np.exp(logevidence-max_logevidence)
                    PIP[i] = Results[i][3]
                    Ls[i] = np.abs(Results[i][0][0])
                    Ss[i] = Results[i][1]
                    Sigs[i] = Results[i][2]
                weights = weights/weights.sum()
                l = [Ls.T @ weights]
                s = np.sum(Ss*weights)
                sig = np.sum(Sigs*weights)
                lmbda = PIP.T @ weights
                Runtime[j] += time.time()-start
            
            # Optional post-fit run
            if post_fit[j]:
                # Updating post_fit args and function_args
                post_fit_args = ["init_GP_iter", "min_VBEM_iter", "max_VBEM_iter", "l0", "s0", "sig0", "lmbda0", "subsample", "post_fit"]
                post_fit_argvals = [0, 0, 0, l[0], s, sig, lmbda, post_fit_subsample, True]
                function_args = [y,X]+list(master_arg_defaults)
                for i in range(len(args[j])):
                               index = np.where(args[j][i]==np.array(master_args))[0][0]
                               function_args[index] = arg_vals[j][i]
                for i in range(len(post_fit_args)):
                               index = np.where(post_fit_args[i]==np.array(master_args))[0][0]
                               function_args[index] = post_fit_argvals[i]
                function_args = tuple(function_args)
                
                # Running algorithm for post-fit
                start = time.time()
                l, s, sig, lmbda, v0, elbo =  algorithm_training(*function_args)
                Runtime[j] += time.time()-start
            
            # Taking lengthscales out of list object
            if len(l)>1:
                Sig = l[1]
            else:
                Sig = 0
            l = l[0]

            # Storing length-scales and PIP:
            if order_relevant_vars:
                lmbda_q = sorted(lmbda[:q], reverse = True)
                l_q = sorted(np.abs(l)[:q], reverse = True)
            else:
                lmbda_q = lmbda[:q]
                l_q = np.abs(l)[:q]
            if order_irrelevant_vars:
                lmbda_pq = sorted(lmbda[q:], reverse = True)
                l_pq = sorted(np.abs(l)[q:], reverse = True)
            else:
                lmbda_pq = lmbda[q:]
                l_pq = np.abs(l)[q:]
            Lambda[j,:] =  np.concatenate((lmbda_q,lmbda_pq))
            L[j,:] =  np.concatenate((l_q, l_pq))
            

            # Computing MSE
            start = time.time()
            if model_select[j] or post_fit[j]:
                if MC_pred[j]:
                    fpost_mean, ypost_mean = 0,0
                    for i in range(MC_pred_draws):
                        if np.all(Sig==0):
                            select_ = np.repeat(True,p)
                            lpred = diagnostics.get_ls(l, lmbda)

                        else:
                            select_ = np.abs(l)>0
                            q = np.sum(select_)
                            mu = diagnostics.get_ls(l[select_], lmbda[select_])
                            lpred = np.random.multivariate_normal(mu,Sig @ Sig.T,1).reshape(q,)
                        fm, fv, ym, yv = algorithm_testing(y=y,X=X[:,select_],Xtest = Xtest[:,select_], l=lpred,s=s,sig=sig)
                        fpost_mean+=fm/MC_pred_draws
                        ypost_mean+=ym/MC_pred_draws
                else:
                    if predict_selected[j]:
                        select_ = lmbda>=VS_threshs[j][0]
                    else:
                        select_ = np.repeat(True,p)
                    fpost_mean, fpost_var, ypost_mean, ypost_var = algorithm_testing(y=y,X=X[:,select_], Xtest = Xtest[:,select_], l=l[select_],s=s,sig=sig)
            
            else:
                # ONLY DO THIS IF MODEL AVERAGING and no post-fit
                fpost_mean, ypost_mean = diagnostics.get_BMA_predictions(y,X,Xtest,algorithm_testing, Results,weights, MC_samples=1000)
                        
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
                thresh_var = np.abs(l)

            if len(np.unique(thresh_var))>1:
                AUC[j] = roc_auc_score(select,thresh_var)

            for i in range(len(VS_threshs[j])):
                GP_select=(thresh_var>=VS_threshs[j][i])*1
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
        return Runtime, Lambda, L, L1norm, L2norm, MSE_F, MSE_Y, Acc, Weighted_Acc, TPR, TNR, PPV, NPV, AUC, MCC
        
    def simulator(nruns, runstart,n,ntest, p, q, corr, r2, sigma2, sigma2x, ltrue, strue, kern_draw, parametric, scale_var_parametric, plot_YX, dupe,
                 algorithm_training, algorithm_testing, nmodels, args, arg_vals, post_fit_GP, SS_GP, order_relevant_vars, order_irrelevant_vars, VS_threshs, predict_selected, print_=True):

        m = nmodels
        t = len(VS_threshs[0])
        
        # Storage objects
        Runtime=np.zeros((nruns, m))
        Lambda = np.zeros((nruns, m, p))
        L = np.zeros((nruns, m, p))
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
                lselect=[]
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


            Runtime[run], Lambda[run], L[run], L1norm[run], L2norm[run], MSE_F[run], MSE_Y[run], Acc[run], Weighted_Acc[run], TPR[run], TNR[run], PPV[run], NPV[run], AUC[run] = simulations.do_simulation_VBEMSSGP(
                                       y, X, ftest, ytest, Xtest, q, algorithm_training = algorithm_training, algorithm_testing = algorithm_testing, 
                                       nmodels = nmodels, args = args, arg_vals = arg_vals, post_fit_GP = post_fit_GP, SS_GP = SS_GP, 
                                       order_relevant_vars = order_relevant_vars, order_irrelevant_vars = order_irrelevant_vars, VS_threshs = VS_threshs, predict_selected = predict_selected, select = select, ltrue=lselect)
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
        
        Names = ["Runtime", "Lambda", "L", "L1norm", "L2norm", "MSE_F", "MSE_Y", "Acc", "Weighted_Acc", "TPR", "TNR", "PPV", "NPV", "AUC"]
        
        return Names, Runtime, Lambda, L, L1norm, L2norm, MSE_F, MSE_Y, Acc, Weighted_Acc, TPR, TNR, PPV, NPV, AUC
