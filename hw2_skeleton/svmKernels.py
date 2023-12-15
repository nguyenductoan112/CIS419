import numpy as np


_polyDegree = 2
_gaussSigma = 1


def myPolynomialKernel(X1, X2):
    '''
        Arguments:  
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    ''' 
    d = X1.shape[1]
    assert X2.shape[1] == d

    X1_X2T = np.dot(X1, X2.T)
    f = np.vectorize(lambda x: (x+1)**d)
    return f(X1_X2T)

    return #TODO

def norm(X1, X2):
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    kernel = np.zeros((n1, n2))
    # norms_ = np.zeros((n1, n2))
    for i in range(n1):
        # difference = np.array([X1[i, :],]*n2) - X2
        # sq = np.vectorize(lambda x: x**2)
        # norms_[i, :] = np.sum(sq(difference))
        for j in range(n2):
            difference = X1[i, :] - X2[j, :]
            sq = np.vectorize(lambda x: x**2)
            kernel[i, j] = np.sum(sq(difference))
    return kernel

def myGaussianKernel(X1, X2):
    '''
        Arguments:
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''
    gaussian = np.vectorize(lambda x: exp(-x/(2*_gaussSigma**2)))
    return gaussian(norm(X1, X2))

 



def myCosineSimilarityKernel(X1,X2):
    '''
        Arguments:
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    kernel = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            norm_xi, norm_xj = (np.linalg.norm(v) for v in (X1[i], X2[j]))
            XiT_Xj= np.dot(X1[i], X2[j])
            kernel[i, j] = XiT_Xj/(norm_xi*norm_xj)
    return kernel #TODO (CIS 519 ONLY)

