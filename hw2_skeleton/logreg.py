'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton
'''

import numpy as np

class LogisticRegression:

    def __init__(self, alpha = 0.01, regLambda=0.01, epsilon=0.0001, maxNumIters = 10000):
        '''
        Constructor
        '''
        self.alpha = alpha
        self.regLambda = regLambda
        self.epsilon = epsilon
        self.maxNumIters = maxNumIters
        self.theta = None

    def computeCost(self, theta, X, y, regLambda):
        '''
        Computes the objective function
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            a scalar value of the cost  ** make certain you're not returning a 1 x 1 matrix! **
        '''
        n,d = X.shape
        cost = (-y.T * np.log(self.sigmoid(X * theta)) - (1.0 - y).T * 
            np.log(1.0 - self.sigmoid(X * theta)))/n + regLambda/(2.0 * n) * (theta.T * theta)
        return cost.item((0,0))
    
    def computeGradient(self, theta, X, y, regLambda):
        '''
        Computes the gradient of the objective function
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            the gradient, an d-dimensional vector
        '''
        n, d = X.shape
        gradient = (X.T * (self.sigmoid(X * theta) - y) + regLambda*theta) / n
        # l,f = theta.shape
        # print "n = %d" % (n)
        # print "d = %d" % (d)
        # print "l = %d" % (l)
        # print "f = %d" % (f)

        # don't regularize the theta_0 parameter
        gradient[0] = sum(self.sigmoid(X * theta) - y) / n
        return gradient

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
        '''

        a,b = X.shape
        # add the 1's features
        X = np.c_[np.ones((a,1)), X]

        # create a random starting theta
        self.theta = np.mat(np.random.rand(b + 1,1))

        theta_old = self.theta
        theta_new = self.theta
        
        i = 0
        while i < self.maxNumIters:
            theta_new = theta_old - self.alpha * self.computeGradient(theta_new, X, y, self.regLambda)
            if self.hasConverged(theta_new, theta_old):
                self.theta = theta_new
                return
            else: 
                theta_old = np.copy(theta_new)
                i = i + 1
                cost = self.computeCost(theta_new, X, y, self.regLambda)
                # print "cost: ", cost

        self.theta = theta_new

    def hasConverged(self, theta_new, theta_old):
        if np.linalg.norm(theta_new - theta_old) < self.epsilon:
            return True
        else:
            return False


    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy matrix
        Returns:
            an n-dimensional numpy vector of the predictions
        '''
        a,b = X.shape
        # add the 1's features
        X = np.c_[np.ones((a,1)), X]
        # use the sigmoid method to predict the values for X
        return np.array(self.sigmoid(X * self.theta))

    def sigmoid(self, z):
        '''
        This method wasn't provided in the hw template...
        Computes sigmoid for both vectors and matrices
        '''

        # test to verify this works for matrices AND vectors
        # test_z_matrix = np.matrix('1 2; 3 4')
        # test_z_vector = np.array([2,3,1,0])
        # test1 = 1.0 / (1.0 + np.exp(-test_z_vector))
        # test2 = 1.0 / (1.0 + np.exp(-test_z_matrix))
        # print test1
        # print test2

        return 1.0 / (1.0 + np.exp(-z))