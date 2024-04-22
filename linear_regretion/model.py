import numpy as np
class Linearregretion:
    def __init__(self) :
       self.theta=theta


    def linear_function(self,X):
        assert X.ndim > 1
        assert self.theta.ndim > 1
        assert X.shape[1]==self.theta.shape[0],f"the number of columns of X:{X.shape[1]} is different from the number of rows of theta{self.theta.shape[0]}"
        return np.dot(X,self.theta) 
        
    def initialize_theta(D):
         return np.zeros((D,1))



  
    def batch_gradient(self,X, y):
     return -2 * (X.T @ (y - self.linear_function(X, self.theta)))
       

    def update_function(self, grads, step_size):

     return self.theta - step_size*grads

    


    def train_batch_gradient_descent(X, y, num_epochs=100):
        N, D = X.shape
        self.theta = self.initialize_theta(D)
        losses = []
        for epoch in range(num_epochs): # Do some iterations
            ypred = self.linear_function(X,)# make predictions with current parameters
            loss = mean_squared_error(y, ypred)# Compute mean squared error
            grads = self.batch_gradient(X, y,)# compute gradients of loss wrt parameters
            self.update_function(grads)
            losses.append(loss)
            print(f"\nEpoch {epoch}, loss {loss}")
            
    
 
 
 
  
  