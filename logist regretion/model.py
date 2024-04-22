import numpy as np
from accuracy import cross_entropy_loss

class LogisticRegression:

  def __init__(self,lr,n_epochs):
    self.lr = lr
    self.n_epochs = n_epochs
    self.train_losses = []
    self.w = None
    self.weight = []

  def add_ones(self, x):

    ##### WRITE YOUR CODE HERE #####
    return np.hstack((np.ones((x.shape[0],1)),x))
    #### END CODE ####

  def sigmoid(self, x):
    ##### WRITE YOUR CODE HERE ####
    z = x @ self.w
    return 1/(1+np.exp(-z))
    #### END CODE ####

  def predict_proba(self,x):  #This function will use the sigmoid function to compute the probalities
    ##### WRITE YOUR CODE HERE #####
    proba = self.sigmoid(x)
    return proba
    #### END CODE ####

  def predict(self,x):
    ##### WRITE YOUR CODE HERE #####
    probas = self.predict_proba(self.add_ones(x))
    output = (probas >= 0.5).astype(int) #convert the probalities into 0 and 1 by using a threshold=0.5
    return output
    #### END CODE ####

  def fit(self,x,y):

    # Add ones to x
    x=self.add_ones(x)

    # reshape y if needed
    y=y.reshape(-1,1)

    # Initialize w to zeros vector >>> (x.shape[1])
    self.w=np.zeros((x.shape[1],1))

    for epoch in range(self.n_epochs):
      # make predictions
      y_predict=self.sigmoid(x)

      #compute the gradient
      grads=-(1/x.shape[0])*( x.T @ (y-y_predict))

      #update rule
      self.w -= self.lr*grads

      #Compute and append the training loss in a list
      loss = self.cross_entropy(x,y)
      self.train_losses.append(loss)

      if epoch%1000 == 0:
        print(f'loss for epoch {epoch}  : {loss}')

  