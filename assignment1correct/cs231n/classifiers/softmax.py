import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  softmax = lambda x: np.exp(x-np.max(x))/np.sum(np.exp(x-np.max(x)))
  num_classes = W.shape[1]
  num_train   = X.shape[0]
  print 
  for i in xrange(num_train):
    score = X[i].dot(W)
    numerator = np.exp(score[y[i]])
    denom     = np.sum(np.exp(score))
    loss += -np.log(numerator/denom)
    m = np.max(score)
    v = score - m

    for j in xrange(num_classes):

      if j == y[i]:
        dW[:,j] +=  X[i].T * (np.exp(v[j])/np.sum(np.exp(v)) - 1)
      else:
        dW[:,j] += X[i].T * (np.exp(v[j])/np.sum(np.exp(v)))
  loss /= num_train
  dW /= num_train
  loss += 0.5 * reg * np.sum(W*W)
  dW   += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  num_train = X.shape[0]

  scores = X.dot(W.T)   #apply weight matrix to the inputs

  numerator = np.exp(scores[np.arange(num_train),y])
  denom     = np.sum(np.exp(scores[np.arange(num_train),:]),axis=1)
  loss += np.sum(-np.log(numerator/denom))/num_train

  colmxs = np.max(scores,axis=1)
  colmxs = colmxs[...,np.newaxis]
  stablescores = scores - colmxs
  softscores = np.exp(stablescores)/np.sum(np.exp(stablescores),axis=1)[...,np.newaxis]
  softscores[np.arange(num_train),y] += -1
  
  dW = softscores.T.dot( X )/num_train

                     
  loss += 0.5 * reg * np.sum(W*W)

  dW   += reg * W
  #############################################################################
  #                          End OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

