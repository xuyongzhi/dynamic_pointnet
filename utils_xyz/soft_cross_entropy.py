# created in 9/1/2018, Ben
import numpy as np
from sklearn.metrics import log_loss
# import tensorflow as tf

def softmax(x):
    """
    x: num_sample * num_classes
    """
    exp_x = np.exp(x - np.max(x, axis=1, keepdims = True))
    return exp_x / np.sum(exp_x, axis = 1, keepdims = True)

def softmaxloss(predictions,labels):
    """
    predictions: num_sample * num_classes
    labels:  num_sample * 1
    """
    assert predictions.shape[0] == labels.shape[0]
    labels = labels.astype(int)
    m = predictions.shape[0]
    p = softmax(predictions)
    log_likelihood = -np.log( p[range(m), labels])
    loss = np.sum(log_likelihood) / m

    return loss

if __name__=='__main__':
    a = np.array([[0.2,0.4,0.3,0.1,0.9,0.8,0.7,0.6]])
    b = 1.0 - a
    X = (np.concatenate((a,b), axis=0))
    X = np.transpose(X)
    y = np.array([0,0,0,0,1,1,1,1])  # only one bracket
    # y = np.transpose(y)
    loss = softmaxloss(X,y)
    loss_2 = log_loss(y, X)
    X = tf.convert_to_tensor(X)
    y = tf.convert_to_tensor(y)
    loss_3 = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=y, logits=X))
    sess = tf.Session()
    print(sess.run(loss_3))
