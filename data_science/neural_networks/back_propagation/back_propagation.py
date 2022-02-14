# Imports ----------------------------------------------------------------------
import numpy as np


# Objects ----------------------------------------------------------------------
class neural_network:
    def __init__(self,input_size,layer_sizes):
        self.input_size = input_size
        self.weights = []
        self.gradients = []
        self.delta = None
        for output_size in layer_sizes:
            self.weights.append(np.random.rand(input_size+1,output_size))
            self.gradients.append(0)
            input_size = output_size
    
    def forward(self,X):
        H = X.reshape(-1,self.input_size)
        for i, w in enumerate(self.weights):
            H = np.concatenate((H,np.ones(H.shape[0]).reshape(-1,1)),axis=1)
            self.gradients[i] = H[:]
            H = np.dot(H,w)
            # activation
        p = self.activation(H)
        self.delta = p-p**2
        return p

    def activation(self,X):
        return 1/(1+np.exp(-X))

    def zero_weights(self):
        self.gradients = []
        for w in self.weights:
            self.gradients.append(0)

    def __call__(self,*args,**kwargs):
        return self.forward(*args,**kwargs)

    def backprop(self, mae, learning_rate=0.001):
        # TODO # print('"loss"{}:'.format(mae.shape))
        # TODO # print(mae)
        # TODO # print('Stored Delta {}:'.format(self.delta.shape))
        # TODO # print(self.delta)
        self.delta *= 2*(mae)
        # TODO # print('Delta {}:'.format(self.delta.shape))
        # TODO # print(self.delta)
        for i, w in list(enumerate(self.weights))[::-1]:
            # TODO # print('i:',i,'| w:',w.shape,'grad:',self.gradients[i].shape)
            grad = self.gradients[i].reshape(*self.gradients[i].shape,1)
            # TODO # print('grad:',grad)
            grad = np.repeat(grad,w.shape[1],axis=2)
            # TODO # print('grad {}:'.format(grad.shape))
            # TODO # print(grad)
            update = np.repeat(self.delta.reshape(self.delta.shape[0],1,self.delta.shape[1]),w.shape[0],axis=1)
            update *= learning_rate*grad
            # TODO # print('update {}:'.format(update.shape))
            # TODO # print(update)
            # TODO # print(update.sum(0))
            w += update.sum(0)
            # self.delta = (self.delta*w*np.repeat(activation_derivative,w.shape[1],axis=0)) #TODO: calc activation; figure out repeat shape (might need to rearrange due to bias not being an activation)
            # print('Delta:',self.delta.shape)
            # print(self.delta)
            # # Delta needs to be reshaped from (2,1) to (1,2), repeated to (4,2), mul, then summed to (4,1)
            # print('Update:',np.repeat(self.gradients[i].sum(axis=0).reshape(1,4),4,axis=0).shape)
        pass

def train(X,Y, epochs=1, learning_rate=0.01):
    model = neural_network(2,[3,1])
    loss = Y-model(X)


# Main -------------------------------------------------------------------------
if __name__ == '__main__':
    model = neural_network(3,[2])
    model.weights[0] = np.array([[ 10, -5],
                                 [ -2,  1],
                                 [0.5,  0],
                                 [ -2, 20],
                                ],dtype=float)

    # ==========================================================================
    # x = np.array([5.4,4.5,-85])
    # y = np.array([0,1])
    #
    # x = np.array([5.7,9.5,-75])
    # y = np.array([1,0])
    # x = np.array([[5.4,4.5,-85],
    #               [5.7,9.5,-75],
    #              ])
    # y = np.array([[0,1],
    #               [1,0],
    #              ])
    print('\n')
    print(''.join(['=']*30))
    x = np.array([[5.4,4.5,-85],
                  [5.7,9.5,-75],
                 ])
    y = np.array([[0,1],
                  [1,0],
                 ])
    p = model(x)
    print('Initial Pred:')
    print(p)
    print(''.join(['-']*30))
    for _ in range(3000):
        p = model(x)
        print(((y-p)**2).sum())
        model.backprop(y-p,0.001)
    print(''.join(['-']*30))
    # print('X:',x)
    # print('p:',p)
    # print('"loss":',((y-p)**2).sum())
    # model.backprop(y-p)
    print('Final Pred:')
    print(model(x))
    print('Actual:')
    print(y)


    # # ==========================================================================
    # print('\n')
    # print(''.join(['=']*30))


    # x = np.array([[1,2,3],
    #               [1,0,0],
    #              ])
    # y = np.array([[0,1],
    #               [1,0],
    #              ])
    # p = model(x)

    # print('X:',x)
    # print('p:',p)
    # print('"loss":',p-y)
    # model.backprop(p-y)
