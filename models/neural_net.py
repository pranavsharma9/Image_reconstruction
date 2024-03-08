"""Neural network model."""

from typing import Sequence

import numpy as np


class NeuralNetwork:

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
        opt: str,
    ):
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers

        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]
        print(sizes)
        self.params = {}
        self.t = 0
        self.m = {}
        self.v = {}
        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(sizes[i - 1], sizes[i]) / np.sqrt(sizes[i - 1])
            self.params["b" + str(i)] = np.zeros(sizes[i])

            self.m["W" + str(i)] = np.zeros(self.params["W" + str(i)].shape)
            self.v["W" + str(i)] = np.zeros(self.params["W" + str(i)].shape)
            self.m["b" + str(i)] = np.zeros(self.params["b" + str(i)].shape)
            self.v["b" + str(i)] = np.zeros(self.params["b" + str(i)].shape)

            
    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        
        return np.matmul(X,W)+b
    
    def linear_grad(self, W: np.ndarray, X: np.ndarray, b: np.ndarray, de_dz: np.ndarray, reg, N) -> np.ndarray:
        
        de_dw=np.matmul(X.T,de_dz)/N+reg*W/N
        de_dx=np.matmul(de_dz,W.T)/N
        de_db=np.sum(de_dz, keepdims=True, axis=0)/N

        return de_dw, de_db, de_dx

    def relu(self, X: np.ndarray) -> np.ndarray:
        

        a=np.maximum(0,X)
        return a

    def relu_grad(self, X: np.ndarray) -> np.ndarray:
        
        a = np.where(X>0,1,0)
        # print(a)
        return a

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        
        sig = np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

        return sig

    def sigmoid_grad(self, X: np.ndarray) -> np.ndarray:
        
        ans = np.multiply(X,(1-X))
        return ans

    def mse(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        
        return np.mean(np.square(y-p))
    
    def mse_grad(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        
        return 2.0*(p-y)/y.size
        
    
    def mse_sigmoid_grad(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        
        sig_grad = self.sigmoid_grad(p)
        mse_grad = self.mse_grad(y, p)
        return np.multiply(mse_grad,sig_grad)

    def forward(self, X: np.ndarray) -> np.ndarray:
        
        self.outputs = {}
        self.outputs["layer_0"]=X
        for i in range(1,self.num_layers):
          W_i=self.params["W" + str(i)]
          b_i=self.params["b" + str(i)]
          y_i=self.linear(W_i,X,b_i)
          self.outputs["layer_h_"+str(i)]=y_i
          X=self.relu(y_i)
          self.outputs["layer_"+str(i)]=X

        W_lst=self.params["W"+str(self.num_layers)]
        b_lst=self.params["b"+str(self.num_layers)]
        y_lst=self.linear(W_lst,X,b_lst)
        
        X=self.sigmoid(y_lst)
        self.outputs["layer_"+str(self.num_layers)]=X
        # print(X.shape)

        return X

    def backward(self, y: np.ndarray) -> float:
        
        self.gradients = {}
     
        N=y.shape[0]
        
        
        de_dz=self.mse_sigmoid_grad(y,self.outputs["layer_" + str(self.num_layers)])
        loss = self.mse(y, self.outputs["layer_" + str(self.num_layers)])
        # print(de_dz)
        for i in range(self.num_layers, 0, -1):
            W = self.params["W" + str(i)]
            b = self.params["b" + str(i)]
            
            X = self.outputs["layer_" + str(i - 1)]  
  
            if i != self.num_layers:  
                de_dz = np.multiply(de_dx, self.relu_grad(self.outputs["layer_"+str(i)]))
                de_dw, de_db, de_dx = self.linear_grad(W, X, b, de_dz, .0001, 1)
            else:  
              # if i!=1:
              de_dw, de_db, de_dx = self.linear_grad(W, X, b, de_dz, .0001, 1)

            
            self.gradients["W" + str(i)] = de_dw
            self.gradients["b" + str(i)] = de_db
            # print(de_dw)
            # print(de_db)
        # print(self.gradients["W1"])
        return loss
        
    def update(
        self,
        lr: float = 0.001,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        opt: str = 'Adam'
    ):
        

        if opt == "SGD":
          for i in range(1, self.num_layers + 1):
            weight_decay=0.0001
            lr_new = lr / (1 + weight_decay * i)
            self.params["W" + str(i)] -= lr_new * self.gradients["W" + str(i)]
            self.params["b" + str(i)] -= lr_new * np.squeeze(self.gradients["b" + str(i)])
        elif opt == "Adam":
            self.t += 1
            for param_key in self.gradients.keys():
                self.adam(lr, b1, b2, eps, param_key)
        return
    
    def adam(self, lr: float, b1: float, b2: float, eps: float, param_key: str):
      grad = self.gradients[param_key]
#       weight_decay=0.0001
      self.m[param_key] = b1 * self.m[param_key] + (1 - b1) * grad
      self.v[param_key] = b2 * self.v[param_key] + (1 - b2) * grad ** 2
      m_hat = self.m[param_key] / (1 - b1 ** self.t)
      v_hat = self.v[param_key] / (1 - b2 ** self.t)

      self.params[param_key] -= lr * np.squeeze(m_hat / (np.sqrt(v_hat) + eps))