import pickle
import tensorflow as tf
import numpy as np


class Model:
    pass

def load_model(network, sess, filters, kernels, strides, paddings, act=tf.nn.relu):
    with open('networks/' + network + '.pkl', 'rb') as file:
        param_vals = pickle.load(file)
        
    model = Model()

    def predict(inputs,act_fn=act):
        x = inputs
        layers = [x]
        #Define network
        for i, (l,k,s,p) in enumerate(zip(filters,kernels,strides,paddings)):
            if type(s) is str: #Residual
                s = int(s[1:])
                W,b = param_vals[i]
                x = tf.nn.conv2d(act_fn(x),W,[1,s,s,1],p)+b

                if x.shape != layers[-2].shape:
                    last_x = layers[-2]
                    scale = int(last_x.shape[1])//int(x.shape[1])
                    if scale != 1:
                        last_x = tf.nn.avg_pool(last_x, [1,scale,scale,1],[1,scale,scale,1], 'VALID')
                    last_x = tf.pad(last_x, [[0, 0], [0, 0], [0, 0],
                                 [int(x.shape[3]-last_x.shape[3])//2, int(x.shape[3]-last_x.shape[3])//2]])
                    x += last_x
                else:
                    x += layers[-2]
                layers.append(x)
            elif l == 'pool':
                x = tf.nn.max_pool(x,ksize = [1,k,k,1],
                                strides=[1,s,s,1],padding=p)
                layers.append(x)
            else: #Conv
                W,b = param_vals[i]
                if i == 0:
                    x = tf.nn.conv2d(x,W,[1,s,s,1],p)+b
                else:
                    x = tf.nn.conv2d(act_fn(x),W,[1,s,s,1],p)+b
                layers.append(x)
        logits = tf.layers.flatten(x)
        
        return logits

    def ibp(L,U,act_fn=act):
        layers = [(L,U)]
        #Define base network
        for i, (l,k,s,p) in enumerate(zip(filters,kernels,strides,paddings)):
            if i == len(filters) - 1:
                break
            if l == 'pool':
                U = tf.nn.max_pool(U,ksize = [1,k,k,1],
                                strides=[1,s,s,1],padding=p)
                L = tf.nn.max_pool(L,ksize = [1,k,k,1],
                                strides=[1,s,s,1],padding=p)
                layers.append((L,U))
            elif type(s) is str:#Residual
                s = int(s[1:])
                W,b = param_vals[i]
                W_plus = np.maximum(W,0)
                W_minus = np.minimum(W,0)
                
                U,L = act(U),act(L)
                U,L = tf.nn.conv2d(U,W_plus,[1,s,s,1],p)+tf.nn.conv2d(L,W_minus,[1,s,s,1],p)+b+layers[-2][2],\
                      tf.nn.conv2d(U,W_minus,[1,s,s,1],p)+tf.nn.conv2d(L,W_plus,[1,s,s,1],p)+b+layers[-2][0]
                layers.append((L,U))
            else: #Conv
                W,b = param_vals[i]
                W_plus = np.maximum(W,0)
                W_minus = np.minimum(W,0)

                if i == 0:
                    U,L = tf.nn.conv2d(U,W_plus,[1,s,s,1],p)+tf.nn.conv2d(L,W_minus,[1,s,s,1],p)+b,\
                        tf.nn.conv2d(U,W_minus,[1,s,s,1],p)+tf.nn.conv2d(L,W_plus,[1,s,s,1],p)+b
                else:
                    U,L = act(U),act(L)
                    U,L = tf.nn.conv2d(U,W_plus,[1,s,s,1],p)+tf.nn.conv2d(L,W_minus,[1,s,s,1],p)+b,\
                        tf.nn.conv2d(U,W_minus,[1,s,s,1],p)+tf.nn.conv2d(L,W_plus,[1,s,s,1],p)+b
                layers.append((L,U))
        i = len(filters) - 1
        #Find margin bounds
        W,b = param_vals[i]
        l,k,s,p = filters[i],kernels[i],strides[i],paddings[i]
        ub = []
        lb = []
        U,L = act(U),act(L)
        for j in range(W.shape[-1]):
            ubs = []
            lbs = []
            for k in range(W.shape[-1]):
                W_margin = (W[:,:,:,j] - W[:,:,:,k])[:,:,:,np.newaxis]
                W_plus = np.maximum(W_margin,0)
                W_minus = np.minimum(W_margin,0)
                b_margin = b[j] - b[k]
                
                U_margin, L_margin = tf.nn.conv2d(U,W_plus,[1,s,s,1],p)+tf.nn.conv2d(L,W_minus,[1,s,s,1],p)+b_margin,\
                    tf.nn.conv2d(U,W_minus,[1,s,s,1],p)+tf.nn.conv2d(L,W_plus,[1,s,s,1],p)+b_margin
                ubs.append(U_margin)
                lbs.append(L_margin)
            ub.append(ubs)
            lb.append(lbs)
        return lb, ub
        
    
    model.predict = predict
    model.ibp = ibp
    return model
