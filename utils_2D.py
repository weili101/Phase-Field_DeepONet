import tensorflow as tf
import numpy as np
import scipy
from scipy import spatial, interpolate


class Net_FNN(tf.keras.Model):
  
  def __init__(self, paras):
    super(Net_FNN, self).__init__()
    self.__init_layers(paras)

  def __init_layers(self, paras):

    Layers = []
    for layer in paras:
        for n_nodes in layer['nodes']:
            Layers.append(tf.keras.layers.Dense(n_nodes, 
                                                activation=layer['activation']))

    self.Layers = Layers
    
  def call(self, x):
    
    y = x
    for layer in self.Layers:
        y= layer(y)

    return y

class Net_CNN(tf.keras.Model):

    def __init__(self, paras, fnn=None):
        super(Net_CNN, self).__init__()
        
        self.fnn = fnn
        self.__init_layers(paras)

    def __init_layers(self, paras):
        
        Layers = []
        for layer in paras:
            filters = layer['filters']
            kernels = layer['kernels']
            strides = layer['strides']
            padding = layer['padding']
            activation = layer['activation']

            Layers.append(tf.keras.layers.Conv2D(
                filters = filters,
                kernel_size = kernels,
                strides= strides,
                padding= padding,
                data_format=None,
                dilation_rate=(1, 1),
                groups=1,
                activation= activation,
                use_bias=True,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
            ))

        self.Layers = Layers

    def call(self, x):
        
        y = x
        for layer in self.Layers:
            y = layer(y)
        y = tf.keras.layers.Flatten()(y)

        if self.fnn != None:
            y = self.fnn(y)

        return y

class ONet(tf.keras.Model):
  
  def __init__(self, trunk, branch):
    super(ONet, self).__init__()
    
    self.trunk = trunk
    self.branch = branch

  def call(self, u_sensor, x, return_grad = False):

    #y_trunk = self.trunk(x)

    #x_sensor = np.expand_dims(x_sensor, 0)
    u_branch = self.branch(u_sensor)
    
    if return_grad:
        with tf.GradientTape(persistent=True) as g1:
            g1.watch(x)
            u_trunk = self.trunk(x)
            #y_out = tf.tensordot(y_branch, y_trunk, axes=([1], [1]))
            u = tf.unstack(u_trunk, axis=1)
        du_x = []
        du_y = []
        for us in u:
        #u = tf.expand_dims(u, axis=-1)
            du = tf.unstack(g1.gradient(us, x), axis=1)
            du_x.append(du[0])
            du_y.append(du[1])
        du_x, du_y = tf.stack(du_x), tf.stack(du_y)

        #print(u_branch.shape, u_trunk.shape, du_x.shape)
        u_out = tf.tensordot(u_branch, u_trunk, axes=([1], [1]))
        du_x  = tf.tensordot(u_branch, du_x, axes=([1], [0]))
        du_y  = tf.tensordot(u_branch, du_y, axes=([1], [0]))

        u_out = tf.tanh(u_out)
        du_x = ( 1. - tf.tanh(u_out) ) * du_x
        du_y = ( 1. - tf.tanh(u_out) ) * du_y

        return u_out, du_x, du_y

    else:
        u_trunk = self.trunk(x)
        u_out = tf.tensordot(u_branch, u_trunk, axes=([1], [1]))

        u_out = tf.tanh(u_out)
        
        return u_out

def gaussian_process_2d(x, n_grid, n_samples, length_scale_list, u_mean=0.):
    '''
    x -  tuple or list of upoer and lower limits [x0, x1]
    num_points - Number points in each curve (number of of features)
    num_curves - Number of curves to sample (number of samples)
    
    '''
    x1, x2, y1, y2 = x
    # Independent variable samples
    xn, yn = np.meshgrid(np.linspace(x1, x2, n_grid), np.linspace(y1, y2, n_grid))
    X = np.hstack([xn.ravel().reshape(-1,1), yn.ravel().reshape(-1,1)])

    # convert X to numpy float32
    #X = X.astype(np.float32)

    #length_scale_list = [0.02, 0.2, 2.0, 20]
   
    ys = []
    # Draw samples from the prior at our data points.
    # Assume a mean of 0 for simplicity
    for _ in range(n_samples):
        length_scale = np.random.choice(length_scale_list) # Length scale of kernel randomly drawn from a list
        cov = exp_quadratic(X, X, length_scale) # Kernel of data points
        yst = np.random.multivariate_normal( mean=u_mean * np.ones(n_grid**2), cov=cov, size=1)
        if len(ys) == 0:
            ys = yst
        else:
            ys = np.vstack((ys, yst))

    #ys = tf.math.multiply(ys, ((X+x0)*(X+x1)).T)
    #if (np.max(np.abs(ys)) > 1):
    #    ys = np.divide(ys, np.reshape(np.max(np.abs(ys), 1), (-1,1)) )

    return X, ys

def normalize(ys):
    '''
    ys - N X M matrix of M curves with N points each
    Normalize the data to be between -1 and 1
    '''
    if (np.max(np.abs(ys)) > 1):
        ys = np.divide(ys, np.reshape(np.max(np.abs(ys), 1), (-1,1)) )
    return ys


def exp_quadratic(xa, xb, length_scale):
    """Exponentiated quadratic  with σ=1"""
    # L2 distance (Squared Euclidian)
    sq_norm = -0.5 * spatial.distance.cdist(xa, xb, 'sqeuclidean') / length_scale**2
 
    return np.exp(sq_norm)

def solve_allen_cahn(p, x, n_grid, tao, eps):
    
    p_num = np.array(memoryview(p))
    pt = p_num
    
    x1, x2, y1, y2 = x
    Lx = x2 - x1
    Ly = y2 - y1
    nx = ny = n_grid # number of computational grids
    dx, dy = Lx/(nx-1), Ly/(ny-1) # spacing of computational grid [m]
    dt = 0.001 # time increment for a time step [s]
    nsteps = int(tao/dt) # total number of time step
    
    for t, p in enumerate(p_num[:,:,:,0]):
        
        for _ in range(nsteps-1):
            for j in range(ny):
                for i in range(nx):
                    ip = i + 1
                    im = i - 1
                    jp = j + 1
                    jm = j - 1
                    if ip > nx - 1:
                        ip = nx -1
                    if im < 0:
                        im = 0
                    if jp > ny - 1:
                        jp = ny -1
                    if jm < 0:
                        jm = 0
                    p[i,j] = p[i,j] + ( - 1./eps**2 * (p[i,j]**3 - p[i,j]) + # 4./eps**2*p[t,i,j]*(1.-p[t,i,j])*(p[t,i,j]-0.5) 
                                                ((p[ip,j] - 2*p[i,j] + p[im,j])/dx/dx + (p[i,jp] - 2*p[i,j] + p[i,jm])/dy/dy) ) * dt
        pt[t, :, :, 0] = p
        
    return tf.constant(pt, dtype=tf.float32)

import pickle
def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as inp: 
        data = pickle.load(inp)
    return data

