from scipy import spatial, interpolate
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#from sklearn import gaussian_process as gp
#import scipy
from scipy.sparse import diags
import pickle

class ONet(tf.keras.Model):
  
  def __init__(self, trunk, branch):
    super(ONet, self).__init__()
    
    self.trunk = trunk
    self.branch = branch

  def call(self, u_sensor, x):

    y_trunk = self.trunk(x)

    #x_sensor = np.expand_dims(x_sensor, 0)
    y_branch = self.branch(u_sensor)
    
    y_out = tf.tensordot(y_branch, y_trunk, axes=([1], [1]))

    #y_out = tf.reduce_sum(y_out, keepdims=True, axis=1)
    
    return tf.math.multiply(y_out, tf.reshape((x+1)*(x-1), [1, -1]) )
        

class Data():
    '''du/dt=-ku(x,t), -1<=x<=1
        input u(x, t0)
       output u(x, t1)
    '''
    def __init__(self, x, sensor_in, sensor_out, length_scale, train_num, test_num):
        self.x = x
        self.sensor_in = sensor_in
        self.sensor_out = sensor_out
        self.length_scale = length_scale
        self.train_num = train_num
        self.test_num = test_num
        self.__init_data()
        
    def __init_data(self):
        
        features = 100
        u_train = gaussian_process(self.x, features, self.train_num, self.length_scale)
        u_test = gaussian_process(self.x, features, self.test_num,  self.length_scale)
        #self.X_train = sense(train_data, self.sensor_in) #.reshape([-1, self.sensor_in, 1])
        #self.y_train = self.solve(train_data) #.reshape([-1, self.sensor_out, 1])
        #self.X_test = sense(test_data, self.sensor_in)   #.reshape([-1, self.sensor_in, 1])
        #self.y_test = self.solve(test_data)   #.reshape([-1, self.sensor_out, 1])

        x0, x1 = self.x

        X = np.linspace(x0, x1, num=features, dtype=np.float32)

        self.u_train = u_train*(1. - X)*X
        self.u_test = u_test*(1. - X)*X
        self.X = tf.constant(X.reshape(-1, 1), dtype=tf.float32)
    
    
    def solve(self, gps, K, dt):
        u0 = sense(gps, self.sensor_in)
        return np.exp(-dt*K)*u0

# Function to generate the random input of x as the training data
# Uniform distribution between -1 and 1
def x_train_data(N, xd=[-1, 1], random=False):

    x0, x1 = xd
    if random:
        X = np.random.uniform(x0, x1, N).reshape(-1, 1)
    else:
        X = np.linspace(x0, x1, N).reshape(-1, 1)
    
    X = tf.constant(X, dtype=tf.float32)

    return X

def gaussian_process(x, num_points, num_curves, length_scale_list, u_mean=0.):
    '''
    x -  tuple or list of upoer and lower limits [x0, x1]
    num_points - Number points in each curve (number of of features)
    num_curves - Number of curves to sample (number of samples)
    
    '''
    x0, x1 = x

    X = np.expand_dims(np.linspace(x0, x1, num_points), 1)

    #length_scale_list = [0.02, 0.2, 2.0, 20]
   
    ys = []
    # Draw samples from the prior at our data points.
    # Assume a mean of 0 for simplicity
    for _ in range(num_curves):
        length_scale = np.random.choice(length_scale_list) # Length scale of kernel randomly drawn from a list
        cov = exp_quadratic(X, X, length_scale) # Kernel of data points
        yst = np.random.multivariate_normal( mean=u_mean * np.ones(num_points), cov=cov, size=1)
        if len(ys) == 0:
            ys = yst
        else:
            ys = np.vstack((ys, yst))

    #ys = tf.math.multiply(ys, ((X+x0)*(X+x1)).T)
    #if (np.max(np.abs(ys)) > 1):
    #    ys = np.divide(ys, np.reshape(np.max(np.abs(ys), 1), (-1,1)) )

    return ys

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

def sense(gps, sensor_in):
    x = np.linspace(0, 1, num=gps.shape[1])
    res = map(
        lambda y: interpolate.interp1d(x, y, kind='cubic', copy=False, assume_sorted=True
        )(np.linspace(0, 1, num=sensor_in)),
        gps)
    return tf.constant(np.vstack(list(res)), dtype=tf.float32)

class FNN(tf.keras.Model):
  
  def __init__(self, n_output, n_layer, n_nodes, activation):
    super(FNN, self).__init__()
    
    layers = []
    for _ in range(n_layer):
        layers.append(tf.keras.layers.Dense(n_nodes, activation=activation))
    self.hidden = layers

    self.out = tf.keras.layers.Dense(n_output, activation='linear')
    self.n_layer = n_layer
    self.n_nodes = n_nodes
    self.activation = activation
    
  def call(self, x):
    
    y = self.hidden[0](x)
    for layer in self.hidden[1:]:
        y= layer(y)
    y = self.out(y)

    return y
  def get_config(self):
        return {"n_layer": self.n_layer, "n_nodes": self.n_nodes, "activation": self.activation}

  @classmethod
  def from_config(cls, config):
      return cls(**config)

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as inp: 
        data = pickle.load(inp)
    return data

def solve_Cahn_Hilliard(cn, eps, dt, dx, N):
    
    Nx = cn.shape[1]

    A = diags([1, -2, 1], [-1, 0, 1], (Nx, Nx)).toarray()
    A[0,0], A[-1,-1] = -1, -1

    #c0 = [np.sign(0.5 - x) for x in np.linspace(0, 1, Nx)]
    #c0 = np.array(c0)
    c_out = []
    for c0 in cn:
        c0 = c0.numpy()
        for _ in range(N):

            u0 = c0**3 - c0 - eps**2* A @ c0 / dx**2
            #u0 = 0.5*(2*c0**3 - 3*c0**2 + c0) - eps**2* A @ c0 / dx**2
            c0 = A @ u0 /dx**2 * dt + c0
            
        c_out.append(c0)

    return tf.constant(c_out, dtype=tf.float32)

def plot_2D(xlim=None, ylim=None, xticks=None, yticks=None, 
            xlabel = 'x', ylabel = 'y',
            figsize=(7.3, 5.9), label_size=18, tick_size=16, spine_width=1.5):
    
    fig, ax = plt.subplots(figsize=figsize) 

    #-----Format Axis --------------------------- 
    # labels and size
    ax.set_xlabel(xlabel, fontsize=label_size)
    ax.set_ylabel(ylabel, fontsize=label_size)
    
    # limits, ticks and size
    if xlim != None:
        ax.set_xlim(*xlim)
    if ylim != None:
        ax.set_ylim(*ylim)
        
    if xticks != None:
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        
    for tick in ax.get_xticklabels():
        #tick.set_fontname('Times New Roman')
        tick.set_fontsize(tick_size)
    for tick in ax.get_yticklabels():
        #tick.set_fontname('Times New Roman')
        tick.set_fontsize(tick_size)

    #---------- Spines -------
    ax.spines["top"].set_linewidth(spine_width)
    ax.spines["left"].set_linewidth(spine_width)
    ax.spines["right"].set_linewidth(spine_width)
    ax.spines["bottom"].set_linewidth(spine_width)
    
    return fig, ax

import multiprocessing as mp

def worker(u_old_subset, eps, dt, dx, N, output):
    u_old_subset = tf.constant(u_old_subset, dtype=tf.float32)
    u_new_subset = solve_Cahn_Hilliard(u_old_subset, eps, dt, dx, N)
    u_new_subset = u_new_subset.numpy()
    output.put(u_new_subset)

def solve_Cahn_Hilliard_mp(u_old, eps, dt, dx, N, num_processes = 16):

    #u_old = u_old.numpy()

    #num_processes = 4  # change this to the desired number of processes
    u_old_subsets = np.array_split(u_old, num_processes)
    #u_old_subsets = [tf.constant(u_old_subsets[i], dtype=tf.float32) for i in range(num_processes)]

    outputs = [mp.Queue() for _ in range(num_processes)]
    processes = [mp.Process(target=worker, args=(u_old_subsets[i], eps, dt, dx, N, outputs[i])) for i in range(num_processes)]

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    u_new_subsets = [outputs[i].get() for i in range(num_processes)]
    u_new = np.concatenate(u_new_subsets)

    return tf.constant(u_new, dtype=tf.float32)