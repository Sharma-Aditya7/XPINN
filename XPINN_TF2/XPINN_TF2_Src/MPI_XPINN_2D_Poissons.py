import sys
sys.path.insert(0, '../Utilities/')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from plotting import newfig, savefig
# from mpl_toolkits.mplot3d import Axes3D
# import time
import matplotlib.tri as tri
import matplotlib.gridspec as gridspec
# from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Polygon

#MPI imports
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

np.random.seed(seed=1234 + rank) # different seed for each rank
tf.random.set_seed(1234 + rank)

import warnings
warnings.filterwarnings('ignore')

# Initalization of Network
def initialize_NN(layers):        
    weights = []
    biases = []
    A = []
    num_layers = len(layers) 
    for l in range(0,num_layers-1):
        W = xavier_init(size=[layers[l], layers[l+1]])
        b = tf.Variable(tf.zeros([1,layers[l+1]]))#, dtype=tf.float32), dtype=tf.float32)
        a = tf.Variable(0.05)#, dtype=tf.float32)
        weights.append(W)
        biases.append(b)  
        A.append(a)
        
    return weights, biases, A

def xavier_init(size):
    in_dim  = size[0]
    out_dim = size[1]
    std     = np.sqrt(2.0/(in_dim + out_dim))
    return tf.Variable(tf.random.truncated_normal(shape=size, stddev = std))

def neural_net(X, weights, biases, A):
    num_layers = len(weights) + 1
    
    H = X 
    for l in range(0,num_layers-2):
        W = weights[l]
        b = biases[l]
        H = tf.tanh(20*A[l]*tf.add(tf.matmul(H, W), b)) 
    W = weights[-1]
    b = biases[-1]
    Y = tf.add(tf.matmul(H, W), b)
    return Y

@tf.function
def pdenn(x1, y1, x2, y2, x3, y3, xi1, yi1, xi2, yi2, W1, W2, W3, b1, b2, b3, A1, A2, A3):
        
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch([x1, y1, x2, y2, x3, y3, xi1, yi1, xi2, yi2])
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch([x1, y1, x2, y2, x3, y3, xi1, yi1, xi2, yi2])
            u1 = neural_net(tf.concat([x1,y1],1), W1, b1, A1)
            u2 = neural_net(tf.concat([x2,y2],1), W2, b2, A2)
            u3 = neural_net(tf.concat([x3,y3],1), W3, b3, A3)
            u1i1 = neural_net(tf.concat([xi1,yi1],1), W1, b1, A1)
            u2i1 = neural_net(tf.concat([xi1,yi1],1), W2, b2, A2)
            u1i2 = neural_net(tf.concat([xi2,yi2],1), W1, b1, A1)
            u3i2 = neural_net(tf.concat([xi2,yi2],1), W3, b3, A3)
        u1_x = tape1.gradient(u1, x1)
        u1_y = tape1.gradient(u1, y1)
        u2_x = tape1.gradient(u2, x2)
        u2_y = tape1.gradient(u2, y2)
        u3_x = tape1.gradient(u3, x3)
        u3_y = tape1.gradient(u3, y3)
        u1i1_x = tape1.gradient(u1i1, xi1)
        u1i1_y = tape1.gradient(u1i1, yi1)
        u2i1_x = tape1.gradient(u2i1, xi1)
        u2i1_y = tape1.gradient(u2i1, yi1)
        u1i2_x = tape1.gradient(u1i2, xi2)
        u1i2_y = tape1.gradient(u1i2, yi2)
        u3i2_x = tape1.gradient(u3i2, xi2)
        u3i2_y = tape1.gradient(u3i2, yi2)
        del tape1
    u1_xx = tape2.gradient(u1_x, x1)
    u1_yy = tape2.gradient(u1_y, y1)
    u2_xx = tape2.gradient(u2_x, x2)
    u2_yy = tape2.gradient(u2_y, y2)
    u3_xx = tape2.gradient(u3_x, x3)
    u3_yy = tape2.gradient(u3_y, y3)
    u1i1_xx = tape2.gradient(u1i1_x, xi1)
    u1i1_yy = tape2.gradient(u1i1_y, yi1)
    u2i1_xx = tape2.gradient(u2i1_x, xi1)
    u2i1_yy = tape2.gradient(u2i1_y, yi1)
    u1i2_xx = tape2.gradient(u1i2_x, xi2)
    u1i2_yy = tape2.gradient(u1i2_y, yi2)
    u3i2_xx = tape2.gradient(u3i2_x, xi2)
    u3i2_yy = tape2.gradient(u3i2_y, yi2)
    del tape2
    
    # Average value (Required for enforcing the average solution along the interface)
    uavgi1 = (u1i1 + u2i1)/2  
    uavgi2 = (u1i2 + u3i2)/2

    # Residuals
    f1 = u1_xx + u1_yy - (tf.exp(x1) + tf.exp(y1))
    f2 = u2_xx + u2_yy - (tf.exp(x2) + tf.exp(y2))
    f3 = u3_xx + u3_yy - (tf.exp(x3) + tf.exp(y3))
    
    # Residual continuity conditions on the interfaces
    fi1 = (u1i1_xx + u1i1_yy - (tf.exp(xi1) + tf.exp(yi1))) - (u2i1_xx + u2i1_yy - (tf.exp(xi1) + tf.exp(yi1))) 
    fi2 = (u1i2_xx + u1i2_yy - (tf.exp(xi2) + tf.exp(yi2))) - (u3i2_xx + u3i2_yy - (tf.exp(xi2) + tf.exp(yi2))) 

    return   f1, f2, f3, fi1, fi2, uavgi1, uavgi2, u1i1, u1i2, u2i1, u3i2

@tf.function
def train_step(W1, W2, W3, b1, b2, b3, A1, A2, A3, X_ub, ub, X_f1, X_f2, X_f3, X_fi1, X_fi2, opt1, opt2, opt3):
    
    x_ub = X_ub[:,0:1]
    y_ub = X_ub[:,1:2]
    
    x_f1 = X_f1[:,0:1]
    y_f1 = X_f1[:,1:2]
    x_f2 = X_f2[:,0:1]
    y_f2 = X_f2[:,1:2]
    x_f3 = X_f3[:,0:1]
    y_f3 = X_f3[:,1:2]
    
    x_fi1 = X_fi1[:,0:1]
    y_fi1 = X_fi1[:,1:2]
    x_fi2 = X_fi2[:,0:1]
    y_fi2 = X_fi2[:,1:2]
    
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([W1, W2, W3, b1, b2, b3, A1, A2, A3])
        
        
        ub1_pred = neural_net(tf.concat([x_ub,y_ub],1), W1, b1, A1)
        
        f1, f2, f3, fi1, fi2, uavgi1, uavgi2, u1i1, u1i2, u2i1, u3i2 = pdenn(x_f1, y_f1, x_f2, y_f2, x_f3, y_f3, x_fi1, y_fi1, x_fi2, y_fi2,W1, W2, W3, b1, b2, b3, A1, A2, A3)       
        

        # Losses in three subdomains        
        loss1 = 20*tf.reduce_mean(tf.square(ub - ub1_pred)) + tf.reduce_mean(tf.square(f1)) + 1*tf.reduce_mean(tf.square(fi1))\
                  + 1*tf.reduce_mean(tf.square(fi2)) + 20*tf.reduce_mean(tf.square(u1i1-uavgi1)) + 20*tf.reduce_mean(tf.square(u1i2-uavgi2))
                
        loss2 = tf.reduce_mean(tf.square(f2)) + 1*tf.reduce_mean(tf.square(fi1))+ 20*tf.reduce_mean(tf.square(u2i1-uavgi1))  
                            
        loss3 = tf.reduce_mean(tf.square(f3)) + 1*tf.reduce_mean(tf.square(fi2))+ 20*tf.reduce_mean(tf.square(u3i2-uavgi2)) 
    
    
    
    
    grads1 = tape.gradient(loss1, W1+b1+A1, unconnected_gradients=tf.UnconnectedGradients.ZERO)
    grads2 = tape.gradient(loss2, W2+b2+A3, unconnected_gradients=tf.UnconnectedGradients.ZERO)
    grads3 = tape.gradient(loss3, W3+b3+A3, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        
    opt1.apply_gradients(zip(grads1, W1+b1+A1))
    opt2.apply_gradients(zip(grads2, W2+b2+A2))
    opt3.apply_gradients(zip(grads3, W3+b3+A3))
    
    
    return loss1, loss2, loss3, ub1_pred

#MPI-specific training functions
@tf.function
def compute_local_residuals(x_f, y_f, W, b, A):
    """Compute PDE residuals for local subdomain"""
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch([x_f, y_f])
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch([x_f, y_f])
            u = neural_net(tf.concat([x_f, y_f], 1), W, b, A)
        u_x = tape1.gradient(u, x_f)
        u_y = tape1.gradient(u, y_f)
        del tape1
    u_xx = tape2.gradient(u_x, x_f)
    u_yy = tape2.gradient(u_y, y_f)
    del tape2
    
    f = u_xx + u_yy - (tf.exp(x_f) + tf.exp(y_f))
    return f, u

@tf.function
def compute_interface_terms(x_i, y_i, W_left, b_left, A_left, W_right, b_right, A_right):
    """Compute interface continuity terms between two subdomains"""
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch([x_i, y_i])
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch([x_i, y_i])
            u_left = neural_net(tf.concat([x_i, y_i], 1), W_left, b_left, A_left)
            u_right = neural_net(tf.concat([x_i, y_i], 1), W_right, b_right, A_right)
        u_left_x = tape1.gradient(u_left, x_i)
        u_left_y = tape1.gradient(u_left, y_i)
        u_right_x = tape1.gradient(u_right, x_i)
        u_right_y = tape1.gradient(u_right, y_i)
        del tape1
    u_left_xx = tape2.gradient(u_left_x, x_i)
    u_left_yy = tape2.gradient(u_left_y, y_i)
    u_right_xx = tape2.gradient(u_right_x, x_i)
    u_right_yy = tape2.gradient(u_right_y, y_i)
    del tape2
    
    # Residual continuity
    f_left = u_left_xx + u_left_yy - (tf.exp(x_i) + tf.exp(y_i))
    f_right = u_right_xx + u_right_yy - (tf.exp(x_i) + tf.exp(y_i))
    f_interface = f_left - f_right
    
    # Solution continuity (average)
    u_avg = (u_left + u_right) / 2
    
    return f_interface, u_avg, u_left, u_right

class MPIXPINNTrainer:
    def __init__(self, layers_list, data_dict):
        self.rank = rank
        self.size = size
        self.comm = comm
        
        # Initialize networks based on rank
        if self.rank < len(layers_list):
            self.layers = layers_list[self.rank]
            self.W, self.b, self.A = initialize_NN(self.layers)
            self.optimizer = tf.optimizers.Adam(learning_rate=0.0008)
        
        # Distribute data
        self.setup_data(data_dict)
    
    def setup_data(self, data_dict):
        """Distribute training data among processes"""
        # Each process gets its own subdomain data
        if self.rank == 0:
            self.X_f = data_dict['X_f1_train']
            self.X_ub = data_dict['X_ub_train']  # Only rank 0 handles boundary
            self.ub = data_dict['ub_train']
            self.interface_neighbors = [1]  # Subdomain 1 connects to subdomain 2
            self.X_interfaces = [data_dict['X_fi1_train']]
            
        elif self.rank == 1:
            self.X_f = data_dict['X_f2_train']
            self.interface_neighbors = [0, 2]  # Subdomain 2 connects to both 1 and 3
            self.X_interfaces = [data_dict['X_fi1_train'], data_dict['X_fi2_train']]
            
        elif self.rank == 2:
            self.X_f = data_dict['X_f3_train']
            self.interface_neighbors = [1]  # Subdomain 3 connects to subdomain 2
            self.X_interfaces = [data_dict['X_fi2_train']]
    
    def exchange_interface_data(self):
        """Exchange network parameters for interface calculations"""
        # Gather all network parameters from all processes
        all_weights = self.comm.allgather(self.W if self.rank < 3 else None)
        all_biases = self.comm.allgather(self.b if self.rank < 3 else None)
        all_A = self.comm.allgather(self.A if self.rank < 3 else None)
        
        return all_weights, all_biases, all_A
    
    @tf.function
    def train_step_local(self, all_W, all_b, all_A):
        """Local training step for each subdomain"""
        x_f = self.X_f[:, 0:1]
        y_f = self.X_f[:, 1:2]
        
        with tf.GradientTape() as tape:
            tape.watch(self.W + self.b + self.A)
            
            # Compute local PDE residual
            f_local, u_local = compute_local_residuals(
                tf.convert_to_tensor(x_f, dtype=tf.float32),
                tf.convert_to_tensor(y_f, dtype=tf.float32),
                self.W, self.b, self.A
            )
            
            loss = tf.reduce_mean(tf.square(f_local))
            
            # Add boundary loss for rank 0
            if self.rank == 0:
                x_ub = tf.convert_to_tensor(self.X_ub[:, 0:1], dtype=tf.float32)
                y_ub = tf.convert_to_tensor(self.X_ub[:, 1:2], dtype=tf.float32)
                ub_pred = neural_net(tf.concat([x_ub, y_ub], 1), self.W, self.b, self.A)
                boundary_loss = 20 * tf.reduce_mean(tf.square(
                    tf.convert_to_tensor(self.ub, dtype=tf.float32) - ub_pred
                ))
                loss += boundary_loss
            
            # Add interface losses
            for i, neighbor_rank in enumerate(self.interface_neighbors):
                if neighbor_rank < len(all_W) and all_W[neighbor_rank] is not None:
                    x_i = tf.convert_to_tensor(self.X_interfaces[i][:, 0:1], dtype=tf.float32)
                    y_i = tf.convert_to_tensor(self.X_interfaces[i][:, 1:2], dtype=tf.float32)
                    
                    f_interface, u_avg, u_local_i, u_neighbor_i = compute_interface_terms(
                        x_i, y_i, self.W, self.b, self.A,
                        all_W[neighbor_rank], all_b[neighbor_rank], all_A[neighbor_rank]
                    )
                    
                    interface_loss = tf.reduce_mean(tf.square(f_interface))
                    continuity_loss = 20 * tf.reduce_mean(tf.square(u_local_i - u_avg))
                    
                    loss += interface_loss + continuity_loss
        
        gradients = tape.gradient(loss, self.W + self.b + self.A)
        self.optimizer.apply_gradients(zip(gradients, self.W + self.b + self.A))
        
        return loss
    
    def train(self, max_iterations=10000):
        """Main training loop with MPI synchronization"""
        if self.rank < 3:  # Only ranks 0, 1, 2 participate in training
            for iteration in range(max_iterations):
                # Exchange network parameters
                all_W, all_b, all_A = self.exchange_interface_data()
                
                # Perform local training step
                loss = self.train_step_local(all_W, all_b, all_A)
                
                # Synchronize (wait for all processes)
                self.comm.Barrier()
                
                if iteration % 100 == 0 and self.rank == 0:
                    print(f"Iteration {iteration}, Rank {self.rank} Loss: {loss.numpy()}")
        
        # Final synchronization
        self.comm.Barrier()
    
    def predict(self, X_test):
        """Make predictions on test data"""
        if self.rank < 3:
            X_tf = tf.convert_to_tensor(X_test, dtype=tf.float32)
            u_pred = neural_net(X_tf, self.W, self.b, self.A)
            return u_pred.numpy()
        return None

def main():
    if rank == 0:
        print(f"Starting MPI XPINN with {size} processes")
    
    # Load data (only on rank 0, then broadcast)
    if rank == 0:
        data = scipy.io.loadmat('../DATA/XPINN_2D_PoissonEqn.mat')
        
        # Prepare data dictionary
        data_dict = {
            'X_f1_train': np.hstack((data['x_f1'].flatten()[:,None], data['y_f1'].flatten()[:,None])),
            'X_f2_train': np.hstack((data['x_f2'].flatten()[:,None], data['y_f2'].flatten()[:,None])),
            'X_f3_train': np.hstack((data['x_f3'].flatten()[:,None], data['y_f3'].flatten()[:,None])),
            'X_fi1_train': np.hstack((data['xi1'].flatten()[:,None], data['yi1'].flatten()[:,None])),
            'X_fi2_train': np.hstack((data['xi2'].flatten()[:,None], data['yi2'].flatten()[:,None])),
            'X_ub_train': np.hstack((data['xb'].flatten()[:,None], data['yb'].flatten()[:,None])),
            'ub_train': data['ub'].flatten()[:,None]
        }
        
        # Sample training points
        N_f1, N_f2, N_f3 = 5000, 1800, 1200
        N_ub, N_I1, N_I2 = 200, 100, 100
        
        # Random sampling
        idx1 = np.random.choice(data_dict['X_f1_train'].shape[0], N_f1, replace=False)
        data_dict['X_f1_train'] = data_dict['X_f1_train'][idx1, :]
        
        idx2 = np.random.choice(data_dict['X_f2_train'].shape[0], N_f2, replace=False)
        data_dict['X_f2_train'] = data_dict['X_f2_train'][idx2, :]
        
        idx3 = np.random.choice(data_dict['X_f3_train'].shape[0], N_f3, replace=False)
        data_dict['X_f3_train'] = data_dict['X_f3_train'][idx3, :]
        
        idx4 = np.random.choice(data_dict['X_ub_train'].shape[0], N_ub, replace=False)
        data_dict['X_ub_train'] = data_dict['X_ub_train'][idx4, :]
        data_dict['ub_train'] = data_dict['ub_train'][idx4, :]
        
        idxi1 = np.random.choice(data_dict['X_fi1_train'].shape[0], N_I1, replace=False)
        data_dict['X_fi1_train'] = data_dict['X_fi1_train'][idxi1, :]
        
        idxi2 = np.random.choice(data_dict['X_fi2_train'].shape[0], N_I2, replace=False)
        data_dict['X_fi2_train'] = data_dict['X_fi2_train'][idxi2, :]
        
    else:
        data_dict = None
    
    # Broadcast data to all processes
    data_dict = comm.bcast(data_dict, root=0)
    
    # Define network architectures for each subdomain
    layers_list = [
        [2, 30, 30, 1],           # Subdomain 1
        [2, 20, 20, 20, 20, 1],   # Subdomain 2
        [2, 25, 25, 25, 1]        # Subdomain 3
    ]
    
    trainer = MPIXPINNTrainer(layers_list, data_dict)
    
    trainer.train(max_iterations=1000)
    
    if rank == 0:
        print("Training completed!")

if __name__ == '__main__':
    main()

'''
if __name__ == '__main__':
    
    # Boundary points from subdomian 1
    N_ub   = 200
    
    # Residual points in three subdomains
    N_f1   = 5000
    N_f2   = 1800
    N_f3   = 1200
    
    # Interface points along the two interfaces
    N_I1   = 100
    N_I2   = 100
    
    # NN architecture in each subdomain
    layers1 = [2, 30, 30, 1]
    layers2 = [2, 20, 20, 20, 20, 1]
    layers3 = [2, 25, 25, 25, 1]
    
    # Load training data (boundary points), residual and interface points from .mat file
    # All points are generated in Matlab
    data = scipy.io.loadmat('DATA\XPINN_2D_PoissonEqn.mat')
    

    x_f1 = data['x_f1'].flatten()[:,None]
    y_f1 = data['y_f1'].flatten()[:,None]
    x_f2 = data['x_f2'].flatten()[:,None]
    y_f2 = data['y_f2'].flatten()[:,None]
    x_f3 = data['x_f3'].flatten()[:,None]
    y_f3 = data['y_f3'].flatten()[:,None]
    xi1  = data['xi1'].flatten()[:,None]
    yi1  = data['yi1'].flatten()[:,None]
    xi2  = data['xi2'].flatten()[:,None]
    yi2  = data['yi2'].flatten()[:,None]
    xb   = data['xb'].flatten()[:,None]
    yb   = data['yb'].flatten()[:,None]

    
    ub_train = data['ub'].flatten()[:,None]
    u_exact  = data['u_exact'].flatten()[:,None]
    u_exact2 = data['u_exact2'].flatten()[:,None]
    u_exact3 = data['u_exact3'].flatten()[:,None]
    
    X_f1_train = np.hstack((x_f1.flatten()[:,None], y_f1.flatten()[:,None]))
    X_f2_train = np.hstack((x_f2.flatten()[:,None], y_f2.flatten()[:,None]))
    X_f3_train = np.hstack((x_f3.flatten()[:,None], y_f3.flatten()[:,None]))
    X_fi1_train = np.hstack((xi1.flatten()[:,None], yi1.flatten()[:,None]))
    X_fi2_train = np.hstack((xi2.flatten()[:,None], yi2.flatten()[:,None]))

    
    X_ub_train = np.hstack((xb.flatten()[:,None], yb.flatten()[:,None]))

    # Points in the whole  domain
    x_total =  data['x_total'].flatten()[:,None] 
    y_total =  data['y_total'].flatten()[:,None]

    X_star1 = np.hstack((x_f1.flatten()[:,None], y_f1.flatten()[:,None]))
    X_star2 = np.hstack((x_f2.flatten()[:,None], y_f2.flatten()[:,None]))
    X_star3 = np.hstack((x_f3.flatten()[:,None], y_f3.flatten()[:,None]))

    # Randomly select the residual points from sub-domains
    idx1 = np.random.choice(X_f1_train.shape[0], N_f1, replace=False)    
    X_f1_train = X_f1_train[idx1,:]
    
    idx2 = np.random.choice(X_f2_train.shape[0], N_f2, replace=False)    
    X_f2_train = X_f2_train[idx2,:]
    
    idx3 = np.random.choice(X_f3_train.shape[0], N_f3, replace=False)    
    X_f3_train = X_f3_train[idx3,:]
    
    # Randomly select boundary points
    idx4 = np.random.choice(X_ub_train.shape[0], N_ub, replace=False)
    X_ub_train = X_ub_train[idx4,:]
    ub_train   = ub_train[idx4,:] 
    
    # Randomly select the interface points along two interfaces
    idxi1 = np.random.choice(X_fi1_train.shape[0], N_I1, replace=False)    
    X_fi1_train = X_fi1_train[idxi1,:]
    
    idxi2 = np.random.choice(X_fi2_train.shape[0], N_I2, replace=False)    
    X_fi2_train = X_fi2_train[idxi2,:]
    
    
    # Initialize NNs
    W1, b1, A1 = initialize_NN(layers1)
    W2, b2, A2 = initialize_NN(layers2)    
    W3, b3, A3 = initialize_NN(layers3)
    

    X_ub_train_tf  = tf.convert_to_tensor(X_ub_train, dtype=tf.float32)
    ub_train_tf    = tf.convert_to_tensor(ub_train, dtype=tf.float32)
    X_f1_train_tf  = tf.convert_to_tensor(X_f1_train, dtype=tf.float32)
    X_f2_train_tf  = tf.convert_to_tensor(X_f2_train, dtype=tf.float32)
    X_f3_train_tf  = tf.convert_to_tensor(X_f3_train, dtype=tf.float32)
    X_fi1_train_tf = tf.convert_to_tensor(X_fi1_train, dtype=tf.float32)
    X_fi2_train_tf = tf.convert_to_tensor(X_fi2_train, dtype=tf.float32)
    
    
    # Learning rate and optimizers
    lr = 0.0008
    optimizer1 = tf.optimizers.Adam(learning_rate=lr)
    optimizer2 = tf.optimizers.Adam(learning_rate=lr)
    optimizer3 = tf.optimizers.Adam(learning_rate=lr)
    
    
    ################################################################# TRAINING
    Max_iter = 100
    m =0 

    Loss1_hist = []
    Loss2_hist = []
    Loss3_hist = []
    
    while m <= Max_iter:
        loss_1, loss_2, loss_3,  u_pred = train_step(W1, W2, W3, b1, b2, b3, A1, A2, A3, X_ub_train_tf, ub_train_tf, X_f1_train_tf,\
         
                                                     X_f2_train_tf, X_f3_train_tf, X_fi1_train_tf, X_fi2_train_tf, optimizer1, optimizer2, optimizer3)
        # if m %20==0:
        print(f"Iteration is : {m},  loss1 is: {loss_1},  loss2 is: {loss_2},  loss3 is: {loss_3}")
        Loss1_hist.append(loss_1)
        Loss2_hist.append(loss_2)
        Loss3_hist.append(loss_3)
        m += 1
    
    ################################################################# PREDICTION
    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)

    X1, Y1 = X_star1[:,0:1], X_star1[:,1:2]
    triang_1 = tri.Triangulation(X1.flatten(), Y1.flatten())
    X2, Y2 = X_star2[:,0:1], X_star2[:,1:2]
    triang_2 = tri.Triangulation(X2.flatten(), Y2.flatten())
    X3, Y3 = X_star3[:,0:1], X_star3[:,1:2]
    triang_3 = tri.Triangulation(X3.flatten(), Y3.flatten())
    x_tot = np.concatenate([X1, X2, X3])
    y_tot = np.concatenate([Y1, Y2, Y3])
    triang_total = tri.Triangulation(x_tot.flatten(), y_tot.flatten())

    
    X1_tf  = tf.convert_to_tensor(X1, dtype=tf.float32)
    X2_tf  = tf.convert_to_tensor(X2, dtype=tf.float32)
    X3_tf  = tf.convert_to_tensor(X3, dtype=tf.float32)
    Y1_tf  = tf.convert_to_tensor(Y1, dtype=tf.float32)
    Y2_tf  = tf.convert_to_tensor(Y2, dtype=tf.float32)
    Y3_tf  = tf.convert_to_tensor(Y3, dtype=tf.float32)
    
    
    
    u_pred1 = neural_net(tf.concat([X1_tf, Y1_tf],1), W1, b1, A1)
    u_pred2 = neural_net(tf.concat([X2_tf, Y2_tf],1), W2, b2, A2)
    u_pred3 = neural_net(tf.concat([X3_tf, Y3_tf],1), W3, b3, A3)

    # Concatenating the solution from subdomains
    u_pred = np.concatenate([u_pred1, u_pred2, u_pred3])
           
    ################################################################# PLOTTING    
    fig, ax = newfig(1.0, 1.1)
    plt.plot(range(0,Max_iter+1,1), Loss1_hist,  'r-', linewidth = 1,label = 'Sub-Net1') 
    plt.plot(range(0,Max_iter+1,1), Loss2_hist,  'b-.', linewidth = 1, label = 'Sub-Net2') 
    plt.plot(range(0,Max_iter+1,1), Loss3_hist,  'g--', linewidth = 1, label = 'Sub-Net3') 
    plt.xlabel('$\#$iterations')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend(loc='upper right')
 #   savefig('./figures/XPINN_PoissonMSEhistory') 
    savefig(r'XPINN_TF2_Src\figures\MPI_XPINN_PoissonMSEhistory')

    # PATCH
    aa1 = np.array([[np.squeeze(xb[-1]),np.squeeze(yb[-1])]])
    aa2 = np.array([[1.8,np.squeeze(yb[-1])], [+1.8,-1.7], [-1.6,-1.7], [-1.6,1.55], [1.8,1.55],[1.8,np.squeeze(yb[-1])]])
    x_domain1 = np.squeeze(xb.flatten()[:,None])
    y_domain1 = np.squeeze(yb.flatten()[:,None])
    aa3 = np.array([x_domain1,y_domain1]).T
    XX = np.vstack((aa3, aa2, aa1))
    triang_total = tri.Triangulation(x_tot.flatten(), y_tot.flatten())

    X_fi1_train_Plot = np.hstack((xi1.flatten()[:,None], yi1.flatten()[:,None]))
    X_fi2_train_Plot = np.hstack((xi2.flatten()[:,None], yi2.flatten()[:,None]))
    
    fig, ax = newfig(1.0, 1.1)
    gridspec.GridSpec(1,1)
    ax = plt.subplot2grid((1,1), (0,0))
    tcf = ax.tricontourf(triang_total, np.squeeze(u_exact), 100 ,cmap='jet')
    ax.add_patch(Polygon(XX, closed=True, fill=True, color='w', edgecolor = 'w'))
    tcbar = fig.colorbar(tcf)
    tcbar.ax.tick_params(labelsize=28)
    ax.set_xlabel('$x$', fontsize = 32)
    ax.set_ylabel('$y$', fontsize = 32)
    ax.set_title('$u$ (Exact)', fontsize = 34)
    ax.tick_params(axis="x", labelsize = 28)
    ax.tick_params(axis="y", labelsize = 28)   
    plt.plot(X_fi1_train_Plot[:,0:1], X_fi1_train_Plot[:,1:2], 'w-', markersize =2, label='Interface Pts')
    plt.plot(X_fi2_train_Plot[:,0:1], X_fi2_train_Plot[:,1:2], 'w-', markersize =2, label='Interface Pts')
    fig.set_size_inches(w=12,h=9)
    #savefig('./figures/XPINN_PoissonEq_ExSol') 
    savefig(r'XPINN_TF2_Src\figures\MPI_XPINN_PoissonEq_ExSol')
    plt.show()  

    fig, ax = newfig(1.0, 1.1)
    gridspec.GridSpec(1,1)
    ax = plt.subplot2grid((1,1), (0,0))
    tcf = ax.tricontourf(triang_total, u_pred.flatten(), 100 ,cmap='jet')
    ax.add_patch(Polygon(XX, closed=True, fill=True, color='w', edgecolor = 'w'))
    tcbar = fig.colorbar(tcf)
    tcbar.ax.tick_params(labelsize=28)
    ax.set_xlabel('$x$', fontsize = 32)
    ax.set_ylabel('$y$', fontsize = 32)
    ax.set_title('$u$ (Predicted)', fontsize = 34)
    ax.tick_params(axis="x", labelsize = 28)
    ax.tick_params(axis="y", labelsize = 28)   
    plt.plot(X_fi1_train_Plot[:,0:1], X_fi1_train_Plot[:,1:2], 'w-', markersize =2, label='Interface Pts')
    plt.plot(X_fi2_train_Plot[:,0:1], X_fi2_train_Plot[:,1:2], 'w-', markersize =2, label='Interface Pts')
    #fig.tight_layout()
    fig.set_size_inches(w=12,h=9)
    #savefig('./figures/XPINN_PoissonEq_Sol')
    savefig(r'XPINN_TF2_Src\figures\MPI_XPINN_PoissonEq_EqSol') 
    plt.show()  
 


    fig, ax = newfig(1.0, 1.1)
    gridspec.GridSpec(1,1)
    ax = plt.subplot2grid((1,1), (0,0))
    tcf = ax.tricontourf(triang_total, abs(np.squeeze(u_exact)-u_pred.flatten()), 100 ,cmap='jet')
    ax.add_patch(Polygon(XX, closed=True, fill=True, color='w', edgecolor = 'w'))   
    tcbar = fig.colorbar(tcf)  
    tcbar.ax.tick_params(labelsize=28)
    ax.set_xlabel('$x$', fontsize = 32)
    ax.set_ylabel('$y$', fontsize = 32)
    ax.set_title('Point-wise Error', fontsize = 34)
    ax.tick_params(axis="x", labelsize = 28)
    ax.tick_params(axis="y", labelsize = 28)   
    plt.plot(X_fi1_train_Plot[:,0:1], X_fi1_train_Plot[:,1:2], 'w-', markersize =2, label='Interface Pts')
    plt.plot(X_fi2_train_Plot[:,0:1], X_fi2_train_Plot[:,1:2], 'w-', markersize =2, label='Interface Pts')
    #fig.tight_layout()
    fig.set_size_inches(w=12,h=9)
    #savefig('./figures/XPINN_PoissonEq_Err') 
    savefig(r'XPINN_TF2_Src\figures\MPI_XPINN_PoissonEq_Err')
    plt.show()
'''