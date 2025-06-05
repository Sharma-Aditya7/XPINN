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
import matplotlib as mpl

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
    def __init__(self, layers_list, data_dict, test_data_dict=None):
        self.rank = rank
        self.size = size
        self.comm = comm
        
        # Initialize networks based on rank
        if self.rank < len(layers_list):
            self.layers = layers_list[self.rank]
            self.W, self.b, self.A = initialize_NN(self.layers)
            self.optimizer = tf.optimizers.Adam(learning_rate=0.0008)
        
        # Store test data for plotting
        self.test_data_dict = test_data_dict
        
        # Initialize loss history tracking
        self.loss_history = []
        
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
                
                # Store loss history
                self.loss_history.append(loss.numpy())
                
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
    
    def gather_all_predictions(self):
        """Gather predictions from all subdomains for plotting"""
        if self.test_data_dict is None:
            return None, None, None
            
        # Get predictions from each subdomain
        predictions = []
        test_points = []
        
        if self.rank == 0:
            X_star1 = self.test_data_dict['X_star1']
            u_pred1 = self.predict(X_star1)
            predictions.append(u_pred1)
            test_points.append(X_star1)
        elif self.rank == 1:
            X_star2 = self.test_data_dict['X_star2']
            u_pred2 = self.predict(X_star2)
            predictions.append(u_pred2)
            test_points.append(X_star2)
        elif self.rank == 2:
            X_star3 = self.test_data_dict['X_star3']
            u_pred3 = self.predict(X_star3)
            predictions.append(u_pred3)
            test_points.append(X_star3)
        else:
            predictions.append(None)
            test_points.append(None)
        
        # Gather all predictions and test points
        all_predictions = self.comm.allgather(predictions[0] if predictions else None)
        all_test_points = self.comm.allgather(test_points[0] if test_points else None)
        
        return all_predictions, all_test_points, self.loss_history
        
    def plot_results(self, u_exact, interface_data):
        """Plot results similar to original XPINN code"""
        if self.rank != 0:  # Only rank 0 handles plotting
            return
            
        # Gather all data for plotting
        all_predictions, all_test_points, _ = self.gather_all_predictions()
        
        # Gather loss histories from all ranks
        all_loss_histories = self.comm.gather(self.loss_history, root=0)
        
        if all_predictions is None or any(p is None for p in all_predictions[:3]):
            print("Warning: Missing prediction data for plotting")
            return
            
        # Extract predictions and test points
        u_pred1, u_pred2, u_pred3 = all_predictions[:3]
        X_star1, X_star2, X_star3 = all_test_points[:3]
        
        # Set up matplotlib
        mpl.rcParams.update(mpl.rcParamsDefault)
        
        # Create triangulations
        X1, Y1 = X_star1[:,0:1], X_star1[:,1:2]
        triang_1 = tri.Triangulation(X1.flatten(), Y1.flatten())
        X2, Y2 = X_star2[:,0:1], X_star2[:,1:2]
        triang_2 = tri.Triangulation(X2.flatten(), Y2.flatten())
        X3, Y3 = X_star3[:,0:1], X_star3[:,1:2]
        triang_3 = tri.Triangulation(X3.flatten(), Y3.flatten())
        
        # Combine all domains
        x_tot = np.concatenate([X1, X2, X3])
        y_tot = np.concatenate([Y1, Y2, Y3])
        triang_total = tri.Triangulation(x_tot.flatten(), y_tot.flatten())
        
        # Concatenate predictions
        u_pred = np.concatenate([u_pred1, u_pred2, u_pred3])
        
        ################################################################# 
        # PLOTTING LOSS HISTORY
        #################################################################
        fig, ax = newfig(1.0, 1.1)
        max_iter = len(self.loss_history) - 1
        
        if all_loss_histories and len(all_loss_histories) >= 3:
            plt.plot(range(0, max_iter+1, 1), all_loss_histories[0], 'r-', linewidth=1, label='Sub-Net1') 
            plt.plot(range(0, max_iter+1, 1), all_loss_histories[1], 'b-.', linewidth=1, label='Sub-Net2') 
            plt.plot(range(0, max_iter+1, 1), all_loss_histories[2], 'g--', linewidth=1, label='Sub-Net3') 
        else:
            plt.plot(range(0, max_iter+1, 1), self.loss_history, 'r-', linewidth=1, label='Combined Loss')
            
        plt.xlabel('$\#$iterations')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend(loc='upper right')
        savefig('./figures/XPINN_PoissonMSEhistory') 
        
        #################################################################
        # SETUP DOMAIN PATCHES
        #################################################################
        # Create domain boundary patches (adapt based on your domain geometry)
        xb, yb = interface_data['xb'], interface_data['yb']
        aa1 = np.array([[np.squeeze(xb[-1]), np.squeeze(yb[-1])]])
        aa2 = np.array([[1.8, np.squeeze(yb[-1])], [+1.8, -1.7], [-1.6, -1.7], 
                       [-1.6, 1.55], [1.8, 1.55], [1.8, np.squeeze(yb[-1])]])
        x_domain1 = np.squeeze(xb.flatten()[:,None])
        y_domain1 = np.squeeze(yb.flatten()[:,None])
        aa3 = np.array([x_domain1, y_domain1]).T
        XX = np.vstack((aa3, aa2, aa1))
        
        # Interface plotting data
        X_fi1_train_Plot = interface_data['X_fi1_train_Plot']
        X_fi2_train_Plot = interface_data['X_fi2_train_Plot']
        
        #################################################################
        # PLOT EXACT SOLUTION
        #################################################################
        fig, ax = newfig(1.0, 1.1)
        gridspec.GridSpec(1,1)
        ax = plt.subplot2grid((1,1), (0,0))
        tcf = ax.tricontourf(triang_total, np.squeeze(u_exact), 100, cmap='jet')
        ax.add_patch(Polygon(XX, closed=True, fill=True, color='w', edgecolor='w'))
        tcbar = fig.colorbar(tcf)
        tcbar.ax.tick_params(labelsize=28)
        ax.set_xlabel('$x$', fontsize=32)
        ax.set_ylabel('$y$', fontsize=32)
        ax.set_title('$u$ (Exact)', fontsize=34)
        ax.tick_params(axis="x", labelsize=28)
        ax.tick_params(axis="y", labelsize=28)   
        plt.plot(X_fi1_train_Plot[:,0:1], X_fi1_train_Plot[:,1:2], 'w-', markersize=2, label='Interface Pts')
        plt.plot(X_fi2_train_Plot[:,0:1], X_fi2_train_Plot[:,1:2], 'w-', markersize=2, label='Interface Pts')
        fig.set_size_inches(w=12, h=9)
        savefig('./figures/XPINN_PoissonEq_ExSol') 
        plt.show()  
        
        #################################################################
        # PLOT PREDICTED SOLUTION
        #################################################################
        fig, ax = newfig(1.0, 1.1)
        gridspec.GridSpec(1,1)
        ax = plt.subplot2grid((1,1), (0,0))
        tcf = ax.tricontourf(triang_total, u_pred.flatten(), 100, cmap='jet')
        ax.add_patch(Polygon(XX, closed=True, fill=True, color='w', edgecolor='w'))
        tcbar = fig.colorbar(tcf)
        tcbar.ax.tick_params(labelsize=28)
        ax.set_xlabel('$x$', fontsize=32)
        ax.set_ylabel('$y$', fontsize=32)
        ax.set_title('$u$ (Predicted)', fontsize=34)
        ax.tick_params(axis="x", labelsize=28)
        ax.tick_params(axis="y", labelsize=28)   
        plt.plot(X_fi1_train_Plot[:,0:1], X_fi1_train_Plot[:,1:2], 'w-', markersize=2, label='Interface Pts')
        plt.plot(X_fi2_train_Plot[:,0:1], X_fi2_train_Plot[:,1:2], 'w-', markersize=2, label='Interface Pts')
        fig.set_size_inches(w=12, h=9)
        savefig('./figures/XPINN_PoissonEq_Sol') 
        plt.show()  
        
        #################################################################
        # PLOT ERROR
        #################################################################
        fig, ax = newfig(1.0, 1.1)
        gridspec.GridSpec(1,1)
        ax = plt.subplot2grid((1,1), (0,0))
        tcf = ax.tricontourf(triang_total, abs(np.squeeze(u_exact)-u_pred.flatten()), 100, cmap='jet')
        ax.add_patch(Polygon(XX, closed=True, fill=True, color='w', edgecolor='w'))   
        tcbar = fig.colorbar(tcf)  
        tcbar.ax.tick_params(labelsize=28)
        ax.set_xlabel('$x$', fontsize=32)
        ax.set_ylabel('$y$', fontsize=32)
        ax.set_title('Point-wise Error', fontsize=34)
        ax.tick_params(axis="x", labelsize=28)
        ax.tick_params(axis="y", labelsize=28)   
        plt.plot(X_fi1_train_Plot[:,0:1], X_fi1_train_Plot[:,1:2], 'w-', markersize=2, label='Interface Pts')
        plt.plot(X_fi2_train_Plot[:,0:1], X_fi2_train_Plot[:,1:2], 'w-', markersize=2, label='Interface Pts')
        fig.set_size_inches(w=12, h=9)
        savefig('./figures/XPINN_PoissonEq_Err') 
        plt.show()

def main():
    if rank == 0:
        print(f"Starting MPI XPINN with {size} processes")
    
    # Load data (only on rank 0, then broadcast)
    if rank == 0:
        data = scipy.io.loadmat('../DATA/XPINN_2D_PoissonEqn.mat')
         
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
        
        # Prepare training data dictionary
        data_dict = {
            'X_f1_train': np.hstack((x_f1.flatten()[:,None], y_f1.flatten()[:,None])),
            'X_f2_train': np.hstack((x_f2.flatten()[:,None], y_f2.flatten()[:,None])),
            'X_f3_train': np.hstack((x_f3.flatten()[:,None], y_f3.flatten()[:,None])),
            'X_fi1_train': np.hstack((xi1.flatten()[:,None], yi1.flatten()[:,None])),
            'X_fi2_train': np.hstack((xi2.flatten()[:,None], yi2.flatten()[:,None])),
            'X_ub_train': np.hstack((xb.flatten()[:,None], yb.flatten()[:,None])),
            'ub_train': data['ub'].flatten()[:,None]
        }
        
        # Prepare test data dictionary for plotting
        test_data_dict = {
            'X_star1': np.hstack((x_f1.flatten()[:,None], y_f1.flatten()[:,None])),
            'X_star2': np.hstack((x_f2.flatten()[:,None], y_f2.flatten()[:,None])),
            'X_star3': np.hstack((x_f3.flatten()[:,None], y_f3.flatten()[:,None]))
        }
        
        # Prepare interface data for plotting
        interface_data = {
            'xb': data['xb'].flatten()[:,None],
            'yb': data['yb'].flatten()[:,None],
            'X_fi1_train_Plot': np.hstack((xi1.flatten()[:,None], yi1.flatten()[:,None])),
            'X_fi2_train_Plot': np.hstack((xi2.flatten()[:,None], yi2.flatten()[:,None]))
        }
        
        # Extract exact solution
        u_exact = np.concatenate([data['u_exact'].flatten(), 
                                 data['u_exact2'].flatten(), 
                                 data['u_exact3'].flatten()])
        
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
        test_data_dict = None
        interface_data = None
        u_exact = None
    
    # Broadcast data to all processes
    data_dict = comm.bcast(data_dict, root=0)
    test_data_dict = comm.bcast(test_data_dict, root=0)
    interface_data = comm.bcast(interface_data, root=0)
    u_exact = comm.bcast(u_exact, root=0)
    
    # Define network architectures for each subdomain
    layers_list = [
        [2, 30, 30, 1],           # Subdomain 1
        [2, 20, 20, 20, 20, 1],   # Subdomain 2
        [2, 25, 25, 25, 1]        # Subdomain 3
    ]
    
    trainer = MPIXPINNTrainer(layers_list, data_dict, test_data_dict)
    
    trainer.train(max_iterations=10)
    print("trainer.train")
    # Plot results (only on rank 0)
    if rank == 0:
        print("Training completed! Generating plots...")
        trainer.plot_results(u_exact, interface_data)
        print("Plotting completed!")

if __name__ == '__main__':
    main()