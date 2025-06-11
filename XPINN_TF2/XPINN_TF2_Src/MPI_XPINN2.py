import sys
sys.path.insert(0, '../Utilities/')
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from plotting import newfig, savefig
import matplotlib.tri as tri
import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon
import warnings
warnings.filterwarnings('ignore')

from mpi4py import MPI
import time

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Set random seeds for reproducibility
np.random.seed(1234 + rank)  # Different seed per rank
torch.manual_seed(1234 + rank)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class XPINN_Network(nn.Module):
    def __init__(self, layers):
        super(XPINN_Network, self).__init__()
        self.layers = nn.ModuleList()
        self.activations = nn.ParameterList()
        
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:  # Don't add activation parameter for output layer
                self.activations.append(nn.Parameter(torch.tensor(0.05)))
        
        # Xavier initialization
        for layer in self.layers[:-1]:  # All layers except the last one
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
        
        # Initialize last layer
        nn.init.xavier_normal_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)
    
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = torch.tanh(20 * self.activations[i] * x)
        
        # Output layer (no activation)
        x = self.layers[-1](x)
        return x

class ParallelXPINNSolver:
    def __init__(self, layers, rank, size):
        self.rank = rank
        self.size = size
        self.net = XPINN_Network(layers).to(device)
        
        # Optimizer
        lr = 0.0008
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        
        # Define neighbors based on rank (assuming 3 subdomains: 0, 1, 2)
        # Subdomain 0 connects to subdomain 1 and 2
        # Subdomain 1 connects to subdomain 0
        # Subdomain 2 connects to subdomain 0
        if rank == 0:
            self.neighbors = [1, 2]
        elif rank == 1:
            self.neighbors = [0]
        elif rank == 2:
            self.neighbors = [0]
        else:
            self.neighbors = []
    
    def pde_loss(self, x, y):
        """Compute PDE residual for Poisson equation: u_xx + u_yy = exp(x) + exp(y)"""
        x.requires_grad_(True)
        y.requires_grad_(True)
        
        u = self.net(torch.cat([x, y], dim=1))
        
        # First derivatives
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), 
                                  create_graph=True, retain_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), 
                                  create_graph=True, retain_graph=True)[0]
        
        # Second derivatives
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), 
                                   create_graph=True, retain_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), 
                                   create_graph=True, retain_graph=True)[0]
        
        # PDE residual
        f = u_xx + u_yy - (torch.exp(x) + torch.exp(y))
        return f, u
    
    def compute_interface_solution(self, xi, yi):
        """Compute solution at interface points"""
        xi.requires_grad_(True)
        yi.requires_grad_(True)
        
        u = self.net(torch.cat([xi, yi], dim=1))
        f, _ = self.pde_loss(xi, yi)
        
        return u, f
    
    def train_step(self, X_ub=None, ub=None, X_f=None, interface_data=None):
        """Single training step for parallel training"""
        
        self.optimizer.zero_grad()
        
        total_loss = 0.0
        
        # Boundary loss (only for rank 0 which has boundary conditions)
        if X_ub is not None and ub is not None and self.rank == 0:
            x_ub, y_ub = X_ub[:, 0:1], X_ub[:, 1:2]
            ub_pred = self.net(torch.cat([x_ub, y_ub], dim=1))
            boundary_loss = torch.mean((ub - ub_pred)**2)
            total_loss += 20 * boundary_loss
        
        # PDE loss for interior points
        if X_f is not None:
            x_f, y_f = X_f[:, 0:1], X_f[:, 1:2]
            f, _ = self.pde_loss(x_f, y_f)
            pde_loss = torch.mean(f**2)
            total_loss += pde_loss
        
        # Interface losses
        interface_loss = 0.0
        if interface_data is not None:
            for neighbor_rank, (X_interface, neighbor_solutions, neighbor_residuals) in interface_data.items():
                if X_interface.shape[0] > 0:
                    x_i, y_i = X_interface[:, 0:1], X_interface[:, 1:2]
                    u_i, f_i = self.compute_interface_solution(x_i, y_i)
                    
                    # Continuity of residuals
                    if neighbor_residuals is not None:
                        residual_continuity = torch.mean((f_i - neighbor_residuals)**2)
                        interface_loss += residual_continuity
                    
                    # Average solution matching
                    if neighbor_solutions is not None:
                        avg_solution = (u_i + neighbor_solutions) / 2
                        avg_loss = torch.mean((u_i - avg_solution)**2)
                        interface_loss += 20 * avg_loss
        
        total_loss += interface_loss
        
        # Backward pass
        if total_loss > 0:
            total_loss.backward()
            self.optimizer.step()
        
        return total_loss.item()
    
    def predict(self, X):
        """Make predictions on test data"""
        self.net.eval()
        
        with torch.no_grad():
            u_pred = self.net(X).cpu().numpy()
        
        return u_pred

def setup_data(rank):
    """Setup training data for each subdomain (executed only by rank 0)"""
    if rank != 0:
        return None
    
    # Parameters
    N_ub = 200
    N_f1 = 5000
    N_f2 = 1800
    N_f3 = 1200
    N_I1 = 100
    N_I2 = 100
    
    # Load data
    data = scipy.io.loadmat('../DATA/XPINN_2D_PoissonEqn.mat')
    
    # Extract data
    x_f1 = data['x_f1'].flatten()[:, None]
    y_f1 = data['y_f1'].flatten()[:, None]
    x_f2 = data['x_f2'].flatten()[:, None]
    y_f2 = data['y_f2'].flatten()[:, None]
    x_f3 = data['x_f3'].flatten()[:, None]
    y_f3 = data['y_f3'].flatten()[:, None]
    xi1 = data['xi1'].flatten()[:, None]
    yi1 = data['yi1'].flatten()[:, None]
    xi2 = data['xi2'].flatten()[:, None]
    yi2 = data['yi2'].flatten()[:, None]
    xb = data['xb'].flatten()[:, None]
    yb = data['yb'].flatten()[:, None]
    
    ub_train = data['ub'].flatten()[:, None]
    u_exact = data['u_exact'].flatten()[:, None]
    u_exact2 = data['u_exact2'].flatten()[:, None]
    u_exact3 = data['u_exact3'].flatten()[:, None]
    
    # Prepare training data for each subdomain
    X_f1_train = np.hstack((x_f1, y_f1))
    X_f2_train = np.hstack((x_f2, y_f2))
    X_f3_train = np.hstack((x_f3, y_f3))
    X_fi1_train = np.hstack((xi1, yi1))
    X_fi2_train = np.hstack((xi2, yi2))
    X_ub_train = np.hstack((xb, yb))
    
    # Random sampling
    idx1 = np.random.choice(X_f1_train.shape[0], N_f1, replace=False)
    X_f1_train = X_f1_train[idx1, :]
    
    idx2 = np.random.choice(X_f2_train.shape[0], N_f2, replace=False)
    X_f2_train = X_f2_train[idx2, :]
    
    idx3 = np.random.choice(X_f3_train.shape[0], N_f3, replace=False)
    X_f3_train = X_f3_train[idx3, :]
    
    idx4 = np.random.choice(X_ub_train.shape[0], N_ub, replace=False)
    X_ub_train = X_ub_train[idx4, :]
    ub_train = ub_train[idx4, :]
    
    idxi1 = np.random.choice(X_fi1_train.shape[0], N_I1, replace=False)
    X_fi1_train = X_fi1_train[idxi1, :]
    
    idxi2 = np.random.choice(X_fi2_train.shape[0], N_I2, replace=False)
    X_fi2_train = X_fi2_train[idxi2, :]
    
    # Organize data by subdomain
    subdomain_data = {
        0: {
            'X_f': X_f1_train,
            'X_ub': X_ub_train,
            'ub': ub_train,
            'interfaces': {1: X_fi1_train, 2: X_fi2_train},
            'X_star': X_f1_train,
            'u_exact': u_exact[idx1, :]
        },
        1: {
            'X_f': X_f2_train,
            'X_ub': None,
            'ub': None,
            'interfaces': {0: X_fi1_train},
            'X_star': X_f2_train,
            'u_exact': u_exact2[idx2, :]
        },
        2: {
            'X_f': X_f3_train,
            'X_ub': None,
            'ub': None,
            'interfaces': {0: X_fi2_train},
            'X_star': X_f3_train,
            'u_exact': u_exact3[idx3, :]
        }
    }
    
    return subdomain_data

def main():
    # Setup data (only rank 0)
    if rank == 0:
        print(f"Setting up data on rank {rank}")
        subdomain_data = setup_data(rank)
        
        # NN architectures for each subdomain
        layers = {
            0: [2, 30, 30, 1],
            1: [2, 20, 20, 20, 20, 1],
            2: [2, 25, 25, 25, 1]
        }
    else:
        subdomain_data = None
        layers = None
    
    # Broadcast data to all processes
    subdomain_data = comm.bcast(subdomain_data, root=0)
    layers = comm.bcast(layers, root=0)
    
    if rank >= 3:  # Only use ranks 0, 1, 2
        return
    
    # Initialize solver for this rank's subdomain
    solver = ParallelXPINNSolver(layers[rank], rank, size)
    
    # Get data for this subdomain
    my_data = subdomain_data[rank]
    
    # Convert to tensors
    if my_data['X_f'] is not None:
        X_f_tensor = torch.FloatTensor(my_data['X_f']).to(device)
    else:
        X_f_tensor = None
    
    if my_data['X_ub'] is not None and my_data['ub'] is not None:
        X_ub_tensor = torch.FloatTensor(my_data['X_ub']).to(device)
        ub_tensor = torch.FloatTensor(my_data['ub']).to(device)
    else:
        X_ub_tensor = None
        ub_tensor = None
    
    # Convert interface data to tensors
    interface_tensors = {}
    for neighbor_rank, X_interface in my_data['interfaces'].items():
        if X_interface is not None:
            interface_tensors[neighbor_rank] = torch.FloatTensor(X_interface).to(device)
    
    # Training loop
    Max_iter = 2
    Loss_hist = []
    
    if rank == 0:
        print("Starting parallel training...")
    
    for epoch in range(Max_iter + 1):
        # Compute interface solutions for neighbors
        interface_data = {}
        
        # Send interface solutions to neighbors
        send_requests = []
        recv_requests = {}
        
        for neighbor_rank in solver.neighbors:
            if neighbor_rank in interface_tensors:
                X_interface = interface_tensors[neighbor_rank]
                if X_interface.shape[0] > 0:
                    u_interface, f_interface = solver.compute_interface_solution(
                        X_interface[:, 0:1], X_interface[:, 1:2]
                    )
                    
                    # Convert to numpy for MPI communication
                    u_interface_np = u_interface.detach().cpu().numpy()
                    f_interface_np = f_interface.detach().cpu().numpy()
                    
                    # Non-blocking send
                    send_requests.append(comm.isend(u_interface_np, dest=neighbor_rank, tag=epoch*10+rank*2))
                    send_requests.append(comm.isend(f_interface_np, dest=neighbor_rank, tag=epoch*10+rank*2+1))

                    # Non-blocking receive
                    recv_u_interface = comm.irecv(source=neighbor_rank, tag=epoch*10 + neighbor_rank*2)
                    recv_f_interface = comm.irecv(source=neighbor_rank, tag=epoch*10 + neighbor_rank*2+1)
                    recv_requests[neighbor_rank] = (recv_u_interface, recv_f_interface)

        # Wait for all send and receive operations to complete
        all_requests = send_requests + [r for pair in recv_requests.values() for r in pair]
        MPI.Request.waitall(all_requests)
        '''
        # Receive interface solutions from neighbors
        neighbor_solutions = {}
        neighbor_residuals = {}
        
        for neighbor_rank in solver.neighbors:
            if neighbor_rank in interface_tensors:
                X_interface = interface_tensors[neighbor_rank]
                if X_interface.shape[0] > 0:
                    # Non-blocking receive
                    recv_req1 = comm.irecv(source=neighbor_rank, tag=epoch*10+neighbor_rank*2)
                    recv_req2 = comm.irecv(source=neighbor_rank, tag=epoch*10+neighbor_rank*2+1)
                    recv_requests.extend([recv_req1, recv_req2])
                    
                    # Store for later processing
                    neighbor_solutions[neighbor_rank] = recv_req1
                    neighbor_residuals[neighbor_rank] = recv_req2
        
        # Wait for all communications to complete
        MPI.Request.waitall(send_requests + recv_requests)
        '''        
        # Process received data
        interface_data = {}
        for neighbor_rank in solver.neighbors:
            if neighbor_rank in interface_tensors and neighbor_rank in recv_requests:
                X_interface = interface_tensors[neighbor_rank]
                if X_interface.shape[0] > 0:
                    recv_u, recv_f = recv_requests[neighbor_rank]
                    u_data = recv_u.wait()
                    f_data = recv_f.wait()

                    if u_data is not None and f_data is not None:
                        u_neighbor = torch.FloatTensor(u_data).to(device)
                        f_neighbor = torch.FloatTensor(f_data).to(device)
                        interface_data[neighbor_rank] = (X_interface, u_neighbor, f_neighbor)
                    else:
                        interface_data[neighbor_rank] = (X_interface, None, None)
        
        # Perform training step
        loss = solver.train_step(X_ub_tensor, ub_tensor, X_f_tensor, interface_data)
        Loss_hist.append(loss)
        
        if epoch % 100 == 0:
            print(f"Rank {rank}, Iteration: {epoch}, Loss: {loss:.6f}")
    
    # Prediction phase
    if rank == 0:
        print("Making predictions...")
    
    X_star_tensor = torch.FloatTensor(my_data['X_star']).to(device)
    u_pred = solver.predict(X_star_tensor)
    
    # Gather all predictions to rank 0 for plotting
    all_predictions = comm.gather(u_pred, root=0)
    all_exact = comm.gather(my_data['u_exact'], root=0)
    all_coords = comm.gather(my_data['X_star'], root=0)
    all_losses = comm.gather(Loss_hist, root=0)
    
    if rank == 0:
        print("Training completed. Generating plots...")
        
        # Combine all predictions
        u_pred_combined = np.concatenate(all_predictions)
        u_exact_combined = np.concatenate(all_exact)
        coords_combined = np.concatenate(all_coords)
        
        # Plotting
        import matplotlib as mpl
        mpl.rcParams.update(mpl.rcParamsDefault)
        
        x_tot = coords_combined[:, 0:1]
        y_tot = coords_combined[:, 1:2]
        print(f"coords_combined shape: {coords_combined.shape}")
        print(f"u_exact_combined shape: {u_exact_combined.shape}")

        triang_total = tri.Triangulation(x_tot.flatten(), y_tot.flatten())
        
        # Loss history plot
        fig, ax = newfig(1.0, 1.1)
        colors = ['r-', 'b-.', 'g--']
        labels = ['Sub-Net0', 'Sub-Net1', 'Sub-Net2']
        for i, (loss_hist, color, label) in enumerate(zip(all_losses, colors, labels)):
            plt.plot(range(0, Max_iter + 1, 1), loss_hist, color, linewidth=1, label=label)
        plt.xlabel('$\#$iterations')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.legend(loc='upper right')
        savefig('./figures/ParallelXPINN_PoissonMSEhistory')
        
        # Load original data for plotting setup
        data = scipy.io.loadmat('../DATA/XPINN_2D_PoissonEqn.mat')
        xb = data['xb'].flatten()[:, None]
        yb = data['yb'].flatten()[:, None]
        xi1 = data['xi1'].flatten()[:, None]
        yi1 = data['yi1'].flatten()[:, None]
        xi2 = data['xi2'].flatten()[:, None]
        yi2 = data['yi2'].flatten()[:, None]
        
        # Create patches for plotting
        aa1 = np.array([[np.squeeze(xb[-1]), np.squeeze(yb[-1])]])
        aa2 = np.array([[1.8, np.squeeze(yb[-1])], [+1.8, -1.7], [-1.6, -1.7], 
                        [-1.6, 1.55], [1.8, 1.55], [1.8, np.squeeze(yb[-1])]])
        x_domain1 = np.squeeze(xb.flatten()[:, None])
        y_domain1 = np.squeeze(yb.flatten()[:, None])
        aa3 = np.array([x_domain1, y_domain1]).T
        XX = np.vstack((aa3, aa2, aa1))
        
        X_fi1_train_Plot = np.hstack((xi1, yi1))
        X_fi2_train_Plot = np.hstack((xi2, yi2))
        
        # Exact solution plot
        fig, ax = newfig(1.0, 1.1)
        gridspec.GridSpec(1, 1)
        ax = plt.subplot2grid((1, 1), (0, 0))
        tcf = ax.tricontourf(triang_total, np.squeeze(u_exact_combined), 100, cmap='jet')
        ax.add_patch(Polygon(XX, closed=True, fill=True, color='w', edgecolor='w'))
        tcbar = fig.colorbar(tcf)
        tcbar.ax.tick_params(labelsize=28)
        ax.set_xlabel('$x$', fontsize=32)
        ax.set_ylabel('$y$', fontsize=32)
        ax.set_title('$u$ (Exact)', fontsize=34)
        ax.tick_params(axis="x", labelsize=28)
        ax.tick_params(axis="y", labelsize=28)
        plt.plot(X_fi1_train_Plot[:, 0:1], X_fi1_train_Plot[:, 1:2], 'w-', markersize=2, label='Interface Pts')
        plt.plot(X_fi2_train_Plot[:, 0:1], X_fi2_train_Plot[:, 1:2], 'w-', markersize=2, label='Interface Pts')
        fig.set_size_inches(w=12, h=9)
        savefig('./figures/ParallelXPINN_PoissonEq_ExSol')
        plt.show()
        
        # Predicted solution plot
        fig, ax = newfig(1.0, 1.1)
        gridspec.GridSpec(1, 1)
        ax = plt.subplot2grid((1, 1), (0, 0))
        tcf = ax.tricontourf(triang_total, u_pred_combined.flatten(), 100, cmap='jet')
        ax.add_patch(Polygon(XX, closed=True, fill=True, color='w', edgecolor='w'))
        tcbar = fig.colorbar(tcf)
        tcbar.ax.tick_params(labelsize=28)
        ax.set_xlabel('$x$', fontsize=32)
        ax.set_ylabel('$y$', fontsize=32)
        ax.set_title('$u$ (Predicted - Parallel)', fontsize=34)
        #ax.tick_params(axis="x", labelsize=28)
        #ax.tick_params(axis="y", labelsize=28)
        plt.plot(X_fi1_train_Plot[:, 0:1], X_fi1_train_Plot[:, 1:2], 'w-', markersize=2, label='Interface Pts')
        plt.plot(X_fi2_train_Plot[:, 0:1], X_fi2_train_Plot[:, 1:2], 'w-', markersize=2, label='Interface Pts')
        fig.set_size_inches(w=12, h=9)
        savefig('./figures/ParallelXPINN_PoissonEq_Sol')
        plt.show()
        
        # Error plot
        fig, ax = newfig(1.0, 1.1)
        gridspec.GridSpec(1, 1)
        ax = plt.subplot2grid((1, 1), (0, 0))
        tcf = ax.tricontourf(triang_total, abs(np.squeeze(u_exact_combined) - u_pred_combined.flatten()), 100, cmap='jet')
        ax.add_patch(Polygon(XX, closed=True, fill=True, color='w', edgecolor='w'))
        tcbar = fig.colorbar(tcf)
        tcbar.ax.tick_params(labelsize=28)
        ax.set_xlabel('$x$', fontsize=32)
        ax.set_ylabel('$y$', fontsize=32)
        ax.set_title('Point-wise Error (Parallel)', fontsize=34)
        ax.tick_params(axis="x", labelsize=28)
        ax.tick_params(axis="y", labelsize=28)
        plt.plot(X_fi1_train_Plot[:, 0:1], X_fi1_train_Plot[:, 1:2], 'w-', markersize=2, label='Interface Pts')
        plt.plot(X_fi2_train_Plot[:, 0:1], X_fi2_train_Plot[:, 1:2], 'w-', markersize=2, label='Interface Pts')
        fig.set_size_inches(w=12, h=9)
        savefig('./figures/ParallelXPINN_PoissonEq_Err')
        plt.show()
        
        print("Parallel XPINN training and visualization completed!")

if __name__ == '__main__':
    main()