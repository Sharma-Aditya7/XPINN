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
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Set random seeds for reproducibility
np.random.seed(1234)
torch.manual_seed(1234)

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

class XPINNSolver:
    def __init__(self, layers1, layers2, layers3):
        self.net1 = XPINN_Network(layers1).to(device)
        self.net2 = XPINN_Network(layers2).to(device)
        self.net3 = XPINN_Network(layers3).to(device)
        
        # Optimizers
        lr = 0.0008
        self.optimizer1 = optim.Adam(self.net1.parameters(), lr=lr)
        self.optimizer2 = optim.Adam(self.net2.parameters(), lr=lr)
        self.optimizer3 = optim.Adam(self.net3.parameters(), lr=lr)
    
    def pde_loss(self, x, y, net):
        """Compute PDE residual for Poisson equation: u_xx + u_yy = exp(x) + exp(y)"""
        x.requires_grad_(True)
        y.requires_grad_(True)
        
        u = net(torch.cat([x, y], dim=1))
        
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
        return f, u, u_xx, u_yy
    
    def interface_continuity_loss(self, xi, yi, net1, net2):
        """Compute interface continuity loss between two networks"""
        xi.requires_grad_(True)
        yi.requires_grad_(True)
        
        # Solutions from both networks at interface
        u1 = net1(torch.cat([xi, yi], dim=1))
        u2 = net2(torch.cat([xi, yi], dim=1))
        
        # Compute PDE residuals at interface
        f1, _, u1_xx, u1_yy = self.pde_loss(xi, yi, net1)
        f2, _, u2_xx, u2_yy = self.pde_loss(xi, yi, net2)
        
        # Interface continuity of residuals
        fi = f1 - f2
        
        # Average solution at interface
        u_avg = (u1 + u2) / 2
        
        return fi, u_avg, u1, u2
    
    def train_step(self, X_ub, ub, X_f1, X_f2, X_f3, X_fi1, X_fi2):
        """Single training step"""
        
        # Extract coordinates
        x_ub, y_ub = X_ub[:, 0:1], X_ub[:, 1:2]
        x_f1, y_f1 = X_f1[:, 0:1], X_f1[:, 1:2]
        x_f2, y_f2 = X_f2[:, 0:1], X_f2[:, 1:2]
        x_f3, y_f3 = X_f3[:, 0:1], X_f3[:, 1:2]
        x_fi1, y_fi1 = X_fi1[:, 0:1], X_fi1[:, 1:2]
        x_fi2, y_fi2 = X_fi2[:, 0:1], X_fi2[:, 1:2]
        
        # Zero gradients
        self.optimizer1.zero_grad()
        self.optimizer2.zero_grad()
        self.optimizer3.zero_grad()
        
        # Boundary loss for network 1
        ub1_pred = self.net1(torch.cat([x_ub, y_ub], dim=1))
        boundary_loss = torch.mean((ub - ub1_pred)**2)
        
        # PDE losses
        f1, _, _, _ = self.pde_loss(x_f1, y_f1, self.net1)
        f2, _, _, _ = self.pde_loss(x_f2, y_f2, self.net2)
        f3, _, _, _ = self.pde_loss(x_f3, y_f3, self.net3)
        
        pde_loss1 = torch.mean(f1**2)
        pde_loss2 = torch.mean(f2**2)
        pde_loss3 = torch.mean(f3**2)
        
        # Interface continuity losses
        fi1, uavgi1, u1i1, u2i1 = self.interface_continuity_loss(x_fi1, y_fi1, self.net1, self.net2)
        fi2, uavgi2, u1i2, u3i2 = self.interface_continuity_loss(x_fi2, y_fi2, self.net1, self.net3)
        
        interface_loss1 = torch.mean(fi1**2)
        interface_loss2 = torch.mean(fi2**2)
        
        # Average solution matching losses
        avg_loss1_i1 = torch.mean((u1i1 - uavgi1)**2)
        avg_loss1_i2 = torch.mean((u1i2 - uavgi2)**2)
        avg_loss2_i1 = torch.mean((u2i1 - uavgi1)**2)
        avg_loss3_i2 = torch.mean((u3i2 - uavgi2)**2)
        
        # Total losses for each network
        loss1 = (20 * boundary_loss + pde_loss1 + 1 * interface_loss1 + 
                1 * interface_loss2 + 20 * avg_loss1_i1 + 20 * avg_loss1_i2)
        
        loss2 = pde_loss2 + 1 * interface_loss1 + 20 * avg_loss2_i1
        
        loss3 = pde_loss3 + 1 * interface_loss2 + 20 * avg_loss3_i2
        
        # Backward pass
        loss1.backward(retain_graph=True)
        loss2.backward(retain_graph=True)
        loss3.backward()
        
        # Update parameters
        self.optimizer1.step()
        self.optimizer2.step()
        self.optimizer3.step()
        
        return loss1.item(), loss2.item(), loss3.item(), ub1_pred.detach()
    
    def predict(self, X1, X2, X3):
        """Make predictions on test data"""
        self.net1.eval()
        self.net2.eval()
        self.net3.eval()
        
        with torch.no_grad():
            u_pred1 = self.net1(X1).cpu().numpy()
            u_pred2 = self.net2(X2).cpu().numpy()
            u_pred3 = self.net3(X3).cpu().numpy()
        
        return u_pred1, u_pred2, u_pred3

def main():
    # Parameters
    N_ub = 200
    N_f1 = 5000
    N_f2 = 1800
    N_f3 = 1200
    N_I1 = 100
    N_I2 = 100
    
    # NN architectures
    layers1 = [2, 30, 30, 1]
    layers2 = [2, 20, 20, 20, 20, 1]
    layers3 = [2, 25, 25, 25, 1]
    
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
    
    # Prepare training data
    X_f1_train = np.hstack((x_f1, y_f1))
    X_f2_train = np.hstack((x_f2, y_f2))
    X_f3_train = np.hstack((x_f3, y_f3))
    X_fi1_train = np.hstack((xi1, yi1))
    X_fi2_train = np.hstack((xi2, yi2))
    X_ub_train = np.hstack((xb, yb))
    
    # Test data
    X_star1 = np.hstack((x_f1, y_f1))
    X_star2 = np.hstack((x_f2, y_f2))
    X_star3 = np.hstack((x_f3, y_f3))
    
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
    
    # Convert to tensors
    X_ub_train_tensor = torch.FloatTensor(X_ub_train).to(device)
    ub_train_tensor = torch.FloatTensor(ub_train).to(device)
    X_f1_train_tensor = torch.FloatTensor(X_f1_train).to(device)
    X_f2_train_tensor = torch.FloatTensor(X_f2_train).to(device)
    X_f3_train_tensor = torch.FloatTensor(X_f3_train).to(device)
    X_fi1_train_tensor = torch.FloatTensor(X_fi1_train).to(device)
    X_fi2_train_tensor = torch.FloatTensor(X_fi2_train).to(device)
    
    # Initialize solver
    solver = XPINNSolver(layers1, layers2, layers3)
    
    # Training
    Max_iter = 200
    Loss1_hist = []
    Loss2_hist = []
    Loss3_hist = []
    
    for m in range(Max_iter + 1):
        loss_1, loss_2, loss_3, u_pred = solver.train_step(
            X_ub_train_tensor, ub_train_tensor, X_f1_train_tensor,
            X_f2_train_tensor, X_f3_train_tensor, X_fi1_train_tensor,
            X_fi2_train_tensor
        )
        
        if m % 100 == 0:
            print(f"Iteration: {m}, loss1: {loss_1:.6f}, loss2: {loss_2:.6f}, loss3: {loss_3:.6f}")
        Loss1_hist.append(loss_1)
        Loss2_hist.append(loss_2)
        Loss3_hist.append(loss_3)
    
    # Prediction
    X1_tensor = torch.FloatTensor(X_star1).to(device)
    X2_tensor = torch.FloatTensor(X_star2).to(device)
    X3_tensor = torch.FloatTensor(X_star3).to(device)
    
    u_pred1, u_pred2, u_pred3 = solver.predict(X1_tensor, X2_tensor, X3_tensor)
    u_pred = np.concatenate([u_pred1, u_pred2, u_pred3])
    
    # Plotting
    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)
    
    X1, Y1 = X_star1[:, 0:1], X_star1[:, 1:2]
    X2, Y2 = X_star2[:, 0:1], X_star2[:, 1:2]
    X3, Y3 = X_star3[:, 0:1], X_star3[:, 1:2]
    
    x_tot = np.concatenate([X1, X2, X3])
    y_tot = np.concatenate([Y1, Y2, Y3])
    triang_total = tri.Triangulation(x_tot.flatten(), y_tot.flatten())
    
    # Loss history plot
    fig, ax = newfig(1.0, 1.1)
    plt.plot(range(0, Max_iter + 1, 1), Loss1_hist, 'r-', linewidth=1, label='Sub-Net1')
    plt.plot(range(0, Max_iter + 1, 1), Loss2_hist, 'b-.', linewidth=1, label='Sub-Net2')
    plt.plot(range(0, Max_iter + 1, 1), Loss3_hist, 'g--', linewidth=1, label='Sub-Net3')
    plt.xlabel('$\#$iterations')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend(loc='upper right')
    savefig('./figures/XPINN_PoissonMSEhistory')
    
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
    tcf = ax.tricontourf(triang_total, np.squeeze(u_exact), 100, cmap='jet')
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
    savefig('./figures/XPINN_PoissonEq_ExSol')
    plt.show()
    
    # Predicted solution plot
    fig, ax = newfig(1.0, 1.1)
    gridspec.GridSpec(1, 1)
    ax = plt.subplot2grid((1, 1), (0, 0))
    tcf = ax.tricontourf(triang_total, u_pred.flatten(), 100, cmap='jet')
    ax.add_patch(Polygon(XX, closed=True, fill=True, color='w', edgecolor='w'))
    tcbar = fig.colorbar(tcf)
    tcbar.ax.tick_params(labelsize=28)
    ax.set_xlabel('$x$', fontsize=32)
    ax.set_ylabel('$y$', fontsize=32)
    ax.set_title('$u$ (Predicted)', fontsize=34)
    ax.tick_params(axis="x", labelsize=28)
    ax.tick_params(axis="y", labelsize=28)
    plt.plot(X_fi1_train_Plot[:, 0:1], X_fi1_train_Plot[:, 1:2], 'w-', markersize=2, label='Interface Pts')
    plt.plot(X_fi2_train_Plot[:, 0:1], X_fi2_train_Plot[:, 1:2], 'w-', markersize=2, label='Interface Pts')
    fig.set_size_inches(w=12, h=9)
    savefig('./figures/XPINN_PoissonEq_Sol')
    plt.show()
    
    # Error plot
    fig, ax = newfig(1.0, 1.1)
    gridspec.GridSpec(1, 1)
    ax = plt.subplot2grid((1, 1), (0, 0))
    tcf = ax.tricontourf(triang_total, abs(np.squeeze(u_exact) - u_pred.flatten()), 100, cmap='jet')
    ax.add_patch(Polygon(XX, closed=True, fill=True, color='w', edgecolor='w'))
    tcbar = fig.colorbar(tcf)
    tcbar.ax.tick_params(labelsize=28)
    ax.set_xlabel('$x$', fontsize=32)
    ax.set_ylabel('$y$', fontsize=32)
    ax.set_title('Point-wise Error', fontsize=34)
    ax.tick_params(axis="x", labelsize=28)
    ax.tick_params(axis="y", labelsize=28)
    plt.plot(X_fi1_train_Plot[:, 0:1], X_fi1_train_Plot[:, 1:2], 'w-', markersize=2, label='Interface Pts')
    plt.plot(X_fi2_train_Plot[:, 0:1], X_fi2_train_Plot[:, 1:2], 'w-', markersize=2, label='Interface Pts')
    fig.set_size_inches(w=12, h=9)
    savefig('./figures/XPINN_PoissonEq_Err')
    plt.show()

if __name__ == '__main__':
    main()