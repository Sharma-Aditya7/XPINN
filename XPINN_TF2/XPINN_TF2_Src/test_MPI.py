import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon, Rectangle
import warnings
warnings.filterwarnings('ignore')
import time

# Set random seeds for reproducibility
np.random.seed(1234)
torch.manual_seed(1234)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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

class NavierStokesXPINNSolver:
    def __init__(self, layers, Re=100, n_subdomains=12):
        # Create 12 networks for 12 subdomains (3x4 grid)
        self.n_subdomains = n_subdomains
        self.networks = nn.ModuleList([XPINN_Network(layers).to(device) for _ in range(n_subdomains)])
        
        self.Re = Re  # Reynolds number
        
        # Optimizers for each network
        lr = 0.001
        self.optimizers = [optim.Adam(net.parameters(), lr=lr) for net in self.networks]
        
        # Subdomain grid configuration (3 columns x 4 rows)
        self.n_cols = 3
        self.n_rows = 4
        
        # Define subdomain boundaries
        self.x_bounds = np.linspace(0, 1, self.n_cols + 1)  # [0, 1/3, 2/3, 1]
        self.y_bounds = np.linspace(0, 1, self.n_rows + 1)  # [0, 1/4, 1/2, 3/4, 1]
    
    def get_subdomain_bounds(self, subdomain_id):
        """Get the bounds for a given subdomain (0-indexed)"""
        row = subdomain_id // self.n_cols
        col = subdomain_id % self.n_cols
        
        x_min, x_max = self.x_bounds[col], self.x_bounds[col + 1]
        y_min, y_max = self.y_bounds[row], self.y_bounds[row + 1]
        
        return x_min, x_max, y_min, y_max
    
    def navier_stokes_residual(self, x, y, net):
        """Compute Navier-Stokes residual for 2D incompressible flow"""
        x.requires_grad_(True)
        y.requires_grad_(True)
        
        # Network outputs: [u, v, p]
        output = net(torch.cat([x, y], dim=1))
        u, v, p = output[:, 0:1], output[:, 1:2], output[:, 2:3]
        
        # First derivatives
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), 
                                  create_graph=True, retain_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), 
                                  create_graph=True, retain_graph=True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), 
                                  create_graph=True, retain_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), 
                                  create_graph=True, retain_graph=True)[0]
        p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), 
                                  create_graph=True, retain_graph=True)[0]
        p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), 
                                  create_graph=True, retain_graph=True)[0]
        
        # Second derivatives
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), 
                                   create_graph=True, retain_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), 
                                   create_graph=True, retain_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), 
                                   create_graph=True, retain_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), 
                                   create_graph=True, retain_graph=True)[0]
        
        # Navier-Stokes equations
        # Momentum equation in x-direction
        f_u = u * u_x + v * u_y + p_x - (1/self.Re) * (u_xx + u_yy)
        
        # Momentum equation in y-direction  
        f_v = u * v_x + v * v_y + p_y - (1/self.Re) * (v_xx + v_yy)
        
        # Continuity equation
        f_c = u_x + v_y
        
        return f_u, f_v, f_c, u, v, p
    
    def boundary_loss(self, x_b, y_b, net, bc_type, bc_values=None):
        """Compute boundary condition losses"""
        output = net(torch.cat([x_b, y_b], dim=1))
        u_pred, v_pred, p_pred = output[:, 0:1], output[:, 1:2], output[:, 2:3]
        
        if bc_type == 'wall':
            # No-slip boundary condition: u = v = 0
            loss_u = torch.mean(u_pred**2)
            loss_v = torch.mean(v_pred**2)
            return loss_u + loss_v
        
        elif bc_type == 'lid':
            # Moving lid: u = 1, v = 0
            loss_u = torch.mean((u_pred - 1.0)**2)
            loss_v = torch.mean(v_pred**2)
            return loss_u + loss_v
        
        else:
            return torch.tensor(0.0, device=device)
    
    def interface_continuity_loss(self, xi, yi, net1, net2):
        """Compute interface continuity loss between two networks"""
        xi.requires_grad_(True)
        yi.requires_grad_(True)
        
        # Solutions from both networks at interface
        output1 = net1(torch.cat([xi, yi], dim=1))
        output2 = net2(torch.cat([xi, yi], dim=1))
        
        u1, v1, p1 = output1[:, 0:1], output1[:, 1:2], output1[:, 2:3]
        u2, v2, p2 = output2[:, 0:1], output2[:, 1:2], output2[:, 2:3]
        
        # Continuity of solution values
        loss_u = torch.mean((u1 - u2)**2)
        loss_v = torch.mean((v1 - v2)**2)
        loss_p = torch.mean((p1 - p2)**2)
        
        # Compute residuals at interface
        f_u1, f_v1, f_c1, _, _, _ = self.navier_stokes_residual(xi, yi, net1)
        f_u2, f_v2, f_c2, _, _, _ = self.navier_stokes_residual(xi, yi, net2)
        
        # Continuity of residuals
        loss_f_u = torch.mean((f_u1 - f_u2)**2)  
        loss_f_v = torch.mean((f_v1 - f_v2)**2)
        loss_f_c = torch.mean((f_c1 - f_c2)**2)
        
        return loss_u + loss_v + loss_p + loss_f_u + loss_f_v + loss_f_c
    
    def train_step(self, collocation_data, boundary_data, interface_data):
        """Single training step"""
        
        # Zero gradients for all optimizers
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        
        total_losses = [torch.tensor(0.0, device=device) for _ in range(self.n_subdomains)]
        
        # PDE losses for each subdomain
        for i in range(self.n_subdomains):
            X_f = collocation_data[f'X_f{i}']
            x_f, y_f = X_f[:, 0:1], X_f[:, 1:2]
            f_u, f_v, f_c, _, _, _ = self.navier_stokes_residual(x_f, y_f, self.networks[i])
            
            pde_loss = torch.mean(f_u**2) + torch.mean(f_v**2) + torch.mean(f_c**2)
            total_losses[i] += pde_loss
        
        # Boundary losses
        for bc_data in boundary_data:
            X_b = bc_data['coords']
            bc_type = bc_data['type']
            subdomain = bc_data['subdomain']
            
            x_b, y_b = X_b[:, 0:1], X_b[:, 1:2]
            bc_loss = self.boundary_loss(x_b, y_b, self.networks[subdomain], bc_type)
            total_losses[subdomain] += 10 * bc_loss
        
        # Interface losses
        for interface in interface_data:
            X_i = interface['coords']
            net_pair = interface['networks']
            
            x_i, y_i = X_i[:, 0:1], X_i[:, 1:2]
            
            net1_id, net2_id = net_pair
            interface_loss = self.interface_continuity_loss(x_i, y_i, 
                                                          self.networks[net1_id], 
                                                          self.networks[net2_id])
            total_losses[net1_id] += 5 * interface_loss
            total_losses[net2_id] += 5 * interface_loss
        
        # Backward pass for all networks
        for i in range(self.n_subdomains - 1):
            total_losses[i].backward(retain_graph=True)
        total_losses[-1].backward()  # Last one doesn't need retain_graph
        
        # Update parameters
        for optimizer in self.optimizers:
            optimizer.step()
        
        return [loss.item() for loss in total_losses]
    
    def predict(self, X_test):
        """Make predictions on test data"""
        for net in self.networks:
            net.eval()
        
        predictions = {}
        
        with torch.no_grad():
            for i in range(self.n_subdomains):
                X = X_test[f'X{i}']
                if X is not None and len(X) > 0:
                    pred = self.networks[i](X).cpu().numpy()
                    predictions[f'pred{i}'] = pred
        
        return predictions

def save_model(solver, filename='xpinn_lid_driven_cavity.pth'):
    """Save all XPINN subdomain networks to a file"""
    model_state = {
        'networks': [net.state_dict() for net in solver.networks],
        'Re': solver.Re,
        'n_subdomains': solver.n_subdomains
    }
    torch.save(model_state, filename)
    print(f"Model saved to {filename}")

def generate_cavity_data():
    """Generate training data for lid-driven cavity with 12 subdomains"""
    
    # Domain: [0,1] x [0,1] divided into 3x4 grid
    n_cols, n_rows = 3, 4
    x_bounds = np.linspace(0, 1, n_cols + 1)
    y_bounds = np.linspace(0, 1, n_rows + 1)
    
    # Collocation points for PDE
    N_f = 500  # points per subdomain (reduced for 12 subdomains)
    
    collocation_data = {}
    
    # Generate collocation points for each subdomain
    for row in range(n_rows):
        for col in range(n_cols):
            subdomain_id = row * n_cols + col
            
            x_min, x_max = x_bounds[col], x_bounds[col + 1]
            y_min, y_max = y_bounds[row], y_bounds[row + 1]
            
            # Add small overlap for better continuity
            overlap = 0.01
            x_min_overlap = max(0, x_min - overlap)
            x_max_overlap = min(1, x_max + overlap)
            y_min_overlap = max(0, y_min - overlap)
            y_max_overlap = min(1, y_max + overlap)
            
            x = np.random.uniform(x_min_overlap, x_max_overlap, N_f)
            y = np.random.uniform(y_min_overlap, y_max_overlap, N_f)
            X_f = np.column_stack([x, y])
            
            collocation_data[f'X_f{subdomain_id}'] = torch.FloatTensor(X_f).to(device)
    
    # Boundary conditions
    N_b = 50  # points per boundary segment
    boundary_data = []
    
    # Bottom wall: y = 0
    for col in range(n_cols):
        subdomain_id = (n_rows - 1) * n_cols + col  # Bottom row subdomains
        x_min, x_max = x_bounds[col], x_bounds[col + 1]
        
        x_bottom = np.linspace(x_min, x_max, N_b)
        y_bottom = np.zeros_like(x_bottom)
        boundary_data.append({
            'coords': torch.FloatTensor(np.column_stack([x_bottom, y_bottom])).to(device),
            'type': 'wall',
            'subdomain': subdomain_id
        })
    
    # Top wall (lid): y = 1
    for col in range(n_cols):
        subdomain_id = col  # Top row subdomains
        x_min, x_max = x_bounds[col], x_bounds[col + 1]
        
        x_top = np.linspace(x_min, x_max, N_b)
        y_top = np.ones_like(x_top)
        boundary_data.append({
            'coords': torch.FloatTensor(np.column_stack([x_top, y_top])).to(device),
            'type': 'lid',
            'subdomain': subdomain_id
        })
    
    # Left wall: x = 0
    for row in range(n_rows):
        subdomain_id = row * n_cols  # Leftmost subdomains
        y_min, y_max = y_bounds[row], y_bounds[row + 1]
        
        x_left = np.zeros(N_b)
        y_left = np.linspace(y_min, y_max, N_b)
        boundary_data.append({
            'coords': torch.FloatTensor(np.column_stack([x_left, y_left])).to(device),
            'type': 'wall',
            'subdomain': subdomain_id
        })
    
    # Right wall: x = 1
    for row in range(n_rows):
        subdomain_id = row * n_cols + (n_cols - 1)  # Rightmost subdomains
        y_min, y_max = y_bounds[row], y_bounds[row + 1]
        
        x_right = np.ones(N_b)
        y_right = np.linspace(y_min, y_max, N_b)
        boundary_data.append({
            'coords': torch.FloatTensor(np.column_stack([x_right, y_right])).to(device),
            'type': 'wall',
            'subdomain': subdomain_id
        })
    
    # Interface data
    N_i = 30
    interface_data = []
    
    # Vertical interfaces (between columns)
    for row in range(n_rows):
        for col in range(n_cols - 1):
            subdomain1 = row * n_cols + col
            subdomain2 = row * n_cols + col + 1
            
            x_interface = x_bounds[col + 1]
            y_min, y_max = y_bounds[row], y_bounds[row + 1]
            
            x_i = x_interface * np.ones(N_i)
            y_i = np.linspace(y_min, y_max, N_i)
            interface_data.append({
                'coords': torch.FloatTensor(np.column_stack([x_i, y_i])).to(device),
                'networks': (subdomain1, subdomain2)
            })
    
    # Horizontal interfaces (between rows)
    for row in range(n_rows - 1):
        for col in range(n_cols):
            subdomain1 = row * n_cols + col
            subdomain2 = (row + 1) * n_cols + col
            
            y_interface = y_bounds[row + 1]
            x_min, x_max = x_bounds[col], x_bounds[col + 1]
            
            x_i = np.linspace(x_min, x_max, N_i)
            y_i = y_interface * np.ones_like(x_i)
            interface_data.append({
                'coords': torch.FloatTensor(np.column_stack([x_i, y_i])).to(device),
                'networks': (subdomain1, subdomain2)
            })
    
    return collocation_data, boundary_data, interface_data

def generate_test_data():
    """Generate test data for visualization"""
    nx, ny = 60, 80  # Adjusted for 3x4 grid
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    
    n_cols, n_rows = 3, 4
    x_bounds = np.linspace(0, 1, n_cols + 1)
    y_bounds = np.linspace(0, 1, n_rows + 1)
    
    test_data = {}
    masks = {}
    
    # Generate test points for each subdomain
    for row in range(n_rows):
        for col in range(n_cols):
            subdomain_id = row * n_cols + col
            
            x_min, x_max = x_bounds[col], x_bounds[col + 1]
            y_min, y_max = y_bounds[row], y_bounds[row + 1]
            
            # Create mask for this subdomain
            mask = (X >= x_min) & (X <= x_max) & (Y >= y_min) & (Y <= y_max)
            masks[f'mask{subdomain_id}'] = mask
            
            if np.any(mask):
                X_test = torch.FloatTensor(np.column_stack([X[mask].flatten(), Y[mask].flatten()])).to(device)
                test_data[f'X{subdomain_id}'] = X_test
            else:
                test_data[f'X{subdomain_id}'] = None
    
    test_data.update({
        'grid': (X, Y),
        'masks': masks,
        'bounds': (x_bounds, y_bounds)
    })
    
    return test_data

def visualize_subdomains():
    """Visualize the subdomain decomposition"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 10))
    
    n_cols, n_rows = 3, 4
    x_bounds = np.linspace(0, 1, n_cols + 1)
    y_bounds = np.linspace(0, 1, n_rows + 1)
    
    colors = plt.cm.Set3(np.linspace(0, 1, 12))
    
    for row in range(n_rows):
        for col in range(n_cols):
            subdomain_id = row * n_cols + col
            
            x_min, x_max = x_bounds[col], x_bounds[col + 1]
            y_min, y_max = y_bounds[row], y_bounds[row + 1]
            
            rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                           facecolor=colors[subdomain_id], alpha=0.7, edgecolor='black')
            ax.add_patch(rect)
            
            # Add subdomain label
            ax.text((x_min + x_max) / 2, (y_min + y_max) / 2, f'S{subdomain_id}',
                   ha='center', va='center', fontsize=12, fontweight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('12 Cartesian Subdomains (3×4 Grid)')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()

def main():
    # Visualize subdomain decomposition
    print("Visualizing subdomain decomposition...")
    visualize_subdomains()
    
    # Parameters
    Re = 100  # Reynolds number
    
    # NN architecture - output 3 variables: [u, v, p]
    layers = [2, 80, 80, 80, 80, 80, 3]  # Slightly smaller networks due to more subdomains
    
    print("Generating training data...")
    collocation_data, boundary_data, interface_data = generate_cavity_data()
    
    print("Generating test data...")
    test_data = generate_test_data()
    
    print("Initializing XPINN solver with 12 subdomains...")
    solver = NavierStokesXPINNSolver(layers, Re=Re, n_subdomains=12)
    
    # Training
    Max_iter = 10000  # Increased iterations for more complex problem
    Loss_hist = [[] for _ in range(12)]
    
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(Max_iter + 1):
        losses = solver.train_step(collocation_data, boundary_data, interface_data)
        
        for i, loss in enumerate(losses):
            Loss_hist[i].append(loss)
        
        if epoch % 200 == 0:
            avg_loss = np.mean(losses)
            print(f"Epoch: {epoch:5d}, Average Loss: {avg_loss:.6f}, Max Loss: {np.max(losses):.6f}")
    
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")

    save_model(solver)
    
    # Prediction
    print("Making predictions...")
    predictions = solver.predict(test_data)
    
    # Combine predictions from all subdomains
    X, Y = test_data['grid']
    masks = test_data['masks']
    
    # Initialize solution arrays
    U = np.zeros_like(X)
    V = np.zeros_like(X)
    P = np.zeros_like(X)
    count = np.zeros_like(X)
    
    # Fill in predictions from each subdomain
    for i in range(12):
        if f'pred{i}' in predictions:
            mask = masks[f'mask{i}']
            U[mask] += predictions[f'pred{i}'][:, 0]
            V[mask] += predictions[f'pred{i}'][:, 1]
            P[mask] += predictions[f'pred{i}'][:, 2]
            count[mask] += 1
    
    # Average overlapping regions
    U = np.divide(U, count, out=np.zeros_like(U), where=count!=0)
    V = np.divide(V, count, out=np.zeros_like(V), where=count!=0)
    P = np.divide(P, count, out=np.zeros_like(P), where=count!=0)
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # U velocity
    im1 = axes[0,0].contourf(X, Y, U, levels=50, cmap='RdBu_r')
    axes[0,0].set_title('U Velocity', fontsize=14)
    axes[0,0].set_xlabel('x')
    axes[0,0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0,0])
    
    # Add subdomain boundaries
    x_bounds, y_bounds = test_data['bounds']
    for x_line in x_bounds:
        axes[0,0].axvline(x_line, color='white', alpha=0.5, linewidth=0.5)
    for y_line in y_bounds:
        axes[0,0].axhline(y_line, color='white', alpha=0.5, linewidth=0.5)
    
    # V velocity
    im2 = axes[0,1].contourf(X, Y, V, levels=50, cmap='RdBu_r')
    axes[0,1].set_title('V Velocity', fontsize=14)
    axes[0,1].set_xlabel('x')
    axes[0,1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[0,1])
    
    # Add subdomain boundaries
    for x_line in x_bounds:
        axes[0,1].axvline(x_line, color='white', alpha=0.5, linewidth=0.5)
    for y_line in y_bounds:
        axes[0,1].axhline(y_line, color='white', alpha=0.5, linewidth=0.5)
    
    # Pressure
    im3 = axes[1,0].contourf(X, Y, P, levels=50, cmap='RdBu_r')
    axes[1,0].set_title('Pressure', fontsize=14)
    axes[1,0].set_xlabel('x')
    axes[1,0].set_ylabel('y')
    plt.colorbar(im3, ax=axes[1,0])
    
    # Add subdomain boundaries
    for x_line in x_bounds:
        axes[1,0].axvline(x_line, color='white', alpha=0.5, linewidth=0.5)
    for y_line in y_bounds:
        axes[1,0].axhline(y_line, color='white', alpha=0.5, linewidth=0.5)
    
    # Streamlines
    speed = np.sqrt(U**2 + V**2)
    axes[1,1].contourf(X, Y, speed, levels=50, cmap='viridis')
    axes[1,1].streamplot(X, Y, U, V, color='white', density=1.5, linewidth=1)
    axes[1,1].set_title('Streamlines', fontsize=14)
    axes[1,1].set_xlabel('x')
    axes[1,1].set_ylabel('y')
    
    # Add subdomain boundaries
    for x_line in x_bounds:
        axes[1,1].axvline(x_line, color='white', alpha=0.3, linewidth=0.5)
    for y_line in y_bounds:
        axes[1,1].axhline(y_line, color='white', alpha=0.3, linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('xpinn_velocity_pressure_fields.png', dpi=300)
    plt.show()
    
    # Loss history
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, 12))
    
    for i in range(12):
        plt.semilogy(Loss_hist[i], color=colors[i], label=f'Subdomain {i}', linewidth=1.5)
    
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss History for 12 Subdomains')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('xpinn_loss_history.png', dpi=300)
    plt.show()
    
    # Average loss plot
    plt.figure(figsize=(10, 6))
    avg_losses = [np.mean([Loss_hist[i][j] for i in range(12)]) for j in range(len(Loss_hist[0]))]
    plt.semilogy(avg_losses, 'r-', linewidth=2, label='Average Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Average Loss')
    plt.title('Average Training Loss Across All Subdomains')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('xpinn_avg_loss.png', dpi=300)
    plt.show()
    
    print("Simulation completed!")
    print(f"Final average loss: {avg_losses[-1]:.6f}")

if __name__ == '__main__':
    main()