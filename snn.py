import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
import numpy as np
import matplotlib.pyplot as plt

# Training parameters
num_epochs = 10
num_classes = 10
batch_size = 128
learning_rate = 5e-4
num_steps = 25  # Number of time steps

# Neuromorphic parameters
beta = 0.95  # neuron decay rate
spike_grad = surrogate.fast_sigmoid(slope=25)  # surrogate gradient

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load MNIST dataset
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, 
                                         download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, 
                                        download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                         batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                        batch_size=batch_size, shuffle=False)

# Define the Convolutional SNN
class CSNN(nn.Module):
    def __init__(self):
        super(CSNN, self).__init__()
        
        # Initialize layers
        self.conv1 = nn.Conv2d(1, 12, 5)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        self.conv2 = nn.Conv2d(12, 64, 5)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        self.fc1 = nn.Linear(64*4*4, 10)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
    def forward(self, x):
        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        
        # Record the final layer
        spk3_rec = []
        mem3_rec = []
        
        for step in range(num_steps):
            cur1 = F.max_pool2d(self.conv1(x), 2)
            spk1, mem1 = self.lif1(cur1, mem1)
            
            cur2 = F.max_pool2d(self.conv2(spk1), 2)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            cur3 = self.fc1(spk2.view(batch_size, -1))
            spk3, mem3 = self.lif3(cur3, mem3)
            
            spk3_rec.append(spk3)
            mem3_rec.append(mem3)
            
        return torch.stack(spk3_rec, dim=0), torch.stack(mem3_rec, dim=0)

# Alternative simpler implementation
class SimpleCSNN(nn.Module):
    def __init__(self):
        super(SimpleCSNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64*7*7, 128)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
        self.fc2 = nn.Linear(128, 10)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        
    def forward(self, x):
        # Initialize hidden states
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        
        # Record spikes and membrane potentials
        spk_rec = []
        mem_rec = []
        
        for step in range(num_steps):
            # Convolutional layers
            cur1 = self.pool1(self.conv1(x))
            spk1, mem1 = self.lif1(cur1, mem1)
            
            cur2 = self.pool2(self.conv2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)
            
            # Fully connected layers
            cur3 = self.fc1(spk2.view(spk2.size(0), -1))
            spk3, mem3 = self.lif3(cur3, mem3)
            
            cur4 = self.fc2(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)
            
            spk_rec.append(spk4)
            mem_rec.append(mem4)
            
        return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)

# Initialize network
net = SimpleCSNN().to(device)
print(f"Network parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad)}")

# Loss and optimizer
loss_fn = SF.ce_rate_loss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# Training function
def train_epoch(net, loader, optimizer, loss_fn, device):
    net.train()
    train_loss = 0
    train_acc = 0
    
    for data, targets in loader:
        data = data.to(device)
        targets = targets.to(device)
        
        # Forward pass
        spk_rec, mem_rec = net(data)
        
        # Calculate loss
        loss_val = loss_fn(spk_rec, targets)
        
        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        
        # Store loss history for future plotting
        train_loss += loss_val.item()
        
        # Calculate accuracy
        acc = SF.accuracy_rate(spk_rec, targets)
        train_acc += acc
        
    return train_loss/len(loader), train_acc/len(loader)

# Testing function
def test_epoch(net, loader, loss_fn, device):
    net.eval()
    test_loss = 0
    test_acc = 0
    
    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device)
            targets = targets.to(device)
            
            # Forward pass
            spk_rec, mem_rec = net(data)
            
            # Calculate loss
            loss_val = loss_fn(spk_rec, targets)
            test_loss += loss_val.item()
            
            # Calculate accuracy
            acc = SF.accuracy_rate(spk_rec, targets)
            test_acc += acc
            
    return test_loss/len(loader), test_acc/len(loader)

# Training loop
print("Starting training...")
train_losses = []
test_losses = []
train_accs = []
test_accs = []

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(net, train_loader, optimizer, loss_fn, device)
    test_loss, test_acc = test_epoch(net, test_loader, loss_fn, device)
    
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    
    print(f'Epoch [{epoch+1}/{num_epochs}] '
          f'Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} '
          f'Test Loss: {test_loss:.4f} Test Acc: {test_acc:.4f}')

# Plot training results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(train_losses, label='Train Loss')
ax1.plot(test_losses, label='Test Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Test Loss')
ax1.legend()

ax2.plot(train_accs, label='Train Accuracy')
ax2.plot(test_accs, label='Test Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Training and Test Accuracy')
ax2.legend()

plt.tight_layout()
plt.show()

# Visualize spiking activity
def visualize_spikes(net, test_loader, device, num_samples=5):
    net.eval()
    
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device)
            
            spk_rec, mem_rec = net(data)
            
            # Plot first few samples
            for i in range(min(num_samples, data.size(0))):
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
                
                # Original image
                img = data[i].cpu().squeeze()
                ax1.imshow(img, cmap='gray')
                ax1.set_title(f'Input Image (Label: {targets[i]})')
                ax1.axis('off')
                
                # Spike raster plot
                spikes = spk_rec[:, i, :].cpu().numpy()
                spike_times, spike_neurons = np.where(spikes > 0)
                ax2.scatter(spike_times, spike_neurons, s=1, alpha=0.7)
                ax2.set_xlabel('Time Steps')
                ax2.set_ylabel('Output Neurons')
                ax2.set_title('Spike Raster Plot')
                ax2.set_ylim(-0.5, 9.5)
                
                # Membrane potential
                mem = mem_rec[:, i, :].cpu().numpy()
                for neuron in range(10):
                    ax3.plot(mem[:, neuron], label=f'Neuron {neuron}', alpha=0.7)
                ax3.set_xlabel('Time Steps')
                ax3.set_ylabel('Membrane Potential')
                ax3.set_title('Membrane Potentials')
                ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                
                plt.tight_layout()
                plt.show()
            
            break  # Only process first batch

# Visualize results
print("\nVisualizing spiking patterns...")
visualize_spikes(net, test_loader, device, num_samples=3)

print(f"\nFinal Test Accuracy: {test_accs[-1]:.4f}")