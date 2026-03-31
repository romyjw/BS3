import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt





def cox_de_boor(x, knots, degree, i):
	"""
	Compute the value of the i-th B-spline basis function of given degree at x using the Cox-de Boor recursion formula.
	
	Parameters:
	x (torch.Tensor): The input value(s) where the B-spline is evaluated.
	knots (torch.Tensor): The knot sequence.
	degree (int): The degree of the B-spline.
	i (int): The index of the basis function.
	
	Returns:
	torch.Tensor: The value of the basis function at x.
	"""

	if degree == 0:
		return ((knots[i] <= x) & (x < knots[i + 1]) ).float()
	
	denom1 = knots[i + degree] - knots[i]
	denom2 = knots[i + degree + 1] - knots[i + 1]
	
	term1 = 0 if denom1 == 0 else ((x - knots[i]) / denom1) * cox_de_boor(x, knots, degree - 1, i)
	term2 = 0 if denom2 == 0 else ((knots[i + degree + 1] - x) / denom2) * cox_de_boor(x, knots, degree - 1, i + 1)
	
	return term1 + term2


def wrapped_cox_de_boor(x,knots, degree, i):
	return torch.maximum( torch.maximum(cox_de_boor(x,knots, degree, i), cox_de_boor(x+(2*torch.pi),knots, degree, i)), cox_de_boor(x-(2*torch.pi),knots, degree, i))



class MLP(nn.Module):
	def __init__(self, layer_sizes):
		super(MLP, self).__init__()
		layers = []
		for i in range(len(layer_sizes) - 1):
			layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
			if i < len(layer_sizes) - 2:  # No activation on last layer
				layers.append(nn.Tanh())
		self.network = nn.Sequential(*layers)
	
	def forward(self, x):
		return self.network(x)

class BlendedMLP(nn.Module):
	def __init__(self, n, layer_sizes, degree, knots):
		super(BlendedMLP, self).__init__()
		self.mlps = nn.ModuleList([MLP(layer_sizes) for _ in range(n)])
		self.n=n
		self.degree=degree
		self.knots=knots
		self.wrapped_knots = torch.cat([self.knots, self.knots[1:]+2*torch.pi])
		print(self.wrapped_knots)
		
		
		x = torch.arange(self.knots[0],self.knots[-1],0.001)
		
		for i in range(n):
			plt.plot(x, wrapped_cox_de_boor(x, self.wrapped_knots[i:i + self.degree+2], self.degree, 0))
			plt.show()
		y=torch.tensor([0]*(len(knots)))
		plt.scatter(knots, y)
		plt.show()
	
	def forward(self, x):
		return sum(self.mlps[i](x) * wrapped_cox_de_boor(x, self.wrapped_knots[i : i + self.degree + 2], self.degree, 0) for i in range(self.n) )
	   
# Example usage

layer_sizes = [1, 16, 16, 2]  # Example structure: 1 input, two hidden layers (16 neurons), 1 output
degree=3
#knots = torch.tensor([0,0.5,0.8,1.0,1.5,2.0,2.5,3.0,3.5 4.0, 5.0, 5.8, 2*torch.pi])
knots = 2*torch.pi * torch.arange(0,1.1,0.1)
n = len(knots)  # Number of MLPs


blended_mlp = BlendedMLP(n, layer_sizes, degree, knots = knots)





#f = lambda theta: torch.stack([torch.cos(theta), torch.sin(theta)]).squeeze().transpose(0,1)
f = lambda theta: torch.stack([torch.cos(theta), torch.sin(theta)]).squeeze().transpose(0,1) * (1+ 0.1*torch.sin(10*theta))

# Training all models jointly
optimizer = optim.Adam(blended_mlp.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Dummy training data
theta_train = 2 * torch.pi * torch.rand(10000, 1)
target_train = f(theta_train)

#print(target_train.shape)


# Training loop
for epoch in range(1000):
	optimizer.zero_grad()
	output_train = blended_mlp(theta_train)
	loss = criterion(output_train, target_train)
	loss.backward()
	optimizer.step()
	
	if epoch % 10 == 0:
		print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
		#print(output_train.shape)

# Test
theta_test = 2 * torch.pi * torch.rand(100000, 1)
target_test = f(theta_test)
output_knots = blended_mlp(         knots.unsqueeze(-1)       )

output_test = blended_mlp(theta_test)
plt.scatter(target_test.detach()[:,0], target_test.detach()[:,1], color='red', s=0.1)
    
plt.scatter(output_test.detach()[:,0], output_test.detach()[:,1], color='blue', s=0.1)
plt.scatter(output_knots.detach()[:,0], output_knots.detach()[:,1], color='blue', alpha=0.5, s=100.0)
for i, (x, y) in enumerate(output_knots.detach()):
    plt.text(x, y, str(knots[i]), fontsize=12, ha='right', va='bottom', color='black')

plt.show()


#Interpret
#blended_mlp.view_patches()
