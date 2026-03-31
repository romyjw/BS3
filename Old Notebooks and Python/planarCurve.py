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
		return ((knots[i] <= x) & (x < knots[i + 1])).float()
	
	denom1 = knots[i + degree] - knots[i]
	denom2 = knots[i + degree + 1] - knots[i + 1]
	
	term1 = 0 if denom1 == 0 else ((x - knots[i]) / denom1) * cox_de_boor(x, knots, degree - 1, i)
	term2 = 0 if denom2 == 0 else ((knots[i + degree + 1] - x) / denom2) * cox_de_boor(x, knots, degree - 1, i + 1)
	
	return term1 + term2

'''
knots = torch.arange(-10,10,1)
x = torch.arange(-3,3,0.01)
plt.plot(x, cox_de_boor(x, knots, degree=0,i=6))
plt.plot(x, cox_de_boor(x, knots, degree=1,i=7))
plt.plot(x, cox_de_boor(x, knots, degree=2,i=8))
plt.plot(x, cox_de_boor(x, knots, degree=3,i=9))
plt.show()
'''




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
	def __init__(self, n, layer_sizes, degree, knots, scale):
		super(BlendedMLP, self).__init__()
		self.mlps = nn.ModuleList([MLP(layer_sizes) for _ in range(n)])
		self.n=n
		self.degree=degree
		self.knots=knots
		self.scale = scale
		
		x = torch.arange(-10,10,0.01)
		for i in range(n):
			plt.plot(x, cox_de_boor(x, self.knots, self.degree, i))
		y=torch.tensor([0]*(n+degree+1))
		print(knots, y)
		plt.scatter(knots, y)
		plt.show()
	
	def forward(self, x):
		return sum( self.mlps[i](x) * cox_de_boor(x*self.scale, self.knots, self.degree, i) for i in range(self.n) )
		
	   
	'''
	def view_patches(self):
		x = torch.arange(-10,10,0.01).unsqueeze(-1)
		y=torch.tensor([0]*(n+degree+1))
		
		
		for i in range(n):
			plt.scatter(self.knots, y)
			color = np.random.rand(3,)
			output_coords = self.mlps[i](x).detach() * cox_de_boor(x, self.knots, self.degree, i)
			plt.plot(output_coords[0], output_coords[1], color=color)
			#plt.plot(x, self.mlps[i](x).detach(), color=color, linestyle=':')
			#plt.ylim(-50,50)
		plt.show()
	'''

# Example usage
n = 7  # Number of MLPs
layer_sizes = [1, 16, 16, 2]  # Example structure: 1 input, two hidden layers (16 neurons), 1 output
degree=3
knots = torch.arange(-n//2, n//2 + degree + 1,1, dtype=torch.float32)
print(knots)
blended_mlp = BlendedMLP(n, layer_sizes, degree, knots = knots, scale=3.0)





#f = lambda theta: torch.stack([torch.cos(theta), torch.sin(theta)]).squeeze().transpose(0,1)
f = lambda theta: torch.stack([torch.cos(theta), torch.sin(theta)]).squeeze().transpose(0,1) * (1+ 0.1*torch.sin(10*theta))

# Training all models jointly
optimizer = optim.Adam(blended_mlp.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Dummy training data
theta_train = 6 * torch.rand(10000, 1) - 3
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
theta_test = 14 * torch.rand(1000, 1) - 7
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
blended_mlp.view_patches()
