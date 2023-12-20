import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
	def __init__(self, in_features, out_features):
		super(GCNLayer, self).__init__()
		self.linear = nn.Linear(in_features, out_features)

	def forward(self, adjacency_matrix, node_features):
		# GCN layer operation
		support = torch.matmul(adjacency_matrix, node_features)
		output = self.linear(support)
		return output


class GCNClassifier(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim):
		super(GCNClassifier, self).__init()
		self.gcn1 = GCNLayer(input_dim, hidden_dim)
		self.gcn2 = GCNLayer(hidden_dim, output_dim)

	def forward(self, adjacency_matrix, node_features):
		h1 = F.relu(self.gcn1(adjacency_matrix, node_features))
		h2 = self.gcn2(adjacency_matrix, h1)
		return h2


def train_gcn_model(adjacency_matrix, node_features, labels, num_epochs):
	# Define the model
	model = GCNClassifier(input_dim=node_features.shape[1], hidden_dim=16, output_dim=2)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
	criterion = nn.CrossEntropyLoss()

	for epoch in range(num_epochs):
		# Forward pass
		output = model(adjacency_matrix, node_features)
		loss = criterion(output, labels)

		# Backpropagation and optimization
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

	return model


# Function to get node representations
def get_node_representations(model, adjacency_matrix, node_features):
	with torch.no_grad():
		node_embeddings = model(adjacency_matrix, node_features)
	return node_embeddings


# Sample data (you should replace this with your own data)
adjacency_matrix = torch.tensor([[0, 1, 0, 0],
								 [1, 0, 1, 0],
								 [0, 1, 0, 1],
								 [0, 0, 1, 0]], dtype=torch.float32)
node_features = torch.rand(4, 3)  # Random node features
labels = torch.LongTensor([0, 1, 0, 1])  # Binary labels

# Train the GCN model
model = train_gcn_model(adjacency_matrix, node_features, labels, num_epochs=100)

# Get node representations
node_representations = get_node_representations(model, adjacency_matrix, node_features)
print("Node Representations:", node_representations)
