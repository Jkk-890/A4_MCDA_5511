from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict
import numpy as np 
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class NewModel(nn.Module):
    def __init__(self, *args):
        super().__init__(*args)
        #self.pretrained = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
        #self.tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
        self.pretrained = AutoModelForCausalLM.from_pretrained("unsloth/phi-4")  # Use the Phi-4 model
        self.tokenizer = AutoTokenizer.from_pretrained("unsloth/phi-4")
        self.output_layers = [1]
        self.selected_out = OrderedDict()
        self.fhooks = []

        for i,l in enumerate(list(self.pretrained._modules.keys())):
            if i in self.output_layers:
                self.fhooks.append(getattr(self.pretrained,l).register_forward_hook(self.forward_hook(l)))

    def forward_hook(self,layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = output
        return hook

    def forward(self, x):
        out = self.pretrained(x)
        return out, self.selected_out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Now, move the model to the appropriate device
model = NewModel().to(device)

prompt = (
    "Where is the best place to go for a vacation in the summer?"
)

input_ids = model.tokenizer(prompt, return_tensors="pt").input_ids.to(device)

gen_tokens = model.pretrained.generate(
    input_ids,
    do_sample=True,
    temperature=0.9,
    max_length=100,
)
gen_text = model.tokenizer.batch_decode(gen_tokens)[0]

print(gen_text)
# Print the raw tensor output (before flattening)
print([value for value in model.selected_out.values()])

# Extract the tensor from the selected output layers (assuming it's the first item)
tensor = list(model.selected_out.values())[0]

# Move the tensor to CPU (if it's on GPU)
tensor_cpu = tensor.cpu()

# Convert the tensor to a NumPy array
numpy_array = tensor_cpu.detach().numpy()  # Detach from computation graph

# Flatten the numpy array and save it to a text file
np.savetxt('tensor_output.txt', numpy_array.flatten())  # Flatten to store as a single line of numbers

##############

class ActivationDatasetGenerator:
    def __init__(self, model_name="unsloth/phi-4", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.activations = []

        def hook_fn(module, input, output):
            self.activations = output[0].detach().cpu().numpy()  # Overwrite, not append

        self.hook_layer = self.model.transformer.h[16]  # Middle layer (32 layers total)
        self.hook_handle = self.hook_layer.register_forward_hook(hook_fn)

    def generate_text_and_activations(self, prompt, max_length=50):
        self.activations = []
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                pad_token_id=self.tokenizer.eos_token_id
            )
        text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return text, self.activations if self.activations.size else None

    def process_dataset(self, prompts, save_path="activations_data.json"):
        dataset = []
        for idx, prompt in enumerate(prompts):
            print(f"Processing {idx+1}/{len(prompts)}: {prompt[:50]}...")
            text, activation = self.generate_text_and_activations(prompt)
            if activation is not None:
                dataset.append({
                    "prompt": prompt,
                    "generated_text": text,
                    "activations": activation.tolist()
                })
        with open(save_path, "w") as f:
            json.dump(dataset, f, indent=4)
        print(f"Dataset saved to {save_path}")

# Example usage

prompts = [
    "What is the best way to learn machine learning?",
    "Explain quantum computing in simple terms.",
    "Tell me a joke about programming.",
    "What happens if you fall into a black hole?",
    "Describe the future of artificial intelligence.",
    "Write a Python function to reverse a linked list.",
    "Explain why recursion is useful in programming.",
    "What is the role of a CPU in modern computing?",
    "Explain how virtual memory works in an operating system.",
    "What are the ethical challenges of artificial intelligence?",
    "How does a neural network learn from data?",
    "How does gradient descent optimize a machine learning model?",
    "Explain the difference between supervised and unsupervised learning.",
    "write python function to generate random numbers",
    "What happens when you type a URL into a browser?"
    "Explain the concept of cloud computing in simple terms."
    
]

generator = ActivationDatasetGenerator()
generator.process_dataset(prompts)

##############

with open("activations_data.json", "r") as f:
    data = json.load(f)
for i, item in enumerate(data):
    print(f"Prompt {i+1} activation shape: {np.array(item['activations']).shape}")

##############

# Function to pad activations
def pad_activations(activations_batch):
    max_len = max([a.size(1) for a in activations_batch])
    padded_activations = [F.pad(a.squeeze(0), (0, 0, 0, max_len - a.size(1))) for a in activations_batch]
    return torch.stack(padded_activations)

# Sparse Autoencoder class
class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim=2560, encoder_dim=5120, sparsity_penalty=0.001):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, encoder_dim), nn.ReLU(True))
        self.decoder = nn.Sequential(nn.Linear(encoder_dim, input_dim), nn.ReLU(True))
        self.sparsity_penalty = sparsity_penalty

    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed, z

    def l1_regularization(self, z):
        return torch.sum(torch.abs(z))

# Load activations from Task 2
with open("activations_data.json", "r") as f:
    data = json.load(f)
activations_batch = [torch.tensor(item["activations"], dtype=torch.float32) for item in data]
padded_activations = pad_activations(activations_batch)

# Split into train and validation (80-20 split)
train_size = int(0.8 * len(padded_activations))
train_data = padded_activations[:train_size]
val_data = padded_activations[train_size:]

# Hyperparameters
input_dim = 2560  # GPT-Neo-2.7B hidden size
encoder_dim = 5120  # 2x input_dim
sparsity_penalty = 0.001

# Instantiate model and optimizer
autoencoder = SparseAutoencoder(input_dim, encoder_dim, sparsity_penalty)
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# Training loop with early stopping
num_epochs = 50
patience = 5
best_val_loss = float("inf")
patience_counter = 0

for epoch in range(num_epochs):
    autoencoder.train()
    optimizer.zero_grad()

    # Forward pass (training)
    train_input = train_data.view(-1, input_dim)  # [batch_size * seq_len, hidden_dim]
    reconstructed, encoded = autoencoder(train_input)
    recon_loss = F.mse_loss(reconstructed, train_input)
    l1_loss = autoencoder.l1_regularization(encoded)
    total_loss = recon_loss + sparsity_penalty * l1_loss

    # Backpropagation
    total_loss.backward()
    optimizer.step()

    # Validation
    autoencoder.eval()
    with torch.no_grad():
        val_input = val_data.view(-1, input_dim)
        val_reconstructed, val_encoded = autoencoder(val_input)
        val_loss = F.mse_loss(val_reconstructed, val_input)

    # Sparsity metric
    sparsity = (encoded.abs() < 0.01).float().mean().item()  # Fraction of near-zero values

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Sparsity: {sparsity:.4f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(autoencoder.state_dict(), "best_sparse_autoencoder.pth")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered")
            break

# Load best model
autoencoder.load_state_dict(torch.load("best_sparse_autoencoder.pth", weights_only=True))
print("Training complete. Best model saved.")

##############

# Assuming 'encoded' contains the compressed activations after training
encoded_data = encoded.detach().cpu().numpy()  # Moving to CPU if using GPU

# Apply PCA or t-SNE to reduce dimensionality for visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(encoded_data)

# Plot the 2D projection of the encoded features
plt.scatter(pca_result[:, 0], pca_result[:, 1])
plt.title('PCA of Encoded Features')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.savefig('pca_plot.png')

##############

# Load trained autoencoder
autoencoder.eval()

# Load activation data
with open("activations_data.json", "r") as f:
    data = json.load(f)

# Store encoded representations and their prompts
encoded_outputs = []
text_snippets = []

with torch.no_grad():
    for item in data:
        text = item["prompt"]  # Extract the original prompt
        act_tensor = torch.tensor(item["activations"], dtype=torch.float32)  # Convert to tensor

        # Reshape to match expected input shape
        act_tensor = act_tensor.view(1, -1)  # Shape: [1, 2560]

        # Pass through encoder
        z = autoencoder.encoder(act_tensor)  # Shape: [1, encoder_dim]

        # Convert to numpy for analysis
        encoded_outputs.append(z.squeeze(0).cpu().numpy())  # Shape: [encoder_dim]
        text_snippets.append(text)

# Convert to numpy array
encoded_outputs = np.stack(encoded_outputs)  # Shape: [num_samples, encoder_dim]

print("Encoded features extracted successfully!")

##############

def get_top_activating_texts(encoded_outputs, text_snippets, top_k=5):
    num_features = encoded_outputs.shape[1]  # Number of encoded dimensions
    features = []

    for dim in range(num_features):
        dim_activations = encoded_outputs[:, dim]  # Get activations for this feature
        top_indices = np.argsort(dim_activations)[-top_k:]  # Indices of top activations
        top_texts = [text_snippets[i] for i in reversed(top_indices)]  # Highest first

        features.append({
            "dimension": dim,
            "top_texts": top_texts,
            "activations": [dim_activations[i] for i in reversed(top_indices)]
        })

    return features

# Get top 5 text prompts per feature dimension
interpretable_features = get_top_activating_texts(encoded_outputs, text_snippets, top_k=5)

# Print results for inspection
for feat in interpretable_features[:10]:  # Check first 10 dimensions
    print(f"--- Dimension {feat['dimension']} ---")
    for i, (text, score) in enumerate(zip(feat["top_texts"], feat["activations"])):
        print(f"{i+1}. ({score:.4f}) {text}")
    print("\n")

##############

# Print results for inspection
for feat in interpretable_features[:50]:  # Check first 10 dimensions
    print(f"--- Dimension {feat['dimension']} ---")
    for i, (text, score) in enumerate(zip(feat["top_texts"], feat["activations"])):
        print(f"{i+1}. ({score:.4f}) {text}")
    print("\n")
