import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np


# -------------------------------
# Part 2: Tensor Output Plot
# -------------------------------
# Load tensor outputs
with open("tensor_output.txt", "r") as f:
    tensor_values = [float(line.strip()) for line in f if line.strip()]

# Plot histogram
plt.figure(figsize=(10, 5))
sns.histplot(tensor_values, bins=50, kde=True, color='skyblue')
plt.title("Distribution of Final Layer Activation Values (Part 2)")
plt.xlabel("Activation Value")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------------
# Part 3: Activation Vectors (Prompt-wise)
# -------------------------------
# Load activations
with open("activations_data.json", "r") as f:
    data = json.load(f)

# Extract activation vectors and labels (prompts)
activation_vectors = []
prompt_labels = []

for item in data:
    if item['activations']:  # Safe check
        vector = item['activations'][0][0]  # nested list
        activation_vectors.append(vector)
        prompt_labels.append(item['prompt'][:30] + "...")  # Trim prompt for label

# Use PCA (or t-SNE) for dimensionality reduction
#pca = PCA(n_components=2)
#reduced_vectors = pca.fit_transform(activation_vectors)


activation_vectors = np.array(activation_vectors)
tsne = TSNE(n_components=2, random_state=42, perplexity=3)
reduced_vectors = tsne.fit_transform(activation_vectors)


# Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=reduced_vectors[:, 0], y=reduced_vectors[:, 1], hue=prompt_labels, palette='tab10', s=100)
plt.title("2D PCA of Activation Vectors per Prompt (Part 3)")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.legend(loc='best', bbox_to_anchor=(1.05, 1), title='Prompt')
plt.tight_layout()
plt.show()
