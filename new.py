import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.manifold import TSNE
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
# -------------------------------
# Part 1: Select Specific Prompts
# -------------------------------
selected_prompts = [
    "What is the best way to learn machine learning?",
    "Explain quantum computing in simple terms.",
    "Describe the future of artificial intelligence.",
    "Write a Python function to reverse a linked list.",
    "Explain why recursion is useful in programming.",
    "What is the role of a CPU in modern computing?",
    "Explain how virtual memory works in an operating system.",
    "What are the ethical challenges of artificial intelligence?",
    "How does a neural network learn from data?",
    "How does gradient descent optimize a machine learning model?",
    "Explain the difference between supervised and unsupervised learning.",
    "Write python function to generate random numbers",
    "What happens when you type a URL into a browser?",
    "How do compiled languages differ from interpreted ones?",
    "Can you explain what a variable is in the context of programming?",
    "What role do comments play in your code?",
    "What exactly is a function, and how is it used?",
    "How would you define an algorithm?",
    "What distinguishes a stack from a queue in programming?",
    "What is recursion, and when would you use it?",
    "In what ways do a for loop and a while loop differ?",
    "What does the term 'data structure' mean in programming?",
    "Can you explain the difference between an array and a linked list?",
    "What does object-oriented programming (OOP) involve?",
    "What are the four core principles of object-oriented programming?"
]

new_script=[
"How do Python and Java differ from one another?",
"What is dynamic typing, and how does it work in languages like Python?",
"What purpose does the `self` keyword serve in Python?",
"What is a lambda function in Python, and when is it useful?",
"Write a Python function to reverse a linked list.",
"Write python function to generate random numbers",
"Describe the future of artificial intelligence.",
"What is the best way to learn machine learning?",
    "What are the ethical challenges of artificial intelligence?",
"How does gradient descent optimize a machine learning model?",
"Explain quantum computing in simple terms."
]
# -------------------------------
# Part 2: Load Activation Data
# -------------------------------
# Load activations from the JSON file
with open("activations_data.json", "r") as f:
    data = json.load(f)

# Extract activation vectors and labels (prompts)
activation_vectors = []
prompt_labels = []

for item in data:
    if item['prompt'] in new_script and item['activations']:  # Filter for selected prompts
        vector = item['activations'][0][0]  # Nested list (assuming the format is correct)
        activation_vectors.append(vector)
        prompt_labels.append(item['prompt'][:30] + "...")  # Trim prompt for label

# -------------------------------
# Part 3: Dimensionality Reduction (t-SNE)
# -------------------------------
activation_vectors = np.array(activation_vectors)
tsne = TSNE(n_components=2, random_state=42, perplexity=10)  # Adjust perplexity if necessary
reduced_vectors = tsne.fit_transform(activation_vectors)

# -------------------------------
# Part 4: Plot Activation Vectors
# -------------------------------
plt.figure(figsize=(12, 8))
sns.scatterplot(x=reduced_vectors[:, 0], y=reduced_vectors[:, 1], hue=prompt_labels, palette='tab10', s=100)
plt.title("2D t-SNE of Activation Vectors for Selected Prompts")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend(loc='best', bbox_to_anchor=(1.05, 1), title='Prompt')
plt.tight_layout()
plt.show()

cosine_sim_matrix = cosine_similarity(activation_vectors)

# Plot as a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(cosine_sim_matrix, annot=True, fmt=".2f", cmap="YlGnBu",
            xticklabels=prompt_labels, yticklabels=prompt_labels)
plt.title("Cosine Similarity Between Prompts")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
