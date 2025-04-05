import json
import numpy as np
import matplotlib.pyplot as plt
import torch


# --- Plot 1: Activation Magnitude Distribution (Part 2) ---
def plot_activation_distribution(json_path="activations_data.json", output_file="activation_distribution.png"):
    # Load the activation data from Part 2
    with open(json_path, "r") as f:
        data = json.load(f)

    # Extract all activation values into a flat list
    all_activations = []
    for item in data:
        activations = np.array(item["activations"]).flatten()  # Shape (1, 1, 4096) -> 4096 values
        all_activations.extend(activations)

    # Plot histogram of activation magnitudes
    plt.figure(figsize=(8, 6))
    plt.hist(np.abs(all_activations), bins=50, log=True, color="skyblue", edgecolor="black")
    plt.title("Distribution of Activation Magnitudes (Layer 10)")
    plt.xlabel("Absolute Activation Value")
    plt.ylabel("Frequency (Log Scale)")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_file)
    plt.close()
    print(f"Saved activation distribution plot to {output_file}")


# --- Plot 2: Training Loss Curve (Part 3) ---
def plot_training_loss(output_file="training_loss.png"):
    # Hardcoded from your sample output; replace with your actual logs if available
    epochs = [1, 2, 3, 4, 5, 6]
    train_losses = [17.2336, 1.2163, 0.0396, 0.0135, 0.0133, 0.0133]
    val_losses = [0.0140, 0.0142, 0.0143, 0.0143, 0.0143, 0.0142]

    # Plot training and validation loss
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label="Training Loss", marker="o", color="blue")
    plt.plot(epochs, val_losses, label="Validation Loss", marker="o", color="orange")
    plt.title("Training and Validation Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_file)
    plt.close()
    print(f"Saved training loss plot to {output_file}")


# --- Plot 3: Sparsity Over Epochs (Part 3) ---
def plot_sparsity_over_epochs(output_file="sparsity_curve.png"):
    # Hardcoded from your sample output; replace with your actual logs if available
    epochs = [1, 2, 3, 4, 5, 6]
    sparsity_values = [0.5578, 0.9535, 0.9987, 1.0000, 1.0000, 1.0000]

    # Plot sparsity
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, sparsity_values, marker="o", color="green")
    plt.title("Sparsity of Encoded Features Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Fraction of Near-Zero Values (<0.01)")
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3)
    plt.savefig(output_file)
    plt.close()
    print(f"Saved sparsity plot to {output_file}")


# --- Run the plotting functions ---
if __name__ == "__main__":
    # Plot for Part 2
    plot_activation_distribution()

    # Plots for Part 3
    #plot_training_loss()
    #plot_sparsity_over_epochs()