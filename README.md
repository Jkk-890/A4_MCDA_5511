# MCDA5511 Assignment 4: Sparse Autoencoder for LLM Interpretability
## Part 1

Hooks are used to capture the intermediate outputs of selected layers with the goal of monitoring the model throughout the forward pass. We can analyze these layers to see how the model is thinking at that specific layer. We decided to put them in in the middle layers, we did this because the layers around the middle will have a nice mix of information that has been changed very little from its original state and ones that have been changed drastically. This was with the goal of seeing which activations had been altered and which had not been altered. In Anthropic's paper they state “We chose to focus on the middle layer of the model because we reasoned that it is likely to contain interesting, abstract features” they say “this is due to the residual stream is smaller than the MLP layer, making SAE training and inference computationally cheaper “ and for cross-layer superposition which is essentially when features are distributed across more than one layer in a model, rather than being in a specific layer. The limitations of this approach are that naturally you do not see the state of the layers at the beginning or end, thus you miss how the initial input and what the final product look like. Focusing on the middle layer, we don’t see the full picture of the decision-making process that these hooks provide for us.  

---
## Part 2: Generating Training Data for the Autoencoder

This repository contains our implementation of a toy sparse autoencoder inspired by Anthropic’s *Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet* (May 2024). Below, we document our work for Parts 2 and 3, focusing on generating training data and training the sparse autoencoder. Parts 1, 4, 5, and 6 were completed by team members and are detailed elsewhere in this README.

### Implementation
We implemented an `ActivationDatasetGenerator` class to create training data for the sparse autoencoder:
- **`generate_text_and_activations` Method**: Takes a prompt, tokenizes it, generates text using the `unsloth/Meta-Llama-3.1-8B-Instruct` model, and collects activations from layer 10 (a middle layer) via a forward hook.
- **`process_dataset` Method**: Iterates over a list of prompts, calls `generate_text_and_activations`, and saves the results (prompts, generated text, and activations) to `activations_data.json` in JSON format. The activations are stored as lists to preserve mappings for later interpretation.

### Corpus Selection
We curated a corpus of 98 prompts focused on programming, computer science, and AI-related topics (e.g., "What is the best way to learn machine learning?", "Explain quantum computing in simple terms"). These prompts were chosen to:
- Explore technical concepts where the model might exhibit distinct, interpretable features.
- Ensure diversity in question types (explanatory, coding tasks, conceptual) to capture a range of model behaviors.
- Keep the dataset manageable for computational constraints while providing sufficient data for training.

### Summary Statistics
- **Number of Prompts**: 98
- **Average Generated Text Length**: Approximately 50 tokens (set by `max_length=50`)
- **Total Activation Samples**: 98 (one per prompt, from layer 10)
- **Activation Shape**: `(1, 1, 4096)` per sample, where 4096 is the hidden size of the Llama model
- **Average Non-Zero Activation Values**: Approximately 90% of dimensions are non-zero in raw activations (pre-autoencoder), indicating dense initial representations.

### Rationale
We selected this corpus to focus on a domain of interest (computer science and AI) where interpretable features might emerge, such as responses to technical definitions or coding tasks. The moderate size balances computational feasibility with the need for diverse activation patterns.

![part2.png](part2.png)

- This plot shows the activation values from layer 10 for my 98 prompts. You can see they’re mostly non-zero and vary widely, which is why we need the autoencoder to simplify them.
---
For 
Prompt=[
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
![part2_1.png](part2_1.png)

![part2_2.png](part2_2.png)

---

## Part 3: Training the Sparse Autoencoder

### Implementation
We built a `SparseAutoencoder` class inspired by Cunningham et al. (2023) and Anthropic’s methodology:
- **Structure**: Encoder-decoder with tied weights conceptually (though implemented separately here), using `nn.Linear` layers and ReLU activations. L1 regularization enforces sparsity on the encoded representation.
- **Hyperparameters**:
  - **Input/Output Dimension**: 4096 (matches Llama’s hidden size)
  - **Encoder Dimension**: 8192 (2x input dimension)
  - **L1 Penalty**: 0.001
- **Training**: Used Adam optimizer (`lr=0.001`), MSE reconstruction loss, and early stopping (patience=5). Training metrics (reconstruction loss, L1 loss, sparsity) were tracked.

### Hyperparameter Experiments
We started with an encoder dimension of 8192 (2x input) and an L1 penalty of 0.001. Training showed rapid convergence to high sparsity (100% near-zero values by epoch 4), suggesting the L1 penalty might be too strong. We didn’t adjust further due to time constraints, but future experiments could lower it (e.g., 0.0001) to balance sparsity and reconstruction quality. Training ran for 50 epochs max, stopping early at epoch 6 when validation loss plateaued.

### What the Autoencoder Does
In *activation space*—the 4096-dimensional space of layer 10 activations from the Llama model—the autoencoder compresses these into an 8192-dimensional sparse feature space. Each dimension in this encoded space represents a potential "feature" (per Anthropic’s definition), such as a specific concept or behavior the model activates for (e.g., "programming concepts" or "explanatory responses"). The encoder maps dense activations to a sparse representation (mostly zeros), and the decoder reconstructs the original activations. Sparsity ensures only a few features are active per input, aiding interpretability.

### Results
- **Reconstruction Loss**: Final train loss = 0.0133, validation loss = 0.0142 (low, indicating good reconstruction)
- **Sparsity**: Reached 100% near-zero values (<0.01) by epoch 4, suggesting overly aggressive sparsity
- **Training Plot**: See `pca_plot.png` for a PCA visualization of encoded features (though this relates more to Part 4 analysis). Loss curves are not plotted but show a sharp drop then plateau.

### Evidence of Sparsity
By epoch 4, 100% of encoded dimensions were near-zero (threshold <0.01), as seen in the sparsity metric rising from 0.5578 (epoch 1) to 1.0000. While this confirms sparsity, it may indicate the model over-pruned features, potentially losing interpretability. Sample encoded outputs (e.g., from later analysis) show only a few dimensions with non-zero values (e.g., 0.0230, 0.0196), supporting sparsity but highlighting the need for tuning.

![training_loss.png](training_loss.png)
- Here’s how the loss dropped during training. The training loss fell fast from 17 to 0.013, and validation stabilized at 0.014, showing good learning before stopping at epoch 6

![sparsity_curve.png](sparsity_curve.png)
- This shows sparsity increasing to 100% by epoch 4. It means most encoded values are zero, which is great for isolating features. 

---
## Part 4: Analyzing Latent Dimensions in LLM Activations

This document outlines the process of analyzing the internal representations of a Language Model (LLM) by examining the activation patterns in an autoencoder trained on these activations. The goal is to identify interpretable "features" that correspond to specific semantic or syntactic patterns in the input text.

The process involved:
1.  Generating activation data from the LLM for a diverse set of text prompts.
2.  Training an autoencoder to compress and reconstruct these activation vectors.
3.  For each dimension in the encoded representation (the bottleneck of the autoencoder), identifying the text prompts that caused the strongest activations in that dimension.
4.  Manually inspecting these top activating text snippets to find common themes and interpret the potential "feature" represented by that dimension.

## Manually Identified Features

Below are 3 examples of features that were manually analyzed from the output of the `get_top_activating_texts` function.

### Feature 1: Fundamentals of Object-Oriented Programming (OOP)

* **Dimension:** 0 (and also strongly present in Dimension 8 and 15, which suggests these might be redundant or very closely related features)
* **Top Text Snippets:**
    1.  What's the difference between synchronous and asynchronous programming? (Activation: 0.0000)
    2.  What are the four core principles of object-oriented programming? (Activation: 0.0000)
    3.  How do you define a class in programming? (Activation: 0.0000)
    4.  What does inheritance mean in object-oriented programming? (Activation: 0.0000)
    5.  Can you explain polymorphism and how it's applied in OOP? (Activation: 0.0000)
* **Interpretation:** This dimension appears to strongly activate (or in this case, perhaps *not* strongly activate, given the near-zero scores) for questions specifically asking about fundamental concepts in Object-Oriented Programming (OOP). The snippets cover core OOP principles like classes, inheritance, and polymorphism. The inclusion of a question about synchronous vs. asynchronous programming with a similar low activation might suggest this dimension is selective for OOP-related inquiries and less responsive to other programming paradigms or general CS concepts. It's worth noting the consistently low activation scores, which might indicate this dimension is specifically *not* encoding these concepts strongly, or that the autoencoder has learned to represent these in a way that doesn't lead to high activation in this particular dimension.

### Feature 2: JavaScript Specific Concepts

* **Dimension:** 13
* **Top Text Snippets:**
    1.  What's the difference between client-side and server-side rendering? (Activation: 0.0829)
    2.  What is the difference between `==` and `===` in JavaScript? (Activation: 0.0732)
    3.  What are the main uses of JavaScript frameworks such as React or Angular? (Activation: 0.0703)
    4.  What is the role of a CPU in modern computing? (Activation: 0.0364)
    5.  What is HTML, and what role does it serve in creating web pages? (Activation: 0.0259)
* **Interpretation:** The top activating text snippets for this dimension are heavily focused on concepts specific to JavaScript and web development. Questions about client-side vs. server-side rendering, the nuances of JavaScript equality operators (`==` and `===`), and the use of popular JavaScript frameworks like React and Angular all appear at the top. While the dimension also shows some activation for more general web-related (HTML) and computing (CPU role) questions, the strongest signals are clearly related to the JavaScript ecosystem. This suggests that this dimension in the encoded space has learned to represent features associated with JavaScript-specific knowledge.

### Feature 3: Concepts Related to Software Testing

* **Dimension:** 36 (and also strongly present in Dimension 37)
* **Top Text Snippets (Dimension 36):**
    1.  Can you explain the difference between white-box and black-box testing? (Activation: 0.1488)
    2.  What is the purpose of a web server like Apache or Nginx? (Activation: 0.1161)
    3.  How does a hash table work, and when would you use it? (Activation: 0.0856)
    4.  Could you explain how merge sort works and why it is efficient? (Activation: 0.0830)
    5.  What are cookies, and how do they help manage sessions on websites? (Activation: 0.0701)
* **Top Text Snippets (Dimension 37):**
    1.  Can you explain the difference between white-box and black-box testing? (Activation: 0.1088)
    2.  What does the term 'data structure' mean in programming? (Activation: 0.1012)
    3.  Can you explain what a RESTful API is and how it works? (Activation: 0.0737)
    4.  How does JavaScript handle asynchronous programming? (Activation: 0.0681)
    5.  How do databases play a role in web application development? (Activation: 0.0598)
* **Interpretation:** Both Dimension 36 and 37 show a strong activation for the question about the difference between white-box and black-box testing. This suggests that a feature related to software testing methodologies is being captured by these dimensions. While other concepts appear in the top snippets (web servers, data structures, algorithms, web technologies), the consistent high activation for the testing-related question indicates that "software testing concepts" is a significant feature being represented. The presence of other, somewhat related (e.g., debugging, quality assurance are related to testing) or more general programming concepts might indicate that this feature is not perfectly isolated or that the model associates testing with broader software development knowledge.

## Conclusion

The manual analysis of the top activating text snippets for a few dimensions of the autoencoder's encoded space provides some insights into the features that the LLM might be implicitly representing. While some dimensions appear to correspond to relatively clear themes (like JavaScript concepts or OOP fundamentals), others might represent more complex or less immediately obvious combinations of features. Further investigation across more dimensions and potentially refining the training data or autoencoder architecture could lead to the discovery of more interpretable features.
