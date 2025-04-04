# Part 1

Hooks are used to capture the intermediate outputs of selected layers with the goal of monitoring the model throughout the forward pass. We can analyze these layers to see how the model is thinking at that specific layer. We decided to put them in in the middle layers, we did this because the layers around the middle will have a nice mix of information that has been changed very little from its original state and ones that have been changed drastically. This was with the goal of seeing which activations had been altered and which had not been altered. In Anthropic's paper they state “We chose to focus on the middle layer of the model because we reasoned that it is likely to contain interesting, abstract features” they say “this is due to the residual stream is smaller than the MLP layer, making SAE training and inference computationally cheaper “ and for cross-layer superposition which is essentially when features are distributed across more than one layer in a model, rather than being in a specific layer. The limitations of this approach are that naturally you do not see the state of the layers at the beginning or end, thus you miss how the initial input and the final product look like. Focusing on these few layers, we don’t see the full picture of the decision-making process that these hooks provide for us.  

# Analyzing Latent Dimensions in LLM Activations

## Overview

This project explores the interpretability of latent dimensions within a trained Large Language Model (LLM) by mapping encoder activations back to their corresponding input text snippets. The goal is to understand the underlying features the LLM has learned by analyzing which questions most strongly activate specific latent dimensions. By identifying patterns in these activating questions, we aim to uncover meaningful and interpretable representations within the model's latent space.

## Findings

Our analysis of multiple latent dimensions revealed several recurring patterns in the text snippets that trigger them. Some dimensions appear to capture distinct and well-defined features, while others exhibit redundancy or lack clear interpretability. Below are the key insights from our investigation:

### 1. Dimensions with Clear Interpretability

Certain dimensions demonstrated a strong correlation with specific thematic clusters:

* **Dimension 9: Diverse High-Level Concepts**
    * **Examples:**
        ```
        Explain the difference between supervised and unsupervised learning.
        What happens if you fall into a black hole?
        What are the ethical challenges of artificial intelligence?
        Explain why recursion is useful in programming.
        Tell me a joke about programming.
        ```
    * **Interpretation:** This dimension seems to capture a broad understanding of diverse, high-level concepts, spanning both AI/ML fundamentals and broader scientific topics. The inclusion of a humor-related question suggests a possible link to abstract reasoning or curiosity-driven inquiries.

* **Dimension 21: Programming Humor and Ethics**
    * **Examples:**
        ```
        Tell me a joke about programming.
        Explain the difference between supervised and unsupervised learning.
        What are the ethical challenges of artificial intelligence?
        ```
    * **Interpretation:** This dimension potentially focuses on the intersection of technology-related humor and ethical considerations. It groups questions that require a level of reasoning beyond simple factual recall.

* **Dimension 22: Computer Science Fundamentals**
    * **Examples:**
        ```
        What is the role of a CPU in modern computing?
        Explain quantum computing in simple terms.
        Explain the difference between supervised and unsupervised learning.
        ```
    * **Interpretation:** This dimension likely captures core concepts related to computer hardware and theoretical computation.

* **Dimension 30: Recursion, ML, and Ethics**
    * **Examples:**
        ```
        Explain why recursion is useful in programming.
        How does gradient descent optimize a machine learning model?
        What are the ethical challenges of artificial intelligence?
        ```
    * **Interpretation:** This dimension suggests a grouping of algorithmic concepts (recursion, gradient descent) with AI ethics. The underlying connection between these topics requires further investigation.

### 2. Dimensions with Redundancy or Repetition

Several dimensions contained highly similar or even identical sets of questions, indicating a lack of clear differentiation in the learned features:

* **Dimension 0, 16, 24, 48:** Contain nearly identical, generic programming and ML-related questions.
    * **Examples:**
        ```
        What happens when you type a URL into a browser?
        Write a Python function to generate random numbers.
        Explain the difference between supervised and unsupervised learning.
        ```
    * **Interpretation:** These dimensions may be capturing general technical knowledge but lack the granularity to distinguish between different types of technical inquiries.

* **Dimension 46, 47:** Overlapping ML and ethics themes.
    * **Examples:**
        ```
        What are the ethical challenges of artificial intelligence?
        What is the best way to learn machine learning?
        ```
    * **Interpretation:** These dimensions appear to encode broad AI-related themes without a clear separation of concerns within the AI domain.

### 3. Dimensions with Mixed Topics

Some dimensions exhibited a mixture of seemingly unrelated concepts, making their interpretation challenging:

* **Dimension 45:**
    * **Examples:**
        ```
        Explain the difference between supervised and unsupervised learning.
        What happens when you type a URL into a browser?
        How does a neural network learn from data?
        ```
    * **Interpretation:** This dimension appears to be poorly structured, as it combines a theoretical machine learning concept with a web-related question, suggesting a lack of a cohesive underlying feature.


## Analysis of Responses Across Dimensions

This section details the analysis of the responses observed across the 50 latent dimensions examined in this project. We focused on the uniqueness, frequency, and scoring patterns of the questions that most strongly activate each dimension.

## Key Observations

Out of the 50 dimensions analyzed, we identified a total of **17 unique responses**.

* **Frequency of Responses:**
    * **16 responses** appeared in only **one** dimension each.
    * **1 response** – "What happens when you type a URL into a browser? Explain the concept of cloud computing in simple terms." – was repeated across the remaining **33 dimensions**. This response consistently received a score of **0.0000** in each of its occurrences.

* **Uniqueness of Responses:**
    * Only **17 out of the 50 dimensions** contained distinct sets of top-activating responses.
    * Each of these 17 dimensions presented a unique combination of responses, with the exception of the consistently repeated question mentioned above.

* **Variety in Questions:**
    * The unique questions spanned a diverse range of topics, including:
        * Machine Learning
        * Artificial Intelligence
        * Programming
        * Computer Science Fundamentals
        * Physics
    * **Examples of Unique Questions:**
        ```
        Explain the difference between supervised and unsupervised learning.
        What happens if you fall into a black hole?
        Explain why recursion is useful in programming.
        Write a Python function to reverse a linked list.
        ```

* **Scoring Pattern:**
    * The top-ranked question within each dimension generally exhibited a relatively high activation score, with some scores exceeding **10.0**.
    * Lower-ranked questions within the same dimension displayed gradually decreasing scores.
    * The consistent presence of the zero-scored, repeated response suggests the possibility of **padding or default values** within the underlying dataset.

## Challenges and Limitations

While this analysis successfully identified some potentially interpretable dimensions, several limitations and challenges were encountered:

* **Redundant and overlapping dimensions:** The presence of dimensions with nearly identical activating questions reduces the distinctiveness and interpretability of the latent space.
* **Lack of fine-grained topic separation:** Some dimensions group together unrelated topics, making it difficult to assign a precise and meaningful feature to them.
* **Need for better hyperparameter tuning:** The characteristics of the learned latent dimensions are likely influenced by the model's architecture, training data, and hyperparameters (e.g., `latent_dim`). Further experimentation with these factors could potentially yield more interpretable results.

## Conclusion

This preliminary analysis offers insights into how latent dimensions in LLMs can encode different types of knowledge. While some dimensions appear to align with interpretable themes such as AI ethics and programming concepts, others suffer from redundancy or a lack of clear separation between features. Future research could focus on fine-tuning the model's hyperparameters and carefully curating the dataset to achieve a more disentangled and interpretable latent space. This would involve exploring different values for `latent_dim` and potentially re-training the model with a dataset designed to encourage clearer feature separation. The analysis reveals a mixed landscape in the learned latent dimensions. While a subset of dimensions (17 out of 50) demonstrates diversity in the top-activating responses, a significant portion (33 out of 50) is dominated by a repeated, zero-scored question. This pattern strongly suggests a potential issue with the dataset generation process, where a substantial number of dimensions may not have been effectively populated with meaningful data. The presence of these default responses could obscure the true underlying features captured by those dimensions and warrants further investigation into the data generation pipeline. The dimensions with unique, high-scoring responses offer more promising avenues for understanding the LLM's internal representations of different knowledge domains.
