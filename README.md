# Interpretable Features from LLM Activations

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
