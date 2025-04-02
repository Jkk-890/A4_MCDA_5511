import os
import torch
import pickle
from tqdm import tqdm

class TrainingDataGenerator:
    def __init__(self, model, prompts, output_dir, batch_size=8):
        """
        Initializes the TrainingDataGenerator.
        
        model: The model you trained (e.g., NewModel).
        prompts: List of strings containing the corpus of prompts.
        output_dir: Directory to store the training data (activations and text).
        batch_size: The number of prompts to process in a batch.
        """
        self.model = model
        self.prompts = prompts
        self.output_dir = output_dir
        self.batch_size = batch_size

        # Make sure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

    def generate_activations(self, prompt):
        """
        Generates text and collects activations for a single prompt.
        """
        # Tokenize the input prompt
        input_ids = self.model.tokenizer(prompt, return_tensors="pt").input_ids.to('cuda:0')
        
        # Generate text and collect activations
        with torch.no_grad():
            gen_tokens = self.model.pretrained.generate(
                input_ids,
                do_sample=True,
                temperature=0.9,
                max_length=100,
            )
            gen_text = self.model.tokenizer.batch_decode(gen_tokens)[0]
        
        # Retrieve activations from the model (from the hooks)
        activations = self.model.selected_out
        
        return gen_text, activations

    def save_training_data(self):
        """
        Generates and saves the training data (text and activations) for the autoencoder.
        """
        all_data = []
        
        # Process prompts in batches
        for i in tqdm(range(0, len(self.prompts), self.batch_size)):
            batch_prompts = self.prompts[i:i + self.batch_size]
            
            for prompt in batch_prompts:
                gen_text, activations = self.generate_activations(prompt)
                
                # Store the generated text and activations in a dictionary
                data_point = {
                    'prompt': prompt,
                    'generated_text': gen_text,
                    'activations': activations
                }
                
                all_data.append(data_point)
        
        # Save the collected data to a file
        output_file = os.path.join(self.output_dir, 'training_data.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump(all_data, f)
        
        print(f"Training data saved to {output_file}")
        
# Example usage:

# Example corpus of prompts (you can replace this with your own corpus)
prompts = [
    "In a shocking finding, scientists discovered a herd of unicorns living in a remote, previously unexplored valley.",
    "The discovery of an ancient civilization beneath the ocean waves has left scientists baffled.",
    "A new species of bird has been discovered in the Amazon rainforest, known for its brightly colored feathers."
]

# Initialize the TrainingDataGenerator
data_generator = TrainingDataGenerator(model=model, prompts=prompts, output_dir='./training_data')

# Generate and save the training data
data_generator.save_training_data()
