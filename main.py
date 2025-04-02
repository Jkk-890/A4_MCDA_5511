from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np

class NewModel(nn.Module):
    def __init__(self, *args):
        super().__init__(*args)
        self.pretrained = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
        self.tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
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