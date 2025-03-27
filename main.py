from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import torch
import torch.nn as nn
from collections import OrderedDict

class NewModel(nn.Module):
    def __init__(self, output_layers, *args):
        super().__init__(*args)
        self.pretrained = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
        self.tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
        self.output_layers = output_layers
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
    

model = NewModel(output_layers = [7, 8]).to('cuda:0')

prompt = (
    "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
    "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
    "researchers was the fact that the unicorns spoke perfect English."
)

input_ids = model.tokenizer(prompt, return_tensors="pt").input_ids

gen_tokens = model.pretrained.generate(
    input_ids,
    do_sample=True,
    temperature=0.9,
    max_length=100,
)
gen_text = model.tokenizer.batch_decode(gen_tokens)[0]

print(gen_text)
print(model.selected_out.keys())