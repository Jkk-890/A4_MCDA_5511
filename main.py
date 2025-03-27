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