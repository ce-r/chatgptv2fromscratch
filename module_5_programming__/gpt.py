import torch
from linear import CustomLinear
from embedding import CustomEmbedding
from mha import CustomMHA

'''
Complete this module which handles a single "block" of our model
as described in our lecture. You should have two sections with
residual connections around them:

1) norm1, mha
2) norm2, a two-layer MLP, dropout

It is perfectly fine to use pytorch implementations of layer norm and dropout,
as well as activation functions (torch.nn.LayerNorm, torch.nn.Dropout, torch.nn.ReLU).

For layer norm, you just need to pass in D-model: self.norm1 = torch.nn.LayerNorm((d_model,))

'''
class TransformerDecoderBlock(torch.nn.Module):

    def __init__(self, d_model, n_heads):
        super().__init__()
        # TODO

    
    def forward(self, x):
        '''
        param x : (tensor) a tensor of size (batch_size, sequence_length, d_model)
        returns the computed output of the block with the same size.
        '''
        # TODO


'''
Create a full GPT model which has two embeddings (token and position),
and then has a series of transformer block instances (layers). Finally, the last 
layer should project outputs to size [vocab_size].
'''
class GPTModel(torch.nn.Module):

    
    def __init__(self, d_model, n_heads, layers, vocab_size, max_seq_len):
        '''
        param d_model : (int) the size of embedding vectors and throughout the model
        param n_heads : (int) the number of attention heads, evenly divides d_model
        param layers : (int) the number of transformer decoder blocks
        param vocab_size : (int) the final output vector size
        param max_seq_len : (int) the longest sequence the model can process.
            This is used to create the position embedding- i.e. the highest possible
            position to embed is max_seq_len
        '''

        super().__init__()
        # TODO
        # hint: for a stack of N layers look at torch ModuleList or torch Sequential

    
    def forward(self, x):
        '''
        param x : (long tensor) an input of size (batch_size, sequence_length) which is
            filled with token ids

        returns a tensor of size (batch_size, sequence_length, vocab_size), the raw logits for the output
        '''
        # TODO
        # hint: x contains token ids, but you will also need to build a tensor of position ids here




if __name__ == "__main__":

    # example of building the model and doing a forward pass
    D = 128
    H = 8
    L = 4
    model = GPTModel(D, H, L, 1000, 512)
    B = 32
    S = 48 # this can be less than 512, it just cant be more than 512
    x = torch.randint(1000, (B, S))
    y = model(x) # this should give us logits over the vocab for all positions

    # should be size (B, S, 1000)
    print(y)
