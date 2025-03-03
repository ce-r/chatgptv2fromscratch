import torch
import numpy as np
from gpt import GPTModel
import matplotlib.pyplot as plt


# since we didn't really cover how to do this in lecture-
# this creates a learning rate schedule for you. Refer to the 
# pytorch docs for more info on using a scheduler.

# This one is designed for you to call scheduler.step() on every
# model update step. 
def cosine_with_warmup_lr_scheduler(opt, total_steps, warmup_steps):
    def thunk(stepnum):
        if stepnum <= warmup_steps:
            # go from ~0 to 1.0
            prog = float(stepnum)/float(warmup_steps)
            lrmult = 0.00001 + prog
        else:
            # go from 1.0 to ~0
            steps_after_peak = stepnum-warmup_steps
            tail_steps = total_steps-warmup_steps
            prog = float(steps_after_peak) / float(tail_steps)
            lrmult = ((np.cos(3.141592*prog)+1.0)*0.5)*0.9 + 0.1
        return max(lrmult, 0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=thunk)
    return scheduler

# ===========================================================================

'''
Complete the following method which trains a GPT model and saves a loss curve.

To reiterate: you don't need to worry about weight decay, weight initialization, grad accumulation, or weight tying.
Use whatever batch size you are able, even something like 2 or 4 is fine.
Use a few hundred warmup steps and a peak learning rate that is (something x 10-4).
'''
def train():

    device = torch.device("cuda") # use "cpu" if not gpu available

    # adjust as needed
    model = GPTModel(d_model=512, n_heads=16, layers=8, vocab_size=10000, max_seq_len=256)
    param_count = sum(p.numel() for p in model.parameters())
    print("Model has", param_count, "parameters.")

    model = model.to(device)

    '''
    # TODO
    
    pseudocode:
    opt = torch.optim.AdamW(...
    scheduler = cosine_with_warmup_lr_scheduler(...

    for batch in dataset:
        opt.zero_grad()

        <run batch through model and compute loss>

        # VERY IMPORTANT ---
        torch.nn.CrossEntropyLoss expects classes in the 2nd dimension.
        You may need to transpose dimensions around to make this work.
        It also expects unnormalized logits (no softmax).
        # ---

        loss.backward()

        # clip the gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # step the optimizer and scheduler
        opt.step()
        scheduler.step() 

        # log total tokens and loss
        # periodically save a plot of loss vs tokens      
        
    '''

    # save model weights if you want
    torch.save(model.state_dict(), "./model_weights.pt")



if __name__ == "__main__":
    train()