import torch
from tqdm import tqdm
import numpy as np
from hftokenizer import HFTokenizer


def construct_dataset(data_txt_file, sequence_length=256):
    '''
    data_txt_file : a string path to a text file containing training data, one sample per line
    sequence_length : int, the desired length of each training sequence

    This method should use the trained tokenizer to convert samples to token_ids, and
    then pack them into a training set represented as a 2D array of size (sequences, sequence_length+1).
    The +1 is very important! It lets us compare our model outputs to the sequence shifted by one.

    You can save this training set in whatever format you wish for loading into the training script.
    I recommend using numpy's np.save() method or the pickle module.

    The saved data should be shuffled so we can directly load it and train on it in the training script.
    '''

    # construct tokenizer
    tokenizer = HFTokenizer()
    tokenizer.load()

    # get all samples
    f = open(data_txt_file, "r")
    samples = f.readlines()
    samples = [x.replace("\n", "") for x in samples]

    # ----------------------------------------
    # TODO
    # - add '<|endoftext|>' to each sample or add tokenizer.eos_token_id after tokenizing.
    # - use tokenizer.encode() to tokenize each sample
    # - pack into sequences of length sequence_length
    # - shuffle
    # - save out data


if __name__ == "__main__":
    construct_dataset("./data.txt", 256)