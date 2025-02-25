# !pip install datasets

from datasets import load_dataset

ds = load_dataset("ubaada/booksum-complete-cleaned", "chapters")["train"]

lines = []
for entry in ds:
	text = entry["text"]
	text = text.replace("\n", " ") # remove newline formatting
	text = " ".join(text.split()) # remove sequences of whitespace
	lines.append(text+"\n")

	# you can get even more data by increasing this above 100.
	# The highest is ~5600.
	if len(lines) == 100:
		break

f = open("data.txt", "w")
f.writelines(lines)
f.close()

def save_vocab(vocab):
  '''
  vocabulary - a list of words in the vocabulary
  '''
  with open("vocab.txt", "w", encoding="utf-8") as f:
    for token in sorted(vocab):
      f.write(f"{token}\n")

def save_merges(merges):
  '''
  merges - a list of tuples: (str, str2) representing a merge
  '''
  with open("merges.txt", "w", encoding="utf-8") as f:
    for merge in merges:
      f.write(f"{merge[0]} {merge[1]}\n")

import heapq
from collections import Counter, defaultdict

# method uses Counter(), a specialized dict to keep
# the frequencies for the dict key adjacent pairs
def count_pairs(corpus):
  counts = Counter()
  for word in corpus:
    for i in range(len(word) - 1):
      counts[(word[i], word[i + 1])] += 1
  return counts

# does the merging of pairs and updating of the corpus
def merge_pairs(corpus, pair):
  # unpack tuple elements for matching
  a, b = pair
  # merge for appending to merges
  merged_pair = a + b
  new_corpus = []

  for word in corpus:
    new_tokens = []
    i = 0
    # iterating indices by i+1 or i+2, if adjacent values match or no match
    while i < len(word):
      if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
        new_tokens.append(merged_pair)
        i += 2
      else:
        new_tokens.append(word[i])
        i += 1
    new_corpus.append(new_tokens)
  return new_corpus

# does everything by calling count_pairs, which works on the dictionary
# and calls merge_pairs, which works on a list and tuple and updates
# the vocab list by passing the counts dict to a heap structure, updating the
# vocab set. heap gets popped and the merges list gets populated by accessing
# the max_pair tuple that was previously stored in the dict counts structures
# returns essential structures, vocab and merges for the save in train_tokenizer()
def char_bpe(corpus, vocab_size):
  corpus = [list(word) for word in corpus.replace(" ", "|").split("|")]
  vocab = set(char for word in corpus for char in word)
  merges = []

  counts = count_pairs(corpus)
  # from Counter() to heapify structure, both optimized structure
  # access max heap value instead of min heap value by negation and
  # return key-value pair
  heap = [(-freq, pair) for pair, freq in counts.items()]
  # heapify stores structures with access complexity of log n instead of n
  heapq.heapify(heap)

  while len(vocab) < vocab_size and heap: # stopping point for training
    # find max pair in the heapify priority queue
    _, max_pair = heapq.heappop(heap)
    corpus = merge_pairs(corpus, max_pair)
    new_token = "".join(max_pair)
    vocab.add(new_token)
    merges.append((max_pair[0], max_pair[1]))  # merges list appends tuples

    # update counts dictionary and heap pq
    counts = count_pairs(corpus)
    heap = [(-freq, pair) for pair, freq in counts.items()]
    heapq.heapify(heap)

  return vocab, merges

# implements the BPE tokeniker calling pertinent methods
# saves vocab and merges structs to files in current wd
def train_tokenizer(file_path, vocab_size):
  with open(file_path, "r", encoding="utf-8") as f:
    corpus = f.read()

  vocab, merges = char_bpe(corpus, vocab_size)
  save_vocab(vocab)
  save_merges(merges)

train_tokenizer("data.txt", vocab_size=1095)
# 24 mins, vocab_size=1095/len(base)+1000

