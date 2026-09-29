import re

def encode(text, vocab):
    parts = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    tokens = [p.strip() for p in parts if p is not None and p.strip() != '']
    return [vocab[t] for t in tokens]


def decode(ids, vocab):
    inv_vocab = {v: k for k, v in vocab.items()}
    tokens = [inv_vocab[i] for i in ids]
    text = ' '.join(tokens)
    text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
    return text