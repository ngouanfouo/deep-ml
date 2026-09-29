import re

def tokenize(vocab, text, mode):
    """
    vocab: dict[str, int] containing '<|unk|>' and '<|endoftext|>'
    text: str (if mode='encode') or list[int] (if mode='decode')
    mode: 'encode' or 'decode'
    """
    if mode == 'encode':
        parts = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        tokens = [p.strip() for p in parts if p is not None and p.strip() != '']

        unk_id = vocab['<|unk|>']
        return [vocab.get(t, unk_id) for t in tokens]

    elif mode == 'decode':
        inv_vocab = {v: k for k, v in vocab.items()}
        tokens = [inv_vocab[i] for i in text]
        result = ' '.join(tokens)
        # Remove space before specified punctuation (including : and ;)
        result = re.sub(r'\s+([,.?!"()\':;])', r'\1', result)
        return result

    else:
        raise ValueError(f"Unknown mode: {mode}")