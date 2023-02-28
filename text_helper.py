import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from collections import Counter


ss = SnowballStemmer('english')
sw = stopwords.words('english')

def split_tokens(row):
    row['all_tokens'] = [ss.stem(i) for i in
                         re.split(r" +",
                         re.sub(r"[^a-z@# ]", "",
                         row['tweet'].lower()))
                         if (i not in sw) and len(i)]
    return row

def get_info(dataset):
    counts = Counter([i for s in dataset['train']['all_tokens'] for i in s])
    counts = {k:v for k, v in counts.items() if v>10}
    vocab = list(counts.keys())
    n_v = len(vocab)
    id2tok = dict(enumerate(vocab))
    tok2id = {token: id for id, token in id2tok.items()}

    return counts, vocab, n_v, id2tok, tok2id

def remove_rare_tokens(row, vocab):
    row['tokens'] = [t for t in row['all_tokens'] if t in vocab]
    return row

def windowizer(row, tok2id, wsize=3):
    """
    Windowizer function for Word2Vec. Converts sentence to sliding-window
    pairs.
    """
    doc = row['tokens']
    wsize = 3
    out = []
    for i, wd in enumerate(doc):
        target = tok2id[wd]
        window = [i+j for j in
                  range(-wsize, wsize+1, 1)
                  if (i+j >= 0) &
                     (i+j < len(doc)) &
                     (j != 0)]

        out+=[(target, tok2id[doc[w]]) for w in window]
    row['moving_window'] = out
    return row

