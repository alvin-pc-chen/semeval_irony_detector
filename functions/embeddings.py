from typing import List, Dict, Tuple
import torch


def read_pretrained_embeddings(
    embeddings_path: str,
    vocab_path: str
) -> Tuple[Dict[str, int], torch.FloatTensor]:
    """Read the embeddings matrix and make a dict hashing each word.
    Note that we have provided the entire vocab for train and test, so that for practical purposes
    we can simply load those words in the vocab, rather than all 27B embeddings.
    Args:
        embeddings_path (str): Glove embeddings downloaded by wget, pretrained run on 200d.
        vocab_path (str): Vocab file created from tokenizer.
    Returns:
        Tuple[Dict[str, int], torch.FloatTensor]: One hot vector of vocab, embeddings tensor.
    """
    word2i = {}
    vectors = []
    with open(vocab_path, encoding='utf8') as vf:
        vocab = set([w.strip() for w in vf.readlines()]) 
    print(f"Reading embeddings from {embeddings_path}...")
    with open(embeddings_path, "r") as f:
        i = 0
        for line in f:
            word, *weights = line.rstrip().split(" ")
            if word in vocab:
                word2i[word] = i
                vectors.append(torch.Tensor([float(w) for w in weights]))
                i += 1
    
    return word2i, torch.stack(vectors)


def get_glove_embeddings(
    embeddings_path: str,
) -> Tuple[Dict[str, int], torch.FloatTensor]:
    """Read all glove embeddings and make a dict hashing each word instead.
    Args:
        embeddings_path (str): Glove embeddings downloaded by wget, pretrained run on 200d.
    Returns:
        Tuple[Dict[str, int], torch.FloatTensor]: One hot vector of vocab, embeddings tensor.
    """
    word2i = {}
    vectors = []
    print(f"Reading embeddings from {embeddings_path}...")
    with open(embeddings_path, "r") as f:
        i = 0
        for line in f:
            word, *weights = line.rstrip().split(" ")
            word2i[word] = i
            vectors.append(torch.Tensor([float(w) for w in weights]))
            i += 1

    return word2i, torch.stack(vectors)


def get_oovs(vocab_path: str, word2i: Dict[str, int]) -> List[str]:
    """Find the vocab items that do not exist in the glove embeddings (in word2i).
    Return the List of such (unique) words.
    Args:
        vocab_path: List of batches of sentences.
        word2i (Dict[str, int]): See above.
    Returns:
        List[str]: Words not in glove.
    """
    with open(vocab_path, encoding='utf8') as vf:
        vocab = set([w.strip() for w in vf.readlines()])
    glove_and_vocab = set(word2i.keys())
    vocab_and_not_glove = vocab - glove_and_vocab
    
    return list(vocab_and_not_glove)


def intialize_new_embedding_weights(num_embeddings: int, dim: int) -> torch.FloatTensor:
    """Xavier initialization for the embeddings of words in train, but not in glove."""
    weights = torch.empty(num_embeddings, dim)
    torch.nn.init.xavier_uniform_(weights)
    
    return weights


def update_embeddings(
    glove_word2i: Dict[str, int],
    glove_embeddings: torch.FloatTensor,
    oovs: List[str]
) -> Tuple[Dict[str, int], torch.FloatTensor]:
    i = len(glove_word2i)
    for oov in oovs:
        if oov not in glove_word2i:
            glove_word2i[oov] = i
            i += 1
    new_embeddings = torch.cat((glove_embeddings, intialize_new_embedding_weights(len(oovs), glove_embeddings.shape[1])))
    
    return glove_word2i, new_embeddings