import os
import shutil
import urllib
import subprocess
from typing import List, Dict, Tuple, Optional
import random

from tqdm.notebook import tqdm as tqdm
from numpy import logical_and, sum as t_sum
import numpy as np
import emoji
import torch
import spacy

#### READ AND LOAD DATASET ####
SEM_EVAL_FOLDER = 'SemEval2018-Task3'
TRAIN_FILE = SEM_EVAL_FOLDER + '/datasets/train/SemEval2018-T3-train-taskA_emoji.txt'
TEST_FILE = SEM_EVAL_FOLDER + '/datasets/goldtest_TaskA/SemEval2018-T3_gold_test_taskA_emoji.txt'

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


def download_zip(url: str, dir_path: str):
    import zipfile
    if url.endswith('zip'):
        print('Downloading dataset file')
        path_to_zip_file = 'downloaded_file.zip'
        urllib.request.urlretrieve(url, path_to_zip_file)
        with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
            zip_ref.extractall('./')
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        if os.path.exists(path_to_zip_file):
            os.remove(path_to_zip_file)
        if os.path.exists(dir_path + '-master'):
            shutil.move(dir_path + '-master', dir_path)
        elif os.path.exists(dir_path + '-main'):
            shutil.move(dir_path + '-main', dir_path)
    print(f'Downloaded dataset to {dir_path}')


def git(*args):
    return subprocess.check_call(['git'] + list(args))


def download_irony():
    if os.path.exists(SEM_EVAL_FOLDER):
        return
    else:
        try:
            git('clone', 'https://github.com/Cyvhee/SemEval2018-Task3.git')
            return
        except OSError:
            pass
    print('Downloading dataset')
    download_zip('https://github.com/Cyvhee/SemEval2018-Task3/archive/master.zip', SEM_EVAL_FOLDER)


def read_dataset_file(file_path):
    with open(file_path, 'r', encoding='utf8') as ff:
        rows = [line.strip().split('\t') for line in ff.readlines()[1:]]
        _, labels, texts = zip(*rows)
    clean_texts = [emoji.demojize(tex) for tex in texts]
    unique_labels = sorted(set(labels))
    lab2i = {lab: i for i, lab in enumerate(unique_labels)}
    return clean_texts, labels, lab2i


def load_datasets():
    download_irony()
    train_texts, train_labels, label2i = read_dataset_file(TRAIN_FILE)
    test_texts, test_labels, _ = read_dataset_file(TEST_FILE)
    
    return train_texts, train_labels, test_texts, test_labels, label2i


#### TRAIN AND RUN MODEL ####
def split_data(sentences, labels, split=0.8):
    """Splits the training data into training and development sets."""
    data = list(zip(sentences, labels))
    random.shuffle(data)
    sents, labs = zip(*data)
    index = int(len(sents) * split)
    
    return sents[:index], labs[:index], sents[index:], labs[index:]


def make_batches(sequences: List[str], batch_size: int) -> List[List[str]]:
    """Yield batch_size chunks from sequences."""
    batches = []
    for i in range(0, len(sequences), batch_size):
        batches.append(sequences[i:i + batch_size])
    
    return batches


class Tokenizer:
    """Tokenizes and pads a batch of input sentences."""
    def __init__(self, pad_symbol: Optional[str] = "<PAD>"):
        """Initializes the tokenizer.
        Args:
            pad_symbol (Optional[str], optional): The symbol for a pad. Defaults to "<PAD>".
        """
        self.pad_symbol = pad_symbol
        self.nlp = spacy.load("en_core_web_sm")
    
    def __call__(self, batch: List[str]) -> List[List[str]]:
        """Tokenizes each sentence in the batch, and pads them if necessary so
        that we have equal length sentences in the batch.
        Args:
            batch (List[str]): A List of sentence strings.
        Returns:
            List[List[str]]: A List of equal-length token Lists.
        """
        batch = self.tokenize(batch)
        batch = self.pad(batch)
        return batch

    def tokenize(self, sentences: List[str]) -> List[List[str]]:
        """Tokenizes the List of string sentences into a Lists of tokens using spacy tokenizer.
        Args:
            sentences (List[str]): The input sentence.
        Returns:
            List[str]: The tokenized version of the sentence.
        """
        tokens = []
        for sentence in sentences:
            t = ["<SOS>"]
            for token in self.nlp(sentence):
                t.append(token.text)
            t.append("<EOS>")
            tokens.append(t)
        return tokens

    def pad(self, batch: List[List[str]]) -> List[List[str]]:
        """Appends pad symbols to each tokenized sentence in the batch such that
        every List of tokens is the same length. This means that the max length sentence
        will not be padded.
        Args:
            batch (List[List[str]]): Batch of tokenized sentences.
        Returns:
            List[List[str]]: Batch of padded tokenized sentences. 
        """
        maxlen = max([len(sent) for sent in batch])
        for sent in batch:
            for i in range(maxlen - len(sent)):
                sent.append(self.pad_symbol)
        return batch
    

def encode_sentences(batch: List[List[str]], word2i: Dict[str, int]) -> torch.LongTensor:
    """Encode the tokens in each sentence in the batch with a dictionary.
    Args:
        batch (List[List[str]]): The padded and tokenized batch of sentences.
        word2i (Dict[str, int]): The encoding dictionary.
    Returns:
        torch.LongTensor: The tensor of encoded sentences.
    """
    UNK_IDX = word2i["<UNK>"]
    tensors = []
    for sent in batch:
        tensors.append(torch.LongTensor([word2i.get(w, UNK_IDX) for w in sent]))
        
    return torch.stack(tensors)


def encode_labels(labels: List[int]) -> torch.FloatTensor:
    """Turns the batch of labels into a tensor.
    Args:
        labels (List[int]): List of all labels in the batch.
    Returns:
        torch.FloatTensor: Tensor of all labels in the batch.
    """
    return torch.LongTensor([int(l) for l in labels])


class IronyDetector(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embeddings_tensor: torch.FloatTensor,
        pad_idx: int,
        output_size: int,
        dropout_val: float = 0.5,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.pad_idx = pad_idx
        self.dropout_val = dropout_val
        self.output_size = output_size
        # Initialize the embeddings from the weights matrix.
        # Check the documentation for how to initialize an embedding layer
        # from a pretrained embedding matrix. 
        # Be careful to set the `freeze` parameter!
        # Docs are here: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html#torch.nn.Embedding.from_pretrained
        self.embeddings = torch.nn.Embedding.from_pretrained(embeddings_tensor, True, pad_idx)
        # Dropout regularization
        # https://jmlr.org/papers/v15/srivastava14a.html
        self.dropout_layer = torch.nn.Dropout(p=self.dropout_val, inplace=False)
        # Bidirectional 2-layer LSTM. Feel free to try different parameters.
        # https://colah.github.io/posts/2015-08-Understanding-LSTMs/
        self.lstm = torch.nn.LSTM(
            self.input_dim,
            self.hidden_dim,
            num_layers=3,
            dropout=dropout_val,
            batch_first=True,
            bidirectional=True,
        )
        # For classification over the final LSTM state.
        self.classifier = torch.nn.Linear(hidden_dim*2, self.output_size)
        self.log_softmax = torch.nn.LogSoftmax(dim=2)
    
    def encode_text(
        self,
        symbols: torch.Tensor
    ) -> torch.Tensor:
        """Encode the (batch of) sequence(s) of token symbols with an LSTM.
            Then, get the last (non-padded) hidden state for each symbol and return that.
        Args:
            symbols (torch.Tensor): The batch size x sequence length tensor of input tokens.
        Returns:
            torch.Tensor: The final hiddens tate of the LSTM, which represents an encoding of
                the entire sentence.
        """
        # First we get the embedding for each input symbol
        embedded = self.embeddings(symbols)
        embedded = self.dropout_layer(embedded)
        # Packs embedded source symbols into a PackedSequence.
        # This is an optimization when using padded sequences with an LSTM
        lens = (symbols != self.pad_idx).sum(dim=1).to("cpu")
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            embedded, lens, batch_first=True, enforce_sorted=False
        )
        # -> batch_size x seq_len x encoder_dim, (h0, c0).
        packed_outs, (H, C) = self.lstm(packed)
        encoded, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_outs,
            batch_first=True,
            padding_value=self.pad_idx,
            total_length=None,
        )
        # Now we have the representation of each token encoded by the LSTM.
        encoded, (H, C) = self.lstm(embedded)
        # This part looks tricky. All we are doing is getting a tensor
        # That indexes the last non-PAD position in each tensor in the batch.
        last_enc_out_idxs = lens - 1
        # -> B x 1 x 1.
        last_enc_out_idxs = last_enc_out_idxs.view([encoded.size(0)] + [1, 1])
        # -> 1 x 1 x encoder_dim. This indexes the last non-padded dimension.
        last_enc_out_idxs = last_enc_out_idxs.expand(
            [-1, -1, encoded.size(-1)]
        )
        # Get the final hidden state in the LSTM
        last_hidden = torch.gather(encoded, 1, last_enc_out_idxs)
        return last_hidden
    
    def forward(
        self,
        symbols: torch.Tensor,
    ) -> torch.Tensor:
        encoded_sents = self.encode_text(symbols)
        output = self.classifier(encoded_sents)
        return self.log_softmax(output)
    

def predict(model: torch.nn.Module, sequences: List[torch.Tensor]):
    """Run prediction with the model
    Args: model (torch.nn.Module): Model to use.
        dev_sequences (List[torch.Tensor]): List of encoded sequences to analyze.
    Returns:
        List[int]: List of predicted labels.
    """
    preds = []
    with torch.no_grad():
        model.eval()
        logits = model(sequences)
        preds = list(torch.argmax(logits, axis=2).squeeze().numpy())
    return preds


def training_loop(
    num_epochs,
    train_features,
    train_labels,
    dev_features,
    dev_labels,
    optimizer,
    model,
    label2i,
):
    print("Training...")
    loss_func = torch.nn.NLLLoss()
    batches = list(zip(train_features, train_labels))
    random.shuffle(batches)
    for i in range(num_epochs):
        losses = []
        for features, labels in tqdm(batches):
            # Empty the dynamic computation graph
            optimizer.zero_grad()
            logits = model(features).squeeze(1)
            loss = loss_func(logits, labels)
            # Backpropogate the loss through our model
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"epoch {i+1}, loss: {sum(losses)/len(losses)}")
        # Estimate the f1 score for the development set
        print("Evaluating dev...")
        preds = predict(model, dev_features)
        dev_f1 = f1_score(preds, dev_labels, label2i['1'])
        dev_avg_f1 = avg_f1_score(preds, dev_labels, set(label2i.values()))
        print(f"Dev F1 {dev_f1}")
        print(f"Avg Dev F1 {dev_avg_f1}")
    # Return the trained model
    return model


def accuracy(predicted_labels, true_labels):
    """Accuracy is correct predictions / all predicitons"""
    correct_count = 0
    for pred, label in zip(predicted_labels, true_labels):
        correct_count += int(pred == label)
    
    return correct_count/len(true_labels) if len(true_labels) > 0 else 0.


def precision(predicted_labels, true_labels, which_label=1):
    """Precision is True Positives / All Positives Predictions"""
    pred_which = np.array([pred == which_label for pred in predicted_labels])
    true_which = np.array([lab == which_label for lab in true_labels])
    denominator = t_sum(pred_which)
    if denominator:
        return t_sum(logical_and(pred_which, true_which))/denominator
    else:
        return 0.


def recall(predicted_labels, true_labels, which_label=1):
    """Recall is True Positives / All Positive Labels"""
    pred_which = np.array([pred == which_label for pred in predicted_labels])
    true_which = np.array([lab == which_label for lab in true_labels])
    denominator = t_sum(true_which)
    if denominator:
        return t_sum(logical_and(pred_which, true_which))/denominator
    else:
        return 0.


def f1_score(predicted_labels, true_labels, which_label=1):
    """F1 score is the harmonic mean of precision and recall"""
    P = precision(predicted_labels, true_labels, which_label=which_label)
    R = recall(predicted_labels, true_labels, which_label=which_label)
    if P and R:
        return 2*P*R/(P+R)
    else:
        return 0.


def avg_f1_score(predicted_labels: List[int], true_labels: List[int], classes: List[int]):
    """
    Calculate the f1-score for each class and return the average of it
    :return: float
    """
    return sum([f1_score(predicted_labels, true_labels, which_label=which_label) for which_label in classes])/len(classes)


#### SAVE AND LOAD MODEL ####
MODEL = "./pretrained_model/pretrained_irony_detector.pth"
EMBEDDINGS = "./pretrained_model/pretrained_embeddings.pth"
WORD2I = "./pretrained_model/word2i.pth"
LABEL2I = "./pretrained_model/label2i.pth"

def save_model(model=MODEL, embeddings=EMBEDDINGS, word2i=WORD2I, label2i=LABEL2I):
    torch.save(model.state_dict(), MODEL)
    torch.save(embeddings, EMBEDDINGS)
    torch.save(word2i, WORD2I)
    torch.save(label2i, LABEL2I)
    print("Saved model at: " + str(os.getcwd()) + "/pretrained_model/")


def load_model(pretrained_model=MODEL, embeddings=EMBEDDINGS, hidden_dim=128, word2i=WORD2I, label2i=LABEL2I):
    embeddings = torch.load(embeddings)
    word2i = torch.load(word2i)
    label2i = torch.load(label2i)
    model = IronyDetector(
        input_dim=embeddings.shape[1],
        hidden_dim=hidden_dim,
        embeddings_tensor=embeddings,
        pad_idx=word2i["<PAD>"],
        output_size=len(label2i),
    )
    model.load_state_dict(torch.load(pretrained_model))
    model.eval()