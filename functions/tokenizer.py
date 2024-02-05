from typing import List, Optional
import spacy


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