import os
import torch
from ironymodel import IronyDetector


#### MODEL PATHS ####
MODEL = "./pretrained_model/pretrained_irony_detector.pth"
EMBEDDINGS = "./pretrained_model/pretrained_embeddings.pth"
WORD2I = "./pretrained_model/word2i.pth"
LABEL2I = "./pretrained_model/label2i.pth"

def save_model(model, path=MODEL, embeddings=EMBEDDINGS, word2i=WORD2I, label2i=LABEL2I):
    torch.save(model.state_dict(), path)
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