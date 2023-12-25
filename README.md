### Summary
 - Learned to train, save, and load PyTorch modules and LSTMs
 - Learned to use spaCy tokenizers and encode text data with static embeddings
 - Learned to conduct hyperperameter tuning and evaluation for optimal results

### Use
 1. for ideal results use python=3.9
 2. run pip install -r requirements.txt in the repo root directory
 3. run the train_from_scratch.ipynb notebook as is

### Project Details
 The Irony Detector ([a SemEval 2018 subtask](https://github.com/Cyvhee/SemEval2018-Task3)) was completed as part of my NLP class at CU Boulder in Fall 2023. In this project, I created a spaCy tokenizer to learn the vocabulary across the dataset (a selection of tweets), after which I used [GloVe embeddings](https://nlp.stanford.edu/projects/glove/) to encode the data. The vocabulary is saved into a text file for future use while out of vocabulary words are initialized randomly using the [Xavier](https://pytorch.org/docs/stable/_modules/torch/nn/init.html#xavier_uniform_) embedding initialization method, which creates randomized embeddings for more efficient learning (compared to zeroed weights). I then developed a language model module using PyTorch, comprising several BiLSTM layers, a linear layer, and a softmax layer to produce the classification result. Finally, I set up a training loop and trained the model using a variety of hyperperameters in order to get the ideal results, achieving an F1 of roughly 0.70 which was the [top result](https://competitions.codalab.org/competitions/17468#results) on the SemEval subtask (see Evaluation Task A). For this model, I wrote a function to save the model weights, embedding weights, and index so that the tokenizer and the model can be initialized from scratch.

### Hyperparameter Results
| **Test Avg F1** | **Epochs** | **Learning Rate** | **Optimizer** | **Hidden Layers** | **Weight Decay** |
|-----------------|------------|-------------------|---------------|-------------------|------------------|
| **0.7000**      | 20         | 0.00005           | AdamW         | 128               | 0                |
| **0.6947**      | 20         | 0.00005           | AdamW         | 256               | 0                |
| **0.6947**      | 15         | 0.00005           | AdamW         | 256               | 0                |
| **0.6798**      | 15         | 0.00005           | AdamW         | 128               | 0.01             |
| **0.6657**      | 15         | 0.00005           | AdamW         | 128               | 0.1              |
| **0.6899**      | 15         | 0.0001            | AdamW         | 64                | 0.1              |
| **0.6525**      | 15         | 0.001             | AdamW         | 64                | 0.1              |
| **0.5535**      | 10         | 0.001             | AdamW         | 64                | 0.1              |
| **0.5556**      | 10         | 0.1               | SGD           | 32                | 0                |
| **0.2840**      | 20         | 0.00001           | RAdam         | 128               | 0.1              |
| **0.6846**      | 15         | 0.001             | RAdam         | 64                | 0.0001           |
| **0.3758**      | 12         | 0.01              | RAdam         | 128               | 0.0001           |
| **0.7129**      | 10         | 0.00005           | AdamW         | 128               | 0                |

While working on the training loop I used high learning rates (0.1), low epoch counts (<5) and few hidden layers (~30), which yielded low F1's (I also slightly adjusted batch size to 16). As a result, I started hyperparameter tuning with much higher epochs, lower learning rates, and more hidden layers (I also changed train/dev split to 0.85 for all tests). Compared to the top results in the 2018 SemEval shared task (F1 = 0.7054), my first test with the above hyperparameters already performed incredibly well. I hypothesized that the performance was due to using much higher hidden layers, so I tried with double the hidden layer count. While the resulting F1's are not significantly different, I was not able to get an F1 higher than 128, so I tried instead with lower layers. As seen, lower hidden layers yielded a significantly worse result. On the other hand, I had not used any weight decay for the first few tests, so I tested this as well. An important finding was that if the weight decay was significantly greater than the learning rate, the dev F1's during training would not change between epochs (or sometimes yield 0.0), which suggested that the model was not training (the sessions terminated mid-training are not shown). This also occurred with learning rates lower than 0.00005, as shown in the test with F1 = 0.2840 (other tests without weight decay also failed to train and were terminated before completion). Higher learning rates also did not necessarily provide worse results, but certainly did not improve on the original score. Finally, I tried several different optimizers, however training failed in multiple sessions and were not recorded. Since there were so few completed trainings, there may be some room for optimization on that front. For the final test, I noticed that in earlier trials dev F1's would decrease in the final few epochs, which suggested overfitting (although final train F1's were not significantly higher than dev or test F1's), so I tested with a much lower epoch rate. The end result was an incredibly high F1 compared to the SemEval 2018 results and compared to the other testing.
