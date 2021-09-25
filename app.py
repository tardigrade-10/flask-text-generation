import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, render_template
import numpy as np
from nltk.tokenize import RegexpTokenizer
import pickle


class NgramModel(nn.Module):
    def __init__(self, vocab_size, context_size, n_dim):
        super(NgramModel, self).__init__()
        self.n_word = vocab_size
        self.embedding = nn.Embedding(self.n_word, n_dim)
        self.linear1 = nn.Linear(context_size * n_dim, 128)
        self.linear2 = nn.Linear(128, self.n_word)

    def forward(self, x):
        emb = self.embedding(x)
        emb = emb.view(1,-1)
        out = self.linear1(emb)
        out = F.relu(out)
        out = self.linear2(out)
        log_prob = F.log_softmax(out, dim = 1)
        return log_prob

file =  open('movie_review_corpus.txt')
text = file.read().lower()
tokenizer = RegexpTokenizer(r'\w+')
corpus = tokenizer.tokenize(text)
text = corpus[:100000]
pentagram = [((text[i:i+5]), text[i+5]) for i in range(len(text) - 5)]
vocab = set(text)
wrd_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_wrd = {wrd_to_idx[word]:word for word in vocab}


model = NgramModel(len(wrd_to_idx), 5, 100)
# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
model = torch.load("text_gen_model_weights.pth", map_location=torch.device('cpu'))
# model.load_state_dict(torch.load("text_gen_model_weights.pth", map_location=torch.device('cpu')))
model.eval()

# model.load_state_dict(torch.load("text_gen_model_weights.pth", map_location=torch.device('cpu'))
# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

def text_generator(n, input_text):
    start_seq = input_text
    final_seq = start_seq
    start = start_seq.split(" ")
    final = final_seq.split(" ")
    for i in range(int(n)):
        start_tensor = torch.tensor([wrd_to_idx[i] for i in start], dtype = torch.long)
        out = model(start_tensor)
        _, predict_label = torch.max(out, 1)
        predict_word = idx_to_wrd[predict_label.item()]
        final.append(str(predict_word))
        start = final[i+1:]
    final = " ".join(final)
    return final
    
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def index():
    if request.method == 'POST':
        input = request.form['input']
        num = request.form['num']
        result = text_generator(num, input)

    return render_template('output.html', result = result)

if __name__ == '__main__':
	app.run(debug=True)
    # inp = "i love the way you"
    # num = 20
    # print(text_generator(num, inp))



