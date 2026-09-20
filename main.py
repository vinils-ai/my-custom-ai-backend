from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import random
import nltk
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Deep Neural Network Architecture
class PowerfulNeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(PowerfulNeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        out = self.relu(self.l1(x))
        out = self.dropout(out)
        out = self.relu(self.l2(out))
        out = self.dropout(out)
        out = self.relu(self.l3(out))
        out = self.l4(out)
        return out

checkpoint = torch.load("custom_model.pth", map_location=torch.device('cpu'))

input_size = checkpoint["input_size"]
hidden_size = checkpoint["hidden_size"]
output_size = checkpoint["output_size"]
all_words = checkpoint["all_words"]
tags = checkpoint["tags"]
intents = checkpoint["intents"]

model = PowerfulNeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(checkpoint["model_state"])
model.eval()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = [0.0] * len(all_words)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return torch.tensor(bag, dtype=torch.float32).unsqueeze(0)

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def home():
    return {"status": "VinilsAi API is fully operational!"}

@app.post("/predict")
def predict_response(request: ChatRequest):
    sentence = tokenize(request.message)
    X = bag_of_words(sentence, all_words)
    
    output = model(X)
    probs = torch.softmax(output, dim=1)
    prob, predicted = torch.max(probs, dim=1)
    
    tag = tags[predicted.item()]
    
    # 70% Confidence Threshold
    if prob.item() > 0.70:
        for intent in intents['intents']:
            if tag == intent['tag']:
                reply = random.choice(intent['responses'])
                return {"reply": reply, "tag": tag, "confidence": prob.item()}
                
    return {
        "reply": "I am not fully trained on that question yet! Vinil is continuously expanding my dataset.",
        "tag": "unknown",
        "confidence": prob.item()
    }