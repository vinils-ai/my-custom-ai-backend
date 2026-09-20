from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import random
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.relu(self.l1(x))
        out = self.relu(self.l2(out))
        out = self.l3(out)
        return out

checkpoint = torch.load("custom_model.pth", map_location=torch.device('cpu'))

input_size = checkpoint["input_size"]
hidden_size = checkpoint["hidden_size"]
output_size = checkpoint["output_size"]
all_words = checkpoint["all_words"]
tags = checkpoint["tags"]
intents = checkpoint["intents"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(checkpoint["model_state"])
model.eval()

def clean_sentence(sentence):
    sentence = sentence.lower()
    return re.findall(r'\b\w+\b', sentence)

def bag_of_words(tokenized_sentence, all_words):
    bag = [0.0] * len(all_words)
    for w in tokenized_sentence:
        if w in all_words:
            bag[all_words.index(w)] = 1.0
    return torch.tensor(bag, dtype=torch.float32).unsqueeze(0)

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def home():
    return {"status": "VinilsAi API is live!"}

@app.post("/predict")
def predict_response(request: ChatRequest):
    sentence = clean_sentence(request.message)
    X = bag_of_words(sentence, all_words)
    
    output = model(X)
    probs = torch.softmax(output, dim=1)
    prob, predicted = torch.max(probs, dim=1)
    
    tag = tags[predicted.item()]
    
    # Confidence threshold (75% minimum prediction accuracy required)
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:
                reply = random.choice(intent['responses'])
                return {"reply": reply, "tag": tag, "confidence": prob.item()}
                
    return {
        "reply": "I don't fully understand that yet. Vinil is still expanding my dataset!",
        "tag": "unknown",
        "confidence": prob.item()
    }