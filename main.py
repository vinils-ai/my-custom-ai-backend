from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import random

app = FastAPI()

# Enable CORS so your website can make requests to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Define the exact same Neural Network structure
class ChatbotModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ChatbotModel, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.relu(self.l1(x))
        out = self.relu(self.l2(out))
        out = self.l3(out)
        return out

# 2. Load the custom model file saved from Colab
checkpoint = torch.load("custom_model.pth", map_location=torch.device('cpu'))

input_size = checkpoint["input_size"]
hidden_size = checkpoint["hidden_size"]
output_size = checkpoint["output_size"]
all_words = checkpoint["all_words"]
tags = checkpoint["tags"]
intents = checkpoint["intents"]

model = ChatbotModel(input_size, hidden_size, output_size)
model.load_state_dict(checkpoint["model_state"])
model.eval()

# 3. Helper to convert text into Bag of Words tensor
def bag_of_words(tokenized_sentence, all_words):
    bag = [0.0] * len(all_words)
    for w in tokenized_sentence:
        if w in all_words:
            bag[all_words.index(w)] = 1.0
    return torch.tensor(bag).unsqueeze(0)

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def home():
    return {"status": "AI Backend is running online!"}

@app.post("/predict")
def predict_response(request: ChatRequest):
    sentence = request.message.lower().split()
    X = bag_of_words(sentence, all_words)
    
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    
    # Match predicted tag with intent responses
    for intent in intents['intents']:
        if tag == intent['tag']:
            reply = random.choice(intent['responses'])
            return {"reply": reply, "tag": tag}
            
    return {"reply": "I am not sure how to answer that yet.", "tag": "unknown"}