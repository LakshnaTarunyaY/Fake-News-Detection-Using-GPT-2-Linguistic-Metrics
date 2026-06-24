from transformers import AutoTokenizer,AutoModelForCausalLM
import torch
import math

def load_model():
    tokenizer=AutoTokenizer.from_pretrained('gpt2')
    model=AutoModelForCausalLM.from_pretrained('gpt2')
    model.eval()
    return tokenizer,model

def encode(texts,tokenizer):
    encoded=tokenizer(texts,return_tensors='pt',truncation=True,max_length=1024)
    return encoded['input_ids']

def perplexity(texts,tokenizer,model):
    
    tokens=encode(texts,tokenizer)
    with torch.no_grad():
        outputs=model(tokens,labels=tokens)
    loss=outputs.loss
    per=torch.exp(loss)
    return per.item()

def token_prediction_accuracy(texts,tokenizer,model):
    tokens=encode(texts,tokenizer)
    if tokens.size(1)<2:
        return 0.0
    with torch.no_grad():
        outputs=model(tokens)
    logits=outputs.logits
    predictions=torch.argmax(logits,dim=-1)
    correct=(predictions[:,:-1]==tokens[:,1:]).sum().item()
    total=tokens.size(1)-1
    accuracy=correct/total
    return accuracy

def extract(texts,tokenizer,model):
    features=[]
    per=[]
    acc=[]
    for i in texts:
        p=perplexity(i,tokenizer,model)
        a=token_prediction_accuracy(i,tokenizer,model)
        features.append([float(p),float(a)])
        per.append(float(p))
        acc.append(float(a))
    return features,per,acc
