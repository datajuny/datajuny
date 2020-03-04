from torch import nn
from torch.utils.data import Dataset, DataLoader
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from tqdm import tqdm, tqdm_notebook

import pandas as pd
import re
import torch
import torch.nn.functional as F
import torch.optim as optim
import gluonnlp as nlp
import numpy as np



class BERTDataset_infer(Dataset):

    def __init__(self,
                 dataset,
                 sent_idx,
                 bert_tokenizer,
                 max_len,
                 pad,
                 pair):
        
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, 
            max_seq_length = max_len, 
            pad = pad, 
            pair = pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i])


    
class BERTClassifier(nn.Module):
    
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=2,
                 dr_rate = None,
                 params = None):

        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(
            input_ids = token_ids, 
            token_type_ids = segment_ids.long(), 
            attention_mask = attention_mask.float().to(token_ids.device))
        
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

        

def BERT_inference(text):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    _, vocab = get_pytorch_kobert_model(device)

    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    max_len = 80
    batch_size = 64
    warmup_ratio = 0.1
    num_epochs = 10
    max_grad_norm = 1
    log_interval = 30
    learning_rate = 5e-5


    # 1. 로드방법 : 학습한 파라미터만 Load하는 방법
    #new_save_path = 'v3_model_only_parameter_0302.pt'
    #model = BERTClassifier(bertmodel, dr_rate=0.1)
    #model.load_state_dict(new_save_path)
    #model.eval()
    
    # 2. 로드방법 : 모델 전체 저장한것 Load
    save_path = 'v2_model_0302.pt'
    model = torch.load(save_path)
    model.eval()

    infer_data = BERTDataset_infer(text, 0, tok, max_len, True, False)
    infer_data = torch.tensor(next(iter(infer_data))[0]).reshape(1, -1)

    segments_tensors = torch.zeros(len(infer_data[0]))
    segments_tensors = segments_tensors.reshape(1, -1)

    valid_length = torch.tensor(len(infer_data[0]))
    valid_length = valid_length.reshape(1, -1)

    infer_data = infer_data.long().to(device)
    segments_tensors = segments_tensors.long().to(device)
    valid_length = valid_length.long().to(device)


    with torch.no_grad():
        outputs = model(infer_data, valid_length, segments_tensors)
   
    print("딥러닝 최종 inference : ", torch.argmax(outputs[0]))
   
    return torch.argmax(outputs[0])

