import os
import re
import json
import gdown

import numpy as np
import torch
import torch.nn as nn
from underthesea import word_tokenize
from transformers import AutoTokenizer
        
class PhoBERT_classification(nn.Module):
  def __init__(self, phobert):
    super(PhoBERT_classification, self).__init__()
    
    self.phobert = phobert
    self.dropout = nn.Dropout(0.2)
    self.relu = nn.ReLU()
    self.fc1 = nn.Linear(768, 512, device=self.DEVICE)
    self.fc2 = nn.Linear(512, self.classes.__len__(), device=self.DEVICE)
    self.softmax = nn.Softmax(dim=1)

  def forward(self, input_ids, attention_mask):
    last_hidden_states, cls_hs = self.phobert(input_ids=input_ids, \
                                              attention_mask=attention_mask, \
                                              return_dict=False)

    x = self.fc1(last_hidden_states[:, 0, :])
    x = self.relu(x)
    x = self.dropout(x)

    x = self.fc2(x)
    x = self.softmax(x)

    return x



class CategoryModel():
    def __init__(self, config):
        self.DEVICE = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = json.load(open("./config/classes.json", "r"))
        self.id2label = {v: k for k, v in self.classes.items()}
    
        self.config = config
        self.get_model()

    def get_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
        
        if not os.path.isfile(self.config.model.path):
            gdown.download(self.config.model.url, self.config.model.path, quiet=True)
        self.model = torch.load(self.config.model.path, map_location=self.DEVICE)
        self.model.eval()


    def predict(self, paragraph):
        
        def clean_string(input_string):
            # Sử dụng biểu thức chính quy để tìm và loại bỏ các ký tự không phải là chữ cái, khoảng trắng và số
        
            input_string = input_string.replace("\n", " ")
            split_string = input_string.split()
            input_string = " ".join([text.title() if text.isupper() else text for text in split_string ])
            cleaned_string = re.sub(r'[^\w\s]', '', input_string)
            return cleaned_string
        
        def input_tokenizer(text):
            text = clean_string(text)
            segment_text = word_tokenize(text, format="text")
            tokenized_text = self.tokenizer(segment_text, \
                                        padding="max_length", \
                                        truncation=True, \
                                        max_length=256, \
                                        return_tensors="pt")
            tokenized_text = {k: v.to(self.DEVICE) for k, v in tokenized_text.items()}
            return tokenized_text
        
        def get_top_acc(predictions, thre):
            results = {}
            indexes = np.where(predictions[0] > thre)[0]
            for index in indexes:
                results[self.id2label[index]] = float(predictions[0][index])
            results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)}
            
            return results
        
        tokenized_text = input_tokenizer(paragraph)
        input_ids = tokenized_text["input_ids"]
        token_type_ids = tokenized_text["token_type_ids"]
        attention_mask = tokenized_text["attention_mask"]
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            
        results = get_top_acc(logits.cpu().numpy(), self.config.model.theshold)
        results_arr = []
        for rs in results:
            results_arr.append({
                "category": rs,
                "score": results[rs]
            })
        return results_arr


# if __name__ == '__main__':
#     src_config = OmegaConf.load('config/config.yaml')
#     CategoryModel = CategoryModel(config=src_config)

#     result = CategoryModel.predict('''''')
#     print(result)



