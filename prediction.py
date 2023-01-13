import argparse 

p = argparse.ArgumentParser() 
p.add_argument('--model_path', required=True)  # 모델 경로 
p.add_argument('--data_path', required=True)  # validation data가 들어 있는 폴더 경로  
config=p.parse_args()


import torch
import pandas as pd 
import nltk
import nltk.data
import json, os, tqdm

nltk.download('punkt')
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

file_name, slang_form, std_form = [], [], []
for (dirpath, dirnames, filenames) in os.walk(config.data_path):
    for filename in filenames:
        with open(dirpath + "/" + filename,'rt', encoding='utf-8') as f:
            title = dirpath + "/" + filename
            if title.endswith(".json"):
                data = json.load(f)
                data_type = str(data['MediaUrl'])[:2]             
                if data_type == "대화":  # 대화일 경우 
                    dialogs = data['Dialogs']
                    for i in range(len(dialogs)):
                        slang_dial = dialogs[i]['SpeakerText'] 
                        std_dial = dialogs[i]['TextConvert']
                        if len(std_dial) > 120:
                            sep_slang = sent_tokenizer.tokenize(slang_dial)
                            sep_std = sent_tokenizer.tokenize(std_dial) 
                            length = len(sep_slang)
                            if len(sep_slang) == len(sep_std) and len(sep_slang[0]) > 15:
                                slang_form.extend(sep_slang)
                                std_form.extend(sep_std)
                                file_name.extend([filename[:-5]] * length)
                                continue 
                        slang_form.append(slang_dial) 
                        std_form.append(std_dial)
                        file_name.append(filename[:-5])
                    else:
                        continue
                else:  # 단어나 문장일 경우 
                    dialogs = data['Dialogs']
                    for i in range(len(dialogs)):
                        if dialogs[i]['WordInfo']:   
                            slang_ws = dialogs[i]['SpeakerText'] 
                            std_ws = dialogs[i]['TextConvert']
                            if len(std_ws) > 120:
                                sep_slang = sent_tokenizer.tokenize(slang_ws)
                                sep_std = sent_tokenizer.tokenize(std_ws) 
                                length = len(sep_slang)
                                if len(sep_slang) == len(sep_std) and len(sep_slang[0]) > 15:
                                    slang_form.extend(sep_slang)
                                    std_form.extend(sep_std)
                                    file_name.extend([filename[:-5]] * length)
                                    continue 
                            slang_form.append(slang_ws)
                            std_form.append(std_ws)
                            file_name.append(filename[:-5])


df = pd.DataFrame()
df['file_name'] = file_name
df['slang'] = slang_form
df['standard'] = std_form 

for i in range(len(df)):
    if type(df['standard'][i]) is float or type(df['slang'][i]) is float:
        df.drop(i, inplace=True)
df = df.reset_index()
df.to_csv("./data/val_data.csv")

data = pd.read_csv("./data/val_data.csv")
data  


# 모델 불러오기 
from transformers import BertTokenizerFast
from transformers import MBartForConditionalGeneration

tokenizer = BertTokenizerFast.from_pretrained(config.model_path, strip_accents=False, 
                                              lowercase=False) 
model = MBartForConditionalGeneration.from_pretrained(config.model_path)  # 기존 베이스 모델
model.load_state_dict(torch.load(config.model_path + "/pytorch_model.bin"))  # 파인튜닝한 weight

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda가 사용 가능하다면 cuda를 사용하고 사용 불가하면 cpu 사용하도록

model.to(device)  # 모델을 디바이스에 불러오기
model.eval()  # elal()은 evaluation 과정에서 사용하지 않아야 하는 layer들을 알아서 off 시키도록 하는 함수


# 텍스트 넣어서 결과 예측하는 함수 
def get_prediction(text):
    embeddings = tokenizer(text, max_length=256, truncation="longest_first", return_attention_mask=False,
                           return_token_type_ids=False, return_tensors='pt')  # 토크나이징
    embeddings.to(device)  # 토큰을 디바이스에 불러오기 
    output = model.generate(**embeddings, max_length=256, bos_token_id=tokenizer.cls_token_id,  eos_token_id=tokenizer.sep_token_id)[0, 0:-1].cpu()  # 모델로 예측 
    result = tokenizer.decode(output[1:])  # 토큰을 글자로 디코딩
    if '[CLS]' in result:
        result = result[6:]
    return result  

# 함수로 예측하기 
from tqdm import tqdm 

val_title, val_slang, val_standard, val_translate, val_BLEU = [], [], [], [], []
for i in tqdm(range(len(data))):
    slang_data = data['slang'][i]
    std_data = data['standard'][i]
    
    val_title.append(data['file_name'][i])
    val_slang.append(slang_data)
    val_standard.append(std_data)
    
    result = get_prediction(slang_data)
    val_translate.append(result)
    BLEU = nltk.translate.bleu_score.sentence_bleu([std_data], result)
    val_BLEU.append(BLEU)

df = pd.DataFrame()
df['title'] = val_title 
df['slang'] = val_slang
df['standard'] = val_standard
df['translated'] = val_translate 
df['BLEU'] = val_BLEU 

df.to_csv("./result.csv")