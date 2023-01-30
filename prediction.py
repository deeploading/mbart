import argparse 

p = argparse.ArgumentParser() 
p.add_argument('--data_path', required=True)  
p.add_argument('--slang_model', type=str, default="DeepLoading/slang-translation") 
config=p.parse_args()


import torch
import pandas as pd 
import nltk
import nltk.data
import json, os, tqdm

nltk.download('punkt')
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# 단어 불러오기 
word_path = config.data_path + "/라벨링데이터/단어"
file_name, file_path, slang_form, std_form = [], [], [], []
for (dirpath, dirnames, filenames) in os.walk(word_path):
    for filename in filenames:
        with open(dirpath + "/" + filename,'rt', encoding='utf-8') as f:
            title = dirpath + "/" + filename
            if title.endswith(".json"):
                data = json.load(f)
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
                                file_path.extend([dirpath + data['MediaUrl'][:-4] + ".json"] * length)
                                continue 
                        slang_form.append(slang_ws)
                        std_form.append(std_ws)
                        file_name.append(filename[:-5])
                        file_path.append(dirpath + data['MediaUrl'][:-4] + ".json")

df_word = pd.DataFrame()
df_word['file_name'] = file_name 
df_word['file_path'] = file_path  
df_word['slang'] = slang_form
df_word['standard'] = std_form 
df_word.to_csv("./data/val_words.csv")

# 문장 불러오기 
file_name, file_path, slang_form, std_form = [], [], [], []
sent_path = config.data_path + "/라벨링데이터/문장"
for (dirpath, dirnames, filenames) in os.walk(sent_path):
    for filename in filenames:
        with open(dirpath + "/" + filename,'r', encoding='utf-8') as f:
            data = json.load(f)
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
                            file_path.extend([dirpath + data['MediaUrl'][:-4] + ".json"] * length)
                            continue 
                    slang_form.append(slang_ws)
                    std_form.append(std_ws)
                    file_name.append(filename[:-5])
                    file_path.append(dirpath + data['MediaUrl'][:-4] + ".json")

import pandas as pd
df_sent = pd.DataFrame()
df_sent['file_name'] = file_name 
df_sent['file_path'] = file_path  
df_sent['slang'] = slang_form
df_sent['standard'] = std_form
df_sent.to_csv("./data/val_sent.csv")

# 대화 불러오기 
file_name, file_path, slang_form, std_form = [], [], [], []
dial_path = config.data_path + "/라벨링데이터/대화"
for (dirpath, dirnames, filenames) in os.walk(dial_path):
    for filename in filenames:
        with open(dirpath + "/" + filename,'r', encoding='utf-8') as f:
            title = dirpath + "/" + filename      
            if title.endswith(".json"):
                data = json.load(f)           
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
                            file_path.extend([dirpath + data['MediaUrl'][:-4] + ".json"] * length)
                            continue 
                    slang_form.append(slang_dial) 
                    std_form.append(std_dial)
                    file_name.append(filename[:-5])
                    file_path.append(dirpath + data['MediaUrl'][:-4] + ".json")

df_conv = pd.DataFrame()
df_conv['file_name'] = file_name 
df_conv['file_path'] = file_path  
df_conv['slang'] = slang_form
df_conv['standard'] = std_form 
df_conv.to_csv("./data/val_conv.csv")


df1 = pd.read_csv("./data/val_words.csv")
df2 = pd.read_csv("./data/val_sent.csv")
df3 = pd.read_csv("./data/val_conv.csv")

df = pd.concat([df1, df2, df3], ignore_index=True)
df = df.drop(['Unnamed: 0'], axis = 1)

for i in range(len(df)):
    if type(df['standard'][i]) is float or type(df['slang'][i]) is float:
        df.drop(i, inplace=True)
df.reset_index(inplace=True, drop=True)
df.to_csv("./data/val_data.csv")

df = pd.read_csv("./data/val_data.csv")


# 모델 불러오기 
from transformers import BertTokenizerFast
from transformers import MBartForConditionalGeneration

tokenizer = BertTokenizerFast.from_pretrained(config.slang_model, strip_accents=False, 
                                              lowercase=False) 
model = MBartForConditionalGeneration.from_pretrained(config.slang_model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

model.to(device) 
model.eval()  


def get_prediction(text):
    embeddings = tokenizer(text, max_length=256, truncation="longest_first", return_attention_mask=False,
                           return_token_type_ids=False, return_tensors='pt') 
    embeddings.to(device) 
    output = model.generate(**embeddings, max_length=256, bos_token_id=tokenizer.cls_token_id,  eos_token_id=tokenizer.sep_token_id)[0, 0:-1].cpu()  # 모델로 예측 
    result = tokenizer.decode(output[1:])
    if '[CLS]' in result:
        result = result[6:]
    return result  


from tqdm import tqdm 

val_title, val_path, val_slang, val_standard, val_translate, val_BLEU = [], [], [], [], [], []
for i in tqdm(range(len(df))):
    slang_data = df['slang'][i]
    std_data = df['standard'][i]
    
    val_title.append(df['file_name'][i])
    val_path.append(df['file_path'][i])
    val_slang.append(slang_data)
    val_standard.append(std_data)
    
    result = get_prediction(slang_data)
    val_translate.append(result)
    BLEU = nltk.translate.bleu_score.sentence_bleu([std_data], result)
    val_BLEU.append(BLEU)

df = pd.DataFrame()
df['title'] = val_title
df['path'] = val_path  
df['slang'] = val_slang
df['standard'] = val_standard
df['translated'] = val_translate 
df['BLEU'] = val_BLEU 

df.to_csv("./result.csv")