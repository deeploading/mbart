import torch
import pandas as pd
from datasets import load_dataset
import evaluate
import regex as re
import transformers
import argparse 
import os
import shutil
import json 

metric = evaluate.load("sacrebleu")

p = argparse.ArgumentParser() 
p.add_argument('--data_path', required=True)
p.add_argument('--model_path', required=True) 
p.add_argument('--model_final', required=True) 
p.add_argument('--num_train_epochs', type=int, default=3) 
p.add_argument('--batch_size', type=int, default=4) 
p.add_argument('--weight_decay', type=float, default=0.05) 
p.add_argument('--learning_rate', type=float, default=5e-5) 
config=p.parse_args() 

os.mkdir('./data')

# 단어 불러오기 
word_path = config.data_path + "/단어"
slang_form, std_form = [], []
for (dirpath, dirnames, filenames) in os.walk(word_path):
    for filename in filenames:
        with open(dirpath + "/" + filename,'rt', encoding='utf-8') as f:
            title = dirpath + "/" + filename
            if title.endswith(".json"):
                data = json.load(f)
                dialogs = data['Dialogs']
                for i in range(len(dialogs)):
                    if dialogs[i]['WordInfo']:   
                        slang_form.append(dialogs[i]['SpeakerText'])
                        std_form.append(dialogs[i]['TextConvert'])

df_word = pd.DataFrame()
df_word['slang'] = slang_form
df_word['standard'] = std_form 
df_word.to_csv("./data/all_words.csv")

# 문장 불러오기 
slang_form, std_form = [], []
sent_path = config.data_path + "/문장"
for (dirpath, dirnames, filenames) in os.walk(sent_path):
    for filename in filenames:
        with open(dirpath + "/" + filename,'r', encoding='utf-8') as f:
            data = json.load(f)
            dialogs = data['Dialogs']
            for i in range(len(dialogs)):
                slang_form += [data['Dialogs'][0]['SpeakerText']]
                std_form += [data['Dialogs'][0]['TextConvert']]

import pandas as pd
df_sent = pd.DataFrame()
df_sent['slang'] = slang_form
df_sent['standard'] = std_form
df_sent.to_csv("./data/all_sent.csv")

# 대화 불러오기 
slang_form, std_form = [], []
dial_path = config.data_path + "/대화"
for (dirpath, dirnames, filenames) in os.walk(dial_path):
    for filename in filenames:
        with open(dirpath + "/" + filename,'r', encoding='utf-8') as f:
            data = json.load(f)
            dialogs = data['Dialogs']
            for i in range(len(dialogs)):
                slang_form += [dialogs[i]['SpeakerText']]
                std_form += [dialogs[i]['TextConvert']]

df_conv = pd.DataFrame()
df_conv['slang'] = slang_form
df_conv['standard'] = std_form 
df_conv.to_csv("./data/all_conv.csv")


data1 = pd.read_csv("./data/all_words.csv")
data2 = pd.read_csv("./data/all_sent.csv")
data3 = pd.read_csv("./data/all_conv.csv")

data = data1.append(data2, ignore_index = True)
data = data.append(data3, ignore_index = True)

import nltk
import nltk.data
nltk.download('punkt')

sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
seperated = sent_tokenizer.tokenize(data['standard'][0])

for i in range(len(data)):
    if type(data['standard'][i]) is float or type(data['slang'][i]) is float:
        data = data.drop(i)
data = data.reset_index()

import nltk.data
slang_form, std_form = [], []
for i in range(len(data)):
    if len(data['standard'][i]) > 120: 
        sep_slang = sent_tokenizer.tokenize(data['slang'][i])
        sep_std = sent_tokenizer.tokenize(data['standard'][i])
        if len(sep_slang) == len(sep_std) and len(sep_slang[0]) > 15:
            slang_form.extend(sep_slang)
            std_form.extend(sep_std) 
            continue
    slang_form.append(data['slang'][i])
    std_form.append(data['standard'][i])

original_len = len(slang_form)
word_len = len(data1)


delete_list = []
for i in range(word_len, len(slang_form)):
    if len(slang_form[i]) < 3:
        delete_list.append(i)

for i in range(len(delete_list) - 1, -1, -1):
    del slang_form[delete_list[i]]
    del std_form[delete_list[i]]

for i in range(len(slang_form)):
    slang_sent = slang_form[i]
    if re.search('[(]', slang_sent):
        for j in range(2):
            if re.search('[(]', slang_sent):
                start = slang_sent.index("(")
                try: 
                    end = slang_sent.index(")")
                except ValueError:
                    continue
                raw_word = slang_sent[start:end + 2]
                try: 
                    start = slang_sent.index("(")
                except ValueError:
                    continue
                try:
                    end = slang_sent.index(")")
                except ValueError:
                    continue
                kor_word = slang_sent[start:end + 1] + "/"
                slang_sent = slang_sent.replace(kor_word, "")
                end = slang_sent.index(")")
                slang_sent = slang_sent[:start] + raw_word[1:-2] + slang_sent[end + 1:]
        slang_form[i] = slang_sent
 
data_split = pd.DataFrame()
data_split['standard'] = pd.DataFrame(std_form)
data_split['slang'] = pd.DataFrame(slang_form)

data_split

data_split.to_csv("./data/slang_dataframe_all.csv")

slang_data = pd.read_csv("./data/slang_dataframe_all.csv")

length = len(slang_data)
standard = slang_data['standard']
slang = slang_data['slang']

import re 
train_standard = []
train_slang = []

for i in range(length):
    new_slang = re.sub("[^ㄱ-ㅎ ㅏ-ㅣ 가-힣 0-9 .,!?]","", slang[i])
    train_slang.append(new_slang)
    
for i in range(length):
    new_standard = re.sub("[^ㄱ-ㅎ ㅏ-ㅣ 가-힣 0-9 .,!?]","",standard[i])
    train_standard.append(new_standard)

slang_data['standard'] = train_standard
slang_data['slang'] = train_slang
slang_data[-10:]

import random as rand
train_list = []
for i in range(len(data)):
    temptemp = {}
    try:
        temptemp['standard'] = slang_data['standard'][i]
    except KeyError:
        continue
    temptemp['slang'] = slang_data['slang'][i]
    train_list.append(temptemp)

import random as rand
test_list = []
for i in range(int(0.1 * len(data))):
    index = rand.randint(0, len(data))
    temptemp = {} 
    try: 
        temptemp['standard'] = slang_data['standard'][index]
    except KeyError:
        continue 
    temptemp['slang'] = slang_data['slang'][index]
    test_list.append(temptemp)

from datasets import Dataset, DatasetDict
import datasets

df_train = pd.DataFrame({"translation" : train_list})
df_test = pd.DataFrame({"translation" : test_list})

datasets = DatasetDict({
    "train": Dataset.from_pandas(df_train),
    "test" : Dataset.from_pandas(df_test)
    })

datasets

datasets['train'][0]


import re 
tok_slang, tok_std = [], []
for i in range(len(slang_data)):
    slang_sent = slang_data['slang'][i]
    std_sent = slang_data['standard'][i]
    slang_nums = re.findall(r'\d+', slang_sent)
    std_nums = re.findall(r'\d+', std_sent)
    for num in slang_nums:
        if len(num) > 1:
            num_space = " ".join(num)
            slang_sent = slang_sent.replace(num, num_space)
    for num in std_nums:
        if len(num) > 1:
            num_space = " ".join(num)
            std_sent = std_sent.replace(num, num_space)
    tok_slang.append(slang_sent)
    tok_std.append(std_sent)

tok_data = pd.DataFrame()
tok_data['slang'] = tok_slang
tok_data['standard'] = tok_std

only_sentences = tok_data[["slang", "standard"]]

len(only_sentences)

only_sentences.to_csv("./data/slang_sentences.csv")

import os
import transformers
from tokenizers import BertWordPieceTokenizer

tokenizer = BertWordPieceTokenizer(strip_accents=False, lowercase=False, clean_text=True, wordpieces_prefix="##")  # 악센트 제거 x, 대소문자 구분 x

corpus_file   = './data/slang_sentences.csv'
vocab_size    = 32000
limit_alphabet= 60000 
min_frequency = 3 

tokenizer.train(files=corpus_file,  
               vocab_size=vocab_size, 
               min_frequency=min_frequency, 
               limit_alphabet=limit_alphabet,
               show_progress=True)

os.mkdir('./tokenizer')
tokenizer.save_model('./tokenizer')

from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained('./tokenizer', strip_accents=False, lowercase=False) 
                                                       
tokenized_input_for_pytorch = tokenizer(slang_data['standard'][0], truncation=True,
                                        return_tensors="pt", 
                                        max_length=50,  
                                        padding=True)  

max_input_length = 50
max_target_length = 50

def preprocess_function(samples):  
    inputs = [s["slang"] for s in samples["translation"]]  
    targets = [s["standard"] for s in samples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True) 

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"] 
    return model_inputs

preprocess_function(datasets['train'][:1])

tokenized_datasets = datasets.map(preprocess_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["token_type_ids"])


from transformers import MBartForConditionalGeneration 
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25")

batch_size = 4
args = Seq2SeqTrainingArguments(  
    config.model_final,
    evaluation_strategy = 'epoch',  
    overwrite_output_dir = 'True',
    learning_rate=config.learning_rate,  
    num_train_epochs=config.num_train_epochs, 
    per_device_train_batch_size=config.batch_size,
    per_device_eval_batch_size=config.batch_size, 
    weight_decay=config.weight_decay, 
    save_total_limit=1, 
    predict_with_generate=True, 
)

from transformers import DataCollatorForSeq2Seq  
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

import numpy as np

def postprocess_text(preds, labels): 
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(eval_preds): 
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds] 
    result["gen_len"] = np.mean(prediction_lens)  
    result = {k: round(v, 4) for k, v in result.items()}
    return result

trainer = Seq2SeqTrainer(  
    model, 
    args, 
    train_dataset=tokenized_datasets["train"], 
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,  
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.save_model(config.model_final)
