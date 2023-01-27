# slang-translation

## 모델 설명
![image](https://user-images.githubusercontent.com/59904000/212801767-34924e5d-588d-41ac-918f-1b39b8a4d241.png)

slang-translation 모델은 은어∙속어를 표준어로 번역해주는 한국어 언어모델입니다. 은어와 속어가 포함되어 있는 연령대별 특징적 발화를 누구나 이해할 수 있는 표준어로 번역하여 나타냅니다. 

## 모델 아키텍처
![image](https://user-images.githubusercontent.com/59904000/212256214-6749fb35-f0c2-4d83-a00a-49be7faa7ef2.png)

mBART(Bidirectional Auto-Regressive Transformers)는 2019년 페이스북에서 개발하였으며, 양방향 인코더와 자동 회귀 디코더를 가진 기계 번역을 목적으로 만들어진 seq2seq 모델입니다. 텍스트 이해와 생성이 모두 가능하기 때문에 번역과 요약 태스크에 적합하며, 한국어를 포함한 50여개의 언어로 pre-trained 되어 있습니다. pre-training 단계는 텍스트를 임의적인 노이즈 함수로 오염시킨 이후 기존 텍스트를 복원하기 위해 학습하는 두 가지 단계로 이루어져 있습니다. 

## 모델 입출력
● 입력: 텍스트 데이터  
● 출력: 텍스트 데이터  
 
## 모델 태스크
기계 번역   

## 테스트 시스템 사양
```
Windows 10 Pro
Python 3.8.15
Torch 1.13.1
CUDA 11.7
cuDnn 8.5.0
Docker 20.10.21
```

## 학습 데이터셋
연령대별 특징적 발화(은어∙속어 등) 라벨링 데이터   

## 파라미터
### 모델 학습  
● data_path: 데이터가 들어 있는 폴더 경로   
● slang_model: 모델 저장 경로   
● num_train_epochs: epoch 개수     
● batch_size: batch 사이즈    
● weight_decay: 가중치 감쇠, 기존 값 0.05  
● learning_rate: 학습률, 기존 값 5e-5    
### 모델로 예측
● slang_model: 학습한 모델 로컬 경로, 입력하지 않을 시 huggingface에 업로드 된 모델 사용    
● data_path: 데이터가 들어 있는 폴더 경로   


## 실행 방법
### 모델 학습 
```
python train.py \
--data_path=./라벨링데이터 \
--slang_model=./slang_model \
--num_train_epochs=3 \
--batch_size=4 \
--weight_decay=0.05 \
--learning_rate=5e-5 
```   
[./data_path]의 데이터로 모델이 학습을 한 후 [./slang_model]에 모델을 저장합니다.  
--data_path : 데이터 저장 경로   
--slang_model : 모델 저장 경로   
--num_train_epochs : epoch 숫자  
--batch_size : batch 크기   
--weight_decay : 가중치 감쇠   
--learning_rates : 학습률    

### 모델으로 예측 
```
python prediction.py \
--data_path=./라벨링데이터 \
--slang_model=./slang_model 
```  
[./slang_model] 모델을 사용하여 라벨링데이터 폴더의 json 파일을 읽어와 은어와 속어가 포함된 문장을 표준어로 번역한다.   
--data_path : 데이터 저장 경로   
--slang_model : 학습한 모델 경로     

## 평가 기준
기계 번역의 성능을 측정하는 지표인 BLEU(Bilingual Evaluation Understudy) score를 사용하여 성능을 측정하였으며, 모델의 BLEU 스코어 결과는 86.9입니다.   

## License
MIT License

Copyright (c) Deep Loading, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
