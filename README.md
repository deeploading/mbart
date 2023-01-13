# slang-translation

## 모델 설명
slang-translation 모델은 은어∙속어를 표준어로 번역해주는 한국어 언어모델입니다.   

## 모델 아키텍처
![image](https://user-images.githubusercontent.com/59904000/212256214-6749fb35-f0c2-4d83-a00a-49be7faa7ef2.png)

mBART는 양방향 인코더와 자동 회귀 디코더를 가진 기계 번역 모델입니다. 텍스트 이해와 생성이 모두 가능하며, 50여개의 언어로 pre-trained 되어 있습니다.   

## 모델 입출력
● 입력: 텍스트 데이터  
● 출력: 텍스트 데이터  
 
## 모델 태스크
기계 번역   

## 학습 데이터셋
연령대별 특징적 발화(은어∙속어 등) 라벨링 데이터   

## 하이퍼파라미터
● num_train_epochs: epoch 개수  
● batch_size: batch 사이즈  
● weight_decay: 가중치 감쇠, 기존 값 0.05  
● learning_rate: 학습률, 기존 값 5e-5    

## 평가 기준
BLEU score 



