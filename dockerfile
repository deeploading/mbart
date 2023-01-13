# python:3.8의 이미지로 부터
FROM python:3.8-slim-buster

# 제작자 및 author 기입
LABEL maintainer="smoh@deeploading.com" 

# image의 directory로 이동하고
WORKDIR /slang-translation

# COPY packages /mbart
COPY requirements.txt /slang-translation

# 필요한 의존성 file들 설치
RUN pip3 install -r requirements.txt

COPY model_final /slang-translation/model_final
COPY model_path /slang-translationy/model_path
COPY 라벨링데이터 /slang-translation/라벨링데이터
COPY prediction.py /slang-translation
COPY train.py /slang-translation

# container가 구동되면 실행
CMD ["main.py"]
ENTRYPOINT ["/bin/sh", "-c", "/bin/bash"]
