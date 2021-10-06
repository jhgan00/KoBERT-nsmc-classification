# NSMC classification using KoBERT

- 2021 가을학기 자연어처리와 딥러닝 과제
- KoBERT 를 이용한 네이버 영화리뷰 코퍼스 감정 분류
- [`yonlu`](https://github.com/MinSong2/yonlu), [`KoBERT-nsmc`](https://github.com/monologg/KoBERT-nsmc) 저장소의 코드 참고(조기종료, 텐서보드, 실험 기록 등 편의성 추가)

## Requirements

```
torch==1.8.0 
kobert-transformers==0.5.1
transformers==4.11.2
tensorboard==2.6.0
treform # preprocess.py 를 실행하려면 설치
```

## Usage

모델은 `models/expr_name` 디렉토리에 저장되고, 텐서보드 로그는 `logs/expr_name` 디렉토리에 저장됩니다. 
학습을 마치면 `experiments.json` 파일에 사용한 인자와 검증 데이터셋 성능이 기록됩니다. 
테스트셋에 대해 성능을 측정하려면 학습된 모델의 경로를 명시해주세요.

```bash
$ python preprocess.py  # nsmc 데이터셋을 전처리하고 학습, 검증 데이터셋 분리 
$ python main.py --mode train  # 기본 파라미터로 학습
$ tensorboard --logdir ./logs
$ python main.py --mode test --load_model "models/2021-10-05 21:02:20.792095/best_model_states.bin"
``` 

## Results

기본 파라미터로 실행시 검증 데이터셋에 대해 약 0.8896, 평가 데이터셋 에 대해 약 0.885 의 성능을 보입니다. 

```bash
$ cat experiments.json
[
    {
        "mode": "train",
        "expr_name": "2021-10-05 21:02:20.792095",
        "train": "./data/train.txt",
        "valid": "./data/valid.txt",
        "test": "./data/test.txt",
        "max_len": 100,
        "train_batch_size": 64,
        "valid_batch_size": 1024,
        "lr": 2e-05,
        "beta_1": 0.9,
        "beta_2": 0.999,
        "weight_decay": 0.5,
        "num_warmup_steps": 500,
        "num_epochs": 10,
        "eval_every": 500,
        "early_stopping_rounds": 5,
        "bert_model_name": "monologg/kobert",
        "load_model": null,
        "device": "cuda",
        "seed": 42,
        "valid_accuracy": 0.8889629654321811
    }
]
$ python main.py --mode test --load_model "models/2021-10-05 21:02:20.792095/best_model_states.bin"
Loading model: models/2021-10-05 21:02:20.792095/best_model_states.bin
test accuracy: 0.8850
```


## References

- [KoBERT](https://github.com/SKTBrain/KoBERT)
- [KoBERT-Transformers](https://github.com/monologg/KoBERT-Transformers)
- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [NSMC](https://github.com/e9t/nsmc)
- [Treform](https://github.com/MinSong2/treform)
- [Yonlu](https://github.com/MinSong2/yonlu)
