## 뉴스 토픽 분류 AI 경진대회 
---
### 하이퍼 파라미터
---

- max_len = 32 
- batch_size = 64 # 모델 크게이 따라 유동적 
- warmup_ratio = 0.1
- num_epochs = 5
- max_grad_norm = 1
- log_interval = 200
- learning_rate = 5e-5

### 모델 성능
---
|모델|k-fold|전처리|public score|private score|
|---|-----|---|-----|----|
|kobert|5-fold|O|0.8560|
|kobert|5-fold|x|0.84600|
|kobert|X|O|0.8491|
|kobert|x|x|0.8538|
|kobert|x|x|0.8567|
|ko-electra|x|x|0.8470|
|ko-electra|x|o|0.8475|
|ko_electra|o|x|0.8578|
|ko_electra|O|O|0.8562|
|xlm-roberta-large|x|x|0.8468784227820373|
|xlm-roberta-large|x|x|0.8468784227820373|
