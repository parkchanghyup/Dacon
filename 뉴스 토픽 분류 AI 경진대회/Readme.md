## 뉴스 토픽 분류 AI 경진대회 
25/256
### 하이퍼 파라미터


- max_len = 32 
- batch_size = 64 # 모델 크게이 따라 유동적 
- warmup_ratio = 0.1
- num_epochs = 5
- max_grad_norm = 1
- log_interval = 200
- learning_rate = 5e-5
- soft voting 

### 모델 성능
---
|모델|k-fold|전처리|public score|private score|
|---|-----|---|-----|----|
|kobert|3-fold|x|0.8637|0.830
|kobert|5-fold|O|0.8560|
|kobert|X|O|0.8562|
|kobert|x|x|0.8569|
|ko-electra|x|x|0.8470|
|ko-electra|x|o|0.8475|
|ko_electra|O|O|0.8562|
|ko_electra|3-fold|o|0.88536|
|ko_electra|5-fold|o|0.8547|
|xlm-roberta-large|x|x|0.8468784227820373|
|xlm-roberta-large|x|x|0.8468784227820373|

