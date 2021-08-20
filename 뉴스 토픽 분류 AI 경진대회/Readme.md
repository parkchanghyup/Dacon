## 뉴스 토픽 분류 AI 경진대회 

🏅 최종 순위 25등 (25/256 (상위 10%))

### 대회 개요 

#### 주제  

한국어 📰뉴스 헤드라인을 이용하여, 뉴스의 주제를 분류하는 알고리즘 개발  

#### 배경 

텍스트 주제를 추론하는 것은 언어 이해 시스템이 보유해야 하는 핵심 기능입니다. YNAT(주제 분류를 위한 연합뉴스 헤드라인) 데이터 세트를 활용해 주제 분류 알고리즘을 개발해 주세요.  
국내 최초 오픈 데이터 세트인 KLUE(Korean Language Understanding Evaluation) 데이터 세트를 이용하여 다양한 언어 모델의 성능을 비교해 한국어 자연어처리 분야의 발전에 기여할 것으로 예상합니다.  

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

### 회고
- 무거운 모델이라고 다 성능이 좋게 나오진 않음.
- 데이터 증강을 어떻게 해야 할지 고민했는데 결국 적용 하지 못함.
  - 다른 사람 코드 봤는데 한글 -> 영어 -> 한글 순으로 번역하여 데이터 증강.. 
- 여러 모델을 앙상블하여 결과 도출 해봤지만, 성능이 좋지 않음
- 데이터가 제한 적이라 낮은 epoch 에서도 수렴. 아마 데이터 증강을 했더라면 epoch 을 더 높게 해서 학습 하고 테스트 데이터 셋에서도 더 좋은 결과물을 얻을 수 있지 않을까 싶음.
