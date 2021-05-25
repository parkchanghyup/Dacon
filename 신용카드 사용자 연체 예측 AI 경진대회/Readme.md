
# 신용카드 사용자 연체 예측 AI 경진대회
- 2021.04.05 ~ 2021.05.24
- private : 0.67995 (165등, 상위 22%)



## Feature Engineering
---
- 결측치 -> NaN 으로 대체
- child_num(자녀 수) 와 family_size(가족 수) 상관도가 높아서 child_num 제거.
- `DAYS_BIRTH`,`DAYS_EMPLOYED` : 일별 데이터를 년도별 데이터로 변환 (div 365)
- `BEGIN_MONTH` : 음수를 양수로 바꿔줌.
- 정규화 적용 : `income_total`,`DAYS_BIRTH`,`DAYS_EMPLOYED` 
    - 모델을 돌려 본 후 변수 중요도가 높은 것 위주로 적용
- 데이터 형 변환 : `FLAG_MOBIL`,`work_phone`,`phone`,`email`
    - 휴대폰 존재 여부 와 같은 0 / 1 로 이루어진 변수는 category 형태로 변환.

### 적용 하지 않은 feature Engineering
- 중복 제거
    - credit 제외 중복제거
    - credit, begin_month 제외 중복제거

위 두가지를 모두 적용해 보았지만, 제거되는 행 갯수가 너무많아 오히려 과적합이 심하게 발생됨.

- occyp_type 제거
    - occyp_type에만 결측치가 존재 하여서 제거 해 보았지만 성능 하락
- log 변환
    - 음수 데이터를 100 % 양수 데이터로 변환 시키기에 애매한 부분이 존재 해서 적용 x
    - 대회가 끝난 후 든 생각이지만 특정 변수는 log 변환을 적용 시켜도 되지 않았을 까 라고 생각함.
- 이상치 제거
    - 변수 중요도가 높은 것을 기준으로 이상치를 제거 해보았지만, 어떻게 제거를 하든지, 성능이 않좋아 져서 적용하지 않음.

## Cross Validation 전략
---
- Stratified K-Fold : label이 불균형 하였기 때문에 Stratified k-fold 적용
- 평가 방식이 log-loss 이기 때문에 10-fold를 이용하여 과적합을 최대한 방지

## 하이퍼 파라미터 튜닝
---
- Catboost 의 경우 모델 자체가 웬만해서는 최적화 된 상태로 학습이 되기 때문에, 튜닝을 거의 하지 않음
- LGBM, RF, XGB 의 경우 Bayesian optimization 을 적용

## 모델링
---

- Catboost, LGBM, RF, XGB를 앙상블 하여 최종 모델 생성 , 그러나 모델 4개를 조합한 결과 RF,XGB, LGBM 3개 모델만 앙상블 하였을때 성능이 제일 높게 나옴


| model   |10-fold|public LB|
|----------|:-------------:|------:|
| RandomForest |  0.7126|0.703|
| XGBoost |    0.71967 |   0.7175166 |
| LightGBM | 0.737739 |   0.724728783 |
| Catboost | 0.7293812 |   0.724960769 |