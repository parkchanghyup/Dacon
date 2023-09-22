import pandas as pd
import numpy as np

def get_data(path='./data'):
    """
    df, test 데이터와 building 데이터를 merge하여 return
    각 데이터의 컬럼명 변경 및 building 건물 유형 label encoding 적용
    """

    # 데이터 로딩
    df = pd.read_csv(f'{path}/df.csv')
    test = pd.read_csv(f'{path}/test.csv')
    building = pd.read_csv(f'{path}/building_info.csv')

    columns = {'건물번호': 'building_number', '일시': 'date_time', '기온(C)': 'temperature', '강수량(mm)': 'rainfall',
               '풍속(m/s)': 'windspeed', '습도(%)': 'humidity', '일조(hr)': 'sunshine', '일사(MJ/m2)': 'solar_radiation',
               '전력소비량(kWh)': 'power_consumption'}

    building_columns = {
                        '건물번호': 'building_number',
                        '건물유형': 'building_type',
                        '연면적(m2)': 'total_area',
                        '냉방면적(m2)': 'cooling_area',
                        '태양광용량(kW)': 'solar_power_capacity',
                        'ESS저장용량(kWh)': 'ess_capacity',
                        'PCS용량(kW)': 'pcs_capacity'
                        }

    # 컬럼명 변경 및 불필요한 컬럼 제거
    df = df.rename(columns=columns)
    df.drop(['num_date_time', 'sunshine', 'solar_radiation'], axis=1, inplace=True)
    
    # 컬럼명 변경 및 불필요한 컬럼 제거
    test = test.rename(columns=columns)
    test.drop('num_date_time', axis=1, inplace=True)

    # 컬럼명 변경
    building = building.rename(columns=building_columns)

    # label encoding
    translation_dict = {
                        '건물기타': '1',
                        '공공': '2',
                        '대학교': '3',
                        '데이터센터': '4',
                        '백화점및아울렛': '3',
                        '병원': '4',
                        '상용': '5',
                        '아파트': '6',
                        '연구소': '7',
                        '지식산업센터': '8',
                        '할인마트': '9',
                        '호텔및리조트': '10'
                       }
    building['building_type'] = building['building_type'].replace(translation_dict)

    # data merge
    df = pd.merge(df, building, on='building_number', how='left')
    test = pd.merge(test, building, on='building_number', how='left')
    return df, test


def pre_proceccing(df, phase = 'df'):
    """
    데이터 전처리 적용 함수
    """
    df['date_time'] = pd.to_datetime(df['date_time'], format='%Y%m%d %H')

    # date time feature 생성
    df['hour'] = df['date_time'].dt.hour
    df['day'] = df['date_time'].dt.weekday
    df['month'] = df['date_time'].dt.month
    # df['week'] = df['date_time'].dt.weekofyear

    # 공휴일 변수 추가
    df['holiday'] = df['day'].apply(lambda x: 0 if x < 5 else 1)
    df.loc[('2022-06-06' <= df.date_time) & (df.date_time < '2022-06-07'), 'holiday'] = 1
    df.loc[('2022-08-15' <= df.date_time) & (df.date_time < '2022-08-16'), 'holiday'] = 1

    # hour변수 전처리
    df['sin_time'] = np.sin(2 * np.pi * df.hour / 24)
    df['cos_time'] = np.cos(2 * np.pi * df.hour / 24)

    # CDH 변수 생성
    def CDH(xs):
        ys = []
        for i in range(len(xs)):
            if i < 11:
                ys.append(np.sum(xs[:(i + 1)] - 26))
            else:
                ys.append(np.sum(xs[(i - 11):(i + 1)] - 26))
        return np.array(ys)

    cdhs = np.array([])

    for num in range(1, 101):
        temp = df[df['building_number'] == num]
        cdh = CDH(temp['temperature'].values)
        cdhs = np.concatenate([cdhs, cdh])

    df['CDH'] = cdhs

    df['building_type'] = df['building_type'].apply(lambda x: int(x))
    df = df.fillna(0)
    
    # train과 test에 다르게 적용 
    if phase == 'train':
        df.drop(['date_time', 'hour', 'solar_power_capacity', 'ess_capacity', 'pcs_capacity'], axis=1, inplace=True)
    else :
        df.drop(['date_time', 'hour'], axis=1, inplace=True)

    return df

