def ensemble(df1, df2):
    """
    두 개의 submission을 앙상블하는 함수
    """
    df1['answer'] = (df1['answer'] + df2['answer'])/2
    return df1
