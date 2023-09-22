def ensemble(df1, df2):
    df1['answer'] = (df1['answer'] + df2['answer'])/2
    return df1