import pandas as pd
def pad_nan(X_missing_reg:pd.DataFrame)
    "X_missing_reg: 缺失值DataFrame"
    for i in sortindex:
    # 构建我们的新特征矩阵 （没有被选中去填充的特征  + 原始标签） 和新标签（被选中 去 填充 的特征）
    df = X_missing_reg
    # 新标签  Y
    fillc = df.iloc[:,i]
    # 新特征矩阵  除了 标签 列    +  y_full (原来的 target)
    df= pd.concat([df.iloc[:,df.columns!=i],pd.DataFrame(y_full)],axis=1)
    # 在新特征矩阵中，对含有缺失值的列，进行补0的填充
    df_0 = SimpleImputer(missing_values=np.nan,strategy="constant",fill_value=0).fit_transform(df)
    # 找出我们的训练集和测试集
    Ytrain = fillc[fillc.notnull()]
    Ytest  =  fillc[fillc.isnull()]
    Xtrain = df_0[Ytrain.index,:]
    Xtest = df_0[Ytest.index,:]
    
    # 用随机森林回归来填补缺失值
    rfc = RandomForestRegressor(n_estimators=100)
    rfc.fit(Xtrain,Ytrain)
    Ypredict = rfc.predict(Xtest)
    
    # 将填补好的特征返回到我们原始的特征矩阵中
    X_missing_reg.loc[X_missing_reg.iloc[:,i].isnull(),i] = Ypredict
    return X_missing_reg