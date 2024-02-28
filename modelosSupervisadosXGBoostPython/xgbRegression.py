#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xgboost
import random


# In[9]:


# create xgboost matrices

def makeXGBMatrix(data, label):

    return xgboost.DMatrix(data,
                           label)


# In[17]:


def findXGBRegression(trainXGB,
                          testXGB,
                          xtrain,
                          ytrain,
                          xtest,
                          ytest,
                          iters):
    
    # iniciar el valor de la metrica del cual partimos (alto para metricas que queremos reducir
    # y viceversa
    
    best_metric = 100000
    best_params = {}
    
    # muestrear numeros de forma aleatoria (dentro de un rango) para los hyperparámetros 
    # cada iteracion 
    
    for iteration in range(iters):

        params = {
                'tree_method' : 'exact',
                'booster' : 'gbtree', # 'gblinear'
                'eta' : random.uniform(0.01, 0.3),
                'max_depth' : random.randint(5,14),
                'reg_lambda' : random.uniform(0.01, 0.4),
                'reg_alpha' : random.uniform(0.01, 0.4),
                'gamma' : random.randint(0, 20),
                'subsample' : random.uniform(0.5, 1),
                'colsample_bytree' : random.uniform(0.5, 1),
                'objective' : 'reg:squarederror',
                'eval_metric' : 'rmse'
            }
    
    # ajustar un modelo utilizando validacion cruzada para probar los hiperparámetros en
    # todas las regiones de los datos de entrenamiento
    
        xgb_cv = xgboost.cv(
            params = params, 
            dtrain = trainXGB, 
               nfold=10,
               metrics={'rmse'}, 
               seed=2345,
               callbacks=[xgboost.callback.EvaluationMonitor(show_stdv=True),
                          xgboost.callback.EarlyStopping(2)]
        )
 
    # registrar la metrica
    
        rmse = xgb_cv.iloc[-1,2]
    
    # si la metrica alcanzada fuera mejor que la actual, reemplazar la actual  y guardar los
    # hiperparámetros del modelo que llegó a ella.
    # Nota : la metrica debe ser menor si se esta reduciendo (ej rmse) o mayor si se esta aumentando (ej auc)
    
        if rmse < best_metric:

            best_metric = rmse
            best_params = params
    
    # Ajustar un modelo final con los mejores hiperparámetros, probándolo ahora en el test set
    
    final_model = xgboost.XGBRegressor( 
        eval_metric='rmse',
        early_stopping_rounds=2,
        n_estimators=1000000
    )

    final_model.set_params(**best_params)

    final_model.fit(
        X=xtrain,
        y=ytrain,
        eval_set = [(xtrain, ytrain),(xtest, ytest)]
    )
    
    return final_model

