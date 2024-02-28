# load packages

library(caret)
library(tidyverse)
library(xgboost)
library(rsample)
library(hrbrthemes)

# load all supervised modelling functions
source('modulo_2/modelosSupervisadosXGBOOST/cross_validation_xgb_binaryclassification.R')
source('modulo_2/modelosSupervisadosXGBOOST/cross_validation_xgb_linear_regression.R')
source('modulo_2/modelosSupervisadosXGBOOST/cross_validation_xgb_multiclassclassification.R')
source('modulo_2/modelosSupervisadosXGBOOST/fitXGBoost.R')
source('modulo_2/modelosSupervisadosXGBOOST/viz.R')

# leer datos

df <-
  read_csv(
    'modulo_2/ejerciciosKPIsXGBoost/deteccion_transacciones_fraudulentas/datos/data_fraude.csv'
  ) %>%
  select(-...1)

# pre-process datos (si es necesario)

df <-
  df %>%
  rename("fraude" = "0...21")

# dividir train / test

split <- initial_split(df, prop = 0.80)

df_train <- training(split)
df_test <- testing(split)

# crear matrices de xgboost

xvars <- (df %>% names)[df %>% names != 'fraude']
yvar <- 'fraude'


# make XGB matrices

xgbTrain <- makeXGBMatrix(xvars = xvars,
                          yvar = yvar,
                          df = df_train)

xgbTest <- makeXGBMatrix(xvars = xvars,
                         yvar = yvar,
                         df = df_test)

# fit model

modelo <- 
  fitXGB(
  xgbTrain,
  xgbTest,
  iterations = 10,
  model_type = 'binary')


cm <-
  caret::confusionMatrix(as_factor(predict(modelo, xgbTest)),
                         as_factor(df_test$fraude))

# visualoise error at training

error_plot <-
  visualise_error(evaluation_log = modelo$evaluation_log,
                  error_metric = 'auc')


shap_viz <- make_shap_viz(df, modelo, 'fraude', 'contribucion_variables', target = 'binary') 


var_importance_viz  <- make_importance_viz(modelo)
