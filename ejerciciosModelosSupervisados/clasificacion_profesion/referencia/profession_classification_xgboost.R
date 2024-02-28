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


# pre-process all data

df_train <-
  read_csv('modulo_2/ejerciciosKPIsXGBoost/clasificacion_profesion/datos/train.csv') %>%
  select(-Var_1, -Segmentation) %>%
  mutate(
    Ever_Married = recode(Ever_Married, 'No' = 0, 'Yes' = 1),
    Profession = recode(
      Profession,
      'Healthcare' = 0,
      'Engineer' = 1,
      'Lawyer' = 2,
      'Entertainment' = 3,
      'Artist' = 4,
      'Executive' = 5,
      'Doctor' = 6,
      'Homemaker' = 7,
      'Marketing' = 8
    )
  ) %>%
  drop_na(Profession)

df_test <-
  read_csv('modulo_2/ejerciciosKPIsXGBoost/clasificacion_profesion/datos/test.csv') %>%
  select(-Var_1) %>%
  mutate(
    Ever_Married = recode(Ever_Married, 'No' = 0, 'Yes' = 1),
    Profession = recode(
      Profession,
      'Healthcare' = 0,
      'Engineer' = 1,
      'Lawyer' = 2,
      'Entertainment' = 3,
      'Artist' = 4,
      'Executive' = 5,
      'Doctor' = 6,
      'Homemaker' = 7,
      'Marketing' = 8
    )
  ) %>%
  drop_na(Profession)


xvars <- (df_train %>% names)[df_train %>% names != 'Profession']
yvar <- 'Profession'

# create xgboost matrices

xgbTrain <- makeXGBMatrix(xvars = xvars,
                          yvar = yvar,
                          df = df_train)

xgbTest <- makeXGBMatrix(xvars = xvars,
                         yvar = yvar,
                         df = df_test)

modelo <- fitXGB(xgbTrain,
                 xgbTest,
                 iterations = 5,
                 model_type = 'multiclass')

# visualise confusion matrix

cm <-
  caret::confusionMatrix(as_factor(predict(modelo, xgbTest)), as_factor(df_test$Profession))

# visualoise error at training 

plot <-
  visualise_error(evaluation_log = modelo$evaluation_log,
                  error_metric = 'mlogloss')

shap_viz <- make_shap_viz(df_train, modelo, 'Profession', 'contribucion_variables', target = 'multiclass') 


var_importance_viz  <- make_importance_viz(modelo)




