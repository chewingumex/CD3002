# load packages

library(tidyverse)
library(xgboost)
library(readxl)
library(hrbrthemes)
library(fastDummies)
library(rsample)
library(caret)

# load all supervised modelling functions
source('Documents/CD3002/modelosSupervisadosXGBoostR/cross_validation_xgb_binaryclassification.R')
source('Documents/CD3002/modelosSupervisadosXGBoostR/cross_validation_xgb_linear_regression.R')
source('Documents/CD3002/modelosSupervisadosXGBoostR/cross_validation_xgb_multiclassclassification.R')
source('Documents/CD3002/modelosSupervisadosXGBoostR/fitXGBoost.R')
source('Documents/CD3002/modelosSupervisadosXGBoostR/viz.R')


df <-
  read_excel(
    'Documents/CD3002/ejerciciosModelosSupervisados/ausentismo_laboral/datos/Absenteeism_at_work_Project.xls'
    )

# check variables

df |> 
  str()


# fix naming conventions

df <-
  df |> 
  rename_all(
    .funs = list(
      ~ str_replace_all(pattern = ' ',
                    replacement = '_',
                    tolower(.)
      )
    )
  )

#####################################################################
#### DATA VIZ #######################################################
#####################################################################

# absenteeism vs age 

df |> 
  select(age, absenteeism_time_in_hours) |> 
  drop_na() |> 
  group_by(age) |> 
  summarise(
    total_hours = sum(absenteeism_time_in_hours, na.rm = T),
    average_hours = sum(absenteeism_time_in_hours, na.rm = T) / length(age)
  ) |> 
  ungroup() |> 
  ggplot(aes(age, total_hours)) +
  geom_point() +
  theme_ipsum()


# absenteeism vs age (with a lm fitted)

df |> 
  select(age, absenteeism_time_in_hours) |> 
  drop_na() |> 
  group_by(age) |> 
  summarise(
    average_hours = sum(absenteeism_time_in_hours, na.rm = T) / length(age)
  ) |> 
  ungroup() |> 
  ggplot(aes(age, average_hours)) +
  geom_point() +
  geom_smooth(method = 'lm', se= T) +
  theme_ipsum()

# probability distribution of absentism by 5 age buckets and 20 absenteeism levels

df |> 
  select(age, absenteeism_time_in_hours) |> 
  drop_na() |> 
  mutate(age = as.factor(ntile(age, 5)),
         absentism_level = as.factor(ntile(absenteeism_time_in_hours, 20))
         ) |> 
  ggplot(aes(absentism_level, group = age, fill=age)) +
  geom_density(alpha=0.4) +
  theme_ipsum()

# absenteeism by month 

df |> 
  select(month_of_absence, absenteeism_time_in_hours, age) |> 
  drop_na() |> 
  group_by(month_of_absence) |> 
  summarise(
    total_hours = sum(absenteeism_time_in_hours, na.rm = T)
  ) |> 
  ungroup() |> 
  filter(month_of_absence > 0) |> 
  mutate(month_of_absence = as.factor(month_of_absence)) |> 
  ggplot(aes(month_of_absence, total_hours)) +
  geom_bar(stat = 'identity') +
  theme_ipsum()

# absenteeism by workload

df |> 
  ggplot(aes(absenteeism_time_in_hours,`work_load_average/day`), group=age, colour=age) +
  geom_jitter() +
  geom_smooth(method = 'lm', se= T) +
  theme_minimal() + 
  facet_wrap(~age)
  
#####################################################################
#### MODELLING ######################################################
#####################################################################

df <-
  read_excel(
    'Documents/CD3002/ejerciciosModelosSupervisados/ausentismo_laboral/datos/Absenteeism_at_work_Project.xls'
  ) |> 
  rename_all(
    .funs = list(
      ~ str_replace_all(pattern = ' ',
                        replacement = '_',
                        tolower(.)
      )
    )
  )

# drop variables which uniquely identify observations

df <- df |> select(-id)

# tranasform categorical variables to one-hot-encoded 

df <- 
  df |> 
    dummy_cols(remove_selected_columns = T,
               remove_first_dummy = T,
               select_columns = c('reason_for_absence', 
                                  'month_of_absence', 
                                  'day_of_the_week', 
                                  'seasons'),
                ) |> 
  drop_na(absenteeism_time_in_hours)

# dividir train / test 

split <- initial_split(df, prop = 0.80)

train <- training(split)
test <- testing(split)

# seleccionar las variables dependiente e independiente [!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!]

xvars <- (df %>% names)[df %>% names != 'absenteeism_time_in_hours']
yvar <- 'absenteeism_time_in_hours'

# matrcies xgboost

xgbTrain <- makeXGBMatrix(xvars=xvars,
                          yvar=yvar,
                          df=train)

xgbTest <- makeXGBMatrix(xvars=xvars,
                         yvar=yvar,
                         df=test)

# fit model (regression as it's a linear target)

modelo <- 
  fitXGB(xgbTrain, 
         xgbTest, 
         iterations=20,
         model_type = 'regression')

# visualise error during fitting 

visualise_error(evaluation_log = modelo$evaluation_log,
                error_metric = 'rmse')


 make_shap_viz(test, 
                          modelo, 
                          'absenteeism_time_in_hours', 
                          'contribucion_variables', 
                          target='continuous') 


make_importance_viz(modelo)