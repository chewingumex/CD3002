# fit model with crossvalidation (choose model type between regression, binary or multiclass)

fitXGB <- function(xgbTrain, xgbTest, iterations, model_type){
  
  if(model_type=='regression'){
    
    return(fitXGBRegression(xgbTrain=xgbTrain, xgbTest=xgbTest, iterations=iterations))
    
  }
  
  else if(model_type=='binary'){
    
    return(fitXGBBinary(xgbTrain=xgbTrain, xgbTest=xgbTest, iterations=iterations))
    
  }
  
  else if(model_type=='multiclass'){
    
    return(fitXGBMulticlass(xgbTrain=xgbTrain, xgbTest=xgbTest, iterations=iterations))
    
  }
  
}