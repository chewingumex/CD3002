# SHAP visualisation

make_shap_viz <-
  function(df, finalXGB, yvar, shap_viz_name, target = 'continuous') {
    dataMatrix <-
      df %>%
      select(-all_of((yvar))) %>%
      as.matrix
    
    shap_viz <-
      xgb.ggplot.shap.summary(model = finalXGB,
                              top_n = 20,
                              data = dataMatrix)
    
    total_length <-
      max(shap_viz[[1]]$feature_value) - min(shap_viz[[1]]$feature_value)
    
    increase <- total_length / 8
    
    if (target == 'continuous') {
      shap_viz <-
        xgb.ggplot.shap.summary(model = finalXGB,
                                top_n = 20,
                                data = dataMatrix) +
        theme_ipsum() +
        scale_colour_gradient(
          low = "#FFC300",
          high = "#2171B5",
          space = "Lab",
          na.value = "grey50",
          guide = "colourbar",
          aesthetics = "colour",
          name = 'xvar'
        ) +
        labs(y = yvar,
             x = 'yvar') +
        ggtitle(shap_viz_name) +
        theme(
          axis.text.x = element_text(size = 10),
          axis.text.y = element_text(size = 10),
          legend.title = element_text(size = 10)
          
        )
    }
    
    else if (target == 'binary') {
      shap_viz <-
        xgb.ggplot.shap.summary(model = finalXGB,
                                top_n = 20,
                                data = dataMatrix) +
        theme_ipsum() +
        scale_color_gradientn(
          colours = c('#FFC300',
                      "#2171B5"),
          name = "xvar",
          breaks = c(
            min(shap_viz[[1]]$feature_value) + increase,
            max(shap_viz[[1]]$feature_value) - increase
          ),
          labels = c("low", "high")
        ) +
        labs(y = yvar,
             x = 'yvar') +
        ggtitle(shap_viz_name) +
        theme(
          axis.text.x = element_text(size = 10),
          axis.text.y = element_text(size = 10),
          legend.title = element_text(size = 10)
          
        )
    }
    
    else if (target == 'multiclass') {
      top_3_classes <-
        df |>
        group_by(Profession) |>
        summarise(n = n()) |>
        arrange(desc(n)) |>
        slice_head(n = 3) |>
        select(all_of(yvar))
      
      shap_viz_list <- list()
      counter = 1
      for (i_class in top_3_classes[[yvar]]) {
        shap_viz_list[[counter]] <-
          xgb.ggplot.shap.summary(
            model = finalXGB,
            top_n = 20,
            data = dataMatrix,
            target_class = as.numeric(i_class)
          ) +
          theme_ipsum() +
          scale_colour_gradient(
            low = "#FFC300",
            high = "#2171B5",
            space = "Lab",
            na.value = "grey50",
            guide = "colourbar",
            aesthetics = "colour",
            name = 'xvar'
          ) +
          labs(y = yvar,
               x = 'yvar') +
          ggtitle(shap_viz_name) +
          theme(
            axis.text.x = element_text(size = 10),
            axis.text.y = element_text(size = 10),
            legend.title = element_text(size = 10)
            
          )
        
        counter + counter + 1
        
      }
      
    }
    return(shap_viz)
  }

# visualisation of error

visualise_error <- function(evaluation_log, error_metric) {
  if (error_metric == 'rmse') {
    return(
      evaluation_log %>%
        pivot_longer(
          cols = c('training_rmse', 'testing_rmse'),
          names_to = 'sample',
          values_to = 'value'
        ) %>%
        ggplot(aes(
          iter, value, group = sample, colour = sample
        )) +
        geom_line() +
        theme_ipsum()
    )
    
  }
  
  else if (error_metric == 'auc') {
    return(
      evaluation_log %>%
        pivot_longer(
          cols = c('training_auc', 'testing_auc'),
          names_to = 'sample',
          values_to = 'value'
        ) %>%
        ggplot(aes(
          iter, value, group = sample, colour = sample
        )) +
        geom_line() +
        theme_ipsum()
    )
    
  }
  
  else if (error_metric == 'mlogloss') {
    return(
      evaluation_log %>%
        pivot_longer(
          cols = c('training_mlogloss', 'testing_mlogloss'),
          names_to = 'sample',
          values_to = 'value'
        ) %>%
        ggplot(aes(
          iter, value, group = sample, colour = sample
        )) +
        geom_line() +
        theme_ipsum()
    )
    
  }
  
}

# variable importance visualisation

make_importance_viz <- function(model) {
  imp_plot <-
    xgb.ggplot.importance(
      importance_matrix =
        xgb.importance(feature_names = model[['feature_names']],
                       model = model),
      top_n = 30,
      rel_to_first = T,
      n_clusters = 4
    ) +
    theme_ft_rc() +
    scale_fill_manual(values = colorRampPalette(c('#FFC300', "#2171B5"))(4)) +
    theme(axis.text.y = element_text(size = 9))
  
  return(imp_plot)
  
}