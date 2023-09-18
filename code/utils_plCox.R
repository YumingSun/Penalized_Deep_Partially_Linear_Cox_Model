library(tidyverse)
library(survival)
library(splines2)
library(rhdf5)

train_test_read_real =  function(all_data,clinic_names,image_name, ids){
  p = dim(all_data)[2]
  n = dim(all_data)[1]
  train_id = ids$Train
  test_id = ids$Test
  
  train_data = all_data[train_id,1:p]
  test_data = all_data[test_id,1:p]
  
  image_train = train_data[,c('PatientID',image_names)]
  image_test = test_data[,c('PatientID',image_names)]
  
  clinic_train = train_data[,c('PatientID',clinic_names)]
  clinic_test = test_data[,c('PatientID',clinic_names)]
  
  image_clinic_train = inner_join(image_train,clinic_train,by = "PatientID") %>%
    select(-c("PatientID"))
  image_clinic_test = inner_join(image_test,clinic_test,by = "PatientID") %>%
    select(-c("PatientID"))
  
  outcome_train = Surv(train_data$diagnosis_osdays,train_data$os_status)
  outcome_test = Surv(test_data$diagnosis_osdays,test_data$os_status)
  
  dataOutput = list('x_train' = image_clinic_train, 'y_train' = outcome_train,
                    'x_test' = image_clinic_test, 'y_test' = outcome_test)
  return(dataOutput)
}

train_test_read = function(dataLoc, fileId, train_sample_size){
  x_h5 = H5Fopen(paste0(dataLoc,'x/',
                        sprintf('select_%03d.h5',fileId)))
  z_h5 = H5Fopen(paste0(dataLoc,'x/',
                        sprintf('no_select_%03d.h5',fileId)))
  y_h5 = H5Fopen(paste0(dataLoc,'y/',
                        sprintf('y_%03d.h5',fileId)))
  x = as_tibble(x_h5$arr1)
  z = as_tibble(z_h5$arr1)
  y = as_tibble(y_h5$arr1)
  
  # x = read_csv(paste0(dataLoc,'x/',
  #                     sprintf('select_%03d.csv',fileId)),col_names = FALSE)
  # 
  # z = read_csv(paste0(dataLoc,'x/',
  #                     sprintf('no_select_%03d.csv',fileId)),col_names = FALSE)
  # 
  # y = read_csv(paste0(dataLoc,'y/',
  #                     sprintf('y_%03d.csv',fileId)),col_names = FALSE)

  # beta = read_csv(paste0(dataLoc,'coef/',
  #                        sprintf('coef_%02d.csv',fileId)),col_names = FALSE)
  
  censor_rate = as.numeric(prop.table(table(y[,1]))[1])
  dead_id = (y[,1] == 1)
  alive_id = (y[,1] == 0)
  test_alive_size = as.integer(1000*censor_rate)
  test_dead_size = as.integer(1000*(1 - censor_rate))
  
  train_alive_size = as.integer(train_sample_size*censor_rate)
  train_dead_size = as.integer(train_sample_size*(1-censor_rate))
  
  x_dead = x[dead_id,]; z_dead = z[dead_id,]; y_dead = y[dead_id,]
  x_alive = x[alive_id,]; z_alive = z[alive_id,]; y_alive = y[alive_id,]
  
  x_dead_train = x_dead[1:train_dead_size,]
  z_dead_train = z_dead[1:train_dead_size,]
  y_dead_train = y_dead[1:train_dead_size,]
  
  x_dead_test = x_dead[(nrow(x_dead) - test_dead_size + 1):nrow(x_dead),]
  z_dead_test = z_dead[(nrow(z_dead) - test_dead_size + 1):nrow(z_dead),]
  y_dead_test = y_dead[(nrow(y_dead) - test_dead_size + 1):nrow(y_dead),]
  
  
  x_alive_train = x_alive[1:train_alive_size,]
  z_alive_train = z_alive[1:train_alive_size,]
  y_alive_train = y_alive[1:train_alive_size,]
  
  x_alive_test = x_alive[(nrow(x_alive) - test_alive_size + 1):nrow(x_alive),]
  z_alive_test = z_alive[(nrow(z_alive) - test_alive_size + 1):nrow(z_alive),]
  y_alive_test = y_alive[(nrow(y_alive) - test_alive_size + 1):nrow(y_alive),]
  
  
  x_train = rbind(x_dead_train,x_alive_train); z_train = rbind(z_dead_train,z_alive_train)
  y_train = rbind(y_dead_train,y_alive_train)
  
  x_test = rbind(x_dead_test,x_alive_test); z_test = rbind(z_dead_test,z_alive_test)
  y_test = rbind(y_dead_test,y_alive_test)
  
  data_x_z_train = cbind(x_train,z_train)
  data_x_z_test = cbind(x_test,z_test) 
  
  y_train_surv = Surv(pull(y_train,2),pull(y_train,1))
  y_test_surv = Surv(pull(y_test,2),pull(y_test,1))
  
  dataOutput = list('x_train' = data_x_z_train, 'y_train' = y_train_surv,
                    'x_test' = data_x_z_test, 'y_test' = y_test_surv)
  return(dataOutput)
  
}

train_test_read_debug = function(dataLoc, fileId, train_sample_size, censor_rate){
  x_h5 = H5Fopen(paste0(dataLoc,
                        sprintf('select_%03d.csv',fileId)))
  z_h5 = H5Fopen(paste0(dataLoc,
                        sprintf('no_select_%03d.csv',fileId)))
  y_h5 = H5Fopen(paste0(dataLoc,
                        sprintf('y_%03d.csv',fileId)))
  x = as_tibble(x_h5$arr1)
  z = as_tibble(z_h5$arr1)
  y = as_tibble(y_h5$arr1)
  # x = read_csv(paste0(dataLoc,'x/',
  #                     sprintf('select_%03d.csv',fileId)),col_names = FALSE)
  # 
  # z = read_csv(paste0(dataLoc,'x/',
  #                     sprintf('no_select_%03d.csv',fileId)),col_names = FALSE)
  # 
  # y = read_csv(paste0(dataLoc,'y/',
  #                     sprintf('y_%03d.csv',fileId)),col_names = FALSE)
  # 
  # beta = read_csv(paste0(dataLoc,'coef/',
  #                        sprintf('coef_%02d.csv',fileId)),col_names = FALSE)
  
  dead_id = (y[,1] == 1)
  alive_id = (y[,1] == 0)
  test_alive_size = 300
  test_dead_size = 700
  
  train_alive_size = as.integer(train_sample_size*censor_rate)
  train_dead_size = as.integer(train_sample_size*(1-censor_rate))
  
  x_dead = x[dead_id,]; z_dead = z[dead_id,]; y_dead = y[dead_id,]
  x_alive = x[alive_id,]; z_alive = z[alive_id,]; y_alive = y[alive_id,]
  
  x_dead_train = x_dead[1:train_dead_size,]
  z_dead_train = z_dead[1:train_dead_size,]
  y_dead_train = y_dead[1:train_dead_size,]
  
  x_dead_test = x_dead[(nrow(x_dead) - test_dead_size + 1):nrow(x_dead),]
  z_dead_test = z_dead[(nrow(z_dead) - test_dead_size + 1):nrow(z_dead),]
  y_dead_test = y_dead[(nrow(y_dead) - test_dead_size + 1):nrow(y_dead),]
  
  
  x_alive_train = x_alive[1:train_alive_size,]
  z_alive_train = z_alive[1:train_alive_size,]
  y_alive_train = y_alive[1:train_alive_size,]
  
  x_alive_test = x_alive[(nrow(x_alive) - test_alive_size + 1):nrow(x_alive),]
  z_alive_test = z_alive[(nrow(z_alive) - test_alive_size + 1):nrow(z_alive),]
  y_alive_test = y_alive[(nrow(y_alive) - test_alive_size + 1):nrow(y_alive),]
  
  
  x_train = rbind(x_dead_train,x_alive_train); z_train = rbind(z_dead_train,z_alive_train)
  y_train = rbind(y_dead_train,y_alive_train)
  
  x_test = rbind(x_dead_test,x_alive_test); z_test = rbind(z_dead_test,z_alive_test)
  y_test = rbind(y_dead_test,y_alive_test)
  
  data_x_z_train = cbind(x_train,z_train)
  data_x_z_test = cbind(x_test,z_test) 
  
  y_train_surv = Surv(y_train[,2],y_train[,1])
  y_test_surv = Surv(y_test[,2],y_test[,1])
  
  dataOutput = list('x_train' = data_x_z_train, 'y_train' = y_train_surv,
                    'x_test' = data_x_z_test, 'y_test' = y_test_surv)
  return(dataOutput)
  
}

scale_test = function(x_train,x_test,scale_id){
  for (i in 1:length(scale_id)) {
    x_train_min = min(x_train[,scale_id[i]]) + 1e-4
    x_train_max = max(x_train[,scale_id[i]]) - 1e-4
    x_test_min = min(x_test[,scale_id[i]])
    x_test_max = max(x_test[,scale_id[i]])
    if (x_test_min == x_test_max){
      x_test[,scale_id[i]] = median(pull(x_train,scale_id[i]))
    } else{
      x_test[,scale_id[i]] = (x_test[,scale_id[i]] - x_test_min)/(x_test_max - x_test_min) * (x_train_max - x_train_min) + x_train_min
    }
  }
  return(x_test)
}

spline_transform = function(x,x_test,qt = c(0.1),degree = 1){
  p = dim(x)[2]
  x_spline = c()
  x_test_spline = c()
  for (i in 1:p) {
    knots = quantile(x[,i],qt)
    for (j in 1:length(knots)){
      if (knots[j] == min(x[,i])){
        knots[j] = knots[j] + abs(knots[j]) * 0.001 + 1e-6
      } else if (knots[j] == max(x[,i])){
        knots[j] = knots[j] - abs(knots[j]) * 0.001 - 1e-6
      }
    }
    bsMat = bSpline(x[,i], knots = knots, degree = degree, intercept = TRUE)
    bsMatTest = predict(bsMat, x_test[,i])
    x_spline = cbind(x_spline,bsMat)
    x_test_spline = cbind(x_test_spline,bsMatTest)
  }
  return(list(x = x_spline,x_test = x_test_spline))
}

spline_nonparam = function(x,x_test,scale_id, qt = c(0.1),degree = 1){
  x_param = x[,-scale_id]
  x_nonparam = x[,scale_id]
  x_test_param = x_test[,-scale_id]
  x_test_nonparam = x_test[,scale_id]
  
  splineData = spline_transform(x_nonparam,x_test_nonparam,
                                qt = qt,degree = degree)
  x_nonparam_spline = splineData$x
  x_test_nonparam_spline = splineData$x_test
  
  x_spline = cbind(x_param,x_nonparam_spline)
  x_test_spline = cbind(x_test_param,x_test_nonparam_spline)
  
  return(list(x = x_spline,x_test = x_test_spline))
}
