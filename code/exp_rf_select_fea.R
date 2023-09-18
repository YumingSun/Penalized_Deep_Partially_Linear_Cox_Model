library(randomForestSRC)
library(tidyverse)
library(survival)
# setwd('/Users/yumsun/study/DFS/')
source('./utils_plCox.R')

cal_sample_size = function(x){
  n = dim(x)[1]
  return(min(n * .632, max(150, n ^ (3/4))))
}


# select_dim = 600
# select_features = 10
rsf_nonlinear = function(x_train,y_train,x_test,y_test,select_dim,
                         select_features){
  y_train =  as.data.frame(as.matrix(y_train))
  y_test = as.data.frame(as.matrix(y_test))
  
  
  train_data = cbind(x_train,y_train)
  test_data = cbind(x_test,y_test)
  p_all = dim(train_data)[2]
  
  colnames(train_data) = paste("var",seq(1,p_all),sep = '')
  colnames(train_data)[(p_all-1):p_all] = c('time','event')
  colnames(test_data) = paste("var",seq(1,p_all),sep = '')
  colnames(test_data)[(p_all-1):p_all] = c('time','event')
  
  
  bestNodeSize = tune.nodesize(Surv(time, event) ~ ., train_data, doBest = TRUE,trace = TRUE)$nsize.opt
  
  rsf = rfsrc(Surv(time, event) ~ ., train_data,
              nodesize = bestNodeSize,
              ntree = 100,nsplit=1)
  
  # rsf = rfsrc(Surv(time, event) ~ ., train_data,
  #             nodesize = bestNodeSize,
  #             ntree = 500,save.memory = TRUE)
  
  fi = subsample(rsf)$vmp$time
  select_fi = fi[1:select_dim,1]
  sort_id = order(select_fi,decreasing = TRUE)
  est_supp = sort_id[1:select_features]
  non_select_supp = select_dim + seq(8)
  select_supp_all = c(est_supp,non_select_supp,c(p_all-1,p_all))
  # select_supp_all = c(1:10,non_select_supp,c(p_all-1,p_all))
  
  false_pos = length(setdiff(est_supp,c(1:10)))
  false_neg = length(setdiff(c(1:10),est_supp))
  
  train_data_final = train_data[,select_supp_all]
  test_data_final = test_data[,select_supp_all]
  
  tune_final = tune(Surv(time, event) ~ ., train_data_final,trace = TRUE)
  
  rsf_final = rfsrc(Surv(time, event) ~ ., train_data_final,ntree = 100, 
              nodesize = tune_final$optimal[1], mtry = tune_final$optimal[2],
              nsplit = 1, sampsize = cal_sample_size(train_data_final))
  
  risk_train = predict(rsf_final, train_data_final,outcome = 'test')$predicted.oob
  risk_test = predict(rsf_final, test_data_final,outcome = 'test')$predicted.oob
  
  c_r_train = 1 - get.cindex(train_data_final$time,train_data_final$event,risk_train)
  c_r_test = 1 - get.cindex(test_data_final$time,test_data_final$event,risk_test)
  
  y_train_mat = as.matrix(y_train)
  y_test_mat = as.matrix(y_test)
  
  calculate_c_stat_train = data.frame(
    `Train Risk` = risk_train,
    `Train Time` = y_train_mat[,1],
    `Train Event` = y_train_mat[,2]
  )
  
  calculate_c_stat_test = data.frame(
    `Test Risk` = risk_test,
    `Test Time` = y_test_mat[,1],
    `Test Event` = y_test_mat[,2]
  )
  
  selection_res = data.frame(
    `FPR` = false_pos,
    `FNR` = false_neg,
    `Select Features` = length(est_supp)
  )
  
  c_r = data.frame(
    `C Train` = c_r_train,
    `C Test` = c_r_test
  )
  
  output = list('C Stat Train' = calculate_c_stat_train, 
                'C Stat Test' = calculate_c_stat_test,
                'C Stat From R' = c_r,
                'Selection' =  selection_res)
  return(output)
  
}
###########################################
args = commandArgs(trailingOnly=TRUE)

expNum = as.numeric(args[1])
train_sample_size = as.integer(args[2])
signal_strength = as.integer(args[3])
select_dim = as.integer(args[4])
linear = as.logical(as.integer(args[5]))

# expNum = 0
# train_sample_size = 500
# signal_strength = 90
# select_dim = 600
if (linear) {
  dataLoc = paste0('/home/yumsun/DFS/data/experiment/linear/',
                   sprintf("p%d/",select_dim))

  resPath = paste0('/home/yumsun/DFS/results/experiment/linear/rsf/',
                   sprintf('N%04dP%04d/',train_sample_size,select_dim),
                   'all_res/')
  num_of_selected_features_loc = paste0('/home/yumsun/DFS/src/selected_features_scad_spline_linear/',
                                      sprintf('N%04dP%04d/',train_sample_size,select_dim),
                                      'all_res/',
                                      sprintf('select_fea_%03d.csv',expNum))
} else{
  dataLoc = paste0('/home/yumsun/DFS/data/experiment/nonlinear/',
                   sprintf("C%d/",signal_strength),
                   sprintf("p%d/",select_dim))
  resPath = paste0('/home/yumsun/DFS/results/experiment/nonlinear/rsf/',
                 sprintf("C%d/",signal_strength),
                 sprintf('N%04dP%04d/',train_sample_size,select_dim),
                 'all_res/')
  num_of_selected_features_loc = paste0('/home/yumsun/DFS/src/selected_features_scad_spline/',
                                      sprintf('N%04dP%04d/',train_sample_size,select_dim),
                                      'all_res/',
                                      sprintf('select_fea_%03d.csv',expNum))
}


# dataLoc = '/Users/yumsun/study/DFS/experiment_debug/data/N0500P0600/'
# num_of_selected_features_loc = paste0('/Users/yumsun/study/DFS/experiment_debug/data/N0500P0600/num_of_selected_fea_scad_spline/',
#                                       sprintf('select_fea_%03d.csv',expNum))

num_of_selected_features = length(read.csv(num_of_selected_features_loc,header = FALSE)$V1)

allData = train_test_read(dataLoc, expNum, train_sample_size)

x_train = allData$x_train
y_train = allData$y_train
x_test = allData$x_test
y_test = allData$y_test


res = rsf_nonlinear(x_train,y_train,x_test,y_test,
                    select_dim,num_of_selected_features)

output_train_risk_path = paste0(resPath,sprintf('risk_scores_train_%03d.csv',expNum))
output_test_risk_paht = paste0(resPath,sprintf('risk_scores_test_%03d.csv',expNum))
output_selection_path = paste0(resPath,sprintf('selection_%03d.csv',expNum))
output_c_stat_path = paste0(resPath,sprintf('c_stat_%03d.csv',expNum))


write_csv(res$`C Stat Train`, output_train_risk_path, append = FALSE, col_names = TRUE)
write_csv(res$`C Stat Test`, output_test_risk_paht, append = FALSE, col_names = TRUE)
write_csv(res$Selection, output_selection_path, append = FALSE, col_names = TRUE)
write_csv(res$`C Stat From R`, output_c_stat_path, append = FALSE, col_names = TRUE)
