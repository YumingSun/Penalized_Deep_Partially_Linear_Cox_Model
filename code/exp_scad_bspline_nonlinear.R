library(splines2)
library(glmnet)
library(tidyverse)
library(survival)
library(ncvreg)
# setwd('D:/study/2021autumn/LungCancer/DFS/')
source('./utils_plCox.R')

scad_spline = function(x_train,y_train,x_test,y_test,basis,select_dim){
  N = dim(x_train)[1]
  p = dim(x_train)[2]
  
  pf = rep(1,p)
  pf[(p-basis+1):p] = 0
  
  LAMBDAS =  exp(seq(log(5), log(0.05), length.out=100))
  scad_cox = ncvsurv(as.matrix(x_train), y_train, penalty = 'SCAD',lambda = LAMBDAS,
                     penalty.factor = pf)
  
  s = apply(scad_cox$beta!=0,2,sum)
  AIC = 2*scad_cox$loss + 2*s
  BIC = 2*scad_cox$loss+ log(N)*s
  EBIC = 2*scad_cox$loss+ (log(N) + log(select_dim))*s
  
  aic_id = which.min(AIC)
  bic_id = which.min(BIC)
  ebic_id = which.min(EBIC)
  
  
  aic_beta = scad_cox$beta[,aic_id]
  bic_beta = scad_cox$beta[,bic_id]
  ebic_beta = scad_cox$beta[,ebic_id]
  
  fpr_aic = sum(aic_beta[11:select_dim] != 0)/(select_dim - 10) * 100
  fpr_bic = sum(bic_beta[11:select_dim] != 0)/(select_dim - 10) * 100
  fpr_ebic = sum(ebic_beta[11:select_dim] != 0)/(select_dim - 10) * 100
  
  fnr_aic = sum(aic_beta[1:10] == 0)/10 * 100
  fnr_bic = sum(bic_beta[1:10] == 0)/10 * 100
  fnr_ebic = sum(ebic_beta[1:10] == 0)/10 * 100
  
  aic_beta_select = aic_beta[1:select_dim]
  bic_beta_select = bic_beta[1:select_dim]
  ebic_beta_select = ebic_beta[1:select_dim]
  
  
  select_feature_aic = sum(aic_beta[1:select_dim] != 0)
  select_feature_bic = sum(bic_beta[1:select_dim] != 0)
  select_feature_ebic = sum(ebic_beta[1:select_dim] != 0)
  
  risk_train_aic = as.matrix(x_train) %*% aic_beta 
  risk_train_bic = as.matrix(x_train) %*% bic_beta 
  risk_train_ebic = as.matrix(x_train) %*% ebic_beta 
  
  risk_test_aic = as.matrix(x_test) %*% aic_beta 
  risk_test_bic = as.matrix(x_test) %*% bic_beta 
  risk_test_ebic = as.matrix(x_test) %*% ebic_beta 
  
  calculate_c_stat_train = data.frame(
    `Train Risk AIC` = risk_train_aic,
    `Train Risk BIC` = risk_train_bic,
    `Train Risk EBIC` = risk_train_ebic,
    `Train Time` = as.matrix(y_train)[,1],
    `Train Event` = as.matrix(y_train)[,2]
  )
  
  calculate_c_stat_test = data.frame(
    `Test Risk AIC` = risk_test_aic,
    `Test Risk BIC` = risk_test_bic,
    `Test Risk EBIC` = risk_test_ebic,
    `Test Time` = as.matrix(y_test)[,1],
    `Test Event` = as.matrix(y_test)[,2]
  )
  
  selection_res = data.frame(
    `FPR AIC` = fpr_aic,
    `FPR BIC` = fpr_bic,
    `FPR EBIC` = fpr_ebic,
    `FNR AIC` = fnr_aic,
    `FNR BIC` = fnr_bic,
    `FNR EBIC` = fnr_ebic,
    `Select Features AIC` = select_feature_aic,
    `Select Features BIC` = select_feature_bic,
    `Select Features EBIC` = select_feature_ebic
  )
  
  select_beta = data.frame(
    `Select Feature AIC` = aic_beta_select,
    `Select Feature BIC` = bic_beta_select,
    `Select Feature EBIC` = ebic_beta_select
  )
  
  output = list('C Stat Train' = calculate_c_stat_train, 
                'C Stat Test' = calculate_c_stat_test,
                'Selection' =  selection_res,
                'Select Feature' = select_beta)
  return(output)
}

#####################################################################
args = commandArgs(trailingOnly=TRUE)

expNum = as.numeric(args[1])
train_sample_size = as.integer(args[2])
signal_strength = as.integer(args[3])
select_dim = as.integer(args[4])
linear = as.logical(as.integer(args[5]))

if (linear) {
  dataLoc = paste0('/home/yumsun/DFS/data/experiment/linear/',
                   sprintf("p%d/",select_dim))
  
  resPath = paste0('/home/yumsun/DFS/results/experiment/linear/scad_spline/',
                   sprintf('N%04dP%04d/',train_sample_size,select_dim),
                   'all_res/')
} else{
  dataLoc = paste0('/home/yumsun/DFS/data/experiment/nonlinear/',
                   sprintf("C%d/",signal_strength),
                   sprintf("p%d/",select_dim))
  
  resPath = paste0('/home/yumsun/DFS/results/experiment/nonlinear/scad_spline/',
                   sprintf("C%d/",signal_strength),
                   sprintf('N%04dP%04d/',train_sample_size,select_dim),
                   'all_res/')
  
}

# dataLoc = 'D:/study/2021autumn/LungCancer/DFS/experiment_debug/data/'
# expNum = 0
# train_sample_size = 400


allData = train_test_read(dataLoc, expNum, train_sample_size)

x_train = allData$x_train
y_train = allData$y_train
x_test = allData$x_test
x_test = scale_test(x_train,x_test,(select_dim + 1):(select_dim + 8))
y_test = allData$y_test

splineData = spline_nonparam(x_train,x_test, (select_dim + 1):(select_dim + 8))
x_train_spline = splineData$x
x_test_spline = splineData$x_test

basisNum = dim(x_train_spline)[2] - select_dim
res = scad_spline(x_train_spline,y_train,x_test_spline,y_test,basisNum,select_dim)
# x_train = x_train_spline
# x_test = x_test_spline
# basis = basisNum
# out$selection

output_train_risk_path = paste0(resPath,sprintf('risk_scores_train_%03d.csv',expNum))
output_test_risk_paht = paste0(resPath,sprintf('risk_scores_test_%03d.csv',expNum))
output_selection_path = paste0(resPath,sprintf('selection_%03d.csv',expNum))
output_select_feature_path = paste0(resPath,sprintf('select_beta_%03d.csv',expNum))

write_csv(res$`C Stat Train`, output_train_risk_path, append = FALSE, col_names = TRUE)
write_csv(res$`C Stat Test`, output_test_risk_paht, append = FALSE, col_names = TRUE)
write_csv(res$Selection, output_selection_path, append = FALSE, col_names = TRUE)
write_csv(res$`Select Feature`, output_select_feature_path, append = FALSE, col_names = TRUE)




