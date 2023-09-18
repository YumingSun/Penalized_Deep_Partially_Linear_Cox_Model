# Penalized Deep Partially Linear Cox Models with Application to CT Scans of Lung Cancer Patients

Table of contents
=================

<!--tc-->
   * [Table of contents](#table-of-contents)
   * [Overview](#overview)
   * [Requirements](#requirements)
   * [Data](#data)
   * [Demo](#demo)
<!--tc-->

Overview
========

Lung cancer is a leading cause of cancer mortality globally, highlighting the importance of understanding its mortality risks to design effective patient-centered therapies. The National Lung Screening Trial (NLST) employed computed tomography texture analysis (CTTA), which provides objective measurements of texture patterns on CT scans, to quantify the mortality risks of lung cancer patients. Partially linear Cox models have gained popularity for survival analysis by dissecting the hazard function into parametric and nonparametric components, allowing for the effective incorporation of both well-established risk factors (such as age and clinical variables) and emerging risk factors (e.g., image features) within a unified framework. However,  when the dimension of parametric components exceeds the sample size, the task of model fitting becomes formidable, while nonparametric modeling grapples with the curse of dimensionality. We propose a novel Penalized Deep Partially Linear Cox Model (Penalized DPLC), which incorporates the SCAD penalty to select important texture features and employs a deep neural network to estimate the nonparametric component of the model. We prove the convergence and asymptotic properties of the estimator and compare it to other methods through extensive simulation studies, evaluating its performance in risk prediction and feature selection. The proposed method is applied to the NLST study dataset to uncover the effects of key clinical and imaging risk factors on patients' survival. Our findings provide valuable insights into the relationship between these factors and survival outcomes.

Requirements
============

The project has been tested on Python 3.7.4 with `PyTorch == 1.12.1+cu102`, `Scikit-learn == 1.1.3` , `Pandas == 1.4.4` and `Numpy == 1.24.2`.


Data
====
The datasets, `no_select.h5`, `select.h5`, and `y.h5`, contain the 500 training samples and 1,000 testing samples simulated as described in the _Simulations_ part of the paper.  `no_select.h5` includes 8 features which are the inputs of the DNN. `select.h5` contains 600 features in which 10 features are active. `y.h5` includes the survival time and censoring label for each patient.

Demo
====
* `exp_dplc_nonlinear.py`: The script used to train the Penalized DPLC and assess the model performance. It trains a DNN with 2 hidden layers followed by two dropout layers. This script takes 4 arguments, the number of hidden layers in the two hidden layers, and the dropout rate of the two dropout layers. It outputs the False Positive Number, False Negative Number, and C-index. The detailed results and trained DPLC model is save to the working directory.
```
$ python -u exp_dplc_nonlinear.py 8 4 0.3 0.3
```
