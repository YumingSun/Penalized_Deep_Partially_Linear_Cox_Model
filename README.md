# Penalized Deep Partially Linear Cox Models with Application to CT Scans of Lung Cancer Patients

Table of contents
=================

<!--tc-->
   * [Table of contents](#table-of-contents)
   * [Overview](#overview)
   * [Requirements](#requirements)
   * [SCAD-penalized Deep Partially Linear Cox Models](#DPLC)
   * [Data](#data)
   * [Demo](#demo)
<!--tc-->

Overview
========

Lung cancer is a leading cause of cancer mortality globally, highlighting the importance of understanding its mortality risks to design effective patient-centered therapies. The National Lung Screening Trial (NLST) employed computed tomography texture analysis (CTTA), which provides objective measurements of texture patterns on CT scans, to quantify the mortality risks of lung cancer patients. Partially linear Cox models have gained popularity for survival analysis by dissecting the hazard function into parametric and nonparametric components, allowing for the effective incorporation of both well-established risk factors (such as age and clinical variables) and emerging risk factors (e.g., image features) within a unified framework. However,  when the dimension of parametric components exceeds the sample size, the task of model fitting becomes formidable, while nonparametric modeling grapples with the curse of dimensionality. We propose a novel Penalized Deep Partially Linear Cox Model (Penalized DPLC), which incorporates the SCAD penalty to select important texture features and employs a deep neural network to estimate the nonparametric component of the model. We prove the convergence and asymptotic properties of the estimator and compare it to other methods through extensive simulation studies, evaluating its performance in risk prediction and feature selection. The proposed method is applied to the NLST study dataset to uncover the effects of key clinical and imaging risk factors on patients' survival. Our findings provide valuable insights into the relationship between these factors and survival outcomes.

Requirements
============

The project has been tested on Python 3.7.4 with `PyTorch == 1.12.1+cu102`, `Scikit-learn == 1.1.3` , `Pandas == 1.4.4` and `Numpy == 1.24.2`.

SCAD-penalized Deep Partially Linear Cox Models: 
===========

A partially linear Cox model assumes a hazard function: 
$$
\lambda(t|\mathbf{x},\mathbf{z}) = \lambda_0(t)\exp(\boldsymbol{\beta}_0^{\top}\mathbf{x} + g_0(\mathbf{z})),
$$


Data
====
The training dataset, `TrainingData.csv`, and the testing dataset, `TestingData.csv`, are simulated as described in the _Simulation Study_ part of the papaer. There are 16,000 observations in the training dataset and 4,000 observations in the testing dataset. Both of the two datasets have one label and three features. 

Demo
====
* `train.py`: The script used to train the INNER model. To train the INNER model run:
```
$ python train.py PATH_TO_TRAINING_DATA PATH_TO_MODEL
```
It trains the INNER model using training data at `PATH_TO_TRAINING_DATA` and saves the trained model at `PATH_TO_MODEL`. Pre-trained model is `pre_trained_model.h5`. The training process takes about one minute.

* `test.py`: The script used to evaluate the performance of the INNER model. To evaluate the performance run:
```
python test.py PATH_TO_TESTING_DATA PATH_TO_MODEL
```
It evaluate the performance of pre-trained model at `PATH_TO_MODEL` using testing data at `PATH_TO_TESTING_DATA`. It prints out the C statistics, accuaracy, sensitivity, specificity and balance accuracy. The testing process takes few seconds.

* `estimate.py`: The script used to estimate the BOT and POT. To estimate these two metrics run:
```
python estimate.py PATH_TO_ESTIMATE_DATA PATH_TO_MODEL PATH_TO_OUTPUT
```
It estimate the BOT and POT for each subject in the data at `PATH_TO_ESTIMATE_DATA` using pre-trained model at `PATH_TO_MODEL`. It prints out the first five results and save all the results at `PATH_TO_OUTPUT`. The estimation process takes few seconds.
