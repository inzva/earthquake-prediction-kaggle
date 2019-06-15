# Kaggle LANL Earthquake Prediction Challenge Project
inzva AI Projects #2 - Earthquake Prediction Kaggle Challenge [1]

## Project Description

In this project, we try and predict the remaining time until the next earthquake occurs in laboratory conditions using a model proposed and used by Andrew Ng in his coursera course deeplearning.ai specialization, sequence models part [2], for detection of trigger words such as "hello google" for Google or "你好百度" for Baidu etc. in smart devices using acoustic voice data as input. This model has a binary output, we will modify it to out a float, representing the time until the next earthquake, using the acoustic data detected by the devices in the laboratory experiment.

## Dataset

The dataset given by LANL consists of 2 columns and approximately 630 million rows.
One column is the acoustic data outputted by the sensors on the laboratory earthquake given as integers and the other one is the tine until the next earthquake, which is preprocessed and not given as output by any device.

<img src="https://www.kaggleusercontent.com/kf/11939338/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..z4ZrFO7g1blxhOOaRm480Q.-tVk-qPtaDB1tGB6822d-aIZxhoqgZqKM9pHeuXmILCA1gc93qSkFczvuap8x8zLgDr0CVTE2UV6p-cB-ymln-9p8giDHf_I-ZGaq_zT0bc2ybvPGuZkkwYpY0cK2S523I54oDIvpjMG_2gYW3-lzEuxmFucb23kc8JI_oKAB74.W_Cz1IoiQPzsaZUkAA4vfw/__results___files/__results___6_1.png" height="400"/>

|	**acoustic_data**	|**time_to_failure** |
|:-----------------:|:--------------:|
|count	6.291455e+08	|6.291455e+08|
|mean	4.519468e+00	|4.477084e-01|
|std	1.073571e+01	|2.612789e+00|
|min	-5.515000e+03	|9.550396e-05|
|max	5.444000e+03	|1.610740e+01|



## Project Dependencies
- Tensorflow 1.12
- NumPy
- Keras

## Models

### Baseline Model
The Trigger Word Detection model described in Ng's Coursera course [2] 
#### How to Run:
Download the dataset and, run **train.py** to train the model.

## References

[1] https://www.kaggle.com/c/LANL-Earthquake-Prediction

[2] Ng, "Trigger Word Detection" *Coursera deeplearning.ai*
https://www.coursera.org/learn/nlp-sequence-models/notebook/cvGhe/trigger-word-detection

