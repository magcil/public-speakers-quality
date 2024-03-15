# public-speakers-quality: A Dataset for Assessing Quality of Public Speakers

## 1. Intro
Automatically assessing the quality of public speakers can be used to build 
coaching applications and training tools that focus on improving presentation
 skills. This is a dataset of audio recordings of more than 350 conference
  presentations, annotated with regards to the corresponding speaking quality. 
To assess the speaking quality we focus on aspects of public speech, such as 
confidence, intonation, flow of speech and the use of pauses and filler words. 
This repo also demonstrates the ability of various machine learning methods to 
automatically assess the speaking quality in conferences over the annotated 
dataset. 

## 2. Data
The dataset includes features from 387 audio recordings of 1.5 mins duration 
each, created by randomly selecting 30 second segments from the start, 
middle and end of the whole video presentations, that were collected from links 
publicly available in the conference websites and YouTube. 
The language of all speeches is English. 22.16% of the speakers are female 
and 77.84\% are male. 38.52% of the 
speakers are native and 233 61.48% non-native. 44.5% of the speakers 
gave a talk in industry-driven conferences, thus we categorized them as 
engineers, while the rest as academics. 
Among the academics 24.2% of the speakers were senior academics
(faculty, research scientists), while 27.4% of the speakers were junior 
academics (post-docs and PhD candidates). 

The annotated aspects of speaking quality are: confidence, intonation, 
flow of speech and the use of pauses and filler words. Also, an overall rating 
was annotated to characterize the public speaking quality as a whole. 
As a scale we use the MOS (Mean Opinion Score) 1-5 degradation, 
also used by earlier works, where the value 3 always corresponds to a 
neutral rating. The following table shows all annotation tasks:

| Speech Aspect  | Rating Scale and Description |
| ------------- | ------------- |
| Confidence  | 1 (insecure) - 5 (convincing) |
| Intonation  | 1 (no or extreme variance) - 5 (expressive)  |
| Flow of Speech  | 1 (unpleasant) - 5 (engaging)  |
| Pauses - Fillers  | 1 (too many) - 5 (none) |
| Overall Rating | 1 (worst) - 5 (best) |

For each task (speech aspect), at least two annotators have provided their 
feedback. To generate the final ground truth for each recording, we compute 
an average annotation rating. We also compute the mean absolute deviation (MAE)
and the corresponding annotation confidence. 
All metadata and ground truth values (average and MAE) for the five tasks 
(overall, confidence, fillers, intonation and flow) are stored in file 
`annotations_metadata.json`. 

Audio features for each recording are given as a separate Numpy file. 
Each recording is represented by either (a) a sequence of short-term feature 
vectors or (b) a mel-spectrogram. All feature files are stored in npy files in 
the `features` folder. Short-term features are extracted using the 
`pyAudioAnalysis` library and they are stored in npy files with the 
`_pyaudioanalysis.npy` postfix. Each file stores a matrix of 4500 68-D 
feature vectors. Mel-spectrograms are stored in npy files with the 
`_melgram.npy` postfix (4500 x 128 feature matrices). The 4500 short-term 
frames in both cases yield from the fact that the adopted short-term step 
(hop size) is 20 mseconds (and the duration of the file is 90 seconds). 
Finally, the adopted window size is 40 mseconds.  


## 3. Basline Methods and models
### 3.1 Setup 
Install the required packages
```
pip install -r requirements.txt
``` 

### 3.2 Inference 
Test the models on your own audio file.

In order to try the pretrained best models using segment-level feature statistics 
from the hand-crafted audio features described in the paper run: 

```
python3 inference.py -i <audio_input> -m <pretrained_model_path> 
```  
 Where:
- audio_input is the path of the input wav file to be tested.
- pretrained_model_path is the path to the pretrained model. Here you can use any
of the models stored in output_models folder (confidence.pt, fillers.pt, flow.pt, intonation.pt, overall.pt)

### 3.2 Train models from scratch 

- #### HCAF+LTA

    This subsection refers to training using segment statistics on hand-crafted audio features (HCAF),
Long-Term Averaging (LTA) and shallow ML regressors.

    ```
    python3 train_regressor.py -g annotations_metadata.json -f features/ -t <task_name> 
    ```  
    Replace <task_name> with any of the following: confidence, fillers, flow, intonation, overall

  The above script will use the stored pyaudioanalysis features to train some algorithms
  (Dummy regressor, svr, gradient boosting, bayesian ridge, linear regressor, xg-boost regressor) using Gridsearch
  for hyperparameter tuning and storing results in a csv file named "results_<task_name>.csv".
- #### HCAF+AGG 
  This subsection refers to training using segment statistics on hand-crafted audio features (HCAF),
Recording-level aggregation (AGG) and shallow ML regression.
  - First of all, you need to specify the parameters of the model you will use in the config.py file.
  - Second, run the following:
    ```
    python3 train_regressor.py -g annotations_metadata.json -f features/ -t <task_name> -m <model_name> -c -o <output_name>
    ```   
    In <model_name> write the name of the model-algorithm you want to run for. The name should be compatible with config file.
    The argument -c is optional and should be added if you want to use cross validation and report the mean cross validated error on aggregated segments.
    Otherwise, the whole input dataset will be used to train the model that will be stored in <output_name>.
- #### MEL+AGG
  This subsection refers to training using mel-spectrograms (MEL), Recording-level aggregation (AGG) and CNNs.
    - First of all, you need to specify the number of epochs, the output folder, the batch size, the mid window and mid step in cnn/config_cnn.py
    - Second, run the following:
      ```
      python3 cnn/train_cnn.py -g annotations_metadata.json -f features/ -t <task_name> -o <output_model_name>
      ```  
      The argument -o is optional and should be used if you want to define a specific output model name. Otherwise a name based on the time will be adopted. 
      The above script will split the input data 20 times. Each of the times a cnn model will be trained and using training subset and validated on validation subset. 
      The final reported error will be the mean (across 20 splits) cross validated error on aggregated segments.

## Detailed statistics and graphs 
A detailed report of all the statistics of the dataset and the metadata of the paper accompanied by the corresponding graphs can be found at the following colab link: 

https://colab.research.google.com/drive/1-FL7HTkEsaWU1yoduqUb2q05lpfaV7mW?usp=sharing

To infer the statistical significances of the differences between the average values of each regression task for male and
female speakers based on the p-value you can run the following script: 

```
python3 metadata_stat_tests.py -i annotations_metadata.json -t <task_name>
```
