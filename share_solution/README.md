In this directory, I would like to share the 1st place solution at Kaggle BirdCLEF 2021 competition.

To overview our solution, please check here.
'solutionへのリンク'

In short, our solution is composed of the three stage training.

1st stage
Making melspectrogram classifier (0:nocall, 1:somebird singing) from freefield1010 data. (hereinafter referred to as "nocall detector")
Then calibrating 2nd stage input data by that. We make 2nd stage input labels weighted with call probablility.
(freefield1010 - https://academictorrents.com/details/d247b92fa7b606e0914367c0839365499dd20121)

2nd stage
Melspectrogram multiclass classifier to identify which birds are singing in a clip(7sec).
training: train_short_audio data
validation: train_soundscapes data

3rd stage
Candidate extraction from 2nd stage output (five birds extracted per clip(7sec))
The train_metadata is added as features and then classification for each of candidates (0:unlikely 1:likely) is performed by lightgbm.
training: train_short_audio data
validation: train_soundscapes data


To reproduce the result, please follow the steps below.

1. BUILD NOCALL DETECTOR
We use the nocall detector for the following two purposes.
A. To calibrate 2nd stage input data. 
B. To attach labels to 3rd stage input data. At this time, threshold was 0.5 (hard labeling).

Check the following code and datasets.
code
../working/build_nocall_detector.ipynb

input
freefield1010 data
https://www.kaggle.com/startjapan/ff1010bird-duration7
timm library
https://www.kaggle.com/startjapan/pytorch-image-models

output
Nocall detector models are outputted.

2. CALIBRATE 2ND STAGE INPUT DATA


3. BUILD MELSPECTROGRAM MULTICLASS CLASSIFIER & USE IT FOR TRAIN_SHORT_AUDIO & TRAIN_SOUNDSCAPES
Check the following code.
'コードのパス'

input
melspectrogram images from train_short_audio
calibrated labels for the images

output
melspectrogram multiclassifier models (Ⅰ)


4. EXTRACT CANDIDATES & ADD FEATURES FROM TRAIN_METADATA & TRAIN LIGHTGBM & FIND A BEST THRESHOLD & MAKE SUBMISSION
Check the following code.
'../working/third_stage.ipynb'

input
birdclef-2021 (original data)
melspectrogram multiclassifier models (Ⅰ)
https://www.kaggle.com/namakemono/birdclef-groupby-author-05221040-728258
https://www.kaggle.com/kami634/clefmodel
train_short_audio birdcall probabilities caliculated by melspectrogram multiclassifier models (Ⅰ)
https://www.kaggle.com/namakemono/metadata-probability-v0525-2100
resnest library
https://www.kaggle.com/ttahara/resnest50-fast-package
sklearn library (To use StratifiedGroupKfold, we have to install scikit-learn 1.0.dev0)
https://www.kaggle.com/namakemono/scikit-learn-10dev0

output
submission.csv


(Appendix) CONVERT TRAIN_SHORT_AUDIO INTO MELSPECTROGRAM IMAGES
Here is a useful code by kneroma@Kaggle (maybe known as kkiller) to perform that.
(https://www.kaggle.com/kneroma/birdclef-mels-computer-public)