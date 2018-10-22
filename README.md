
# Internship2018
The work I have done while doing my Internship at METU CENG ImageLab

## Abstract
The work I have done was trying different models for cognitive state classification and analyzing the Functional Magnetic Resonance Imaging (fMRI) time series data recorded in three dimensional voxel coordinates by clustering the Blood Oxygen Level Dependent (BOLD) responses. The experimental results show that, on a multiclass fMRI dataset,  the Deep Learning models improves the classification accuracy (an average 12.5% increase when using Convolutional Neural Networks and an average 14.43% increase when using Multilayer Perceptrons) compared to a classical multi voxel pattern analysis (MVPA) method. 

** The baseline accuracy of the MVPA 33.73% on average. It is taken from MSc Thesis of [Gunes Sucu](https://scholar.google.com.tr/citations?user=Z97AJZQAAAAJ&hl=tr)

## The Dataset : Emotional Memory Retrieval Dataset

The main dataset we used in this work is a neuroimage (or image) dataset collected from 12 human subjects. For each subject, there are 210 training and 210 testing images. Each image in the dataset belongs to one of four classes.

The dataset was originally collected for an emotional memory retrieval experiment* where the stimuli consisted of two neutral (kitchen utensils and furniture) and two emotional (fear and disgust) categories of images. Each trial started with a 12-second fixation period followed by a 6-second encoding period. In the encoding period, participants were presented with 5 images from the same category, each image lasting 1200 milliseconds on the screen. Following the fifth image, a 12-second delay period was presented in which participants solved three math problems consisting of addition or subtraction of two randomly selected two-digit numbers. Following the third math problem, a 2-second retrieval period started in which participants were presented with a test image from the same category and indicated whether the image was a member of the current study list or not. The reader is referred to paper* for further details. For neuroimage classification, we employed measurements obtained during the encoding and retrieval phases as our training and testing data, respectively.

*[link to paper](http://psycnet.apa.org/record/2015-45624-001)

## Models
I have used two different Deep Learning models which are Convolutional Neural Networks and Multilayer Perceptrons with using some techniques to augment the data. 
 ***
[Code for CNN model PyTorch implementation](/models/CNN/cnn_pytorch.py)
 ***
[Code for CNN model Keras implementation](/models/CNN/cnn_keras.py)
 ***
[Code for MLP model PyTorch implementation](/models/MLP/mlp_timeseries_pytorch.py)
 ***
[Code for MLP model Keras implementation](/models/MLP/mlp_timeseries.py)
 ***
## Results

```markdown
| Subjects 	| TEST ACCURACY FOR MLP WITH AUGMENTATION 	| TEST ACCURACY FOR MLP WITHOUT AUGMENTATION 	|
| --------	|:---------------------------------------:	|:------------------------------------------:	|
|    s1    	|                   41,1                  	|                    43,1                    	|
|    s2    	|                   46,1                  	|                    50,2                    	|
|    s3    	|                   40,2                  	|                    44,8                    	|
|    s4    	|                   40,9                  	|                    44,8                    	|
|    s5    	|                   40,8                  	|                    41,1                    	|
|    s6    	|                   44,2                  	|                    49,1                    	|
|    s7    	|                    63                   	|                     61                     	|
|    s8    	|                   51,9                  	|                    53,8                    	|
|    s9    	|                   49,8                  	|                    53,9                    	|
|    s10   	|                   50,2                  	|                    47,2                    	|
|    s11   	|                   40,7                  	|                    47,9                    	|
|    s12   	|                   44,9                  	|                    41,2                    	|
|    s13   	|                   47,2                  	|                    48,1                    	
|:--------:	|:---------------------------------------:	|:------------------------------------------:	|
| AVERAGES 	|                  46,23                  	|                    48,16                   	|
```
