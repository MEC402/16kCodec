# 16kCodec

## Summary
A compression algorithm to reduce the storage size of the 16k static videos. The algorithm is based on detecting moving objects around a static camera.
Assuming the background is static, at the very beginning, the codec sends a background image and in the posterior video frames it only sends the detected moving objects.


## Introduction
High resolution images take a lot of memory space when storing them in a computer. If these images are taken as frames of a very High Resolution video, the size of the video file will be large.
MPEG compression algorithm considerably decreases the video file size but in the other hand, the resolution decreases too. The goal of this project is to considerably reduce the required space of a High Resolution video
while maintaining the quality.

This algorithm, assumes the background is static and that there are going to be objects moving around the camera. This assumption implies the possibility of using the background, which is not going to change along the scene, to 
detect the moving objects. Using the entire sequence, the algorithm is capable of determining when and where there is an object present in the scene. 

At the first frame of the video, the algorithm sends the background of the scene. In each frame, the moving objects are sent to the destination user and put over the background. This approach ensures, 
high compression of the video.


## Requirements
This project uses some required libraries to work.:

* Item Python 3.5 (I installed from [Anaconda](https://www.anaconda.com/download/))
* Item OpenCV 3.3 (Using Python Pip and a [wheel](http://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv))



## Test cases (subject to change)
To test my program I listed the possible cases I can encounter while detecting the moving objects around the camera. I am going to create some videos to test the following cases:

* Item Empty scenario
* Item One object constantly moving
* Item One object moving and stopping
* Item One still object and then moving
* Item One object moving, it stops and then moves again.
* Item One smooth object moving
* Item Multiple objects moving
* Item Multiple objects moving and one stops
* Item Multiple objects moving, one still and then moves
* Item Multiple objects moving, one stops and then moves again.
* Item Two objects moving and then overlap
* Item Two objects overlapped moving and then separate 



## The paper
Together with the development of this project, there is a paper which explains all the steps this project performs. This paper is located in [overleaf] (https://www.overleaf.com/11703920rystkrknnggx) , it has read and write
permissions.

The paper is still in the writing process and thus it is not fisnished.
