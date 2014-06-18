has-vocals
==========

Detects whether or not an audio segment has vocals

About
-----
This was my final project for Intro to AI. 
The intent was as much to practice implementing a multi-layer perceptron
by hand as to experiment with actually detecting vocals.
In it's current state, it is not fit for use in production. 
However, I intend to port the project to C/C++ with an established AI library 
someday soon.

Right now the project can only estimate whether or not a window (~50 ms) contains vocals.
For this to be really useful, it needs to be hooked into a Hidden Markov Model since
a segment that should be labeled as 'has-vocals' will contain windows with and without vocals.

I am not providing the training data publicly at the moment, but I may in the future.
