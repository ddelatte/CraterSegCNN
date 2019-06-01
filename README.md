# CraterSegCNN

CraterSegCNN is a companion to this paper:
- DeLatte, D. M., Crites, S. T., Guttenberg, N., Tasker, E. J., & Yairi, T. (2019). Segmentation Convolutional Neural Networks for Automatic Crater Detection on Mars. 
- Article DOI: 10.1109/JSTARS.2019.2918302

Other work of ours:
- DeLatte, D. M., Crites, S. T., Guttenberg, N., Tasker, E. J., & Yairi, T. (2018). Experiments in Segmenting Mars Craters Using Convolutional Neural Networks (pp. 1–8). Presented at the i-SAIRAS, Madrid.
- DeLatte, D. M., Crites, S. T., Guttenberg, N., Tasker, E. J., & Yairi, T. (2018). Exploration of Machine Learning Methods for Crater Counting on Mars (pp. 1–2). Presented at the 46th Lunar and Planetary Science Conference. 

How to Use This Repo
- Use our implementation of a Python Keras (Tensorflow) U-Net 
- Use our implementation of Crater U-Net (similar to U-Net, epochs run over twice as fast -- good for testing variations)
- Look at example code for various parts of the crater counting pipeline, especially useful for segmentation applications
- Learn how to use Robbins & Hynek 2012 [2] annotations as training examples

[1] U-Net: https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/]

[2] Robbins & Hynek 2012 Mars annotations (>1 km): http://craters.sjrdesign.net/
