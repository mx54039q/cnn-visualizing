# Visualizing Convolutional Networks in CAFFE
Caffe implmentation of "Visualizing and understanding convolutional networks". (http://arxiv.org/abs/1311.2901). 
This work builds on https://github.com/piergiaj/caffe-deconvnet, and makes minor modifiction according on paper.

---

### Building

Replace the caffe files with the included in the caffe folder. Most of the files just have the code to setup the layers and are simple modifications of caffe code. The layers added are pooling_switches_layer.cpp modifies max pooling to collect the "switch" variables, the slice_half layer splits the output in half (used to remove the switches from output of max pooling). The inv_pooling layer takes the switches and pooling and reconstructs the input as described the paper. Once the files are added to caffe, build caffe again.

### Torun

There is a python example on using a deconvolutional network with AlexNet. This shows the use of the pooling switches layers and the slicing layer to separate the switches from the pooled data. The invdepoly reconstructs the network using the inv_pooling and deconvolution layers. The python file runs AlexNet with an image, gathers the switches and reconstructs the input.

### Some Notations in Chinese
* 不需要减均值：在源码中，读图像预处理有一步是减均值，使得输入的范围大概在[-105,150]；而在反卷积的data反向process之前，源码又对data进行了标准化，使得data的范围在[0,255]，这前后两步存在矛盾。在cnn中，减图像均值并不是这么重要，索性将预处理中的减均值操作去掉。

* 源码中在将正向网络的pool5赋给反向网络的输入时，将小于150的置为0,这一步不能理解，既不是ReLU,所以索性去掉。

* 源码中反向网络都没有加ReLU，与论文不符，所以按照论文加上了ReLU层

* 源码中没有求特定层的最strong特征，只是将该层所有特征加入到反卷积中，所以用sum函数求特征层每一个特征的和，并求出最强特征的下标，并只对该特征进行反卷积

### References
'Zeiler, M. D. and Fergus, R. (2013) Visualizing and Understanding Convolutional Networks.'
http://arxiv.org/abs/1311.2901

