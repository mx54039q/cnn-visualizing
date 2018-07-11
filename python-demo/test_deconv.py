#coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import os
import caffe
from PIL import Image

# Environment Setting
caffe_root = './' #/home/edward/caffe-new1/'
pretrained_model = 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
forward_deploy = 'examples/ECCV_visual/deploy.prototxt'
backward_deploy = 'examples/ECCV_visual/backward_conv5relu.prototxt'
test_img = 'examples/ECCV_visual/test/butterfly.jpg'

if __name__ == "__main__":
    # Set to CPU mode
    caffe.set_mode_cpu()

    # Forward Network and Pretrained Model
    net = caffe.Net(forward_deploy,
                    pretrained_model,
                    caffe.TEST)
                    
    # Backward Network
    invnet = caffe.Net(backward_deploy,caffe.TEST)

    # Preproccess for input image
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2,1,0))

    # Read image and process, feed to 'data' layer of Net
    net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(test_img))
    # Net forward.
    output = net.forward()
    # Get Index of Max Activation Feature Map
    max_index = np.array([i.sum() for i in net.blobs['conv5'].data[0]]).argmax()

    # Share parameters and info of Conv Layer and Pooling Layer
    for b in range(1,5):
        invnet.params['conv'+str(b)][0].data[...] = \
            net.params['conv'+str(b)][0].data.reshape(invnet.params['conv'+str(b)][0].data.shape)
    invnet.params['conv5'][0].data[max_index] = \
        net.params['conv5'][0].data[max_index].reshape(invnet.params['conv5'][0].data[max_index].shape)

    # Feed output of Forward Net to input of Backward
    invnet.blobs['pooled'].data[...] = net.blobs['pool5'].data
    invnet.blobs['switches5'].data[...] = net.blobs['switches5'].data
    invnet.blobs['switches2'].data[...] = net.blobs['switches2'].data
    invnet.blobs['switches1'].data[...] = net.blobs['switches1'].data
    invnet.forward()

    # Visualize Max Activation Feature Map
    # Normal
    def norm(x, scale):
        x -= x.min()
        x /= x.max()
        return x * scale
    feat = norm(invnet.blobs['conv1'].data[0],255.0)
    plt.imshow(transformer.deprocess('data', feat))
    plt.show()

## Some other layer
########################## Visualize Conv4 Layer ##################################
# invnet = caffe.Net(caffe_root+'examples/ECCV_visual/invdeploy_conv4.prototxt',caffe.TEST)             
# for b in invnet.params:
#     invnet.params[b][0].data[...] = net.params[b][0].data.reshape(invnet.params[b][0].data.shape)

# invnet.blobs['conv5'].data[...] = net.blobs['conv4'].data
# invnet.blobs['switches2'].data[...] = net.blobs['switches2'].data
# invnet.blobs['switches1'].data[...] = net.blobs['switches1'].data
# invnet.forward()
# feat = norm(invnet.blobs['conv1'].data[0],255.0)
# plt.imshow(transformer.deprocess('data', feat))
# plt.savefig('conv4.png',dpi=400)
# plt.show()
########################### Visualize Conv3 Layer ##################################
# invnet = caffe.Net(caffe_root+'examples/ECCV_visual/invdeploy_conv3.prototxt',caffe.TEST)             
# for b in invnet.params:
#     invnet.params[b][0].data[...] = net.params[b][0].data.reshape(invnet.params[b][0].data.shape)

# invnet.blobs['conv4'].data[...] = net.blobs['conv3'].data
# invnet.blobs['switches2'].data[...] = net.blobs['switches2'].data
# invnet.blobs['switches1'].data[...] = net.blobs['switches1'].data
# invnet.forward()
# feat = norm(invnet.blobs['conv1'].data[0],255.0)
# plt.imshow(transformer.deprocess('data', feat))
# plt.savefig('conv3.png',dpi=400)
# plt.show()
########################## Visualize Conv2 Layer ##################################
# invnet = caffe.Net(caffe_root+'examples/ECCV_visual/invdeploy_conv2.prototxt',caffe.TEST)             
# for b in invnet.params:
#     invnet.params[b][0].data[...] = net.params[b][0].data.reshape(invnet.params[b][0].data.shape)

# invnet.blobs['norm2'].data[...] = net.blobs['norm2'].data
# invnet.blobs['switches2'].data[...] = net.blobs['switches2'].data
# invnet.blobs['switches1'].data[...] = net.blobs['switches1'].data
# invnet.forward()
# feat = norm(invnet.blobs['conv1'].data[0],255.0)
# plt.imshow(transformer.deprocess('data', feat))
# plt.savefig('conv2.png',dpi=400)
# plt.show()
########################## Visualize Conv1 Layer ##################################
# invnet = caffe.Net(caffe_root+'examples/ECCV_visual/invdeploy_conv1.prototxt',caffe.TEST)             
# for b in invnet.params:
#     invnet.params[b][0].data[...] = net.params[b][0].data.reshape(invnet.params[b][0].data.shape)
# invnet.blobs['norm1'].data[...] = net.blobs['norm1'].data
# invnet.blobs['switches1'].data[...] = net.blobs['switches1'].data
# invnet.forward()
# feat = norm(invnet.blobs['conv1'].data[0],255.0)
# plt.imshow(transformer.deprocess('data', feat))
# plt.savefig('conv1.png',dpi=400)
# plt.show()
########################## conv5加ReLU的提取 ##########################
# invnet = caffe.Net(caffe_root+'examples/ECCV_visual/invdeploy_conv5relu.prototxt',caffe.TEST)             
# for b in invnet.params:
#     invnet.params[b][0].data[...] = net.params[b][0].data.reshape(invnet.params[b][0].data.shape)
# invnet.blobs['pooled'].data[...] = net.blobs['pool5'].data
# invnet.blobs['switches5'].data[...] = net.blobs['switches5'].data
# invnet.blobs['switches2'].data[...] = net.blobs['switches2'].data
# invnet.blobs['switches1'].data[...] = net.blobs['switches1'].data
# invnet.forward()
# feat = norm(invnet.blobs['conv1'].data[0],255.0)
# plt.imshow(transformer.deprocess('data', feat))
# plt.savefig('conv5relu.png',dpi=400)
# plt.show()
