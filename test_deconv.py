#coding=utf-8
'''原版可视化上稍微修改了一下'''
import numpy as np
import matplotlib.pyplot as plt
import os
import caffe
from PIL import Image

caffe_root = '/home/edward/caffe-new1/'
#设置cpu模式
caffe.set_mode_cpu()
#前传网络
net = caffe.Net('examples/ECCV_visual/deploy.prototxt',
                'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                caffe.TEST)
                
#反传网络
invnet = caffe.Net('examples/ECCV_visual/invdeploy_conv5relu.prototxt',caffe.TEST)

# 对图片输入进行预处理的设置
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))

transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2,1,0))

#标准化
def norm(x, s=1.0):
    x -= x.min()
    x /= x.max()
    return x*s
    
#将输入图片进行预处理，并传给‘data’这个blob，再进行前传
net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image('examples/ECCV_visual/test/butterfly_4.jpg'))
out = net.forward()

#求得特定层的最大特征的下标
datasum=np.array([i.sum() for i in net.blobs['conv5'].data[0]])
datanum=datasum.argmax()

#参数共享，将前传网路的参数共享给后传网络，如pooling信息，反卷积信息。
for b in range(1,5):
    invnet.params['conv'+str(b)][0].data[...] = net.params['conv'+str(b)][0].data.reshape(invnet.params['conv'+str(b)][0].data.shape)
invnet.params['conv5'][0].data[datanum]=net.params['conv5'][0].data[datanum].reshape(invnet.params['conv5'][0].data[datanum].shape)


#将前传网络的输出赋给后传网络的输入
invnet.blobs['pooled'].data[...] = net.blobs['pool5'].data
invnet.blobs['switches5'].data[...] = net.blobs['switches5'].data
invnet.blobs['switches2'].data[...] = net.blobs['switches2'].data
invnet.blobs['switches1'].data[...] = net.blobs['switches1'].data
invnet.forward()

#显示可视化图片
plt.clf()
feat = norm(invnet.blobs['conv1'].data[0],255.0)
plt.imshow(transformer.deprocess('data', feat))
plt.show()

##其他卷积层的提取
########################## conv4的提取 ##################################
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
########################### conv3的提取 ##################################
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
########################## conv2的提取 ##################################
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
########################## conv1的提取 ##################################
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
############其他relu的提取只要在原版的基础上加上两处relu即可#############
