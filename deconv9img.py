#coding=utf-8
'''求9/6/4张图片的可视化，并补pad将9张图显示在一个图上;若要修改可视化的卷积层，都需要修改反向网络文件/convnum数字/输入反向网络的数据'''
import numpy as np
import matplotlib.pyplot as plt
import os
import caffe
from PIL import Image

caffe_root = '/home/edward/caffe-new1/'
#设置cpu模式
caffe.set_mode_cpu()
#前传网络/后传网络
net = caffe.Net('examples/ECCV_visual/deploy.prototxt',
                'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                caffe.TEST)
invnet = caffe.Net(caffe_root+'examples/ECCV_visual/invdeploy_conv5relu.prototxt',caffe.TEST)

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
    
#可视化并显示
convnum=5
featall=range(9)
for i in range(9):
    net.blobs['data'].data[...] = transformer.preprocess('data', \
            caffe.io.load_image('examples/ECCV_visual/test/butterfly_'+str(i)+'.jpg'))
    out = net.forward()
    
    #求得特定层的最大特征的下标
    datasum=np.array([j.sum() for j in net.blobs['conv'+str(convnum)].data[0]])
    datanum=datasum.argsort()[-2]
    
    #参数共享，将前传网路的参数共享给后传网络，如pooling信息，反卷积信息。
    #每次输入新图都要重新计算下标，并将其他特征的权值清零
    invnet.params['conv'+str(convnum)][0].data[...]=0
    for b in range(1,convnum):
        invnet.params['conv'+str(b)][0].data[...] = net.params['conv'+ \
            str(b)][0].data.reshape(invnet.params['conv'+str(b)][0].data.shape)
    invnet.params['conv'+str(convnum)][0].data[datanum]=net.params['conv'+str(convnum)][0].data[datanum].reshape( \
            invnet.params['conv'+str(convnum)][0].data[datanum].shape)
            
    #各层数据输入到反卷积网络
    invnet.blobs['pooled'].data[...] = net.blobs['pool5'].data
    invnet.blobs['switches5'].data[...] = net.blobs['switches5'].data
    invnet.blobs['switches2'].data[...] = net.blobs['switches2'].data
    invnet.blobs['switches1'].data[...] = net.blobs['switches1'].data
    invnet.forward()
    feat= norm(invnet.blobs['conv1'].data[0],255.0)
    featall[i]=transformer.deprocess('data', feat)

#输出图像
pad1=np.zeros((227,3,3))
pad2=np.zeros((3,687,3))
pad1[pad1==0]=255
pad2[pad2==0]=255
featall=np.array(featall)
feat=np.concatenate((featall[0],pad1,featall[1],pad1,featall[2]),axis=1)
featshow=np.concatenate((feat,pad2),axis=0)
feat=np.concatenate((featall[3],pad1,featall[4],pad1,featall[5]),axis=1)
featshow=np.concatenate((featshow,feat,pad2),axis=0)
feat=np.concatenate((featall[6],pad1,featall[7],pad1,featall[8]),axis=1)
featshow=np.concatenate((featshow,feat,pad2),axis=0)
plt.clf
plt.imshow(featshow)
plt.axis('off')
plt.savefig('conv'+str(convnum)+'.jpg',dpi=600)
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
