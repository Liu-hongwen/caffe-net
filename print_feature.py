# coding=utf-8

import caffe
import numpy as np

deploy = "/Users/lhw/caffe-project/CNN_age_gender/deploy_gender.prototxt"
caffemodel = "/Users/lhw/caffe-project/CNN_age_gender/gender_net.caffemodel"
image = "/Users/lhw/caffe-project/CNN_age_gender/example_image.jpg"

net = caffe.Net(deploy, caffemodel, caffe.TEST)

# 查看各层的参数值，k表示层的名称，v[0].data表示各层的w值，v[1].data表示各层的b值
# 并不是所有层都有参数，只有卷积层的全连接层才有
print [(k, v[0].data) for k, v in net.params.items()]
print [(k, v[1].data) for k, v in net.params.items()]

# 也可以不查看具体值，只看shape
print [(k, v[0].data.shape) for k, v in net.params.items()]
print [(k, v[1].data.shape) for k, v in net.params.items()]

print net.params['conv1'][0].data.shape  # 提取conv1层参数w
print net.params['conv1'][1].data.shape  # 提取conv1层参数b

# 图片预处理设置
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
# transformer.set_mean('data',np.load(meanfile).mean(1).mean(1))
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2, 1, 0))

# 导入图片
img = caffe.io.load_image(image)
# 执行上面设置的图像预处理操作，并将图片载入blob中
net.blobs['data'].data[...] = transformer.preprocess('data', img)

net.forward()

# 查看各层的数据，k表示层的名称，
print [(k, v.data) for k, v in net.blobs.items()]
# 也可以不查看具体值，只看shape
print [(k, v.data.shape) for k, v in net.blobs.items()]

print net.blobs['fc8'].data # 提取fc8层特征
print net.blobs['prob'].data # 提取softmax层特征

