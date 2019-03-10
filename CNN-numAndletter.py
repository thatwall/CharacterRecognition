from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os

# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# print(mnist.train.next_batch(2))


def readImage(filename, ops='small'):
    if ops == 'small':
        image = Image.open(filename).convert('L')
        img = np.array(image)
        return img.ravel()
    else:
        image = Image.open(filename).convert('L')
        img = np.array(image.resize((227,227),Image.ANTIALIAS))
        return img.ravel()

def intToBitSet(num, total):
    bs = np.zeros(total)
    bs[num] = 1
    return bs


def BitSetToInt(arr):
    length = len(arr)
    maximum = 0
    maxIndex = 0
    for i in range(length):
        if arr[i] > maximum:
            maximum = arr[i]
            maxIndex = i
    return maxIndex


def LeNet5(trainList, testList):
    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 62])

    # 创建w参数
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    # 创建b参数
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # 创建卷积层，步长为1,周围补0，输入与输出的数据大小一样（可得到补全的圈数）
    def conv2d(x, W, padding):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)

    # 创建池化层，kernel大小为2,步长为2,周围补0，输入与输出的数据大小一样（可得到补全的圈数）
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='VALID')

    W_conv1 = weight_variable([5, 5, 1, 6])
    b_conv1 = bias_variable([6])

    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1, 'SAME') + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)  # 14x14

    # 初始化参数
    W_conv2 = weight_variable([5, 5, 6, 16])
    b_conv2 = bias_variable([16])

    # 创建卷积层
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 'VALID') + b_conv2)

    # 创建池化层
    h_pool2 = max_pool_2x2(h_conv2)

    # 初始化权重
    W_fc1 = weight_variable([5 * 5 * 16, 120])
    b_fc1 = bias_variable([120])

    # 铺平图像数据
    h_pool2_flat = tf.reshape(h_pool2, [-1, 5 * 5 * 16])

    # 全连接层计算
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    W_fc2 = weight_variable([120, 84])
    b_fc2 = bias_variable([84])
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    W_fc3 = weight_variable([84, 62])
    b_fc3 = bias_variable([62])

    # keep_prob表示保留不关闭的神经元的比例。
    keep_prob = tf.placeholder(tf.float32)
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
    y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3

    # # 1.计算交叉熵损失
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

    # # 2.创建优化器（注意这里用  AdamOptimizer代替了梯度下降法）
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # # 3. 计算准确率
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # # 4.初始化所有变量
    sess.run(tf.global_variables_initializer())

    # # 5. 执行循环
    for j in range(200):
        trainnum = 0
        for i in range(2265):
            # 每批取出62个训练样本
            batch = trainList.range(i * 62, (i + 1) * 62)
            if batch.empty() == True:
                continue
            # 执行训练模型
            trainnum += 1
            train_step.run(feed_dict={x: batch.values, y_: batch.references, keep_prob: 0.5})

            # 打印
            print("step %d %d, training..." % (j, i))
        TrainAccuracySum = 0
        for i in range(2265):
            batch = trainList.range(i * 62, (i + 1) * 62)
            if batch.empty() == True:
                continue
            TrainAccuracySum += accuracy.eval(feed_dict={
                x: batch.values, y_: batch.references, keep_prob: 1.0})
        print("step {}, training accuracy {}".format(j, TrainAccuracySum/trainnum))
    # 打印测试集正确率
        output = []
        TestAccuracySum = 0
        TestNum = 0
        for i in range(453):
            batch = testList.range(i * 62, (i + 1) * 62)
            if batch.empty() == True:
                continue
            TestNum += 1
            result = sess.run([y_conv, y_], feed_dict={x: batch.values, y_: batch.references, keep_prob: 1.0})
            for k in range(len(batch.values)):
                output.append([i,k,BitSetToInt(result[0][k]), BitSetToInt(result[1][k])])
                print(output[i*62+k])
            TestAccuracySum += accuracy.eval(feed_dict={
                x: batch.values, y_: batch.references, keep_prob: 1.0
            })
        print("step {}, test accuracy {}".format(j, TestAccuracySum/TestNum))
        if j == 199:
            return output

def AlexNet(trainList, testList):
    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, shape=[None, 1024])
    y_ = tf.placeholder(tf.float32, shape=[None, 62])

    # 创建w参数
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    # 创建b参数
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # 创建卷积层，步长为1,周围补0，输入与输出的数据大小一样（可得到补全的圈数）
    def conv2d(x, W, padding):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)

    # 创建池化层，kernel大小为2,步长为2,周围补0，输入与输出的数据大小一样（可得到补全的圈数）
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='VALID')
    x_image = tf.reshape(x, [-1,32,32,1])

    W_conv1 = weight_variable([7,7,1,32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image,W_conv1,[1,1,1,1],'SAME')+b_conv1) #32*32*32
    h_pool1 = tf.nn.max_pool(h_conv1, [1,3,3,1],[1,2,2,1],'VALID') # 20*20*32
    h_norm1 = tf.nn.lrn(h_pool1)  # 20*20*32

    W_conv2 = weight_variable([5,5,32,96])
    b_conv2 = bias_variable([96])
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_norm1,W_conv2,[1,1,1,1],'SAME')+b_conv2) # 20*20*96
    h_pool2 = tf.nn.max_pool(h_conv2,[1,3,3,1],[1,2,2,1],'VALID') # 9*9*96
    h_norm2 = tf.nn.lrn(h_pool2)

    W_conv3 = weight_variable([3,3,96,144])
    b_conv3 = bias_variable([144])
    h_conv3 = tf.nn.relu(tf.nn.conv2d(h_norm2,W_conv3,[1,1,1,1],'SAME')+b_conv3) # 9*9*144

    W_conv4 = weight_variable([3,3,144,144])
    b_conv4 = bias_variable([144])
    h_conv4 = tf.nn.relu(tf.nn.conv2d(h_conv3,W_conv4,[1,1,1,1],'SAME')+b_conv4) # 9*9*144

    W_conv5 = weight_variable([3,3,144,96])
    b_conv5 = bias_variable([96])
    h_conv5 = tf.nn.relu(tf.nn.conv2d(h_conv4,W_conv5,[1,1,1,1],'SAME')+b_conv5) # 9*9*96
    h_pool5 = tf.nn.max_pool(h_conv5, [1,3,3,1], [1,2,2,1],'VALID')  #4*4*96

    W_conv6 = weight_variable([3*3*96, 324])
    b_conv6 = bias_variable([324])
    h_pool5_flat = tf.reshape(h_pool5, [-1, 3*3*96])
    h_fc6 = tf.nn.relu(tf.matmul(h_pool5_flat,W_conv6)+b_conv6)  # 324

    drop6 = tf.placeholder(tf.float32)  # 0.5 or 0.1
    h_fc6_drop = tf.nn.dropout(h_fc6, drop6)  # 324

    W_conv7 = weight_variable([324,324])
    b_conv7 = bias_variable([324])
    h_fc7 = tf.nn.relu(tf.matmul(h_fc6_drop,W_conv7)+b_conv7)  # 324

    drop7 = tf.placeholder(tf.float32)
    h_fc7_drop = tf.nn.dropout(h_fc7,drop7)

    W_conv8 = weight_variable([324,62])
    b_conv8 = bias_variable([62])
    y_conv = tf.matmul(h_fc7_drop,W_conv8)+b_conv8

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

    # # 2.创建优化器（注意这里用  AdamOptimizer代替了梯度下降法）
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # # 3. 计算准确率
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # # 4.初始化所有变量
    sess.run(tf.global_variables_initializer())

    for i in range(20):
        trainnum = 0
        for j in range(5000):
            batch = trainList.range(j*30,(j+1)*30)
            if batch.empty() == True:
                continue
            # 执行训练模型
            trainnum += 1
            train_step.run(feed_dict={x: batch.values, y_: batch.references, drop6: 0.5, drop7: 0.5})
            # 打印
            print("step %d %d, training..." % (i, j))
        TrainAccuracySum = 0
        for j in range(5000):
            batch = trainList.range(j * 30, (j + 1) * 30)
            if batch.empty() == True:
                break
            print(j)
            TrainAccuracySum += accuracy.eval(feed_dict={
                x: batch.values, y_: batch.references, drop6: 1.0, drop7: 1.0})
        trainAC = TrainAccuracySum / trainnum
        print("step {}, training accuracy {}".format(i, TrainAccuracySum / trainnum))
        # 打印测试集正确率
        output = []
        TestAccuracySum = 0
        TestNum = 0
        for j in range(1000):
            batch = testList.range(j * 30, (j + 1) * 30)
            print(j)
            if batch.empty() == True:
                break
            TestNum += 1
            result = sess.run([y_conv, y_], feed_dict={x: batch.values, y_: batch.references, drop6: 1.0, drop7: 1.0})
            for k in range(len(batch.values)):
                output.append([j, k, BitSetToInt(result[0][k]), BitSetToInt(result[1][k])])
                print(output[j * 30 + k])
            TestAccuracySum += accuracy.eval(feed_dict={
                x: batch.values, y_: batch.references, drop6: 1.0, drop7: 1.0
            })
        testAC = TestAccuracySum / TestNum
        print("step {}, test accuracy {}".format(i, TestAccuracySum / TestNum))
        if i == 19:
            return output, trainAC, testAC

def AlexNetStd(trainList, testList):
    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, shape=[None, 51529])
    y_ = tf.placeholder(tf.float32, shape=[None, 62])

    # 创建w参数
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    # 创建b参数
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # 创建卷积层，步长为1,周围补0，输入与输出的数据大小一样（可得到补全的圈数）
    def conv2d(x, W, padding):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)

    # 创建池化层，kernel大小为2,步长为2,周围补0，输入与输出的数据大小一样（可得到补全的圈数）
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='VALID')

    x_image = tf.reshape(x, [-1, 227, 227, 1])

    W_conv1 = weight_variable([11, 11, 1, 96])
    b_conv1 = bias_variable([96])
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, [1, 4, 4, 1], 'VALID') + b_conv1)  # 32*32*32
    h_pool1 = tf.nn.max_pool(h_conv1, [1, 3, 3, 1], [1, 2, 2, 1], 'VALID')  # 20*20*32
    h_norm1 = tf.nn.lrn(h_pool1)  # 20*20*32

    W_conv2 = weight_variable([5, 5, 96, 256])
    b_conv2 = bias_variable([256])
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_norm1, W_conv2, [1, 1, 1, 1], 'SAME') + b_conv2)  # 20*20*96
    h_pool2 = tf.nn.max_pool(h_conv2, [1, 3, 3, 1], [1, 2, 2, 1], 'VALID')  # 9*9*96
    h_norm2 = tf.nn.lrn(h_pool2)

    W_conv3 = weight_variable([3, 3, 256, 384])
    b_conv3 = bias_variable([384])
    h_conv3 = tf.nn.relu(tf.nn.conv2d(h_norm2, W_conv3, [1, 1, 1, 1], 'SAME') + b_conv3)  # 9*9*144

    W_conv4 = weight_variable([3, 3, 384, 384])
    b_conv4 = bias_variable([384])
    h_conv4 = tf.nn.relu(tf.nn.conv2d(h_conv3, W_conv4, [1, 1, 1, 1], 'SAME') + b_conv4)  # 9*9*144

    W_conv5 = weight_variable([3, 3, 384, 256])
    b_conv5 = bias_variable([256])
    h_conv5 = tf.nn.relu(tf.nn.conv2d(h_conv4, W_conv5, [1, 1, 1, 1], 'SAME') + b_conv5)  # 9*9*96
    h_pool5 = tf.nn.max_pool(h_conv5, [1, 3, 3, 1], [1, 2, 2, 1], 'VALID')  # 4*4*96

    W_conv6 = weight_variable([6 * 6 * 256, 4096])
    b_conv6 = bias_variable([4096])
    h_pool5_flat = tf.reshape(h_pool5, [-1, 6 * 6 * 256])
    h_fc6 = tf.nn.relu(tf.matmul(h_pool5_flat, W_conv6) + b_conv6)  # 324

    drop6 = tf.placeholder(tf.float32)  # 0.5 or 0.1
    h_fc6_drop = tf.nn.dropout(h_fc6, drop6)  # 324

    W_conv7 = weight_variable([4096, 4096])
    b_conv7 = bias_variable([4096])
    h_fc7 = tf.nn.relu(tf.matmul(h_fc6_drop, W_conv7) + b_conv7)  # 324

    drop7 = tf.placeholder(tf.float32)
    h_fc7_drop = tf.nn.dropout(h_fc7, drop7)

    W_conv8 = weight_variable([4096, 62])
    b_conv8 = bias_variable([62])
    y_conv = tf.matmul(h_fc7_drop, W_conv8) + b_conv8

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

    # # 2.创建优化器（注意这里用  AdamOptimizer代替了梯度下降法）
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # # 3. 计算准确率
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # # 4.初始化所有变量
    sess.run(tf.global_variables_initializer())

    for i in range(200):
        trainnum = 0
        for j in range(5000):
            batch = trainList.range(j * 30, (j + 1) * 30)
            if batch.empty() == True:
                continue
            # 执行训练模型
            trainnum += 1
            train_step.run(feed_dict={x: batch.values, y_: batch.references, drop6: 0.5, drop7: 0.5})
            # 打印
            print("step %d %d, training..." % (i, j))
        TrainAccuracySum = 0
        for j in range(5000):
            batch = trainList.range(j * 30, (j + 1) * 30)
            if batch.empty() == True:
                break
            print(j)
            TrainAccuracySum += accuracy.eval(feed_dict={
                x: batch.values, y_: batch.references, drop6: 1.0, drop7: 1.0})
        trainAC = TrainAccuracySum / trainnum
        print("step {}, training accuracy {}".format(i, TrainAccuracySum / trainnum))
        # 打印测试集正确率
        output = []
        TestAccuracySum = 0
        TestNum = 0
        for j in range(1000):
            batch = testList.range(j * 30, (j + 1) * 30)
            print(j)
            if batch.empty() == True:
                break
            TestNum += 1
            result = sess.run([y_conv, y_], feed_dict={x: batch.values, y_: batch.references, drop6: 1.0, drop7: 1.0})
            for k in range(len(batch.values)):
                output.append([j, k, BitSetToInt(result[0][k]), BitSetToInt(result[1][k])])
                print(output[j * 30 + k])
            TestAccuracySum += accuracy.eval(feed_dict={
                x: batch.values, y_: batch.references, drop6: 1.0, drop7: 1.0
            })
        testAC = TestAccuracySum / TestNum
        print("step {}, test accuracy {}".format(i, TestAccuracySum / TestNum))
        if i == 199:
            return output, trainAC, testAC

def AlexNet2(trainList, testList):
    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, shape=[None, 1024])
    y_ = tf.placeholder(tf.float32, shape=[None, 1056])

    # 创建w参数
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    # 创建b参数
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # 创建卷积层，步长为1,周围补0，输入与输出的数据大小一样（可得到补全的圈数）
    def conv2d(x, W, padding):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)

    # 创建池化层，kernel大小为2,步长为2,周围补0，输入与输出的数据大小一样（可得到补全的圈数）
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='VALID')
    x_image = tf.reshape(x, [-1,32,32,1])

    W_conv1 = weight_variable([7,7,1,96])
    b_conv1 = bias_variable([96])
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image,W_conv1,[1,1,1,1],'SAME')+b_conv1) #32*32*32
    h_pool1 = tf.nn.max_pool(h_conv1, [1,3,3,1],[1,2,2,1],'VALID') # 20*20*32
    h_norm1 = tf.nn.lrn(h_pool1)  # 20*20*32

    W_conv2 = weight_variable([5,5,96,256])
    b_conv2 = bias_variable([256])
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_norm1,W_conv2,[1,1,1,1],'SAME')+b_conv2) # 20*20*96
    h_pool2 = tf.nn.max_pool(h_conv2,[1,3,3,1],[1,2,2,1],'VALID') # 9*9*96
    h_norm2 = tf.nn.lrn(h_pool2)

    W_conv3 = weight_variable([3,3,256,384])
    b_conv3 = bias_variable([384])
    h_conv3 = tf.nn.relu(tf.nn.conv2d(h_norm2,W_conv3,[1,1,1,1],'SAME')+b_conv3) # 9*9*144

    W_conv4 = weight_variable([3,3,384,384])
    b_conv4 = bias_variable([384])
    h_conv4 = tf.nn.relu(tf.nn.conv2d(h_conv3,W_conv4,[1,1,1,1],'SAME')+b_conv4) # 9*9*144

    W_conv5 = weight_variable([3,3,384,256])
    b_conv5 = bias_variable([256])
    h_conv5 = tf.nn.relu(tf.nn.conv2d(h_conv4,W_conv5,[1,1,1,1],'SAME')+b_conv5) # 9*9*96
    h_pool5 = tf.nn.max_pool(h_conv5, [1,3,3,1], [1,2,2,1],'VALID')  #4*4*96

    W_conv6 = weight_variable([3*3*256, 2048])
    b_conv6 = bias_variable([2048])
    h_pool5_flat = tf.reshape(h_pool5, [-1, 3*3*256])
    h_fc6 = tf.nn.relu(tf.matmul(h_pool5_flat,W_conv6)+b_conv6)  # 324

    drop6 = tf.placeholder(tf.float32)  # 0.5 or 0.1
    h_fc6_drop = tf.nn.dropout(h_fc6, drop6)  # 324

    W_conv7 = weight_variable([2048,2048])
    b_conv7 = bias_variable([2048])
    h_fc7 = tf.nn.relu(tf.matmul(h_fc6_drop,W_conv7)+b_conv7)  # 324

    drop7 = tf.placeholder(tf.float32)
    h_fc7_drop = tf.nn.dropout(h_fc7,drop7)

    W_conv8 = weight_variable([2048,1056])
    b_conv8 = bias_variable([1056])
    y_conv = tf.matmul(h_fc7_drop,W_conv8)+b_conv8

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

    # # 2.创建优化器（注意这里用  AdamOptimizer代替了梯度下降法）
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # # 3. 计算准确率
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # # 4.初始化所有变量
    sess.run(tf.global_variables_initializer())

    for i in range(20):
        trainnum = 0
        for j in range(8000):
            batch = trainList.range(j*30,(j+1)*30)
            if batch.empty() == True:
                break
            # 执行训练模型
            trainnum += 1
            train_step.run(feed_dict={x: batch.values, y_: batch.references, drop6: 0.5, drop7: 0.5})
            # 打印
            print("step %d %d, training..." % (i, j))
        TrainAccuracySum = 0
        for j in range(8000):
            batch = trainList.range(j * 30, (j + 1) * 30)
            if batch.empty() == True:
                break
            print(j)
            TrainAccuracySum += accuracy.eval(feed_dict={
                x: batch.values, y_: batch.references, drop6: 1.0, drop7: 1.0})
        trainAC = TrainAccuracySum / trainnum
        print("step {}, training accuracy {}".format(i, TrainAccuracySum / trainnum))
        # 打印测试集正确率
        output = []
        TestAccuracySum = 0
        TestNum = 0
        for j in range(2500):
            batch = testList.range(j * 30, (j + 1) * 30)
            print(j)
            if batch.empty() == True:
                break
            TestNum += 1
            result = sess.run([y_conv, y_], feed_dict={x: batch.values, y_: batch.references, drop6: 1.0, drop7: 1.0})
            for k in range(len(batch.values)):
                output.append([j, k, BitSetToInt(result[0][k]), BitSetToInt(result[1][k])])
                print(output[j * 30 + k])
            TestAccuracySum += accuracy.eval(feed_dict={
                x: batch.values, y_: batch.references, drop6: 1.0, drop7: 1.0
            })
        testAC = TestAccuracySum / TestNum
        print("step {}, test accuracy {}".format(i, TestAccuracySum / TestNum))
        tf.train.Saver().save(sess, './model.mdl')
        if i == 19:
            tf.train.Saver().save(sess, './model.mdl')
            return output, trainAC, testAC

def AlexNet3(trainList, testList):
    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, shape=[None, 1024])
    y_ = tf.placeholder(tf.float32, shape=[None, 1056])

    # 创建w参数
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    # 创建b参数
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # 创建卷积层，步长为1,周围补0，输入与输出的数据大小一样（可得到补全的圈数）
    def conv2d(x, W, padding):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)

    # 创建池化层，kernel大小为2,步长为2,周围补0，输入与输出的数据大小一样（可得到补全的圈数）
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='VALID')
    x_image = tf.reshape(x, [-1,32,32,1])

    W_conv1 = weight_variable([7,7,1,32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image,W_conv1,[1,1,1,1],'SAME')+b_conv1) #32*32*32
    h_pool1 = tf.nn.max_pool(h_conv1, [1,3,3,1],[1,2,2,1],'VALID') # 20*20*32
    h_norm1 = tf.nn.lrn(h_pool1)  # 20*20*32

    W_conv2 = weight_variable([5,5,32,120])
    b_conv2 = bias_variable([120])
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_norm1,W_conv2,[1,1,1,1],'SAME')+b_conv2) # 20*20*96
    h_pool2 = tf.nn.max_pool(h_conv2,[1,3,3,1],[1,2,2,1],'VALID') # 9*9*96
    h_norm2 = tf.nn.lrn(h_pool2)

    W_conv3 = weight_variable([3,3,120,144])
    b_conv3 = bias_variable([144])
    h_conv3 = tf.nn.relu(tf.nn.conv2d(h_norm2,W_conv3,[1,1,1,1],'SAME')+b_conv3) # 9*9*144

    W_conv4 = weight_variable([3,3,144,144])
    b_conv4 = bias_variable([144])
    h_conv4 = tf.nn.relu(tf.nn.conv2d(h_conv3,W_conv4,[1,1,1,1],'SAME')+b_conv4) # 9*9*144

    W_conv5 = weight_variable([3,3,144,120])
    b_conv5 = bias_variable([120])
    h_conv5 = tf.nn.relu(tf.nn.conv2d(h_conv4,W_conv5,[1,1,1,1],'SAME')+b_conv5) # 9*9*96
    h_pool5 = tf.nn.max_pool(h_conv5, [1,3,3,1], [1,2,2,1],'VALID')  #4*4*96

    W_conv6 = weight_variable([3*3*120, 1080])
    b_conv6 = bias_variable([1080])
    h_pool5_flat = tf.reshape(h_pool5, [-1, 3*3*120])
    h_fc6 = tf.nn.relu(tf.matmul(h_pool5_flat,W_conv6)+b_conv6)  # 324

    drop6 = tf.placeholder(tf.float32)  # 0.5 or 0.1
    h_fc6_drop = tf.nn.dropout(h_fc6, drop6)  # 324

    W_conv7 = weight_variable([1080,1080])
    b_conv7 = bias_variable([1080])
    h_fc7 = tf.nn.relu(tf.matmul(h_fc6_drop,W_conv7)+b_conv7)  # 324

    drop7 = tf.placeholder(tf.float32)
    h_fc7_drop = tf.nn.dropout(h_fc7,drop7)

    W_conv8 = weight_variable([1080,1056])
    b_conv8 = bias_variable([1056])
    y_conv = tf.matmul(h_fc7_drop,W_conv8)+b_conv8

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

    # # 2.创建优化器（注意这里用  AdamOptimizer代替了梯度下降法）
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # # 3. 计算准确率
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # # 4.初始化所有变量
    sess.run(tf.global_variables_initializer())

    for i in range(20):
        trainnum = 0
        for j in range(8000):
            batch = trainList.range(j*30,(j+1)*30)
            if batch.empty() == True:
                break
            # 执行训练模型
            trainnum += 1
            train_step.run(feed_dict={x: batch.values, y_: batch.references, drop6: 0.5, drop7: 0.5})
            # 打印
            print("step %d %d, training..." % (i, j))
        TrainAccuracySum = 0
        for j in range(8000):
            batch = trainList.range(j * 30, (j + 1) * 30)
            if batch.empty() == True:
                break
            print(j)
            TrainAccuracySum += accuracy.eval(feed_dict={
                x: batch.values, y_: batch.references, drop6: 1.0, drop7: 1.0})
        trainAC = TrainAccuracySum / trainnum
        print("step {}, training accuracy {}".format(i, TrainAccuracySum / trainnum))
        # 打印测试集正确率
        output = []
        TestAccuracySum = 0
        TestNum = 0
        for j in range(2500):
            batch = testList.range(j * 30, (j + 1) * 30)
            print(j)
            if batch.empty() == True:
                break
            TestNum += 1
            result = sess.run([y_conv, y_], feed_dict={x: batch.values, y_: batch.references, drop6: 1.0, drop7: 1.0})
            for k in range(len(batch.values)):
                output.append([j, k, BitSetToInt(result[0][k]), BitSetToInt(result[1][k])])
                print(output[j * 30 + k])
            TestAccuracySum += accuracy.eval(feed_dict={
                x: batch.values, y_: batch.references, drop6: 1.0, drop7: 1.0
            })
        testAC = TestAccuracySum / TestNum
        print("step {}, test accuracy {}".format(i, TestAccuracySum / TestNum))
        tf.train.Saver().save(sess, './model.mdl')
        if i == 19:
            tf.train.Saver().save(sess, './model.mdl')
            return output, trainAC, testAC

def GoogleNet(trainList, testList):
    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, shape=[None, 1024])
    y_ = tf.placeholder(tf.float32, shape=[None, 1056])

    # 创建w参数
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    # 创建b参数
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x,W,stride,padding):
        return tf.nn.conv2d(x, W, [1,stride,stride,1],padding)

    def inception():
        pass



    pass

class itemList1:
    values = []
    references = []

    def range(self, start, end):
        new = itemList();
        if end < len(self.values):
            new.values = self.values[start:end]
            new.references = self.references[start:end]
        else:
            new.values = self.values[start:]
            new.references = self.references[start:]
        return new

    def empty(self):
        return len(self.values) == 0 or len(self.references) == 0

class itemList2:
    values = []
    references = []

    def range(self, start, end):
        new = itemList();
        if end < len(self.values):
            new.values = self.values[start:end]
            new.references = self.references[start:end]
        else:
            new.values = self.values[start:]
            new.references = self.references[start:]
        return new

    def empty(self):
        return len(self.values) == 0 or len(self.references) == 0

train = itemList1()
test = itemList2()

# for i in range(1, 6):
#     if 1 <= i <= 4:
#         for j in range(1, 454):
#             for k in range(62):
#                 filename = 'data/dataset{}/{}/{}.bmp'.format(i, j, k)
#                 if os.path.exists(filename):
#                     train.values.append(readImage(filename))
#                     train.references.append(intToBitSet(k, 62))
#     else:
#         for j in range(1, 454):
#             for k in range(62):
#                 filename = 'data/dataset{}/{}/{}.bmp'.format(i, j, k)
#                 if os.path.exists(filename):
#                     test.values.append(readImage(filename))
#                     test.references.append(intToBitSet(k, 62))

for i in range(3):
    for j in range(1056):
        filename = 'dataset2/{}/{}.bmp'.format(i,j)
        if os.path.exists(filename):
            if i <= 1:
                train.values.append(readImage(filename))
                train.references.append(intToBitSet(j,1056))
            else:
                test.values.append(readImage(filename))
                test.references.append(intToBitSet(j,1056))
print(len(train.values))
print(len(test.values))
# trainList : 212256
# testList  : 69696

# output, trainAC, testAC = AlexNet3(train, test)
# with open('saveresult', 'w') as f:
#     for var in output:
#         for l in var:
#             f.write(l)
#             f.write(',')
#         f.write('\n')
#     f.write(trainAC)
#     f.write(' ')
#     f.write(testAC)
#     f.write('\n')
# print(BitSetToInt(np.array([0,1,-1,2])))
# readImage('data/dataset1/1/1.bmp')
