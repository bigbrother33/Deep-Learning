# RCNN论文复现
R-CNN: Regions with Convolutional Neural Network Features
# 工程内容
算法基于tensorflow 1.12实现
# 开发环境
windows10+python3.6+tensorflow1.12+scikit-learning+cv2+tflearn  
i7-6700+gtx1060  
# 数据集
由于设备所限，没有采用原论文中的imagenet数据集，采用较小的17flowers据集, 官网下载：http://www.robots.ox.ac.uk/~vgg/data/flowers/17/

# 程序说明
1、config.py---网络定义、训练与数据处理所需要用到的参数  
2、Networks.py---用于定义Alexnet_Net模型、fineturn模型、SVM模型、边框回归模型   
4、process_data.py---用于对训练数据集与微调数据集进行处理（选择性搜索、数据存取等）  
5、train_and_test.py---用于各类模型的训练与测试、主函数  
6、selectivesearch.py---选择性搜索源码  

## 基本知识介绍
### 1,计算机视觉中的不同任务
![](https://github.com/bigbrother33/Deep-Learning/blob/master/photo/20190101200347.png)<br><br>
### 2,IOU的定义
简单介绍一下IOU。物体检测需要定位出物体的boundingbox，就像下面的图片一样，我们不仅要定位出车辆的bounding box 我们还要识别出bounding box 里面的物体就是车辆。对于boundingbox的定位精度，有一个很重要的概念，因为我们算法不可能百分百跟人工标注的数据完全匹配，因此就存在一个定位精度评价公式：`IOU`  <br>
![](https://github.com/bigbrother33/Deep-Learning/blob/master/photo/20160902124803660.png)   
IOU定义了两个bounding box的重叠度，如下图所示： 

![](https://github.com/bigbrother33/Deep-Learning/blob/master/photo/20160902124815518.png)  
矩形框A、B的一个重合度IOU计算公式为：<br>
>>>>>>>>>>>><a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{100}&space;\LARGE&space;IOU=(A\cap&space;B)/(A\cup&space;B)" target="_blank"><img src="https://latex.codecogs.com/png.latex?\dpi{100}&space;\LARGE&space;IOU=(A\cap&space;B)/(A\cup&space;B)" title="\LARGE IOU=(A\cap B)/(A\cup B)" /></a>  
### 3,非极大值抑制NMS（non-maxmum suppression）
因为一会儿讲RCNN算法，会从一张图片中找出n多个可能是物体的矩形框，然后为每个矩形框为做类别分类概率：  
![](https://github.com/bigbrother33/Deep-Learning/blob/master/photo/20160902124825831.png)  
就像上面的图片一样，定位一个车辆，最后算法就找出了一堆的方框，我们需要判别哪些矩形框是没用的。非极大值抑制：先假设有6个矩形框，根据分类器类别分类概率做排序，从小到大分别属于车辆的概率分别为A、B、C、D、E、F。

* (1)从最大概率矩形框F开始，分别判断A~E与F的重叠度IOU是否大于某个设定的阈值;

* (2)假设B、D与F的重叠度超过阈值，那么就扔掉B、D；并标记第一个矩形框F，是我们保留下来的。

* (3)从剩下的矩形框A、C、E中，选择概率最大的E，然后判断E与A、C的重叠度，重叠度大于一定的阈值，那么就扔掉；并标记E是我们保留下来的第二个矩形框。  

就这样一直重复，找到所有被保留下来的矩形框。

## 算法总体思路
利用候选区域与 CNN 结合做目标定位
借鉴了滑动窗口思想，R-CNN 采用对区域进行识别的方案。
因此paper采用的方法是：首先输入一张图片，我们先定位出若干个物体候选框，然后采用CNN提取每个候选框中图片的特征向量，特征向量为全连接层去处顶层后的输出，接着采用svm算法对各个候选框中的物体进行分类识别。也就是总个过程分为三个程序：

a、找出候选框；

b、利用CNN提取特征向量；

c、利用SVM进行特征向量分类；

具体的流程如下图片所示：
![](https://github.com/bigbrother33/Deep-Learning/blob/master/photo/20160902124834270.png)
### 利用预训练与微调解决标注数据缺乏的问题
采用在 ImageNet 上已经训练好的模型，然后在 PASCAL VOC 数据集上进行 fine-tune。
因为 ImageNet 的图像高达几百万张，利用卷积神经网络充分学习浅层的特征，然后在小规模数据集做规模化训练，从而可以达到好的效果。
现在，我们称之为迁移学习，是必不可少的一种技能。
### 候选区域
能够生成候选区域的方法很多，比如：
* objectness
* selective search
* category-independen object proposals
* constrained parametric min-cuts(CPMC)
* multi-scale combinatorial grouping
* Ciresan  
R-CNN 采用的是 Selective Search 算法。这样我们便得到若干个`region proposals`,即候选框（注：原论文中生成2000个）
### CNN特征提取
R-CNN抽取了一个 4096 维的特征向量，采用的是Alexnet模型，原论文基于 Caffe 进行代码开发，本次复现基于TensorFlow。
需要注意的是 Alextnet 的输入图像大小是 227x227。
而通过 Selective Search 产生的候选区域大小不一，为了与 Alexnet 兼容，R-CNN 采用了非常暴力的手段，那就是无视候选区域的大小和形状，统一变换到 227\*227 的尺寸。
#### 1,网络结构设计阶段
网络架构我们有两个可选方案：第一选择经典的Alexnet；第二选择VGG16。经过测试Alexnet精度为58.5%，VGG16精度为66%。VGG这个模型的特点是选择比较小的卷积核、选择较小的跨步，这个网络的精度高，不过计算量是Alexnet的7倍。后面为了简单起见，我们就直接选用Alexnet，并进行讲解；Alexnet特征提取部分包含了5个卷积层、2个全连接层，在Alexnet中p5层神经元个数为9216、 fc_8、fc_10神经元个数都是4096，通过这个网络训练完毕后，最后提取特征每个输入候选框图片都能得到一个4096维的特征向量。代码如下：
```Python
class Alexnet_Net :
    '''
    此类用来定义Alexnet网络及其参数，之后整体作为参数输入到Solver中
    '''
    def __init__(self, is_training=True, is_fineturn=False, is_SVM=False):
        self.image_size = cfg.Image_size
        self.batch_size = cfg.F_batch_size if is_fineturn else cfg.T_batch_size
        self.class_num = cfg.F_class_num if is_fineturn else cfg.T_class_num
        self.input_data = tf.placeholder(tf.float32,[None, self.image_size, self.image_size,3], name='input')
        self.logits = self.build_network(self.input_data, self.class_num, is_svm=is_SVM, is_training=is_training)

        if is_training == True :
            self.label = tf.placeholder(tf.float32, [None, self.class_num], name='label')
            self.loss_layer(self.logits, self.label)
            self.accuracy = self.get_accuracy(self.logits, self.label)
            self.total_loss = tf.losses.get_total_loss()
            tf.summary.scalar('total_loss', self.total_loss)

    def build_network(self, input, output, is_svm= False, scope='R-CNN',is_training=True, keep_prob=0.5):
        with tf.variable_scope(scope):
            with slim.arg_scope([slim.fully_connected, slim.conv2d],
                                activation_fn=nn_ops.relu,
                                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                weights_regularizer=slim.l2_regularizer(0.0005)):
                net = slim.conv2d(input, 96, 11, stride=4, scope='conv_1')
                net = slim.max_pool2d(net, 3, stride=2, scope='pool_2')
                net = local_response_normalization(net)
                net = slim.conv2d(net, 256, 5, scope='conv_3')
                net = slim.max_pool2d(net, 3, stride=2, scope='pool_2')
                net = local_response_normalization(net)
                net = slim.conv2d(net, 384, 3, scope='conv_4')
                net = slim.conv2d(net, 384, 3, scope='conv_5')
                net = slim.conv2d(net, 256, 3, scope='conv_6')
                net = slim.max_pool2d(net, 3, stride=2, scope='pool_7')
                net = local_response_normalization(net)
                net = slim.flatten(net, scope='flat_32')
                net = slim.fully_connected(net, 4096, activation_fn=self.tanh(), scope='fc_8')
                net = slim.dropout(net, keep_prob=keep_prob,is_training=is_training, scope='dropout9')
                net = slim.fully_connected(net, 4096, activation_fn=self.tanh(), scope='fc_10')
                if is_svm:
                    return net
                net = slim.dropout(net, keep_prob=keep_prob, is_training=is_training, scope='dropout11')
                net = slim.fully_connected(net, output, activation_fn=self.softmax(), scope='fc_11')
        return net

    def loss_layer(self, y_pred, y_true):
        with tf.name_scope("Crossentropy"):
            y_pred = tf.clip_by_value(y_pred, tf.cast(1e-10, dtype=tf.float32),tf.cast(1. - 1e-10, dtype=tf.float32))  #tf.clip_by_value(A, min, max)：输入一个张量A，把A中的每一个元素的值都压缩在min和max之间。小于min的让它等于min，大于max的元素的值等于max。
            cross_entropy = - tf.reduce_sum(y_true * tf.log(y_pred),reduction_indices=len(y_pred.get_shape()) - 1)
            loss = tf.reduce_mean(cross_entropy)
            tf.losses.add_loss(loss)
            tf.summary.scalar('loss', loss)

    def get_accuracy(self, y_pred, y_true):
        y_pred_maxs =(tf.argmax(y_pred,1))
        y_true_maxs =(tf.argmax(y_true,1))
        num = tf.count_nonzero((y_true_maxs-y_pred_maxs))
        result = 1-(num/self.batch_size)
        return result
    def softmax(self):
        def op(inputs):
            return tf.nn.softmax(inputs)
        return op
    def tanh(self):
        def op(inputs):
            return tf.tanh(inputs)
        return op
```
#### 2、网络有监督预训练阶段
参数初始化部分：物体检测的一个难点在于，物体标签训练数据少，如果要直接采用随机初始化CNN参数的方法，那么目前的训练数据量是远远不够的。这种情况下，最好的是采用某些方法，把参数初始化了，然后在进行有监督的参数微调，这边文献采用的是有监督的预训练。所以paper在设计网络结构的时候，是直接用Alexnet的网络，然后连参数也是直接采用它的参数，作为初始的参数值，然后再fine-tuning训练。
```Python
    def train(self):
        for step in range(1, self.max_iter+1):
            if self.is_Reg:
                input, labels = self.data.get_Reg_batch()
            elif self.is_fineturn:
                input, labels = self.data.get_fineturn_batch()
            else:
                input, labels = self.data.get_batch()

            feed_dict = {self.net.input_data:input, self.net.label:labels}
            if step % self.summary_step == 0 :
                summary, loss, _=self.sess.run([self.summary_op,self.net.total_loss,self.train_op], feed_dict=feed_dict)
                self.writer.add_summary(summary, step)
                print("Data_epoch:"+str(self.data.epoch)+" "*5+"training_step:"+str(step)+" "*5+ "batch_loss:"+str(loss))
            else:
                self.sess.run([self.train_op], feed_dict=feed_dict)
            if step % self.save_step == 0 :
                print("saving the model into " + self.ckpt_file)
                self.saver.save(self.sess, self.ckpt_file, global_step=self.global_step)
```
#### 3,fine-tuning阶段
我们接着采用selective search 搜索出来的候选框。首先会逐步读入图片，然后采用seletive search 对读入的图片生成候选区域，再计算每个候选区域和ground truth(代码中的fine_turn_list)的交并比（IOU).当IOU大于阈值时，则认为是当前的候选区域属于正确类。并且将其标定为相应的类别(label)。这样每一个候选区域就会产生相应的label即（image, label). (image, label)就是Fineturn训练的训练集。然后处理到指定大小图片，继续对上面预训练的cnn模型进行fine-tuning训练。假设要检测的物体类别有N类，那么我们就需要把上面预训练阶段的CNN模型的最后一层给替换掉，替换成N+1个输出的神经元(加1，表示还有一个背景)，然后这一层直接采用参数随机初始化的方法，其它网络层的参数不变；接着就可以开始继续SGD训练了。开始的时候，SGD学习率选择0.001，在每次训练的时候，我们batch size大小选择128，其中32个事正样本、96个事负样本（正负样本的定义前面已经提过，不再解释）。
### SVM训练
首先会逐步读入图片，然后采用seletive search 对读入的图片生成候选区域，这时候会生成候选区域候选框的坐标信息， 再计算每个候选区域和ground truth(代码中的fine_turn_list)的交并比（IOU).当IOU大于阈值时，则认为是当前的候选区域属于正确类。并且将其标定为相应的类别(label)， 并将这个label对应的候选区域图（iamge）的候选框坐标信息与ground truth的位置信息作对比，保留相对位置(平移和缩放)信息保留下来（label_bbox），作为训练数据。这样整个训练数据则为（image, label, label_bbox）对。但是在训练SVM时并没有用到label_bbox信息。还只是用来分类。而且需要对每种类型都单独训练一个分类器， 并保存训练好的模型，备用。另外SVM分类器的输入是Alex网络softmax链接层之前的全连接层的输出结果。
```Python
class SVM :
    def __init__(self, data):
        self.data = data
        self.data_save_path = cfg.SVM_and_Reg_save
        self.output = cfg.Out_put
    def train(self):
        svms=[]
        data_dirs = os.listdir(self.data_save_path)
        for data_dir in data_dirs:
            images, labels = self.data.get_SVM_data(data_dir)
            clf = svm.LinearSVC()
            clf.fit(images, labels)
            svms.append(clf)
            SVM_model_path = os.path.join(self.output, 'SVM_model')
            if not os.path.exists(SVM_model_path):
                os.makedirs(SVM_model_path)
            joblib.dump(clf, os.path.join(SVM_model_path,  str(data_dir)+ '_svm.pkl'))
```
### bbox regression网络训练
候选的bbox，与gt的bbox的IoU大于阈值，这个bbox才会作为正类。对于N各类别。训练N个不同的bbox regression。给定一个bbox的中心和长宽，以及cnn学习出来的pool5后的特征，学习gt bbox的中心还有长宽。目标函数:  
<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{80}&space;\huge&space;Loss&space;=&space;\sum_i^N(t_*^i&space;-&space;\hat&space;w_*^T\phi_5(P^i))^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{80}&space;\huge&space;Loss&space;=&space;\sum_i^N(t_*^i&space;-&space;\hat&space;w_*^T\phi_5(P^i))^2" title="\huge Loss = \sum_i^N(t_*^i - \hat w_*^T\phi_5(P^i))^2" /></a>  
`w`即为要学习的参数，利用梯度下降法或者最小二乘法就可以得到。
```Python
class Reg_Net(object):
    def __init__(self, is_training=True):
        self.output_num = cfg.R_class_num
        self.input_data = tf.placeholder(tf.float32, [None, 4096], name='input')
        self.logits = self.build_network(self.input_data, self.output_num, is_training=is_training)
        if is_training:
            self.label = tf.placeholder(tf.float32, [None, self.output_num], name='input')
            self.loss_layer(self.logits, self.label)
            self.total_loss = tf.losses.get_total_loss()
            tf.summary.scalar('total_loss', self.total_loss)

    def build_network(self, input_image, output_num, is_training= True, scope='regression_box', keep_prob=0.5):
        with tf.variable_scope(scope):
            with slim.arg_scope([slim.fully_connected],
                                activation_fn=self.tanh(),
                                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                weights_regularizer=slim.l2_regularizer(0.0005)):
                net = slim.fully_connected(input_image, 4096, scope='fc_1')
                net = slim.dropout(net, keep_prob=keep_prob, is_training=is_training, scope='dropout11')
                net = slim.fully_connected(net, output_num, scope='fc_2')
                return net

    def loss_layer(self,y_pred, y_true):
        no_object_loss = tf.reduce_mean(tf.square((1 - y_true[:, 0]) * y_pred[:, 0]))
        object_loss = tf.reduce_mean(tf.square((y_true[:, 0]) * (y_pred[:, 0] - 1)))

        loss = (tf.reduce_mean(y_true[:, 0] * (
                 tf.reduce_sum(tf.square(y_true[:, 1:5] - y_pred[:, 1:5]), 1))) + no_object_loss + object_loss)
        tf.losses.add_loss(loss)
        tf.summary.scalar('loss', loss)

    def tanh(self):
        def op(inputs):
            return tf.tanh(inputs)
        return op
```
### 测试
读入测试图片，生成候选框，用SVM判断类别，如果不是背景，用Reg_box生成平移缩放值， 然后对生成的候选区域进行调整。最后取所有候选区域调整后的结果的平均值作为最终的标定框。
```Python
if __name__ =='__main__':
    
    Features_solver, svms, Reg_box_solver =get_Solvers()

    img_path = './2flowers/jpg/1/image_1283.jpg'  # or './17flowers/jpg/16/****.jpg'
    imgs, verts = process_data.image_proposal(img_path)
    process_data.show_rect(img_path, verts, ' ')
    features = Features_solver.predict(imgs)
    print(np.shape(features))

    results = []
    results_old = []
    results_label = []
    count = 0
    for f in features:
        for svm in svms:
            pred = svm.predict([f.tolist()])
            # not background
            if pred[0] != 0:
                results_old.append(verts[count])
                #print(Reg_box_solver.predict([f.tolist()]))
                if Reg_box_solver.predict([f.tolist()])[0][0] > 0.5:
                    px, py, pw, ph = verts[count][0], verts[count][1], verts[count][2], verts[count][3]
                    old_center_x, old_center_y = px + pw / 2.0, py + ph / 2.0
                    x_ping, y_ping, w_suo, h_suo = Reg_box_solver.predict([f.tolist()])[0][1], \
                                                   Reg_box_solver.predict([f.tolist()])[0][2], \
                                                   Reg_box_solver.predict([f.tolist()])[0][3], \
                                                   Reg_box_solver.predict([f.tolist()])[0][4]
                    new__center_x = x_ping * pw + old_center_x
                    new__center_y = y_ping * ph + old_center_y
                    new_w = pw * np.exp(w_suo)
                    new_h = ph * np.exp(h_suo)
                    new_verts = [new__center_x, new__center_y, new_w, new_h]
                    results.append(new_verts)
                    results_label.append(pred[0])
        count += 1

    average_center_x, average_center_y, average_w,average_h = 0, 0, 0, 0
    #给预测出的所有的预测框区一个平均值，代表其预测出的最终位置
    for vert in results:
        average_center_x += vert[0]
        average_center_y += vert[1]
        average_w += vert[2]
        average_h += vert[3]
    average_center_x = average_center_x / len(results)
    average_center_y = average_center_y / len(results)
    average_w = average_w / len(results)
    average_h = average_h / len(results)
    average_result = [[average_center_x, average_center_y, average_w, average_h]]
    result_label = max(results_label, key=results_label.count)
    process_data.show_rect(img_path, results_old,' ')
    process_data.show_rect(img_path, average_result,flower[result_label])
```
测试结果如下： 

![](https://github.com/bigbrother33/Deep-Learning/blob/master/photo/1.PNG)  ![](https://github.com/bigbrother33/Deep-Learning/blob/master/photo/2.PNG)  ![](https://github.com/bigbrother33/Deep-Learning/blob/master/photo/3.PNG)  
