import tensorflow as tf
import tensorflow.keras as keras
from Flask_Framework import  model
import os

# 学习率
lr = 1e-3
regression_model_path='./checkpoint/regression_model.ckpt'
nn_model_path='./checkpoint/nn_model'
cnn_model_path='./checkpoint/cnn_model/cnn_model.ckpt'

def loadData(mnistflag):
    if mnistflag:
        (x, y), (x_test, y_test) = keras.datasets.mnist.load_data()
    else:
        (x, y), (x_test, y_test) = keras.datasets.cifar10.load_data()
    return x, y, x_test, y_test

    # 对数据进行预处理
def preprocess(x, y):
        x = tf.cast(x, dtype=tf.float32) / 255.
        y = tf.cast(y, dtype=tf.int32)
        return x, y

def trainRegression(x, y, x_test, y_test):

    # 构建dataset对象，方便对数据的打乱，批处理等超操作
    train_db = tf.data.Dataset.from_tensor_slices((x, y)).batch(32)
    train_db = train_db.map(preprocess)
    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
    test_db = test_db.map(preprocess)
    # epoch表示整个训练集循环的次数 这里循环100次
    for epoch in range(100):
        # step表示当前训练到了第几个Batch
        for step, (x, y) in enumerate(train_db):
            # 把训练集进行打平操作
            x = tf.reshape(x, [-1, 28 * 28])
            # 构建模型并计算梯度
            with tf.GradientTape() as tape:  # tf.Variable
                y_,[w,b]=model.Regression(x)
                # 把标签转化成one_hot编码
                yonehot = tf.one_hot(y, depth=10)
                # 计算MSE
                loss = tf.reduce_mean(tf.square(y_-yonehot))
            # 计算梯度
            grads = tape.gradient(loss, [w,b])

            # w = w - lr * w_grad
            # 利用上述公式进行权重的更新
            w.assign_sub(lr * grads[0])
            b.assign_sub(lr * grads[1])
            # 每训练100个Batch 打印一下当前的loss

        print(epoch,'loss:', float(loss))#'w:',w.numpy(),'b:',b.numpy())
        # 每训练完一次数据集 测试一下准确率
        total_correct, total_num = 0, 0
        for step, (x, y) in enumerate(test_db):
            x = tf.reshape(x, [-1, 28 * 28])
            y_predict= tf.nn.softmax(tf.matmul(x,w)+b)
            # 获取概率最大值得索引位置
            pred = tf.argmax(y_predict, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)

            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)
            # 获取每一个batch中的正确率和batch大小
            total_correct += int(correct)
            total_num += x.shape[0]
        # 计算总的正确率
        acc = total_correct / total_num
        print('test acc:', acc)


def trainNN(x, y, x_test, y_test):
    if os.path.exists(nn_model_path):
        nn_model = tf.keras.models.load_model(nn_model_path)  # load saved model
        print('Loading model.....')
    else:
        nn_model = model.nn(lr)
#    cp_callback=tf.keras.callbacks.ModelCheckpoint(filepath=nn_model_path,save_weights_only=True,save_best_only=True)
    nn_model.fit(x, y, batch_size=32, epochs=5,validation_data=(x_test,y_test),validation_freq=1),#callbacks=[cp_callback])
    nn_model.save(filepath=nn_model_path,save_format='tf') #save model
    nn_model.summary()
    #loss,accu= nn_model.evaluate(x_test,y_test)
    #print('loss:',loss,'accu',accu)
    # old_model = tf.keras.models.load_model(nn_model_path) #load saved model
    # print('loading model...')
    # y=old_model.predict(x_test)
    # print(y)
 #       old_model.predict
#    nn_model.summary()

def trainCNN(x, y, x_test, y_test):
    cnn_model = model.Conv()
    cnn_model.compile(optimizer=tf.keras.optimizers.SGD(lr),
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                     metrics='sparse_categorical_accuracy')
    if os.path.exists(cnn_model_path + '.index'):
        cnn_model.load_weights(cnn_model_path)
        print('loading model...')
    cp_callback=tf.keras.callbacks.ModelCheckpoint(filepath=cnn_model_path,save_weights_only=True,save_best_only=True)
    cnn_model.fit(x, y, batch_size=32, epochs=100,validation_data=(x_test,y_test),validation_freq=1,callbacks=[cp_callback])
    cnn_model.summary()
