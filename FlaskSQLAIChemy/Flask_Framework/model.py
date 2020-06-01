import tensorflow as tf
from tensorflow.keras import Model

# create linear regression model:Y=Wx+b
def Regression(x):
    w=tf.Variable(tf.random.truncated_normal([784, 10],stddev=0.1,seed=1))
    b=tf.Variable(tf.random.truncated_normal([10],stddev=0.1,seed=1))
    y=tf.nn.softmax(tf.matmul(x,w)+b)
    return y,[w,b]

# create nn model
def nn(lr):
    model=tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(10,activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.SGD(lr),
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                     metrics='sparse_categorical_accuracy')
    return model

class MyConvModel(Model):
    def __init__(self):
        super(MyConvModel,self).__init__()
        self.c1=tf.keras.layers.Conv2D(filters=6,kernel_size=(5,5),padding='SAME')
        self.b1=tf.keras.layers.BatchNormalization()
        self.a1=tf.keras.layers.Activation('relu')
        self.p1=tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2,padding='SAME')
        self.d1=tf.keras.layers.Dropout(rate=0.2)

        self.flatten=tf.keras.layers.Flatten()
        self.f1= tf.keras.layers.Dense(128,activation='relu')
        self.d2=tf.keras.layers.Dropout(rate=0.2)
        self.f2= tf.keras.layers.Dense(10,activation='softmax')

    def call(self, x):
        x=tf.cast(x,tf.float32)
        x=self.c1(x)
        x=self.b1(x)
        x=self.a1(x)
        x=self.p1(x)
        x=self.d1(x)

        x=self.flatten(x)
        x=self.f1(x)
        x=self.d2(x)
        y=self.f2(x)
        return y

def Conv():
    model=MyConvModel()
    return model