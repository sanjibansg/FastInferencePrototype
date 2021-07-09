import tensorflow as tf

print('TensorFlow', tf.__version__)

class ResidualBlock(tf.keras.Model):
    def __init__(self, block_type=None, n_filters=None):
        super(ResidualBlock, self).__init__()
        self.n_filters = n_filters
        if block_type == 'identity':
            self.strides = 1
        elif block_type == 'conv':
            self.strides = 2
            self.conv_shorcut = tf.keras.layers.Conv2D(filters=self.n_filters, 
                               kernel_size=1, 
                               padding='same',
                               strides=self.strides,
                               kernel_initializer='he_normal')
            self.bn_shortcut = tf.keras.layers.BatchNormalization(momentum=0.9)

        self.conv_1 = tf.keras.layers.Conv2D(filters=self.n_filters, 
                               kernel_size=3, 
                               padding='same',
                               strides=self.strides,
                               kernel_initializer='he_normal')
        self.bn_1 = tf.keras.layers.BatchNormalization(momentum=0.9)
        self.relu_1 = tf.keras.layers.ReLU()

        self.conv_2 = tf.keras.layers.Conv2D(filters=self.n_filters, 
                               kernel_size=3, 
                               padding='same', 
                               kernel_initializer='he_normal')
        self.bn_2 = tf.keras.layers.BatchNormalization(momentum=0.9)
        self.relu_2 = tf.keras.layers.ReLU()

    def call(self, x, training=False):
        shortcut = x
        if self.strides == 2:
            shortcut = self.conv_shorcut(x)
            shortcut = self.bn_shortcut(shortcut)
        y = self.conv_1(x)
        y = self.bn_1(y)
        y = self.relu_1(y)
        y = self.conv_2(y)
        y = self.bn_2(y)
        y = tf.add(shortcut, y)
        y = self.relu_2(y)
        return y

class ResNet34(tf.keras.Model):
    def __init__(self, include_top=True, n_classes=1000):
        super(ResNet34, self).__init__()

        self.n_classes = n_classes
        self.include_top = include_top
        self.conv_1 = tf.keras.layers.Conv2D(filters=64, 
                                               kernel_size=7, 
                                               padding='same', 
                                               strides=2, 
                                               kernel_initializer='he_normal')
        self.bn_1 = tf.keras.layers.BatchNormalization(momentum=0.9)
        self.relu_1 = tf.keras.layers.ReLU()
        self.maxpool = tf.keras.layers.MaxPool2D(3, 2, padding='same')
        self.residual_blocks = tf.keras.Sequential()
     
        self.extra=ResidualBlock(block_type='conv')
        self.GAP = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(units=self.n_classes)

    def call(self, x, training=False):
        y = self.conv_1(x)
        y = self.bn_1(y)
        y = self.relu_1(y)
        y = self.maxpool(y)
        y = self.residual_blocks(y)
        y=  self.extra(y)
        if self.include_top:
            y = self.GAP(y)
            y = self.fc(y)
        return y

## saving weights
model = ResNet34()
#model.build((1, 224, 224, 3))
print(model.layers)

