import tensorflow as tf

def conv(filters, kernel_size, strides):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            use_bias=False
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU()
    ])

def predict_flow():
    return tf.keras.layers.Conv2D(filters=2, kernel_size=5, strides=1, padding='same', use_bias=False)

def upconv(out_channels):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2DTranspose(
            filters=out_channels,
            kernel_size=4,
            strides=2,
            padding='same', 
            use_bias=False
        ),
        tf.keras.layers.ReLU()
    ])

class FlowNet(tf.keras.Model):
    def __init__(self):
        super(FlowNet, self).__init__()
        self.conv_1 = conv(filters=64, kernel_size=7, strides=2)
        self.conv_2 = conv(filters=128, kernel_size=5, strides=2)
        self.conv_3 = conv(filters=256, kernel_size=5, strides=2)
        self.conv_3_1 = conv(filters=256, kernel_size=3, strides=1)
        self.conv_4 = conv(filters=512, kernel_size=3, strides=2)
        self.conv_4_1 = conv(filters=512, kernel_size=3, strides=1)
        self.conv_5 = conv(filters=512, kernel_size=3, strides=2)
        self.conv_5_1 = conv(filters=512, kernel_size=3, strides=1)
        self.conv_6 = conv(filters=1024, kernel_size=3, strides=2)

        self.predict_6 = predict_flow()
        self.predict_5 = predict_flow()  
        self.predict_4 = predict_flow()  
        self.predict_3 = predict_flow()  
        self.predict_2 = predict_flow()

        self.upconv5=upconv(512)
        self.upconv4=upconv(256)
        self.upconv3=upconv(128)
        self.upconv2=upconv(64)

        self.upconvflow6 = tf.keras.layers.Conv2DTranspose(filters=2,kernel_size=4,strides=2,padding='same',use_bias=False)
        self.upconvflow5 = tf.keras.layers.Conv2DTranspose(filters=2,kernel_size=4,strides=2,padding='same',use_bias=False)
        self.upconvflow4 = tf.keras.layers.Conv2DTranspose(filters=2,kernel_size=4,strides=2,padding='same',use_bias=False)
        self.upconvflow3 = tf.keras.layers.Conv2DTranspose(filters=2,kernel_size=4,strides=2,padding='same',use_bias=False)

    def call(self,inputs, training=False):
        x1=self.conv_1(inputs, training=training)
        x2=self.conv_2(x1, training=training)
        x3=self.conv_3(x2, training=training)
        x3_1=self.conv_3_1(x3, training=training)
        x4=self.conv_4(x3_1, training=training)
        x4_1=self.conv_4_1(x4, training=training)
        x5=self.conv_5(x4_1, training=training)
        x5_1=self.conv_5_1(x5, training=training)
        x6=self.conv_6(x5_1, training=training)

        flow6=self.predict_6(x6)
        up_flow6 = self.upconvflow6(flow6)
        out_upconv5 = self.upconv5(x6, training=training)
        concat5  = tf.concat([x5_1, out_upconv5, up_flow6], axis=-1)
        
        flow5 = self.predict_5(concat5)
        up_flow5 = self.upconvflow5(flow5)
        out_upconv4 = self.upconv4(x5_1, training=training)
        concat4 = tf.concat([x4_1, out_upconv4, up_flow5], axis=-1)

        flow4 = self.predict_4(concat4)
        up_flow4 = self.upconvflow4(flow4)
        out_upconv3 = self.upconv3(x4_1, training=training)
        concat3 = tf.concat([x3_1, out_upconv3, up_flow4], axis=-1)

        flow3 = self.predict_3(concat3)
        up_flow3 = self.upconvflow3(flow3)
        out_upconv2 = self.upconv2(x3_1, training=training)
        concat2 = tf.concat([x2, out_upconv2, up_flow3], axis=-1)

        finalflow = self.predict_2(concat2)

        if training:
            return finalflow, flow3, flow4, flow5, flow6
        else:
            return finalflow

