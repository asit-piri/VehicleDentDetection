import os
import urllib
import tensorflow as tf
from unet.model.custom_model import CustomModel

class unet(CustomModel):
    
    def set_data(self, train_dataset, validation_dataset=None):
        self._train_ds = train_dataset
        # self._train_count = train_count
        self._val_ds = validation_dataset
        # self._val_count = validation_count

        
    def _segmentation_model(self):
        '''
        Defines the final segmentation model by chaining together the encoder and decoder.

        Returns:
        keras Model that connects the encoder and decoder networks of the segmentation model
        '''

        def _block(x, n_convs, filters, kernel_size, activation, pool_size, pool_stride, block_name):
            '''
            Defines a block in the VGG network.

            Args:
            x (tensor) -- input image
            n_convs (int) -- number of convolution layers to append
            filters (int) -- number of filters for the convolution layers
            activation (string or object) -- activation to use in the convolution
            pool_size (int) -- size of the pooling layer
            pool_stried (int) -- stride of the pooling layer
            block_name (string) -- name of the block

            Returns:
            tensor containing the max-pooled output of the convolutions
            '''

            for i in range(n_convs):
                x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding='same', name="{}_conv{}".format(block_name, i + 1))(x)

            x = tf.keras.layers.MaxPooling2D(pool_size=pool_size, strides=pool_stride, name="{}_pool{}".format(block_name, i+1 ))(x)

            return x

        
        def _VGG_16(image_input):
            '''
            This function defines the VGG encoder.

            Args:
            image_input (tensor) - batch of images

            Returns:
            tuple of tensors - output of all encoder blocks plus the final convolution layer
            '''

            # create 5 blocks with increasing filters at each stage. 
            # you will save the output of each block (i.e. p1, p2, p3, p4, p5). "p" stands for the pooling layer.
            x = _block(image_input,n_convs=2, filters=64, kernel_size=(3,3), activation='relu',pool_size=(2,2), pool_stride=(2,2), block_name='block1')
            p1= x

            x = _block(x,n_convs=2, filters=128, kernel_size=(3,3), activation='relu',pool_size=(2,2), pool_stride=(2,2), block_name='block2')
            p2 = x

            x = _block(x,n_convs=3, filters=256, kernel_size=(3,3), activation='relu',pool_size=(2,2), pool_stride=(2,2), block_name='block3')
            p3 = x

            x = _block(x,n_convs=3, filters=512, kernel_size=(3,3), activation='relu',pool_size=(2,2), pool_stride=(2,2), block_name='block4')
            p4 = x

            x = _block(x,n_convs=3, filters=512, kernel_size=(3,3), activation='relu',pool_size=(2,2), pool_stride=(2,2), block_name='block5')
            p5 = x

            # create the vgg model
            vgg  = tf.keras.Model(image_input , p5)

            # load the pretrained weights you downloaded earlier
            vgg_weights_path = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
            if not os.path.exists(vgg_weights_path):
                url = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
                with urllib.request.urlopen(url) as u:
                    with open(vgg_weights_path, 'wb') as f:
                        f.write(u.read())

            vgg.load_weights(vgg_weights_path) 

            # number of filters for the output convolutional layers
            n = 4096

            # our input images are 224x224 pixels so they will be downsampled to 7x7 after the pooling layers above.
            # we can extract more features by chaining two more convolution layers.
            c6 = tf.keras.layers.Conv2D( n , ( 7 , 7 ) , activation='relu' , padding='same', name="conv6")(p5)
            c7 = tf.keras.layers.Conv2D( n , ( 1 , 1 ) , activation='relu' , padding='same', name="conv7")(c6)

            # return the outputs at each stage. you will only need two of these in this particular exercise 
            # but we included it all in case you want to experiment with other types of decoders.
            return (p1, p2, p3, p4, c7)

        
        def _fcn8_decoder(convs, n_classes):
            '''
            Defines the FCN 8 decoder.

            Args:
            convs (tuple of tensors) - output of the encoder network
            n_classes (int) - number of classes

            Returns:
            tensor with shape (height, width, n_classes) containing class probabilities
            '''

            # unpack the output of the encoder
            f1, f2, f3, f4, f5 = convs

            # upsample the output of the encoder then crop extra pixels that were introduced
            o = tf.keras.layers.Conv2DTranspose(n_classes , kernel_size=(4,4) ,  strides=(2,2) , use_bias=False )(f5)
            o = tf.keras.layers.Cropping2D(cropping=(1,1))(o)

            # load the pool 4 prediction and do a 1x1 convolution to reshape it to the same shape of `o` above
            o2 = f4
            o2 = ( tf.keras.layers.Conv2D(n_classes , ( 1 , 1 ) , activation='relu' , padding='same'))(o2)

            # add the results of the upsampling and pool 4 prediction
            o = tf.keras.layers.Add()([o, o2])

            # upsample the resulting tensor of the operation you just did
            o = (tf.keras.layers.Conv2DTranspose( n_classes , kernel_size=(4,4) ,  strides=(2,2) , use_bias=False ))(o)
            o = tf.keras.layers.Cropping2D(cropping=(1, 1))(o)

            # load the pool 3 prediction and do a 1x1 convolution to reshape it to the same shape of `o` above
            o2 = f3
            o2 = ( tf.keras.layers.Conv2D(n_classes , ( 1 , 1 ) , activation='relu' , padding='same'))(o2)

            # add the results of the upsampling and pool 3 prediction
            o = tf.keras.layers.Add()([o, o2])

            # upsample up to the size of the original image
            o = tf.keras.layers.Conv2DTranspose(n_classes , kernel_size=(8,8) ,  strides=(8,8) , use_bias=False )(o)

            # append a softmax to get the class probabilities
            o = (tf.keras.layers.Activation('softmax'))(o)

            return o

        inputs = tf.keras.layers.Input(shape=(224,224,3,))
        convs = _VGG_16(image_input=inputs)
        outputs = _fcn8_decoder(convs, 4)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        return model

    
    def _compile_model(self):
        self._model.compile(loss=self._loss, optimizer=self._optimizer, metrics=['accuracy'])

    
    def set_hyperparameters(self, params=None):
        if params is None:
            return
        self._loss = params['loss']
        self._optimizer = params['optimizer']
        self._epochs = params['epochs']
        self._batch_size = params['batch_size']
        self._checkpoint_path = params['checkpoint_path']
        self.add_checkpoint(self._checkpoint_path, 2)
        if 'train_count' in params:
            self._train_count = params['train_count']
        if 'val_count' in params:
            self._val_count = params['val_count']

    
    def add_checkpoint(self, checkpoint_dir, checkpoint_duration):
        checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.h5')
        if not os.path.exists(checkpoint_dir):
            print('Checkpoint directory does not exists, creating one...')
            os.makedirs(checkpoint_dir)
        
        self._checkpoint_dir = checkpoint_dir

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='loss',
            save_best_only=True,
            save_weights_only=True,
            mode='min',
            save_freq=self._train_count*checkpoint_duration if hasattr(self, '_train_count') else 1
        )
        self._callbacks = [checkpoint_callback]
        self._checkpoint_path = checkpoint_path

        if os.path.exists(self._checkpoint_path):
            '''
                When adding checkpoint path, check if a checkpoint already exists and load it
            '''
            self._compile_model()
            self._model.load_weights(self._checkpoint_path)


    def restore_weights(self, checkpoint_path):
        self._checkpoint_path = checkpoint_path
        self._model.load_weights(self._checkpoint_path)


    def train(self):
        if self._train_ds is None:
            print('Data is not set yet')
            return
        
        steps_per_epoch = self._train_count/self._batch_size
        validation_steps = self._val_count/self._batch_size

        self._compile_model()

        if os.path.exists(self._checkpoint_path):
            self._model.load_weights(self._checkpoint_path)

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.1, patience=10, verbose=1,
            mode='auto', min_delta=0.001, cooldown=0, min_lr=1e-15)

        self._callbacks.append(reduce_lr)

        history = self._model.fit(
            self._train_ds, 
            steps_per_epoch=steps_per_epoch, 
            epochs=self._epochs, 
            validation_data=self._val_ds, 
            validation_steps=validation_steps,
            callbacks=self._callbacks,
            verbose=1
        )

        self.save_model()

        return history

    
    def predict(self, data=None):
        if data is None:
            validation_steps = self._val_count/self._batch_size
            return self._model.predict(self._val_ds, steps=validation_steps)
        else:
            return self._model.predict(data)

    
    def save_model(self, only_weights=True, path=None):
        if path is None:
            path = self._checkpoint_path
        
        if only_weights:
            self._model.save_weights(path, overwrite=True)
        else:
            self._model.save_model(path)