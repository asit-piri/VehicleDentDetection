import os
import cv2
from unet.model.unet import unet
from unet.utils import pipeline
from unet.utils import save_model_config
from unet.utils import testing_util
from unet.utils import viz_utils

class UNet():

    def __init__(self, params=None):
        self._params={}
        # if params is not passed then load the default parameters
        if params is None:
            self._load_default_params()
            self._model = None
        else:
            '''
                if parameter is passed then
                1. load the parameters
                2. check if train_path is provided
                3. if train path is provided then load the data
                4. if load is successful then generate model
            '''
            self.load_hyperparameters(params)
            if 'train_path' in self._params and self._params['train_path'] is not None:
                self.load_data()
                if 'train_count' in self._params:
                    self._generate_model()

    def _load_default_params(self):
        #sgd = tf.keras.optimizers.SGD(lr=1E-2, momentum=0.9, nesterov=True)
        self._params = {
            'loss': 'sparse_categorical_crossentropy',
            'batch_size': 16,
            'optimizer': 'sgd',
            'epochs': 2,
            'image_shape': (500,500),
            'seed': 47,
            'apply_augmentation': True,
            'augmentation_threshold': 0.4,
            'checkpoint_path': '.',
            'train_path': None,
            'val_path': None,
            'test_path': None
        }
    
    def get_parameters(self):
        return self._params

    def load_hyperparameters(self, params):
        '''
        Load the hyperparameter which is given as a dictionary
        the input dictionary should exactly match the dictionary keys used by the whole program.
        For keys look at the self._load_default_params function
        '''
        tuple_keys = ['image_shape', 'train_path', 'val_path', 'test_path']
        for key in params:
            value = params[key]
            if key in tuple_keys and len(value) != 2:
                print('[ERROR]:Didnt load {} as the value was not a valid tuple of size 2'.format(key))
                continue
            self._params[key]=value
        
        print('[INFO]: Parameters loaded')


    
    def load_data(self, train_path=None, validation_path=None, test_path=None):
        '''
        Here we load the preprocess the dataset.
        Input: 
            [optional] train directory tuple -> (train image dir, train masks dir)
            [optional] validation directory tuple -> (validation image dir, validation masks dir)
            [optional] test directory tuple -> (test image dir, test masks dir)

        If none of the above paths are provided then the self object should contain 
        the following in the self._params dictionary 
            train_path,
            val_path,
            test_path
        '''
        if train_path is not None:
            self._params['train_path'] = train_path
        if validation_path is not None:
            self._params['val_path'] = validation_path
        if test_path is not None:
            self._params['test_path'] = test_path

        self._train_ds, self._params['train_count'] = pipeline.execute_data_pipeline(
            self._params['train_path'][0],
            self._params['train_path'][1],
            self._params['image_shape'][0],
            self._params['image_shape'][1],
            self._params['batch_size'],
            for_training=True,
            epochs=self._params['epochs'],
            apply_augmentation=self._params['apply_augmentation'],
            augmentation_threshold= self._params['augmentation_threshold']
        )

        if self._params['val_path'] is not None:
            self._val_ds, self._params['val_count'] = pipeline.execute_data_pipeline(
                self._params['val_path'][0],
                self._params['val_path'][1],
                self._params['image_shape'][0],
                self._params['image_shape'][1],
                self._params['batch_size']
            )
        else:
            self._val_ds = None
        if self._params['test_path'] is not None or test_path is not validation_path:
            self._test_ds, self._params['test_count'] = pipeline.execute_data_pipeline(
                self._params['test_path'][0],
                self._params['test_path'][1],
                self._params['image_shape'][0],
                self._params['image_shape'][1],
                self._params['batch_size']
            )
        else:
            self._test_ds = self._val_ds

        print('[INFO]:Data loaded sucessfully')


    def visualize_data(self, count=10, type='train'):
        if self._train_ds is None:
            print('[ERROR]: Training dataset has not been provided')
        viz_utils.list_show_annotation(self._train_ds)

    def _generate_model(self):
        model = unet()
        if hasattr(self, '_train_ds') and hasattr(self, '_val_ds'):
            model.set_data(self._train_ds, self._val_ds)
        model.set_hyperparameters(self._params)
        self._model = model
        print('[INFO]: model generated successfully')

    def plot_model(self):
        if self._model is None:
            self._generate_model()
        # TODO: add keras.utils.plot_model here
        pass

    def train(self):
        if self._model is None:
            self._generate_model()
        return self._model.train()

    def predict(self, image_path=None):
        if image_path is not None:
            test_image = pipeline.get_testable_data(image_path, self._params['image_shape'][0], self._params['image_shape'][1])
            image = cv2.imread(image_path)
            prediction = self._model.predict(test_image)
            masked_image = pipeline.get_masked_image(image, prediction)
            return prediction
        else:
            prediction = []
            if self._test_ds is None:
                prediction = self._model.predict(self._val_ds)
            else:
                prediction = self._model.predict(self._test_ds)
            return prediction


    def show_prediction_result(self, image_path=None, count=10):
        if image_path is None:
            prediction = self.predict()
            if self._test_ds is None:
                testing_util.show_results(self._val_ds, prediction, self._params)
            else:
                testing_util.show_results(self._test_ds, prediction, self._params)
        else:
            prediction = self.predict(image_path)
            image = cv2.imread(image_path)
            masked_image = pipeline.get_masked_image(image, prediction)
            viz_utils.visualize_single_data(masked_image)
            

    def restore_weights(self):
        if not hasattr(self, '_model'):
            self._generate_model()
        path = os.path.join(self._params['checkpoint_path'], 'checkpoint.h5')
        self._model.restore_weights(path)
        print('[INFO]: Done restoring weights')

    def save_model(self, only_weights=True):
        self._model.save_model(only_weights)
