class CustomModel:
    def __init__(self, params=None):
        self._model = self._segmentation_model()
        self._params = params
        if params is not None:
            self.set_hyperparameters(params)

    
    def _segmentation_model(self):
        print('_segmentation_model function not overriden')
        print('Please override _segmentation_model function')
        pass
    
    
    def set_hyperparameters(self, params=None):
        print('set_hyperparamaters function not overriden')
        print('Please override set_hyperparamaters function')
        pass

    
    def add_checkpoint(self, checkpoint_path, checkpoint_duration):
        print('add_checkpoint function not overriden')
        print('Please override add_checkpoint function')
        pass
    
    
    def _compile_model(self):
        print('_compile_model function not overriden')
        print('Please override _compile_model function')
        pass
    
    
    def train(self):
        print('train function not overriden')
        print('Please override train function')
        pass


    def predict(self, data):
        print('predict function not overriden')
        print('Please override predict function')
        pass