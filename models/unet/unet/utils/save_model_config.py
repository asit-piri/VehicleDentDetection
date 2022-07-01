import os

def save_model_config(params, save_path):

    HEIGHT = params['HEIGHT']
    WIDTH = params['WIDTH']
    apply_data_augmentation = params['apply_augmentation']
    epochs = params['EPOCHS']
    train_count = params['TRAIN_SIZE']
    validation_count = params['VALIDATION_SIZE']
    
    if 'OPTIMIZER' in params:
        optimizer = params['OPTIMIZER']
    else:
        optimizer = 'SGD\(lr=1E-2\)'

    if 'LOSS' in params:
        loss = params['LOSS']
    else:
        loss = 'sparse_categorical_crossentropy' 

    s = '# Training Params \n\n'+\
        '- Image Dimension = {0}x{1}\n'.format(HEIGHT, WIDTH) + \
        '- Augmentation = {}\n'.format(apply_data_augmentation) + \
        '- EPOCHS = {}\n'.format(epochs) + \
        '- Training Count = {}\n'.format(train_count) + \
        '- Validation Count = {}\n'.format(validation_count) + \
        '- Optimizer = {} \n'.format(optimizer) + \
        '- Loss = {}'.format(loss)

    with open(os.path.join(save_path, 'unet_configuration.md'), 'w') as f:
        f.write(s)