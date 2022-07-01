import os
import cv2
import numpy as np
import tensorflow as tf


augmentation_multiplier=16

def get_testable_data(image_path, height, width):
    ann = cv2.imread(image_path)
    ann = ann.astype(np.float)
    ann = cv2.resize(ann, (height, width), interpolation=cv2.INTER_NEAREST)
    ann = np.reshape(ann, (ann.shape[0], ann.shape[1], 3))
    ann = ann/127.5
    ann -= 1
    data = [ann]
    return np.array(data)

def _read_annotations_nparray(paths, height, width):
    annotations = []
    for path in paths:
        ann = np.load(path)
        ann = ann.astype(np.float)
        ann = cv2.resize(ann, (height, width), interpolation=cv2.INTER_NEAREST)
        ann = np.reshape(ann, (ann.shape[0], ann.shape[1], 1))
        annotations.append(ann)
    annotations = np.array(annotations, dtype=np.float)#, dtype='object')
    return annotations


def _apply_augmentation(image, annotation, augmentation_threshold):
    p_rotate = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_flip = tf.random.uniform([], 0, 1.0, dtype=tf.float32)

    def rotate(image, annotation, num_90=1):
        image = tf.image.rot90(image, k=num_90)
        annotation = tf.image.rot90(annotation, k=num_90)
        return image, annotation

    def flip(image, annotation, hor_flip=True, ver_flip=True):
        if hor_flip:
            image = tf.image.flip_left_right(image)
            annotation = tf.image.flip_left_right(annotation)
        if ver_flip:
            image = tf.image.flip_up_down(image)
            annotation = tf.image.flip_up_down(annotation)
        return image, annotation

    if p_rotate >= augmentation_threshold:
        p_rotate = tf.random.uniform([], 0, 3, dtype=tf.int32)
        image, annotation = rotate(image, annotation, p_rotate)
    
    if p_flip >= augmentation_threshold:
        her_flip = tf.cast(tf.random.uniform([], 0, 1, dtype=tf.int32), dtype=bool)
        ver_flip = tf.cast(tf.random.uniform([], 0, 1, dtype=tf.int32), dtype=bool)
        image, annotation = flip(image, annotation, her_flip, ver_flip)

    return image, annotation


def _get_data_preprocessor(height, width, for_training=False, apply_augmentation=False, augmentation_threshold=0.4):
    '''
    Preprocesses the dataset by:
    * resizing the input image and label maps
    * normalizing the input image pixels
    * reshaping the label maps from (height, width, 1) to (height, width, 12)
    '''
    def _preprocess_data(t_filename, anno_raw):
        ''' 
        Args:
        t_filename (string) -- path to the raw input image
        a_filename (string) -- path to the raw annotation (label map) file
        height (int) -- height in pixels to resize to
        width (int) -- width in pixels to resize to

        Returns:
        image (tensor) -- preprocessed image of shape [h, w, 3]
        annotation (tensor) -- preprocessed annotation of shape [h, w, 1]
        '''
        
        # Convert image and mask files to tensors 
        img_raw = tf.io.read_file(t_filename)
        image = tf.image.decode_jpeg(img_raw)
        
        # Resize image and segmentation mask
        image = tf.image.resize(image, (height, width))
        image = tf.reshape(image, (height, width, 3,))
        # annotation = cv2.resize(anno_raw, target_dims, interpolation=cv2.INTER_NEAREST)
        # annotation = annotation.astype('float32')
        annotation = tf.convert_to_tensor(anno_raw)
        annotation = tf.cast(anno_raw, dtype=tf.int32)
        # annotation = tf.image.resize(annotation, (HEIGHT, WIDTH,))
        annotation = tf.reshape(annotation, (height, width, 1,))
        
        # apply annotation if chosen above
        if apply_augmentation and for_training:
            image, annotation = _apply_augmentation(image, annotation, augmentation_threshold)
        
        stack_list = []

        # Reshape segmentation masks
        '''for c in range(len(class_names)):
            mask = tf.equal(annotation[:, :, 0], tf.constant(c))
            stack_list.append(tf.cast(mask, dtype=tf.int32))

        annotation = tf.stack(stack_list, axis=2)
        '''
        # Normalize pixels in the input image
        image = image/127.5
        image -= 1
        
        assert image.shape == (height, width, 3)
        assert annotation.shape == (height, width, 1)

        return image, annotation

    return _preprocess_data


def _get_dataset_slice_paths(image_dir, mask_dir):
    '''
    generates the lists of image and label map paths

    Args:
    image_dir (string) -- path to the input images directory
    mask_dir (string) -- path to the label map directory

    Returns:
    image_paths (list of strings) -- paths to each image file
    label_map_paths (list of strings) -- paths to each label map
    '''
    image_file_list = os.listdir(image_dir)
    label_map_file_list = os.listdir(mask_dir)
    image_paths = [os.path.join(image_dir, fname) for fname in image_file_list]
    label_map_paths = [os.path.join(mask_dir, fname) for fname in label_map_file_list]
    
    image_paths.sort()
    label_map_paths.sort()

    return image_paths, label_map_paths


def _get_dataset(image_path_list, mask_path_list, height, width, batch_size, for_training=False, epochs=2, apply_augmentation=False, augmentation_threshold=0.4):
    '''
    Prepares shuffled batches of the training set.

    Args:
    image_path_list (list of strings) -- paths to each image file in the train set
    mask_path_list (list of strings) -- paths to each label map in the train set

    Returns:
    tf Dataset containing the preprocessed train set
    '''
    
    dataset = tf.data.Dataset.from_tensor_slices((image_path_list, _read_annotations_nparray(mask_path_list, height, width)))
    
    if for_training and apply_augmentation:
        dataset = dataset.repeat(augmentation_multiplier)   # repeatation for augmentation
    
    preprocessor = _get_data_preprocessor(height, width, for_training, apply_augmentation)
    dataset = dataset.map(preprocessor, num_parallel_calls=tf.data.AUTOTUNE)
    #dataset = dataset.shuffle(100, reshuffle_each_iteration=True)
    
    dataset = dataset.repeat(epochs)
    
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def execute_data_pipeline(image_dir, mask_dir, height, width, batch_size, for_training=False, epochs=2, apply_augmentation=False, augmentation_threshold=0.4):
    # load the n data in the directory 
    image_path_list, mask_paths_list = _get_dataset_slice_paths(image_dir, mask_dir)
    dataset_size = len(image_path_list) # n
    # augmentation mult is by default 16 so total 16n
    if for_training and apply_augmentation:
        data_count = dataset_size * augmentation_multiplier # 'augmentation_multiplier' types of augmentation is being added so dataset is 'augmentation_multiplier'x the size
    else:
        data_count = dataset_size
    dataset = _get_dataset(image_path_list, mask_paths_list, height, width, batch_size, for_training, epochs, apply_augmentation, augmentation_threshold)
    return dataset, data_count


def _draw_alpha(img, mask, val, RGB, alpha=0.5):
    new_mask = np.zeros((mask.shape[0], mask.shape[1], 3))
    new_mask[mask == val] = RGB
    new_mask = new_mask.astype('uint8')
    return cv2.addWeighted(img, 1, new_mask, 0.5, 0)


def get_masked_image(image, prediction):
    '''
    This is for applying the mask on the input image.
    The mask is reshaped to match the input image and then applied on it.
    args:
        image: numpy array of shape (h, w, 3)
        prediction: the prediction output by the model of shape (1, 224, 224, 1)
    '''
    predicted_masks = np.argmax(prediction, axis=3)
    prediction_mask = predicted_masks[0].astype('uint8')
    resized_mask = cv2.resize(np.array(prediction_mask), dsize=(image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
    
    mod_image = _draw_alpha(image, resized_mask, 1, (255, 255, 0), 0.01)
    mod_image = _draw_alpha(mod_image, resized_mask, 2, (255, 165, 0), 0.01)
    mod_image = _draw_alpha(mod_image, resized_mask, 3, (255, 0, 0), 0.01)
    
    return mod_image