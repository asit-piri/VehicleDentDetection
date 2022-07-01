import numpy as np
from unet.utils import viz_utils
import matplotlib.pyplot as plt

HEIGHT = 224
WIDTH = 224
batch_size = 32


def _get_images_and_segments_test_arrays(ds):
    '''
    Gets a subsample of the val set as your test set

    Returns:
    Test set containing ground truth images and label maps
    '''
    y_true_segments = []
    y_true_images = []

    ds = ds.unbatch()
    ds = ds.batch(batch_size)

    for image, annotation in ds.take(1):
        y_true_images = image
        y_true_segments = annotation


    # y_true_segments = y_true_segments[:test_count, : ,: , :]
    # y_true_segments = np.argmax(y_true_segments, axis=3)  

    return y_true_images, y_true_segments


def _compute_metrics(y_true, y_pred):
    '''
    Computes IOU and Dice Score.

    Args:
    y_true (tensor) - ground truth label map
    y_pred (tensor) - predicted label map
    '''

    class_wise_iou = []
    class_wise_dice_score = []

    smoothening_factor = 0.00001

    for i in range(4):
        intersection = np.sum((y_pred == i) * (y_true == i))
        y_true_area = np.sum((y_true == i))
        y_pred_area = np.sum((y_pred == i))
        combined_area = y_true_area + y_pred_area
    
        iou = (intersection + smoothening_factor) / (combined_area - intersection + smoothening_factor)
        class_wise_iou.append(iou)
    
        dice_score =  2 * ((intersection + smoothening_factor) / (combined_area + smoothening_factor))
        class_wise_dice_score.append(dice_score)

    return class_wise_iou, class_wise_dice_score


def _show_predictions(image, labelmaps, titles, iou_list, dice_score_list):
    '''
    Displays the images with the ground truth and predicted label maps

    Args:
    image (numpy array) -- the input image
    labelmaps (list of arrays) -- contains the predicted and ground truth label maps
    titles (list of strings) -- display headings for the images to be displayed
    iou_list (list of floats) -- the IOU values for each class
    dice_score_list (list of floats) -- the Dice Score for each vlass
    '''
    
    true_img = viz_utils.give_color_to_annotation(labelmaps[1])
    pred_img = viz_utils.give_color_to_annotation(labelmaps[0])

    image = image + 1
    image = image * 127.5
    images = np.uint8([image, pred_img, true_img])

    metrics_by_id = [(idx, iou, dice_score) for idx, (iou, dice_score) in enumerate(zip(iou_list, dice_score_list)) if iou > 0.0]
    metrics_by_id.sort(key=lambda tup: tup[1], reverse=True)  # sorts in place

    class_names = ['background', 'scratch', 'dent', 'damage']
    display_string_list = ["{}: IOU: {} Dice Score: {}".format(class_names[idx], iou, dice_score) for idx, iou, dice_score in metrics_by_id]
    display_string = "\n\n".join(display_string_list) 

    plt.figure(figsize=(15, 4))

    for idx, im in enumerate(images):
        plt.subplot(1, 3, idx+1)
        if idx == 1:
            plt.xlabel(display_string)
        plt.xticks([])
        plt.yticks([])
        plt.title(titles[idx], fontsize=12)
        plt.imshow(im)


def _prep_data(dataset, prediction):
    # load the ground truth images and segmentation masks
    y_true_images, y_true_segments = _get_images_and_segments_test_arrays(dataset)

    # get the model prediction
    # results = model.predict(validation_dataset, steps=validation_steps)

    # for each pixel, get the slice number which has the highest probability
    predicted_masks = np.argmax(prediction, axis=3)

    reshaped_predicted_masks = np.reshape(predicted_masks, (-1, HEIGHT, WIDTH, 1)).astype('uint8')

    return y_true_images, y_true_segments, reshaped_predicted_masks


def _set_params(params):
    HEIGHT = params['image_shape'][0]
    WIDTH = params['image_shape'][1]


def show_result_for_single_image(dataset, prediction, params, integer_slider):
    _set_params(params)
    y_true_images, y_true_segments, reshaped_predicted_masks = _prep_data(dataset, prediction)
    
    iou, dice_score = _compute_metrics(y_true_segments[integer_slider].numpy(), reshaped_predicted_masks[integer_slider])

    # visualize the output and metrics
    _show_predictions(y_true_images[integer_slider].numpy(), [reshaped_predicted_masks[integer_slider], y_true_segments[integer_slider].numpy()], ["Image", "Predicted Mask", "True Mask"], iou, dice_score)


def show_results(dataset, prediction, params, n=10):
    _set_params(params)
    y_true_images, y_true_segments, reshaped_predicted_masks = _prep_data(dataset, prediction)

    for integer_slider in range(n):
        # compute metrics
        iou, dice_score = _compute_metrics(y_true_segments[integer_slider].numpy(), reshaped_predicted_masks[integer_slider])
        
        # visualize the output and metrics
        _show_predictions(y_true_images[integer_slider].numpy(), [reshaped_predicted_masks[integer_slider], y_true_segments[integer_slider].numpy()], ["Image", "Predicted Mask", "True Mask"], iou, dice_score)