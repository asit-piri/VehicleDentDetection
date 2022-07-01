import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import PIL.Image

class_names = ['scratch', 'dent', 'damage']
# generate a list that contains one color for each class
colors = sns.color_palette(None, len(class_names))

# print class name - normalized RGB tuple pairs
# the tuple values will be multiplied by 255 in the helper functions later
# to convert to the (0,0,0) to (255,255,255) RGB values you might be familiar with
# for class_name, color in zip(class_names, colors):
#    print(f'{class_name} -- {color}')

# Visualization Utilities


def _fuse_with_pil(images):
    '''
    Creates a blank image and pastes input images

    Args:
    images (list of numpy arrays) - numpy array representations of the images to paste

    Returns:
    PIL Image object containing the images
    '''

    widths = (image.shape[1] for image in images)
    heights = (image.shape[0] for image in images)
    total_width = sum(widths)
    max_height = max(heights)

    new_im = PIL.Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        pil_image = PIL.Image.fromarray(np.uint8(im))
        new_im.paste(pil_image, (x_offset,0))
        x_offset += im.shape[1]

    return new_im

def give_color_to_annotation(annotation):
    '''
    Converts a 3-D annotation of shape (height, width, number_of_classes) to a numpy array with shape (height, width, 3) where
    the third axis represents the color channel. The label values are multiplied by
    255 and placed in this axis to give color to the annotation

    Args:
    annotation (numpy array) - label map array

    Returns:
    the annotation array with an additional color channel/axis
    '''
    seg_img = np.zeros((annotation.shape[0], annotation.shape[1], 3)).astype('float')

    for c in range(len(class_names)):
        segc = (annotation == c+1)
        # print('segc', segc.shape)
        a = np.array(segc*( colors[c][0] * 255.0)).reshape((annotation.shape[0], annotation.shape[1]))
        b = np.array(segc*( colors[c][1] * 255.0)).reshape((annotation.shape[0], annotation.shape[1]))
        c = np.array(segc*( colors[c][2] * 255.0)).reshape((annotation.shape[0], annotation.shape[1]))
        # print('segc*colors', a.shape)
        seg_img[:,:,0] += a
        seg_img[:,:,1] += b
        seg_img[:,:,2] += c

    return seg_img.astype('int')


def _show_annotation_and_image(image, annotation):
    '''
    Displays the image and its annotation side by side

    Args:
    image (numpy array) -- the input image
    annotation (numpy array) -- the label map
    '''
    #new_ann = np.argmax(annotation, axis=2)
    seg_img = give_color_to_annotation(annotation)

    image = image + 1
    image = image * 127.5
    image = np.uint8(image)
    images = [image, seg_img]

    images = [image, seg_img]
    fused_img = _fuse_with_pil(images)
    plt.imshow(fused_img)


def list_show_annotation(dataset):
    '''
    Displays images and its annotations side by side

    Args:
    dataset (tf Dataset) - batch of images and annotations
    '''

    ds = dataset.unbatch()
    ds = ds.shuffle(buffer_size=100)

    plt.figure(figsize=(25, 15))
    plt.title("Images And Annotations")
    plt.subplots_adjust(bottom=0.1, top=0.9, hspace=0.05)

    # we set the number of image-annotation pairs to 9
    # feel free to make this a function parameter if you want
    for idx, (image, annotation) in enumerate(ds.take(9)):
        plt.subplot(3, 3, idx + 1)
        plt.yticks([])
        plt.xticks([])
        _show_annotation_and_image(image.numpy(), annotation.numpy())


def visualize_single_data(mod_image):
    plt.figure(figsize=(25, 15))
    plt.title("Image And its Annotation")
    plt.subplots_adjust(bottom=0.1, top=0.9, hspace=0.05)
    plt.yticks([])
    plt.xticks([])
    plt.imshow(mod_image)