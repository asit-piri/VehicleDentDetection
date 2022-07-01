import io
import base64
from PIL import Image
from backend.unet_predictor import get_prediction as unet_predictor
from backend.mask_rcnn_predictor import get_prediction as mask_rcnn_predictor

def get_prediction(data, model):
    # write the string in a temp file
    temp_location = 'temp/unet_temp.jpeg'
    with open('temp/tmp', 'w') as op:
        op.write(data)
    # read the file as binary format
    with open('temp/tmp', 'rb') as ip:
        img_bin = ip.read()
    # decode the binary and store the resultant data as jpeg
    with open(temp_location, 'wb') as f:
        f.write(base64.decodestring(img_bin))
    
    # get masked prediction
    if model == 'unet':
        masked_image = unet_predictor(temp_location)
        print('Executing UNet model')
    elif model == 'mask_rcnn':
        masked_image = mask_rcnn_predictor(temp_location)
        print('Executing Mask RCNN model')

    # convert the prediction to encoded form for response 
    masked_image_pil = Image.fromarray(masked_image.astype('uint8'))
    file_object = io.BytesIO()
    masked_image_pil.save(file_object, 'JPEG')
    encoded_img = base64.b64encode(file_object.getvalue())
    return encoded_img.decode('utf-8')
