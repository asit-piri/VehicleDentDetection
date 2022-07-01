from flask import Flask
from flask import redirect
from flask import url_for
from flask import request
from flask import render_template
from flask import make_response
from flask import jsonify
from flask import send_file
from backend.predict import get_prediction


app = Flask(__name__)

@app.route('/')
def root():
    return render_template('ui.html')


@app.route('/ui', methods = ['GET'])
def show_user_interface():
    return render_template('ui.html')


@app.route('/models', methods = ['GET'])
def get_model_names():
    #TODO: Automate this
    response_data = {
        "models" : ["Masked_RCNN", "UNet", "Detectron2"],
        "ids" : ["masked_rcnn", "unet", "detectron2"]
    }
    response = make_response(jsonify(response_data))
    return response


@app.route('/predict', methods = ['POST'])
def predict():
    image = request.json['image'][22:]
    model_choice = request.json['model']
    encoded_data = get_prediction(image, model_choice)
    #file_obj_pred = get_prediction(image, model_choice)
    # prediction = get_prediction(image, model_choice)
    # response_data = {
    #     'response': prediction
    # }
    # response = make_response(jsonify(response_data))
    # return response
    #return send_file(file_obj_pred, mimetype='image/JPEG')
    return render_template('img.html', img_data = encoded_data)


if __name__ == '__main__':
    host = '127.0.0.1'
    port = 5000
    debug = True
    options = None
    app.run(host, port, debug, options)