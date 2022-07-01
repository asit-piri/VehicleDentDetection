//<img> tag objects
let output_image_holder = document.getElementById('output_image');
let input_image_holder = document.getElementById('input_image');
//<button> or similar tag objects
let predict_button = document.getElementById('predict_button');
let image_upload_button = document.getElementById('input_button');
//<p> tag object to show messages
let output_text = document.getElementById('output_text')
//<select> tag object for model choice
let model_selector = document.getElementById('model_selection_dropdown')
let loading_animation = document.getElementById('loading')


image_upload_button.addEventListener('change', handleFileSelect, false);
predict_button.addEventListener('click', predict, false)

image = []

//Read the file submission
function handleFileSelect(event) {
    //Stop sbmit button from refreshing the page
    event.stopPropagation();
    event.preventDefault();
    file = event.target.files[0]
    var fileReader = new FileReader();
    fileReader.onload = function(load_event) {
        result = fileReader.result
        input_image_holder.src = result;
        image = result;
    };
    fileReader.readAsDataURL(file)
}

async function predict() {
    model_choice = model_selector.selectedOptions[0].value;
    data = {
        "image": image,
        "model": model_choice
    }
    loading_animation.style.display = 'block'; 
    const http = new XMLHttpRequest();
    const url = 'http://127.0.0.1:5000/predict'
    http.open('post', url);
    http.setRequestHeader('Content-Type', 'application/json');
    http.send(JSON.stringify(data))
    output_text.innerHTML = 'Processing!!';
    
    http.onreadystatechange = function() {
        if (this.readyState == 4) {
            loading_animation.style.display = 'none';
            if(this.status == 200) {
                output_text.innerHTML = 'Done';
                div_el = document.createElement('div');
                div_el.innerHTML = http.response;
                output_image_holder.src = div_el.firstChild.src;
            } else {
                output_text.innerHTML = 'Error Occured';
            }
        }
    }
}