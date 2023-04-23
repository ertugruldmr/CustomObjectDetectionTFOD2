import gradio as gr
import tensorflow as tf
import numpy as np
import cv2

# Load the trained TensorFlow Object Detection model
model_path = "/content/fine_tuned_ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8"
model = tf.saved_model.load(model_path)

# Define the input and output signatures of the model
inputs = model.signatures['serving_default'].inputs
outputs = model.signatures['serving_default'].outputs

# defining the params
title='TFOD Custom Object Detection (Covid Mask Detection)'
description='Detect objects in an image using a fine-tuned ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8 model via TFOD 2.0 API'
labels = {2:"With mask",  3:"No mask"}
colors = {2:(0, 255, 0), 3:(255, 0, 0)}

# Define a function to make predictions using the model
def predict(input_image):
    # Preprocess the input image
    input_tensor = tf.convert_to_tensor(input_image)
    input_tensor = input_tensor[tf.newaxis, ...]
    input_tensor = tf.image.resize(input_tensor, [320, 320])
    input_tensor = tf.cast(input_tensor, tf.uint8)

    # Make a prediction using the model
    output_dict = model(input_tensor)

    # Process the output dictionary to extract the bounding boxes and classes
    boxes = output_dict['detection_boxes'][0].numpy()
    classes = output_dict['detection_classes'][0].numpy().astype(np.uint8) + 1
    scores = output_dict['detection_scores'][0].numpy()

    # Draw the bounding boxes on the input image
    output_image = np.array(input_image)
    for box, cls, score in zip(boxes, classes, scores):
        if score > 0.5:
            ymin, xmin, ymax, xmax = box
            ymin = int(ymin * output_image.shape[0])
            xmin = int(xmin * output_image.shape[1])
            ymax = int(ymax * output_image.shape[0])
            xmax = int(xmax * output_image.shape[1])

            cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), colors[cls], 2)
            cv2.putText(output_image, labels[cls], (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[cls], 2, cv2.LINE_AA)

    # Return the output image
    return output_image
# declerating the params
comp_parms = {
  "fn":predict, 
  "inputs":'image',
  "outputs":'image',
  "title":title,
  "description":description,
  "examples":"examples"
}
demo = gr.Interface(**comp_parms)
    # Define the Gradio interface

# Launching the demo
if __name__ == "__main__":
    demo.launch()
