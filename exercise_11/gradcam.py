import numpy as np
import tensorflow as tf
import keras
import matplotlib as mpl


def get_img_array(img, size):
    if len(img.shape) == 3:
        img = np.expand_dims(img, axis=0)
    return img


def make_gradcam_heatmap(img_array, model, base_model, last_conv_layer_name, pred_index=None):
    # Check if this is a nested model or a flat model
    is_nested = base_model is not None and any(layer.name == base_model.name for layer in model.layers)
    
    if is_nested:
        # nestedd model case (e.g.effnetv2b0)
        target_layer = base_model.get_layer(last_conv_layer_name)
        base_grad_model = keras.Model(
            base_model.inputs,
            [target_layer.output, base_model.output]
        )
        
        classifier_layers = []
        found_base = False
        for layer in model.layers:
            if layer.name == base_model.name:
                found_base = True
                continue
            if found_base:
                classifier_layers.append(layer)
        
        classifier_input = keras.Input(shape=base_model.output.shape[1:])
        x = classifier_input
        for layer in classifier_layers:
            x = layer(x)
        classifier_model = keras.Model(classifier_input, x)
        
        with tf.GradientTape() as tape:
            conv_output, base_output = base_grad_model(img_array)
            tape.watch(conv_output)
            preds = classifier_model(base_output)
            
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, conv_output)
        
    else:
        grad_model = keras.models.Model(
            model.inputs, 
            [model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, conv_output)
    

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()


def create_gradcam_visualization(img, heatmap, alpha=0.4, background_intensity=0.5):
    heatmap = np.uint8(255 * heatmap)
    jet = mpl.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img * background_intensity
    superimposed_img = keras.utils.array_to_img(superimposed_img)
    
    return superimposed_img
