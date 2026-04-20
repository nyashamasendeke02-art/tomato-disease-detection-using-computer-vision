import yaml
import tensorflow as tf

#load the config file
with open("config.yaml", "r") as ff:
    config = yaml.safe_load(f)
#use the config to build the model
base_model = tf.keras.applications.MobileNetV3Large(
    input_shape=tuple(config['model']['input_shape']),
    weights=config['model']['weights']
    include_top=config['model']['include_top'],
    alpha=config['model']['alpha']
    ) 


if __name__  == "__main__":
    #create the model using config
    model, config = build_mobilenet("config.yaml")

    #compile it(setting the 'engine' settings)
    model.compile(
        optimizer=config['train']['optimizer'],
        loss=config['train']['loss_function'],
        metrics=config['train']['metrics']
    )
    model.summary()# Prints the architecture to the console
'''
Summary of the Flow
Python reads config.yaml.
TensorFlow downloads the pre-trained MobileNetV3Large weights.
The script chops off the head of the model and adds a new one designed for your specific number of classes (e.g., if you are detecting 3 types of defects, it adds a 3-unit layer).
The Main Loop prepares the model for training or real-time inference.

'''