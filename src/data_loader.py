import tensorflow as tf
import yaml

def get_data_loader(config_path):
    #1.load your configuration 
    with open(config_path , "r") as f:
        cfg = yaml.safe_load(f)
        #Extract settings for easy reading
        img_size = tuple(cfg['model']['input_shape'][:2])#takes (224, 224)
        batch_size  cfg['data']['batch_size']

        #2.  Create the training dataset
        train_ds  tf.keras.utils.image_dataset_from_directory(
            cfg['data']['train_path']
            image_sizeimg_size,
            batch_size=batch_size,
            label_mode='categorical'
            shuffle = True

            #3.create the validation Dataset
            val_ds  = tf.keras.utils.image_dataset_from_directory(
                cg['data']['val_path'],
                image_size=img_size,
                batch_sizzebatch_size,,
                label_mode='categoricsl',
                shuffle=False
            )
            #4.MobileNet3 specific preprocessing
            # MobileNetV3 expects pixels to be scaled (usually done internally, 
             # but we apply the official preprocessing layer to be safe).
             preprocess_input = tf.keras.applications.mobilenet_v3.preprocess_input
    
            train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
             val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y))

             # 5. Performance Optimization
             # 'prefetch' overlaps the training and data augmentation
             train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
             val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

             return train_ds, val_ds
            
            