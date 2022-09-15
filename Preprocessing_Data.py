import pandas as pd
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3


class ProcessingDataset():
    def __init__(self, batch_size: int = 16):
        self.batch_size = batch_size

    @classmethod
    def make_data(self, draw=False):
        file_dogs = os.listdir("./dataset/train/dogs")
        file_cats = os.listdir("./dataset/train/cats")
        filenames = file_cats + file_dogs
        file_path = []
        categories = []

        for filename in filenames:
            if "dog" in filename:
                categories.append(1)
                file_name = './dataset/train/dogs/' + str(filename)
                file_path.append(file_name)
            else:
                categories.append(0)
                file_name = './dataset/train/cats/' + str(filename)
                file_path.append(file_name)

        df = pd.DataFrame({
            'filename': file_path,
            'category': categories
        })
        df.head()
        if draw:
            label = ['cat', 'dog']
            df1 = pd.DataFrame({
                'fileName': file_path,
                'label': label
            })

            df1['label'].value_counts().plot.bar()
            plt.show()

        return df

    def split_dataset(self):
        df = self.make_data()
        df["category"] = df["category"].replace({0: 'cat', 1: 'dog'})
        train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)  # split and shuffle data
        train_df = train_df.reset_index(drop=True)  # sap xep lai thu tu
        validate_df = validate_df.reset_index(drop=True)
        return train_df, validate_df

    def generate_data(self):
        train_df, validate_df = self.split_dataset()

        train_datagen = image.ImageDataGenerator(
            rotation_range=15,
            rescale=1. / 255,
            shear_range=0.1,
            zoom_range=0.2,
            horizontal_flip=True,
            width_shift_range=0.1,
            height_shift_range=0.1
        )
        train_generator = train_datagen.flow_from_dataframe(
            train_df,
            x_col='filename',
            y_col='category',
            target_size=IMAGE_SIZE,
            class_mode='categorical',
            color_mode='rgb',
            shuffle=True,
            batch_size=self.batch_size
        )
        validation_datagen = image.ImageDataGenerator(rescale=1. / 255)

        validation_generator = validation_datagen.flow_from_dataframe(
            validate_df,
            x_col='filename',
            y_col='category',
            target_size=IMAGE_SIZE,
            class_mode='categorical',
            color_mode='rgb',
            batch_size=self.batch_size
        )
        return train_generator, validation_generator


# if __name__ == '__main__':
#     my_class = ProcessingDataset()
#     train_df, validate_df = my_class.split_dataset()
#     print(train_df)
