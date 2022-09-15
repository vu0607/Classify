from Preprocessing_Data import ProcessingDataset
from ModelCNN import define_model, call_back
import argparse


def parser_args():
    """
    Initiating argument parser
    :return: args
    """
    parser = argparse.ArgumentParser(description='Hyperparameter Train model')
    parser.add_argument('--epochs', default=2, type=int, help='Number of Epochs')
    parser.add_argument('--batch_size', default=8, type=int, help='Number of batch size')
    args = parser.parse_args()
    return args


def main(arg):
    process = ProcessingDataset()
    train_df, val_df = process.split_dataset()
    total_train = train_df.shape[0]
    total_validate = val_df.shape[0]
    train_generator, validation_generator = process.generate_data()
    model = define_model()
    model.fit_generator(
        train_generator,
        epochs=arg.epochs,
        validation_data=validation_generator,
        validation_steps=total_validate // arg.batch_size,
        steps_per_epoch=total_train // arg.batch_size,
        callbacks=call_back()
    )
    model.save_weights("model.h5")


if __name__ == '__main__':
    arg = parser_args()
    main(arg)
    print('x')

