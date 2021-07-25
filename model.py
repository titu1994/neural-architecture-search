from keras.models import Model
from keras.layers import Input, Dense, Conv2D, GlobalAveragePooling2D

# generic model design
def model_fn(actions):
    # unpack the actions from the list
    kernel_1, filters_1, kernel_2, filters_2, kernel_3, filters_3, kernel_4, filters_4 = actions

    ip = Input(shape=(32, 32, 3))
    x = Conv2D(filters_1, nb_row=kernel_1, nb_col=kernel_1, subsample=(2, 2), border_mode='same', activation='relu')(ip)
    x = Conv2D(filters_2, nb_row=kernel_2, nb_col=kernel_2, subsample=(1, 1), border_mode='same', activation='relu')(x)
    x = Conv2D(filters_3, nb_row=kernel_3, nb_col=kernel_3, subsample=(2, 2), border_mode='same', activation='relu')(x)
    x = Conv2D(filters_4, nb_row=kernel_4, nb_col=kernel_4, subsample=(1, 1), border_mode='same', activation='relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(10, activation='softmax')(x)

    model = Model(ip, x)
    return model
