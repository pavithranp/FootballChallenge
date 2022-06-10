from xgboost import XGBClassifier, plot_tree, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import plot_confusion_matrix


if __name__ == '__main__':
    listdir = os.listdir('npiy')
    first = True
    for datapath in tqdm(listdir):
        if first:
            dataset = np.load('npiy/'+datapath)
            first = False
        else:
            data = np.load('npiy/'+datapath)
            dataset = np.concatenate([dataset,data],axis = 0)

    X = dataset[0:500000,:40]
    Y = dataset[0:500000,40]
    # split data into train and test sets
    seed = 5
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    # fit model no training data
    model = XGBClassifier()
    setattr(model, 'verbosity', 2)
    model.fit(X_train, y_train,verbose=True)
    print("Done")
    features = ['mean_homex', 'mean_homey', 'mean_homespeed', 'front_home', 'back_home', 'left_home', 'right_home',
                'len(home_players)', 'la_ratio', 'ra_ratio', 'ma_ratio',
                'mean_awayx', 'mean_awayy', 'mean_awayspeed', 'front_away', 'back_away', 'left_away', 'right_away',
                'len(away_players)', 'lh_ratio', 'rh_ratio', 'mh_ratio'
                                                             'refx', 'refy', 'refspeed', 'ball_pos_x', 'ball_pos_y',
                'ball_pos_z', 'prev_pos_x', 'prev_pos_y', 'prev_pos_z',
                'next_pos_x', 'next_pos_y', 'next_pos_z', 'prev_speed_x', 'prev_speed_y', 'prev_speed_z',
                'next_speed_x', 'next_speed_y', 'next_speed_z']
    plot_importance(model, fmap='xgb.fmap')
    plot_tree(model, fmap='xgb.fmap')
    plt.show()
    y_pred = model.predict(X_test)

    predictions = [round(value) for value in y_pred]
    predictions = np.array(predictions)
    # testing accuracy when there is a corner kick
    accuracy = accuracy_score(y_test[np.where(y_test != 0)], predictions[np.where(y_test != 0)])
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    # print(confusion_matrix(y_test, y_pred))
    disp = plot_confusion_matrix(model, X_test, y_test,
                                 display_labels=['no_corner','away_corner','home_corner','away_setup','home_setup'],
                                 cmap=plt.cm.Blues,)
    disp.ax_.set_title('confusion matrix')
    plt.show()
    plt.savefig('direct.png')




