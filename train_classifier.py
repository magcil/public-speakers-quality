
import argparse
import numpy as np
import json
import os
import sklearn.svm

def mid_feature_extraction(short_features, mid_window, mid_step,
                           short_window, short_step):

    n_stats = 2
    n_feats = len(short_features)
    #mid_window_ratio = int(round(mid_window / short_step))
    mid_window_ratio = round((mid_window -
                              (short_window - short_step)) / short_step)
    mt_step_ratio = int(round(mid_step / short_step))

    mid_features, mid_feature_names = [], []
    for i in range(n_stats * n_feats):
        mid_features.append([])
        mid_feature_names.append("")

    # for each of the short-term features:
    for i in range(n_feats):
        cur_position = 0
        num_short_features = len(short_features[i])

        while cur_position < num_short_features:
            end = cur_position + mid_window_ratio
            if end > num_short_features:
                end = num_short_features
            cur_st_feats = short_features[i][cur_position:end]

            mid_features[i].append(np.mean(cur_st_feats))
            mid_features[i + n_feats].append(np.std(cur_st_feats))
            cur_position += mt_step_ratio
    mid_features = np.array(mid_features)
    mid_features = np.nan_to_num(mid_features)
    return mid_features

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g", "--ground_truth", type=str, nargs="?", required=True, 
        help="Ground truth and metadata json file")

    parser.add_argument(
        "-f", "--features_folder", type=str, nargs="?", required=True, 
        help="Folder where the features are stored in npy files")

    flags = parser.parse_args()
    gt_file = flags.ground_truth
    f_dir = flags.features_folder

    with open(gt_file) as fp:
        ground_truth = json.load(fp)


    X, y = [], []

    for ig, g in enumerate(ground_truth):
#        if ig > 60:
#            break
        npy_pyaudioanalysis = os.path.join(f_dir, ground_truth[ig]['name'].replace('.wav', '')) + '_pyaudioanalysis.npy'
        gt_overall = ground_truth[ig]['overall']['mean']
        if gt_overall is not None:
            fv = np.load(npy_pyaudioanalysis)
            mtf = mid_feature_extraction(fv, 2, 2, 0.04, 0.02)
            print(fv.shape)
            print(mtf.shape)
            lt_fv = np.mean(mtf, axis=1)
            print(lt_fv.shape, gt_overall)
            X.append(lt_fv)
            y.append(gt_overall)
    X = np.array(X)
    y = np.array(y)

    print(X.shape, y.shape)

    X_train = X[::2]
    y_train = y[::2]
    X_test = X[1::2]
    y_test = y[1::2]

    baseline_decision = y_train.mean()

    svm = sklearn.svm.SVR(C=1, kernel="rbf")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    svm.fit(X_train, y_train)
    train_preds = svm.predict(X_train)
    test_preds = svm.predict(X_test)

    train_err = np.mean(np.abs(train_preds - y_train))
    test_err = np.mean(np.abs(test_preds - y_test))

    print(f"training error {train_err}")   
    print(f"testing error {test_err}")   
    print(f'baseline error {np.mean(np.abs(baseline_decision - y_test))}')

    import matplotlib.pyplot as plt
    plt.plot(test_preds, y_test, '*')
    plt.show()
