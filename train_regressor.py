import argparse
import numpy
import numpy as np
import json
import os
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LinearRegression
from xgboost.sklearn import XGBRegressor
import pandas as pd
from config import PARAMS
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from utils import save_model

mid_term_window = [1, 2, 3, 4, 5, 6, 7]
mid_term_step = [0.5, 1, 1.5, 2, 2.5, 3, 3.5]

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



def train_pyaudio(gt_file, f_dir, task):
    with open(gt_file) as fp:
        ground_truth = json.load(fp)


    best_params, best_scores, steps, windows, models = [], [], [], [], []
    for mid_window, step in zip(mid_term_window, mid_term_step):
        X, y = [], []
        for ig, g in enumerate(ground_truth):
            #        if ig > 60:
            #            break
            npy_pyaudioanalysis = os.path.join(f_dir, ground_truth[ig]['name'].replace('.wav', '')) + '_pyaudioanalysis.npy'
            gt_overall = ground_truth[ig][task]['mean']
            if gt_overall is not None:
                fv = np.load(npy_pyaudioanalysis)
                mtf = mid_feature_extraction(fv, mid_window, step, 0.04, 0.02)
                lt_fv = np.mean(mtf, axis=1)
                #print(lt_fv.shape, gt_overall)
                X.append(lt_fv)
                y.append(gt_overall)
        X = np.array(X)
        y = np.array(y)

        print(X.shape, y.shape)
        scorer = make_scorer(mean_absolute_error, greater_is_better=False)
        K=10
        '''[{'gradientboostingregressor__learning_rate': [.001, 0.01, .1, 0.5], 'gradientboostingregressor__n_estimators': [1, 64, 100, 500, 1000, 2000],
                           'gradientboostingregressor__max_depth': [1, 2, 4, 20], 'gradientboostingregressor__subsample':[.5, .75, 1]}]'''
        ''''xgbregressor__n_estimators': [100, 500, 900, 1100],
            'xgbregressor__max_depth': [2, 3, 5, 10],
            'xgbregressor__learning_rate': [0.05, 0.1, 0.15],
            'xgbregressor__min_child_weight': [1, 2, 3]'''
        algo_models = [
            ('baseline', DummyRegressor(), {}),
            ('svr', SVR(epsilon=0.01),
             [{'svr__kernel': ['rbf', 'linear'], 'svr__C': [0.001, 0.01, 0.1, 0.5, 1, 5, 10]}]),
            ('gboosting', GradientBoostingRegressor(), {'gradientboostingregressor__learning_rate': [.001, 0.01, .1, 0.5], 'gradientboostingregressor__n_estimators': [1, 64, 100, 500, 1000, 2000]}),
            ('bayesian', BayesianRidge(),
             {'bayesianridge__alpha_init': [1, 1.1, 1.2, 1.3, 1.4],
              'bayesianridge__lambda_init': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]}),
            ('linearreg', LinearRegression(), {}),
            ('XGB', XGBRegressor(), {'xgbregressor__n_estimators': [100, 500, 900, 1100]})
        ]

        for name, model, params in algo_models:
            pipe = make_pipeline(StandardScaler(), model)
            gridsearch = GridSearchCV(pipe, params, cv=K, scoring=scorer)
            gridsearch.fit(X, y)
            print("\n The best " + name + " error:\n", gridsearch.best_score_)
            best_params += [gridsearch.best_params_]
            print(gridsearch.best_params_)
            best_scores += [gridsearch.best_score_]
            steps.append(step)
            windows.append(mid_window)
        models = models + ['baseline', 'svr', 'gradient_boosting', 'bayesian_ridge', 'linear_regression', 'xgboost']
    df2 = pd.DataFrame(list(zip(models, windows, steps, best_params, best_scores)), columns=['model', 'mid_term_window_size', 'mid_term_window_step', 'best_params', 'best_score'])
    df2.to_csv('results_' + task + '.csv', sep='\t', index=False)


    #plt.plot(test_preds, y_test, '*')
    #plt.show()



def train_pyaudio_seg(gt_file, f_dir, model_name, task, cross_val=False, out_model=None):
    with open(gt_file) as fp:
        ground_truth = json.load(fp)


    mid_window = PARAMS['mid_window']
    step = PARAMS['mid_step']
    X, y, grps = [], [], []
    counter = 0
    for ig, g in enumerate(ground_truth):
        #        if ig > 60:
        #            break
        npy_pyaudioanalysis = os.path.join(f_dir, ground_truth[ig]['name'].replace('.wav', '')) + '_pyaudioanalysis.npy'
        gt_overall = ground_truth[ig][task]['mean']
        if gt_overall is not None:
            fv = np.load(npy_pyaudioanalysis)
            mtf = mid_feature_extraction(fv, mid_window, step, 0.04, 0.02)
            counter += 1
            for segment in mtf.T:
                X.append(np.array(segment))
                y.append(gt_overall)
                grps.append(counter)

    X = np.array(X)
    y = np.array(y)

    if model_name == 'svr':
        model = SVR(epsilon=0.01, C=PARAMS['svr']['C'], kernel=PARAMS['svr']['kernel'])
    elif model_name == 'gboosting':
        model = GradientBoostingRegressor(learning_rate=PARAMS['gboosting']['learning_rate'],
                                          n_estimators=PARAMS['gboosting']['n_estimators'])
    elif model_name == 'bayesian':
        model = BayesianRidge(alpha_init=PARAMS['bayesian']['alpha_init'],
                              lambda_init=PARAMS['bayesian']['lambda_init'])
    elif model_name == 'XGB':
        model = XGBRegressor(n_estimators=PARAMS['XGB']['n_estimators'])
    elif model_name == 'linearreg':
        model = LinearRegression()

    print(X.shape)
    scaler = StandardScaler()
    if cross_val:
        #scorer = make_scorer(mean_absolute_error, greater_is_better=False)
        #rs = ShuffleSplit(n_splits=10, random_state=0)
        cv = GroupKFold(n_splits=10)
        errors = []
        for i, (train_index, test_index) in enumerate(cv.split(X, groups=grps)):
            print(f"Fold {i}:")
            print(f"  Train: index={train_index}")
            x_train = X[train_index, :]
            x_train = scaler.fit_transform(x_train)
            y_train = y[train_index]
            x_test = X[test_index, :]
            x_test = scaler.transform(x_test)
            y_test = y[test_index]
            groups_test = [grps[index] for index in test_index]
            model.fit(x_train, y_train)
            preds = model.predict(x_test)
            averaged_preds = []
            averaged_gr_truths = []
            print(y_test)
            for group_id in set(groups_test):
                print(group_id)
                ground_truths_of_group = [i for j, i in enumerate(y_test) if groups_test[j] == group_id]
                print(ground_truths_of_group)
                preds_of_group = [i for j, i in enumerate(preds) if groups_test[j] == group_id]
                print(preds_of_group)
                averaged_preds.append(round(numpy.average(preds_of_group), 2))
                averaged_gr_truths.append(ground_truths_of_group[0])
            error = mean_absolute_error(averaged_gr_truths, averaged_preds)
            print(error)
            errors.append(error)
        mean_error = round(numpy.average(errors), 2)
        print("The mean cross validated error on aggregated segments is: ", mean_error)
    else:
        model_dict = {}
        X = scaler.fit_transform(X)
        model.fit(X, y)
        model_dict['model'] = model
        model_dict['mid_window'] = mid_window
        model_dict['mid_step'] = step
        model_dict['scaler'] = scaler

        out_folder = PARAMS['output_path']

        if out_model is None:
            save_model(model_dict, out_folder, name="basic_regressor")
        else:
            save_model(model_dict, out_folder, out_model=out_model)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-g", "--ground_truth", type=str, nargs="?", required=True, 
        help="Ground truth and metadata json file")

    parser.add_argument(
        "-f", "--features_folder", type=str, nargs="?", required=True, 
        help="Folder where the features are stored in npy files")
    parser.add_argument(
        "-t", "--task_name", type=str, nargs="?", required=True,
        help="Task to be solved")
    parser.add_argument("-s", "--segment_level",  action='store_true', required=False,
        help="whether to use mid-term segments as separate instances")
    parser.add_argument("-m", "--model_name", required=False,
                        help="the name of the algorithm/model to be trained")
    parser.add_argument("-val", "--cross_val", required=False, action='store_true',
                        help="if cross validated is needed to split the data")
    parser.add_argument("-o", "--outputmodelname", required=False,
                        help="name to the final model to be saved")
    
    flags = parser.parse_args()
    gt_file = flags.ground_truth
    f_dir = flags.features_folder
    task = flags.task_name
    segment_level = flags.segment_level
    model_name = flags.model_name
    cross_val = flags.cross_val
    output_name = flags.outputmodelname

    if segment_level:
        train_pyaudio_seg(gt_file, f_dir, model_name, task, cross_val, output_name)
    else:
        train_pyaudio(gt_file, f_dir, task)
