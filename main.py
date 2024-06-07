import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import Pipeline

from output_filters import moving_average, smooth_spline, loess_smoothing, kalman_filter
from utils import load_data
from plots import plot_comparisons
from animation import create_animation

from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import GridSearchCV

import seaborn as sns
from lazypredict.Supervised import LazyRegressor


if __name__ == '__main__':
    # Prepare the train data
    train_folder = 'data/P16'
    train_files = ['T2.txt', 'T3.txt', 'T4.txt', 'T5.txt', 'T6.txt', 'T7.txt', 'T8.txt', 'T9.txt', 'T10.txt']
    metadata, data_angles, data_emg_envelope, data_emg_filtered, data_grf, data_torques, data_torques_norm = load_data(train_folder, train_files)
    print(metadata)
    print("\n")

    # Prepare the test data
    test_folder = 'data/P16'
    test_files = ['T1.txt']
    metadata_test, data_angles_test, data_emg_envelope_test, data_emg_filtered_test, data_grf_test, data_torques_test, data_torques_norm_test = load_data(test_folder, test_files)

    St = 'St1'

    # St1_VL	St2_VL	St1_BF	St2_BF	St1_TA	St2_TA	St1_GAL	St2_GAL
    # data_emg_columns = [St+'_VL', St+'_BF', , St+'_TA', , St+'_GAL']
    data_emg_columns = [St+'_VL', St+'_BF']

    # St1_Pelvis_X	St1_Pelvis_Y	St1_Pelvis_Z	St2_Pelvis_X	St2_Pelvis_Y	St2_Pelvis_Z    ...
    # St1_Hip_X	    St1_Hip_Y	    St1_Hip_Z	    St2_Hip_X	    St2_Hip_Y	    St2_Hip_Z	    ...
    # St1_Knee_X	St1_Knee_Y	    St1_Knee_Z	    St2_Knee_X	    St2_Knee_Y	    St2_Knee_Z      ...
    # St1_Ankle_X	St1_Ankle_Y	    St1_Ankle_Z	    St2_Ankle_X	    St2_Ankle_Y	    St2_Ankle_Z
    data_torques_columns = [St+'_Pelvis_X', St+'_Pelvis_Y', St+'_Pelvis_Z',
                            St+'_Hip_X',    St+'_Hip_Y',    St+'_Hip_Z',
                            St+'_Knee_X',   St+'_Knee_Y',   St+'_Knee_Z',
                            St+'_Ankle_X',  St+'_Ankle_Y',  St+'_Ankle_Z']
    # data_torques_columns = [St+'_Knee_X']
    
    # St1_GRF_X	    St1_GRF_Y	    St1_GRF_Z	    St2_GRF_X	    t2_GRF_Y	    St2_GRF_Z
    data_grf_columns = [St+'_GRF_X', St+'_GRF_Y', St+'_GRF_Z']
    # data_grf_columns = [St+'_GRF_X']

    # St1_Pelvis_X	St1_Pelvis_Y	St1_Pelvis_Z	St2_Pelvis_X	St2_Pelvis_Y	St2_Pelvis_Z    ...
    # St1_Hip_X	    St1_Hip_Y	    St1_Hip_Z	    St2_Hip_X	    St2_Hip_Y	    St2_Hip_Z       ...   
    # St1_Knee_X	St1_Knee_Y	    St1_Knee_Z	    St2_Knee_X	    St2_Knee_Y	    St2_Knee_Z      ...
    # St1_Ankle_X	St1_Ankle_Y	    St1_Ankle_Z	    St2_Ankle_X	    St2_Ankle_Y	    St2_Ankle_Z
    # data_angles_columns = [St+'_Knee_X']
    data_angles_columns = [St+'_Knee_X']

    # Select the columns to use
    data_angles = data_angles[data_angles_columns]
    data_emg_envelope = data_emg_envelope[data_emg_columns]
    data_emg_filtered = data_emg_filtered[data_emg_columns]
    data_grf = data_grf[data_grf_columns]
    data_torques = data_torques[data_torques_columns]
    data_torques_norm = data_torques_norm[data_torques_columns]

    # Prepare the input and target to train the models
    X_train = pd.concat([data_emg_envelope, data_grf], axis=1) # Concatenate the input model data
    y_train = data_angles # Get the target data

    # Select the columns to use to test
    data_angles_test = data_angles_test[data_angles_columns]
    data_emg_envelope_test = data_emg_envelope_test[data_emg_columns]
    data_emg_filtered_test = data_emg_filtered_test[data_emg_columns]
    data_grf_test = data_grf_test[data_grf_columns]
    data_torques_test = data_torques_test[data_torques_columns]
    data_torques_norm_test = data_torques_norm_test[data_torques_columns]

    # Prepare the input and target to test the models
    X_test = pd.concat([data_emg_envelope_test, data_grf_test], axis=1) # Concatenate the input model data
    y_test = data_angles_test # Get the target data

    # Defining the models
    models = {
        'KNN': KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None),

        'Decision Tree': DecisionTreeRegressor(criterion='squared_error', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, ccp_alpha=0.0, monotonic_cst=None),

        'SVM': SVR(kernel='rbf', degree=3, gamma='auto', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1),

        'Random Forest': RandomForestRegressor(n_estimators=100, criterion='squared_error', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=1.0, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None, monotonic_cst=None),
        
        'Gradient Boosting': GradientBoostingRegressor(loss='squared_error', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, init=None, random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0),

        'AdaBoost': AdaBoostRegressor(estimator=None, n_estimators=50, learning_rate=1.0, loss='linear', random_state=None),

        'Ridge': Ridge(alpha=1.0, fit_intercept=True, copy_X=True, max_iter=None, tol=0.0001, solver='auto', positive=False, random_state=None),

        'Lasso': Lasso(alpha=1.0, fit_intercept=True, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'),

        'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5, fit_intercept=True, precompute=False, max_iter=1000, copy_X=True, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic'),
        
        'Extra Trees Regressor' : ExtraTreesRegressor(n_estimators=100, criterion='squared_error', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=1.0, max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=False, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None, monotonic_cst=None),

        'MLP Regressor' : MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000),

        'XGBRegressor' : XGBRegressor()
    }

    # Create the voting regressor based on some of the models
    voting_reg = VotingRegressor(estimators=[ ('Random Forest', models['Random Forest']), ('MLP Regressor', models['MLP Regressor']), ('XGBRegressor', models['XGBRegressor'])])

    # Add the voting regressor to the models
    models['Voting Regressor'] = voting_reg

    # Dictionary to store the predictions and metrics to plot after
    predictions = {}
    metrics = {}

    # Use lazy predict to get a holistc view about the result of a lot os models
    # lazy_model = LazyRegressor()
    # models_lazy, predictions_lazy = lazy_model.fit(X_train, X_test, y_train, y_test)
    # print(models_lazy)

    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"Training and evaluating {model_name}")

        pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # applyng smoth filter
        # y_pred = moving_average(y_pred, window_size=75)
        # y_pred = smooth_spline(y_pred)
        y_pred = loess_smoothing(y_pred, frac=0.09)
        # y_pred = kalman_filter(y_pred)

        predictions[model_name] = y_pred

        # Calculate assessment metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        metrics[model_name] = r2

        # Print the metrics
        print(f"{model_name} - Mean Absolute Error: {mae:.2f}")
        print(f"{model_name} - Mean Squared Error: {mse:.2f}")
        print(f"{model_name} - Root Mean Squared Error: {rmse:.2f}")
        print(f"{model_name} - Mean Absolute Percentage Error: {mape:.2%}")
        print(f"{model_name} - R^2 Score: {r2:.2f}")
        print() # Blank line

    plot_comparisons(y_test, predictions, metrics)

    # Finding the best model
    best_model = max(metrics, key=metrics.get)

    # Prepare the data to create the animation
    emg_anim = {col: data_emg_envelope_test[col].to_numpy() for col in data_emg_columns} # Prepare all EMG signals
    y_test_anim = y_test.to_numpy().ravel()
    y_test_anim = loess_smoothing(y_test_anim, frac=0.09)

    # Run the animation with the best model    
    create_animation(best_model, emg_anim, y_test_anim, predictions[best_model].flatten())