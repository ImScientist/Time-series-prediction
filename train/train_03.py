""" Train the third model.
- xgboost (baseline)
"""
import os
import json
import time
from rossmann.models.model_03 import model_03
import matplotlib.pyplot as plt
import xgboost as xgb
import argparse

if __name__ == '__main__':
    """For more information use: python train/model_02.py --help

    python train/train_03.py \
        --data_dir ../data/rossmann-store-sales/source \
        --num_boost_round 2 \
        --early_stopping_rounds 10
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir',
                        type=str,
                        dest='data_dir',
                        required=True,
                        help='Directory with training data')

    parser.add_argument('--num_boost_round',
                        type=int,
                        dest='num_boost_round',
                        required=True,
                        default=10,
                        help='Number of boosting iterations')

    parser.add_argument('--early_stopping_rounds',
                        type=int,
                        dest='early_stopping_rounds',
                        required=False,
                        default=None,
                        help='Early stopping rounds.')

    args = parser.parse_args()

    cwd = os.getcwd()
    basedir = os.path.abspath(os.path.dirname(__file__))
    output_dir = os.path.join(basedir, '..', 'outputs')
    data_dir = args.data_dir

    params = {
        "objective": "reg:squarederror",
        "booster": "gbtree",
        "eta": 0.03,
        "max_depth": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "verbosity": 1,
        "nthread": 1,
        "seed": 10
    }

    start = time.time()

    result, gbm, train_log = model_03(data_dir=data_dir,
                                      num_boost_round=args.num_boost_round,
                                      early_stopping_rounds=args.early_stopping_rounds,
                                      params=params)

    end = time.time()

    ####################################################################
    # keep relevant information in output directory
    ####################################################################

    params = {
        'model': params,
        'training': dict((k, v) for k, v in args.__dict__.items())
    }
    params['training']['duration'] = end - start
    params['model']['name'] = 'ARIMA'

    os.makedirs(output_dir, exist_ok=True)
    result.to_csv(os.path.join(output_dir, 'predictions.csv'))

    with open(os.path.join(output_dir, 'params.json'), 'w') as f:
        json.dump(params, f, indent=2)

    with open(os.path.join(output_dir, 'train_log.json'), 'w') as f:
        json.dump(train_log, f, indent=2)

    fig = xgb.plot_importance(gbm)
    plt.rcParams['figure.figsize'] = [6, 10]
    fig.get_figure().savefig(os.path.join(output_dir, 'feature_importance_xgb.png'),
                             bbox_inches='tight',
                             pad_inches=1)

    print('Done')
