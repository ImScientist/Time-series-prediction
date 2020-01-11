""" Train the first model.
- ARIMA
"""
import os
import json
import time
from rossmann.models.model_01 import model_01
import argparse

if __name__ == '__main__':
    """For more information use: python train/model_01.py --help
    
    python train/train_01.py \
        --data_dir ../data/rossmann-store-sales/source \
        --max_pdq 4 1 2 \
        --n_stores 4
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir',
                        type=str,
                        dest='data_dir',
                        required=True,
                        help='Directory with training data')

    parser.add_argument('--max_pdq',
                        type=int,
                        dest='max_pdq',
                        required=False,
                        default=[6, 1, 4],
                        nargs=3,
                        help='Max p, d, q for the Grid search of the best ARIMA model.')

    parser.add_argument('--n_stores',
                        type=int,
                        dest='n_stores',
                        required=False,
                        default=None,
                        help='Apply the model to a subset of the stores (only to n_stores of them).'
                             'Used only to test fast if the model works.')

    args = parser.parse_args()

    cwd = os.getcwd()
    basedir = os.path.abspath(os.path.dirname(__file__))
    output_dir = os.path.join(basedir, '..', 'outputs')
    data_dir = args.data_dir

    # print('sys.executable \t', sys.executable)
    # print('cwd \t', cwd)
    # print('basedir \t', basedir)
    # print('folders \t', os.listdir())
    # print('folders cwd \t', os.listdir(cwd))

    start = time.time()

    result, metrics = model_01(data_dir=args.data_dir,
                               p_range=(0, args.max_pdq[0]),
                               d_range=(0, args.max_pdq[1]),
                               q_range=(0, args.max_pdq[2]),
                               n_stores=args.n_stores)

    end = time.time()

    ####################################################################
    # keep relevant information in output directory
    ####################################################################

    params = {
        'model': {
            'name': 'ARIMA'
        },
        'training': dict((k, v) for k, v in args.__dict__.items())
    }
    params['training']['duration'] = end - start

    os.makedirs(output_dir, exist_ok=True)
    result.to_csv(os.path.join(output_dir, 'predictions.csv'))

    with open(os.path.join(output_dir, 'params.json'), 'w') as f:
        json.dump(params, f, indent=2)

    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print('Done')
