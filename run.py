import pandas as pd
import numpy as np
import sys
import gc


if __name__ == '__main__':
    """
    main function, it runs the 2 scripts files, each corresponding to a model,
    and then does the mean of the predictions
    """
    if len(sys.argv) == 1:
        import script

        "this is called when we run ($ python run.py) on the anaconda console"
        print("performing training and predictions for models 1 and 2")
        script.run(num_model = 1)
        print('first model predictions terminated \n starting second model training')
        # trying to reduce memory usage for second model
        gc.collect()
        script.run(num_model = 2)
        print('second model predictions terminated \n combining predictions...')

    elif len(sys.argv) == 2:
        "this is called when we run ($ python run.py merge) on the anaconda console"
        if sys.argv[1] == 'merge':
            print("merging predictions")
        else:
            print("invalid argument")
            sys.exit(0)
    
    # fetches the predictions for both models
    preds1 = pd.read_csv('pred_model_1.csv')
    preds2 = pd.read_csv('pred_model_2.csv')

    # merges them together
    preds = (preds1['prediction'] + preds2['prediction']) / 2
    del preds1['prediction']
    THRESHOLD = 0.28
    preds1['prediction'] = np.where(preds > THRESHOLD, 1, 0)

    # writes out to the final csv
    file_name = 'result.csv'
    preds1.to_csv(file_name, index = False)
    print("""predictions generated in -{}- file""".format(file_name))