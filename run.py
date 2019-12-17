import script1
#import script2
#import script3

import helpers
import pandas as pd

if __name__ == '__main__':
    """
    main function, it runs the 3 scripts files,each corresponding to a model,
    and then does the mean of the predictions
    """
    script1.run_1()
    #script2.run_2()
    #script3.run_3()

    preds1 = pd.read_csv('pred_model_1.csv')
    #preds2 = pd.read_csv('pred_model_2.csv')
    #preds3 = pd.read_csv('pred_model_3.csv')

    preds = preds1

    # preds = (preds1 + preds2 + preds3) / 3

    lambda_ = lambda x : 1 if x > helpers.foreground_threshold else 0
    preds['prediction'] = preds['prediction'].map(lambda_)
    preds.to_csv()