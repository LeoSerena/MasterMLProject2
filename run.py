import script
import helpers
import pandas as pd
import numpy as np

if __name__ == '__main__':
    """
    main function, it runs the 3 scripts files,each corresponding to a model,
    and then does the mean of the predictions
    """

    script.run(num_model = 1)
    script.run(num_model = 2)

    preds1 = pd.read_csv('pred_model_1.csv')
    print('first model predictions terminated \n starting second model triaining')
    preds2 = pd.read_csv('pred_model_2.csv')
    print('second model predictions terminated \n combining predictions...')

    preds = (preds1['prediction'] + preds2['prediction']) / 2
    del preds1['prediction']

    preds1['prediction'] = np.where(preds > helpers.foreground_threshold, 1, 0)

    file_name = 'result.csv'
    preds1.to_csv(file_name, index = False)
    print("""predictions generated in -{}- file""".format(file_name))