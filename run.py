import script
import helpers
import pandas as pd

if __name__ == '__main__':
    """
    main function, it runs the 3 scripts files,each corresponding to a model,
    and then does the mean of the predictions
    """

    script.run(num_model = 1)
    script.run(num_model = 2)

    preds1 = pd.read_csv('pred_model_1.csv')
    preds2 = pd.read_csv('pred_model_2.csv')

    preds = (preds1 + preds2) / 2

    lambda_ = lambda x : 1 if x > helpers.foreground_threshold else 0
    preds['prediction'] = preds['prediction'].map(lambda_)
    preds.to_csv()