import pandas as pd

def gen_submission(review_id, y_pred,csvname='output.csv'):
    """Function to generate submissions files for Yelp challenge

    Args:
        review_id: list or array of review_ids
        y_pred   : list or array of corresponding predicted stars  

    Returns:
        No return value, just prints to screen saved file.

    Test:
    gen_submission(df.review_id[:10].values,df.stars[:10].values)
    """
    df = pd.DataFrame({'review_id': review_id,
                       'stars'    : y_pred})
    df.to_csv(csvname,index=False)
    print 'Submission file saved as ' + csvname

    return