def gen_submission(review_id, y_pred,csvname='output.csv'):
    """Function to generate submissions files for Yelp challenge

    Args:
        review_id: list or array of review_ids
        y_pred   : list or array of corresponding predicted stars  

    Returns:
        No return value, just prints to screen saved file.
    """
    df = pd.DataFrame({'review_id': review_id,
                       'stars'    : stars})
    df.to_csv(csvname,index=False)
    print 'Submission file saved as ' + csvname

    return