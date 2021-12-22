import sklearn

def tofile(logger, y, error_report, network_name, dataset_name):

    logger.info("F1 metric: {}".format(sklearn.metrics.classification_report(y, error_report)))
    logger.info("Confusion matrix: {}".format(sklearn.metrics.confusion_matrix(y, error_report)))
    logger.info("f1_score {}".format(sklearn.metrics.f1_score(y, error_report)))