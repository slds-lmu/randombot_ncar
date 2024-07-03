
# Matthew correlction coefficient
mcc = mlr3misc::crate({function(truth, response) {
    m = table(response = response, truth = truth)

    t_sum = rowSums(m)
    p_sum = colSums(m)
    n_correct = sum(diag(m))
    n_samples = sum(p_sum)

    cov_ytyp = n_correct * n_samples - sum(t_sum * p_sum)
    cov_ypyp = n_samples^2 - sum(p_sum^2)
    cov_ytyt = n_samples^2 - sum(t_sum^2)

    value = if (cov_ypyp * cov_ytyt == 0) return(0)
    cov_ytyp / sqrt(cov_ytyt * cov_ypyp)
}})

# lightgbm measures
lightgbm_bacc_binary = mlr3misc::crate({function(preds, dtrain) {
    truth = factor(lightgbm::get_field(dtrain, "label"))
    response = factor(ifelse(preds > 0.5, 1, 0), levels = c(0, 1))

    list(name = "bacc", value = mlr3measures::bacc(truth, response), higher_better = TRUE)
}})

lightgbm_bacc_multiclass = mlr3misc::crate({function(preds, dtrain) {
    truth = factor(lightgbm::get_field(dtrain, "label"))
    response = matrix(preds, ncol = n_classes)
    response = factor(apply(response, 1, which.max) - 1, levels = 0:(n_classes - 1))

    list(name = "bacc", value = mlr3measures::bacc(truth, response), higher_better = TRUE)
}}, n_classes = n_classes)


lightgbm_mcc_binary = mlr3misc::crate({function(preds, dtrain) {
    truth = factor(lightgbm::get_field(dtrain, "label"))
    response = factor(ifelse(preds > 0.5, 1, 0), levels = c(0, 1))

    list(name = "mcc", value = mcc(truth, response), higher_better = TRUE)
}}, mcc = mcc)

lightgbm_mcc_multiclass = mlr3misc::crate({function(preds, dtrain) {
    truth = factor(lightgbm::get_field(dtrain, "label"))
    response = matrix(preds, ncol = n_classes)
    response = factor(apply(response, 1, which.max) - 1, levels = 0:(n_classes - 1))

    list(name = "mcc", value = mcc(truth, response), higher_better = TRUE)
}}, n_classes = n_classes, mcc = mcc)

lightgbm_mauc_aunu = mlr3misc::crate({function(preds, dtrain) {
  truth = factor(lightgbm::get_field(dtrain, "label"))
  prob = matrix(preds, ncol = n_classes)
  colnames(prob) = 0:(n_classes - 1)

  list(name = "mauc_aunu", value = mlr3measures::mauc_aunu(truth, prob), higher_better = TRUE)
}}, n_classes = n_classes)

# xgboost measures

xgboost_bacc_binary = mlr3misc::crate({function(prediction, dtrain) {
  truth = factor(xgboost::getinfo(dtrain, "label"))
  response = factor(ifelse(prediction > 0.5, 1, 0), levels = c(0, 1))

  list(metric = "bacc", value = mlr3measures::bacc(truth, response))
}})

xgboost_bacc_multiclass = mlr3misc::crate({function(prediction, dtrain) {
  truth = factor(xgboost::getinfo(dtrain, "label"))
  response = matrix(prediction, ncol = n_classes)
  response = factor(apply(response, 1, which.max) - 1, levels = 0:(n_classes - 1))

  list(metric = "bacc", value = mlr3measures::bacc(truth, response))
}}, n_classes = n_classes)

xgboost_mcc_binary = mlr3misc::crate({function(prediction, dtrain) {
  truth = factor(xgboost::getinfo(dtrain, "label"))
  response = factor(ifelse(prediction > 0.5, 1, 0), levels = c(0, 1))

  list(name = "mcc", value = mcc(truth, response))
}}, mcc = mcc)

xgboost_mcc_multiclass = mlr3misc::crate({function(prediction, dtrain) {
  truth = factor(xgboost::getinfo(dtrain, "label"))
  response = matrix(prediction, ncol = n_classes)
  response = factor(apply(response, 1, which.max) - 1, levels = 0:(n_classes - 1))

  list(name = "mcc", value = mcc(truth, response))
}}, n_classes = n_classes, mcc = mcc)