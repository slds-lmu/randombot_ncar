# glmnet
lrn_glmnet = as_learner(po("removeconstants", id = "glmnet_removeconstants") %>>%
  po("imputehist", id = "glmnet_imputehist") %>>%
  po("imputeoor", id = "glmnet_imputeoor") %>>%
  po("fixfactors", id = "glmnet_fixfactors") %>>%
  po("imputesample", affect_columns = selector_type(c("factor", "ordered")), id = "glmnet_imputesample") %>>%
  po("collapsefactors", target_level_count = 100, id = "glmnet_collapse") %>>%
  po("encode", method = "one-hot", id = "glmnet_encode") %>>%
  po("removeconstants", id = "glmnet_post_removeconstants") %>>%
  lrn("classif.glmnet", id = "glmnet"))
lrn_glmnet$id = "glmnet"

# kknn
lrn_kknn = as_learner(po("removeconstants", id = "kknn_removeconstants") %>>%
  po("imputehist", id = "kknn_imputehist") %>>%
  po("imputeoor", id = "kknn_imputeoor") %>>%
  po("fixfactors", id = "kknn_fixfactors") %>>%
  po("imputesample", affect_columns = selector_type(c("factor", "ordered")), id = "kknn_imputesample") %>>%
  po("collapsefactors", target_level_count = 100, id = "kknn_collapse") %>>%
  po("removeconstants", id = "kknn_post_removeconstants") %>>%
  lrn("classif.kknn", id = "kknn"))
lrn_kknn$id = "kknn"

# ranger
lrn_ranger = as_learner(po("removeconstants", id = "ranger_removeconstants") %>>%
  po("imputeoor", id = "ranger_imputeoor") %>>%
  po("fixfactors", id = "ranger_fixfactors") %>>%
  po("imputesample", affect_columns = selector_type(c("factor", "ordered")), id = "ranger_imputesample") %>>%
  po("collapsefactors", target_level_count = 100, id = "ranger_collapse") %>>%
  po("removeconstants", id = "ranger_post_removeconstants") %>>%
  lrn("classif.ranger", id = "ranger"))
lrn_ranger$id = "ranger"

# svm
lrn_svm = as_learner(po("removeconstants", id = "svm_removeconstants") %>>%
  po("imputehist", id = "svm_imputehist") %>>%
  po("imputeoor", id = "svm_imputeoor") %>>%
  po("fixfactors", id = "svm_fixfactors") %>>%
  po("imputesample", affect_columns = selector_type(c("factor", "ordered")), id = "svm_imputesample") %>>%
  po("collapsefactors", target_level_count = 100, id = "svm_collapse") %>>%
  po("encode", method = "one-hot", id = "smv_encode") %>>%
  po("removeconstants", id = "svm_post_removeconstants") %>>%
  lrn("classif.svm", id = "svm", type = "C-classification"))
lrn_svm$id = "svm"

# xgboost
lrn_xgboost = as_learner(po("removeconstants", id = "xgboost_removeconstants") %>>%
  po("imputeoor", id = "xgboost_imputeoor") %>>%
  po("fixfactors", id = "xgboost_fixfactors") %>>%
  po("imputesample", affect_columns = selector_type(c("factor", "ordered")), id = "xgboost_imputesample") %>>%
  po("encodeimpact", id = "xgboost_encode") %>>%
  po("removeconstants", id = "xgboost_post_removeconstants") %>>%
  lrn("classif.xgboost", id = "xgboost"))
lrn_xgboost$id = "xgboost"

# catboost
lrn_catboost = as_learner(po("colapply", applicator = as.numeric, affect_columns = selector_type("integer")) %>>%
  lrn("classif.catboost", id = "catboost"))
lrn_catboost$id = "catboost"

# lightgbm
lrn_lightgbm = as_learner(po("removeconstants", id = "lightgbm_removeconstants") %>>% lrn("classif.lightgbm", id = "lightgbm"))
lrn_lightgbm$id = "lightgbm"

learners = list(
  glmnet = lrn_glmnet, 
  kknn = lrn_kknn, 
  ranger = lrn_ranger, 
  svm = lrn_svm, 
  xgboost = lrn_xgboost, 
  catboost = lrn_catboost, 
  lightgbm = lrn_lightgbm)