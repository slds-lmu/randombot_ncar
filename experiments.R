# problem pars
pdes = replicate(length(reg$problems), list(data.table(measure = c("ce", "bacc", "logloss", "auc", "mcc"))))
pdes = set_names(pdes, reg$problems)

# algorithm pars

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
  lrn("classif.ranger", id = "ranger", num.trees = 500))
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
lrn_xgboost = set_validate(as_learner(po("removeconstants", id = "xgboost_removeconstants") %>>%
  po("imputeoor", id = "xgboost_imputeoor") %>>%
  po("fixfactors", id = "xgboost_fixfactors") %>>%
  po("imputesample", affect_columns = selector_type(c("factor", "ordered")), id = "xgboost_imputesample") %>>%
  po("encodeimpact", id = "xgboost_encode") %>>%
  po("removeconstants", id = "xgboost_post_removeconstants") %>>%
  lrn("classif.xgboost", id = "xgboost", early_stopping_rounds = 100)), validate = "test")
lrn_xgboost$id = "xgboost"

# catboost
lrn_catboost = set_validate(as_learner(po("colapply", applicator = as.numeric, affect_columns = selector_type("integer"), id = "catboost_as_numeric") %>>%
  po("colapply", applicator = as.factor, affect_columns = selector_type("logical"), id = "catboost_as_factor") %>>%
  lrn("classif.catboost", id = "catboost", early_stopping_rounds = 100)), validate = "test")
lrn_catboost$id = "catboost"

# lightgbm
lrn_lightgbm = set_validate(as_learner(po("removeconstants", id = "lightgbm_removeconstants") %>>% 
  lrn("classif.lightgbm", id = "lightgbm", early_stopping_rounds = 100)), validate = "test")
lrn_lightgbm$id = "lightgbm"

learners = list(
  glmnet = lrn_glmnet, 
  kknn = lrn_kknn, 
  ranger = lrn_ranger, 
  svm = lrn_svm, 
  xgboost = lrn_xgboost, 
  catboost = lrn_catboost, 
  lightgbm = lrn_lightgbm)

# token
tokens = list(
  glmnet = list(
    # Bischl et al. (2021)
    glmnet.alpha = to_tune(0, 1),
    # Misc
    glmnet.lambda = to_tune(p_dbl(1e-4, 1e4, logscale = TRUE))
  ),

  kknn = list(
    # Bischl et al. (2021)
    kknn.k = to_tune(1, 100, logscale = TRUE),
    kknn.distance = to_tune(1, 5),
    kknn.kernel = to_tune(c("rectangular", "optimal", "epanechnikov", "biweight", "triweight", "cos",  "inv",  "gaussian", "rank"))
  ),

  ranger = list(
    # Bischl et al. (2021)
    ranger.mtry.ratio       = to_tune(0, 1),
    ranger.replace          = to_tune(),
    ranger.sample.fraction  = to_tune(1e-1, 1),
    # van Rijn and Hutter (2018)
    ranger.min.node.size    = to_tune(1, 50),
    # Misc
    ranger.min.bucket       = to_tune(1, 50)
  ),

  svm = list(
    # Bischl et al. (2021)
    svm.cost    = to_tune(1e-4, 1e4, logscale = TRUE),
    svm.kernel  = to_tune(c("polynomial", "radial", "sigmoid", "linear")),
    svm.degree  = to_tune(2, 5),
    svm.gamma   = to_tune(1e-4, 1e4, logscale = TRUE)
  ),

  xgboost = list(
    # Bischl et al. (2021)
    xgboost.eta               = to_tune(1e-3, 1, logscale = TRUE),
    xgboost.max_depth         = to_tune(1, 20),
    xgboost.colsample_bytree  = to_tune(1e-1, 1),
    xgboost.colsample_bylevel = to_tune(1e-1, 1),
    xgboost.lambda            = to_tune(1e-3, 1e3, logscale = TRUE),
    xgboost.alpha             = to_tune(1e-3, 1e3, logscale = TRUE),
    xgboost.subsample         = to_tune(1e-1, 1),
    xgboost.nrounds           = to_tune(1, 5000, internal = TRUE)
  ),

  catboost = list(
    # Salinas and Erickson (2024)
    catboost.learning_rate                = to_tune(1e-3, 1, logscale = TRUE),
    catboost.depth                        = to_tune(1, 16), # catboost has an upper limit of 16
    catboost.l2_leaf_reg                  = to_tune(1e-3, 1e3),
    catboost.max_ctr_complexity           = to_tune(1, 8),
    catboost.grow_policy                  = to_tune(c("SymmetricTree", "Depthwise", "Lossguide")),
    # SageMaker
    catboost.random_strength              = to_tune(1, 20),
    # catboost.ai
    catboost.bagging_temperature          = to_tune(0, 1),
    catboost.border_count                 = to_tune(32, 255),
    catboost.iterations                   = to_tune(1, 5000, internal = TRUE),
    # Misc
    catboost.min_data_in_leaf             = to_tune(1, 200), # lightgbm
    catboost.one_hot_max_size             = to_tune(p_int(2, 10))
  ),

  lightgbm = list(
    # Salinas and Erickson (2024)
    lightgbm.learning_rate     = to_tune(1e-3, 1, logscale = TRUE),
    lightgbm.feature_fraction  = to_tune(0.1, 1),
    lightgbm.min_data_in_leaf  = to_tune(1, 200),
    lightgbm.num_leaves        = to_tune(10, 255),
    lightgbm.extra_trees       = to_tune(),
    # SageMaker 
    lightgbm.bagging_fraction  = to_tune(0, 1),
    lightgbm.bagging_freq      = to_tune(0, 10),
    # lightgbm.io
    lightgbm.lambda_l1         = to_tune(1e-3, 1e3, logscale = TRUE),
    lightgbm.lambda_l2         = to_tune(1e-3, 1e3, logscale = TRUE),
    lightgbm.min_gain_to_split = to_tune(1e-3, 0.1, logscale = TRUE),
    lightgbm.num_iterations    = to_tune(1, 5000, internal = TRUE)
  )
)

learners = map(learners, function(learner) {
  learner$param_set$set_values(.values = tokens[[learner$id]])
  learner
})

ades = list(
  run_mbo = data.table(learner = learners)
)

addExperiments(prob.design = pdes, algo.design = ades)