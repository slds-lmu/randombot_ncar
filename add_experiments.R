library(batchtools)
library(mlr3)
library(mlr3learners)
library(mlr3extralearners)
library(mlr3oml)
library(paradox)
library(mlr3misc)
library(mlr3batchmark)
library(mlr3pipelines)

unlink("/glade/derecho/scratch/marcbecker/randombot_ncar/registry", recursive = TRUE)

reg = makeExperimentRegistry(
  file.dir = "/glade/derecho/scratch/marcbecker/randombot_ncar/registry", 
  conf.file = "batchtools.conf.R",
  seed = 1)

# reg = loadRegistry(
#   file.dir = "/glade/derecho/scratch/marcbecker/randombot_ncar/registry", 
#   conf.file = "batchtools.conf.R",
#   writeable = TRUE)

# add problems 
# tasks and resamplings

#task_ids_218 = ocl(218)$task_ids
#task_ids_99 = ocl(99)$task_ids
#task_ids = c(task_ids_218, task_ids_99)
# download of task albert fails
#task_ids = task_ids[task_ids != 189356]

# connection to openml often times out
task_ids = c(3L, 12L, 31L, 53L, 3917L, 3945L, 7592L, 7593L, 9952L, 9977L, 
  9981L, 10101L, 14965L, 34539L, 146195L, 146212L, 146606L, 146818L, 
  146821L, 146822L, 146825L, 167119L, 167120L, 168329L, 168330L, 
  168331L, 168332L, 168335L, 168337L, 168338L, 168868L, 168908L, 
  168909L, 168910L, 168911L, 168912L, 189354L, 189355L, 6L, 11L, 
  14L, 15L, 16L, 18L, 22L, 23L, 28L, 29L, 32L, 37L, 43L, 45L, 49L, 
  219L, 2074L, 2079L, 3021L, 3022L, 3481L, 3549L, 3560L, 3573L, 
  3902L, 3903L, 3904L, 3913L, 3918L, 9910L, 9946L, 9957L, 9960L, 
  9964L, 9971L, 9976L, 9978L, 9985L, 10093L, 14952L, 14954L, 14969L, 
  14970L, 125920L, 125922L, 146800L, 146817L, 146819L, 146820L, 
  146824L, 167121L, 167124L, 167125L, 167140L, 167141L)

walk(task_ids[seq(3)], function(id) {
  try({
    otask = otsk(id = id)
    task = as_task(otask)
    resampling = as_resampling(otask)

    addProblem(
      name = otask$data_name,
      data = list(task = task, resampling = resampling),
      fun = function(data, job, measure, ...) {
        c(data, list(measure))
      })

    rm(otask)
    rm(task)
    rm(resampling)
    gc()
  })
})

pdes = replicate(length(reg$problems), list(data.table(measure = msrs(c("classif.ce", "classif.bacc", "classif.logloss", "classif.mauc_au1p")))))
pdes = set_names(pdes, reg$problems)

# add Algorithms
run_mbo = function(data, job, instance, learner, ...) {
  library(mlr3)
  library(mlr3learners)
  library(mlr3tuning)
  library(mlr3mbo)
  library(mlr3pipelines)


  inter_result_path = sprintf("/glade/derecho/scratch/marcbecker/randombot_ncar/registry/inter_results/%s.rds", job$id)

  callback = callback_batch_tuning("bbotk.backup",
    label = "Backup Archive Callback",
    man = "bbotk::bbotk.backup",

    on_optimizer_after_eval = function(callback, context) {
      if (file.exists(callback$state$path)) unlink(callback$state$path)
      saveRDS(context$instance$archive, callback$state$path)
    }
  )

  callback$state$path = inter_result_path

  learner$predict_type = "prob"
  learner$encapsulate = c(train = "callr", predict = "callr")
  learner$fallback = lrn("classif.featureless", predict_type = "prob")
  learner$timeout = c(train = 3600, predict = 3600)

  future::plan("multicore", workers = 10)

  optim_instance = ti(
    task = data$task,
    learner = learner,
    resampling = data$resampling,
    measure = data$measure,
    terminator = trm("evals", n_evals = 5),
    callbacks = callback)

  if (file.exists(inter_result_path)) {
    archive = readRDS(inter_result_path)
    optim_instance$archive = archive
  } else {
    init_design = generate_design_sobol(optim_instance$search_space, n = 10)$data
    optim_instance$eval_batch(init_design)
  }

  tuner = tnr("mbo")
  tuner$optimize(optim_instance)

  instance
} 

addAlgorithm(
  name = "run_mbo",
  fun = run_mbo
)

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

# token
tokens = list(
  glmnet = list(
    glmnet.s     = to_tune(1e-4, 1e4, logscale = TRUE),
    glmnet.alpha = to_tune(0, 1)
  ),

  kknn = list(
    kknn.k = to_tune(1, 50, logscale = TRUE),
    kknn.distance = to_tune(1, 5),
    kknn.kernel = to_tune(c("rectangular", "optimal", "epanechnikov", "biweight", "triweight", "cos",  "inv",  "gaussian", "rank"))
  ),

  ranger = list(
    ranger.mtry.ratio      = to_tune(0, 1),
    ranger.replace         = to_tune(),
    ranger.sample.fraction = to_tune(1e-1, 1),
    ranger.num.trees       = to_tune(1, 2000)
  ),

  svm = list(
    svm.cost    = to_tune(1e-4, 1e4, logscale = TRUE),
    svm.kernel  = to_tune(c("polynomial", "radial", "sigmoid", "linear")),
    svm.degree  = to_tune(2, 5),
    svm.gamma   = to_tune(1e-4, 1e4, logscale = TRUE)
  ),

  xgboost = list(
    xgboost.eta               = to_tune(1e-4, 1, logscale = TRUE),
    xgboost.max_depth         = to_tune(1, 20),
    xgboost.colsample_bytree  = to_tune(1e-1, 1),
    xgboost.colsample_bylevel = to_tune(1e-1, 1),
    xgboost.lambda            = to_tune(1e-3, 1e3, logscale = TRUE),
    xgboost.alpha             = to_tune(1e-3, 1e3, logscale = TRUE),
    xgboost.subsample         = to_tune(1e-1, 1),
    xgboost.nrounds           = to_tune(1, 5000)
  ),

  catboost = list(
    catboost.learning_rate               = to_tune(1e-3, 1, logscale = TRUE),
    catboost.depth                       = to_tune(4, 10),
    catboost.l2_leaf_reg                 = to_tune(1, 10),
    catboost.max_ctr_complexity          = to_tune(1, 5),
    catboost.grow_policy                 = to_tune(c("SymmetricTree", "Depthwise")),
    catboost.iterations                  = to_tune(1, 5000),
    catboost.random_strength             = to_tune(1, 20),
    catboost.bagging_temperature         = to_tune(0, 1),
    catboost.leaf_estimation_iterations  = to_tune(1, 100),
    # catboost.subsample                   = to_tune(0, 1),
    catboost.rsm                         = to_tune(0.1, 1),
    catboost.min_data_in_leaf            = to_tune(1, 5000),
    catboost.border_count                = to_tune(32, 255)
    # catboost.one_hot_max_size          = to_tune(2, 10)
  ),

  lightgbm = list(
    lightgbm.learning_rate     = to_tune(1e-3, 0.2, logscale = TRUE),
    lightgbm.feature_fraction  = to_tune(0.1, 1),
    lightgbm.min_data_in_leaf  = to_tune(2, 200),
    lightgbm.num_leaves        = to_tune(10, 255),
    lightgbm.extra_trees       = to_tune(),
    lightgbm.num_iterations    = to_tune(1, 5000),
    lightgbm.bagging_fraction  = to_tune(0, 1),
    lightgbm.bagging_freq      = to_tune(0, 10),
    lightgbm.max_depth         = to_tune(15, 100),
    lightgbm.lambda_l1         = to_tune(1e-3, 100, logscale = TRUE),
    lightgbm.lambda_l2         = to_tune(1e-3, 100, logscale = TRUE),
    lightgbm.min_gain_to_split = to_tune(1e-3, 0.1, logscale = TRUE)
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
