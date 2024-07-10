run_mbo = function(data, job, instance, learner, ...) {

  fp = file(file.path(job$file.dir, "logs", sprintf("%s.log", job$id)), open = "at")
  sink(file = fp, append = TRUE)
  sink(file = fp, type = "message", append = TRUE)
  on.exit({ sink(type = "message"); sink(type = "output"); close(fp) }, add = TRUE)

  library(mlr3)
  library(mlr3learners)
  library(mlr3tuning)
  library(mlr3mbo)
  library(mlr3pipelines)
  library(bbotk)
  library(mlr3misc)
  library(data.table)

  lgr::get_logger("mlr3")$set_threshold("warn")

  # backup callback
  inter_result_path = file.path(job$external.dir, sprintf("inter_result_%s.rds", job$id))
  callback_backup = callback_batch_tuning("bbotk.backup",
    label = "Backup Archive Callback",
    man = "bbotk::bbotk.backup",

    on_optimizer_after_eval = function(callback, context) {
      start_time = Sys.time()
      tmp_file = tempfile(tmpdir = job$external.dir, fileext = ".rds")
      saveRDS(context$instance$archive$data, tmp_file)
      unlink(callback$state$path)
      file.rename(tmp_file, callback$state$path)
      message(sprintf("Saving intermediate results took %s seconds", difftime(Sys.time(), start_time, units = "s")))
    }
  )
  callback_backup$state$path = inter_result_path

  # prepare learner
  learner$predict_type = "prob"
  learner$encapsulate = c(train = "callr", predict = "callr")
  learner$fallback = lrn("classif.featureless", predict_type = "prob")
  learner$timeout = c(train = 3600, predict = 3600)

  # measure
  source("MeasureClassifMCC.R")
  msr_mcc = MeasureClassifMCC$new()
  msr_mcc$id = "mcc"

  measures = if ("twoclass" %in% data$task$properties) {
    list(
      "ce" = msr("classif.ce", id = "ce"),
      "bacc" = msr("classif.bacc", id = "bacc"),
      "logloss" = msr("classif.logloss", id = "logloss"),
      "auc" = msr("classif.auc", id = "auc"),
      "mcc" =  msr_mcc)
  } else {
    list(
      "ce" = msr("classif.ce", id = "ce"),
      "bacc" = msr("classif.bacc", id = "bacc"),
      "logloss" = msr("classif.logloss", id = "logloss"),
      "auc" = msr("classif.mauc_mu", id = "auc"),
      "mcc" =  msr_mcc)
  }

  callback_measures = callback_batch_tuning("mlr3tuning.measures",
    label = "Additional Measures Callback",
    man = "mlr3tuning::mlr3tuning.measures",

    on_eval_before_archive = function(callback, context) {
      set(context$aggregated_performance, j = callback$state$ids, value = context$benchmark_result$aggregate(callback$state$measures)[, callback$state$ids, with = FALSE])
    }
  )

  extra_measures = measures[setdiff(names(measures), instance$measure)]
  callback_measures$state$measures = extra_measures
  callback_measures$state$ids = names(extra_measures)

  # early stopping metrics
  if (learner$id == "catboost") {
    metric = if ("twoclass" %in% data$task$properties) {
      switch(instance$measure,
        "ce" = "Accuracy",
        "bacc" = "BalancedAccuracy",
        "logloss" = "Logloss",
        "auc" = "AUC",
        "mcc" =  "MCC")
      } else {
      switch(instance$measure,
        "ce" = "Accuracy",
        "bacc" = "Accuracy",
        "logloss" = "MultiClass",
        "auc" = "AUC:type=Mu",
        "mcc" =  "MCC")
      }
    learner$param_set$set_values(catboost.eval_metric = metric)
  } else if (learner$id == "xgboost") {
    metric = if ("twoclass" %in% data$task$properties) {
      switch(instance$measure,
        "ce" = "error",
        "bacc" = msr("classif.bacc"),
        "logloss" = "logloss",
        "auc" = "auc",
        "mcc" =  msr_mcc)
      } else {
      switch(instance$measure,
        "ce" = "merror",
        "bacc" = msr("classif.bacc"),
        "logloss" = "mlogloss",
        "auc" = msr("classif.mauc_mu"),
        "mcc" =  msr_mcc)
      }
    learner$param_set$set_values(xgboost.eval_metric = metric)
  } else if (learner$id == "lightgbm") {
    metric = if ("twoclass" %in% data$task$properties) {
      switch(instance$measure,
        "ce" = "binary_error",
        "bacc" = msr("classif.bacc"),
        "logloss" = "binary_logloss",
        "auc" = "auc",
        "mcc" =  msr_mcc)
    } else {
      switch(instance$measure,
        "ce" = "multi_error",
        "bacc" = msr("classif.bacc"),
        "logloss" = "multi_logloss",
        "auc" = msr("classif.mauc_mu"),
        "mcc" =  msr_mcc)
      }
    learner$param_set$set_values(lightgbm.eval = list(metric))
  }

  # parallel options
  options(parallelly.availableCores.system = 128)
  options(parallelly.maxWorkers.localhost = Inf)
  options(mlr3.exec_chunk_bins = 10)

  future::plan("multisession", workers = 10)

  n_evals = learner$param_set$search_space()$length * 100

  tuning_instance = ti(
    task = data$task,
    learner = learner,
    resampling = data$resampling,
    measure = measures[[instance$measure]],
    terminator = trm("evals", n_evals = n_evals),
    callbacks = list(callback_backup, callback_measures),
    store_benchmark_result = FALSE,
    store_models = FALSE)

  # initial design
  init_design_size = 0.25 * n_evals

  # restore archive
  if (file.exists(inter_result_path)) {
    tryCatch({
      message("Intermediate results found")
      archive = readRDS(inter_result_path)
      tuning_instance$archive$data = archive
    },
    error = function(cond) {
      message("Reading intermediate results failed")
    }) 
  } else {
    message("No intermediate results found")
  }


  # evaluate init design in batches of size 1 to get checkpoints
  if (tuning_instance$archive$n_evals < init_design_size) {
    init_design = generate_design_lhs(tuning_instance$search_space, n = init_design_size)$data

    walk(seq(tuning_instance$archive$n_evals + 1, init_design_size), function(i) {
      tuning_instance$eval_batch(init_design[i])
    })
  }

  # configure mbo
  lrn_mbo = lrn("regr.ranger_mbo",
    predict_type = "se",
    keep.inbag = TRUE,
    se.method = "simple",
    splitrule = "extratrees",
    num.random.splits = 1L,
    num.trees = 10L,
    replace = TRUE,
    sample.fraction = 1,
    min.node.size = 1,
    mtry.ratio = 1
  )

  surrogate = srlrn(as_learner(po("imputesample", affect_columns = selector_type("logical")) %>>%
    po("imputeoor", multiplier = 3, affect_columns = selector_type(c("integer", "numeric", "character", "factor", "ordered"))) %>>%
    po("colapply", applicator = as.factor, affect_columns = selector_type("character")) %>>%
    lrn_mbo), catch_errors = TRUE)

  acq_optimizer = acqo(
    optimizer = opt("focus_search", n_points = 1000L, maxit = 9L),
    terminator = trm("evals", n_evals = 20000L),
    catch_errors = FALSE
  )

  acq_function = acqf("cb", lambda = 1, check_values = FALSE)

  tuner = tnr("mbo",
    loop_function = bayesopt_ego_log,
    surrogate = surrogate,
    acq_function = acq_function,
    acq_optimizer = acq_optimizer,
    args = list(random_interleave_iter = 4, init_design_size = init_design_size)
  )

  # run optimization
  tuner$optimize(tuning_instance)
  tuning_instance
} 

addAlgorithm(
  name = "run_mbo",
  fun = run_mbo
)