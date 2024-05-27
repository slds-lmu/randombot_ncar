library(batchtools)
library(mlr3)
library(mlr3learners)
library(mlr3extralearners)
library(mlr3oml)
library(paradox)
library(mlr3misc)
library(mlr3batchmark)

unlink("/glade/u/home/marcbecker/randombot_ncar/registry", recursive = TRUE)

reg = makeExperimentRegistry(
  file.dir = "registry", 
  conf.file = "batchtools.conf.R",
  seed = 1)

task_ids_218 = ocl(218)$task_ids
task_ids_99 = ocl(99)$task_ids
task_ids = c(task_ids_218, task_ids_99)
# download of task albert fails
task_ids = task_ids[task_ids != 189356]

tasks = list()
resamplings = list()

for (id in task_ids) {
  otask = otsk(id = id)
  tasks = c(tasks, as_task(otask))
  resamplings = c(resamplings, as_resampling(otask))
}

learners = lrns(c("classif.glmnet", "classif.kknn", "classif.ranger", "classif.svm", "classif.xgboost", "classif.catboost", "classif.lightgbm"))

tokens = list(
  classif.glmnet = list(
    s     = to_tune(1e-4, 1e4, logscale = TRUE),
    alpha = to_tune(0, 1)
  ),

  classif.kknn = list(
    k = to_tune(1, 50, logscale = TRUE),
    distance = to_tune(1, 5),
    kernel = to_tune(c("rectangular", "optimal", "epanechnikov", "biweight", "triweight", "cos",  "inv",  "gaussian", "rank"))
  ),

  classif.ranger = list(
    mtry.ratio      = to_tune(0, 1),
    replace         = to_tune(),
    sample.fraction = to_tune(1e-1, 1),
    num.trees       = to_tune(1, 2000)
  ),

  classif.svm = list(
    cost    = to_tune(1e-4, 1e4, logscale = TRUE),
    kernel  = to_tune(c("polynomial", "radial", "sigmoid", "linear")),
    degree  = to_tune(2, 5),
    gamma   = to_tune(1e-4, 1e4, logscale = TRUE)
  ),

  classif.xgboost = list(
    eta               = to_tune(1e-4, 1, logscale = TRUE),
    max_depth         = to_tune(1, 20),
    colsample_bytree  = to_tune(1e-1, 1),
    colsample_bylevel = to_tune(1e-1, 1),
    lambda            = to_tune(1e-3, 1e3, logscale = TRUE),
    alpha             = to_tune(1e-3, 1e3, logscale = TRUE),
    subsample         = to_tune(1e-1, 1),
    nrounds           = to_tune(1, 5000)
  ),

  classif.catboost = list(
    learning_rate               = to_tune(1e-3, 1, logscale = TRUE),
    depth                       = to_tune(4, 10),
    l2_leaf_reg                 = to_tune(1, 10),
    max_ctr_complexity          = to_tune(1, 5),
    grow_policy                 = to_tune(c("SymmetricTree", "Depthwise")),
    iterations                  = to_tune(1, 5000),
    random_strength             = to_tune(1, 20),
    bagging_temperature         = to_tune(0, 1),
    leaf_estimation_iterations  = to_tune(1, 100),
    subsample                   = to_tune(0, 1),
    rsm                         = to_tune(0.1, 1),
    min_data_in_leaf            = to_tune(1, 5000),
    border_count                = to_tune(32, 255)
    # one_hot_max_size            = to_tune(2, 10)
  ),

  classif.lightgbm = list(
    learning_rate     = to_tune(1e-3, 0.2, logscale = TRUE),
    feature_fraction  = to_tune(0.1, 1),
    min_data_in_leaf  = to_tune(2, 200),
    num_leaves        = to_tune(10, 255),
    extra_trees       = to_tune(),
    num_iterations    = to_tune(1, 5000),
    bagging_fraction  = to_tune(0, 1),
    bagging_freq      = to_tune(0, 10),
    max_depth         = to_tune(15, 100),
    lambda_l1         = to_tune(1e-3, 100, logscale = TRUE),
    lambda_l2         = to_tune(1e-3, 100, logscale = TRUE),
    min_gain_to_split = to_tune(1e-3, 0.1, logscale = TRUE)
  )
)

param_values = map(learners, function(learner) {
  learner = learner$clone()
  learner$param_set$set_values(.values = tokens[[learner$id]])
  search_space = learner$param_set$search_space()

  samplers = map(search_space$subspaces(), function(subspace) {
    if (subspace$storage_type == "numeric") {
      m = (subspace$lower - subspace$upper) / 2
      Sampler1DNormal$new(subspace, mean = m, sd = m^2)
    } else {
      Sampler1DUnif$new(subspace)
    }
  })

  sampler = SamplerHierarchical$new(search_space, samplers)
  data = sampler$sample(1)$data
  search_space$check_dt(data)
  transpose_list(data)
})

design = benchmark_grid(
  tasks = tasks,
  learners = learners,
  resamplings = resamplings,
  param_values = param_values,
  paired = TRUE
)

batchmark(design, reg = reg)

submitJobs()
