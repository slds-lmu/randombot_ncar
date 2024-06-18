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

tasks = list()
resamplings = list()

for (id in task_ids) {
  otask = otsk(id = id)
  tasks = c(tasks, as_task(otask))
  resamplings = c(resamplings, as_resampling(otask))
}

addProblem("tasks", data = list(tasks = tasks, resamplings = resamplings))

# add algorithms
eval_config = function(data, job, instance, learner_id, param_values) {

  library(mlr3)
  library(mlr3learners)
  library(mlr3extralearners)
  library(mlr3oml)
  library(paradox)
  library(mlr3misc)
  library(mlr3batchmark)
  library(mlr3pipelines)

  tasks = data$tasks
  resamplings = data$resamplings

  source("learners.R")

  learner = learners[[learner_id]]
  learner$param_set$set_values(.values = param_values)
  learner$predict_type = "prob"
  learner$encapsulate = c(train = "callr", predict = "callr")
  learner$fallback = lrn("classif.featureless", predict_type = "prob")
  learner$timeout = c(train = 3600, predict = 3600)

  future::plan("multicore", workers = 128)

  grid = benchmark_grid(
    tasks = tasks,
    learners = learner,
    resamplings = resamplings,
    paired = TRUE
  )

  benchmark(grid, store_models = FALSE)
}

addAlgorithm(
  name = "eval_config",
  fun = eval_config
)

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

set.seed(1)

source("learners.R")

ades = map_dtr(learners, function(learner) {
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
  design = sampler$sample(5)
  search_space$check_dt(design$data)
  
  data.table(learner_id = learner$id, param_values = design$transpose())
})

addExperiments(
  algo.designs = list(eval_config = ades)
)

submitJobs(1)
