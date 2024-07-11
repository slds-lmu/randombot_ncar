library(brew)
library(batchtools)
library(mlr3misc)
library(uuid)
library(data.table)

reg = loadRegistry(
  file.dir = "/glade/derecho/scratch/marcbecker/randombot_ncar/registry", 
  conf.file = "batchtools.conf.R",
  writeable = FALSE)

job_table = getJobTable()
job_table = unnest(job_table, c("algo.pars", "prob.pars"))[1:140]
job_table[, learner_id := map_chr(learner, "id")]

job_table_learner = split(job_table, job_table$learner_id)

# super archive
iwalk(job_table_learner, function(tab, name) {
  job_ids_done = findDone()$job.id
  job_ids = tab$job.id
  super_archive = map(job_ids, function(job_id) {
    job = makeJob(job_id)
    if (job_id %in% job_ids_done) {
      instance = loadResult(job_id)
      data = as.data.table(instance$archive)
      if ("internal_tuned_values" %in% names(data)) data = unnest(data, "internal_tuned_values")
    } else {
      data = data.table()
    }

    data[, task := job$instance$task$id]
    data[, job_id := job_id]
    data[, learner := name]
    data[, measure := job$instance$measure]
    data
  })
  super_archive = rbindlist(super_archive, fill = TRUE, use.names = TRUE)
  setcolorder(super_archive, c("job_id", "learner", "task", "measure", "batch_nr", "errors", "warnings", "ce", "bacc", "logloss", "auc", "mcc"))

  fwrite(super_archive, sprintf("results/super_archive_%s.csv", name))
})

# best config
iwalk(job_table_learner, function(tab, name) {
  job_ids_done = findDone()$job.id
  job_ids = tab$job.id
  best = map(job_ids, function(job_id) {
    job = makeJob(job_id)
    if (job_id %in% job_ids_done) {
      instance = loadResult(job_id)
      data = instance$archive$best()
      data = unnest(data, "x_domain", prefix = "x_domain_")
      if ("internal_tuned_values" %in% names(data)) data = unnest(data, "internal_tuned_values")
    } else {
      data = data.table()
    }

    data[, task := job$instance$task$id]
    data[, job_id := job_id]
    data[, learner := name]
    data[, measure := job$instance$measure]
    data
  })
  best = rbindlist(best, fill = TRUE, use.names = TRUE)
  setcolorder(best, c("job_id", "learner", "task", "measure", "batch_nr", "errors", "warnings", "ce", "bacc", "logloss", "auc", "mcc"))

  fwrite(best, sprintf("results/best_%s.csv", name))
})

# misc
super_archives = map(list.files("results", "super_archive", full.names = TRUE), function(path) {
  data = fread(path)
})

runtime = map_dtr(super_archives, function(super_archive) {
  super_archive[, list(
    min_runtime = min(runtime_learners, na.rm = TRUE) / 10 , 
    mean_runtime = median(runtime_learners, na.rm = TRUE) / 10, 
    max_runtime = max(runtime_learners, na.rm = TRUE) / 10, 
    total_runtime = sum(runtime_learners, na.rm = TRUE) / 10 / 60,
    total_config = .N,
    total_errors = sum(errors)), by = list(learner, task, measure)]
})

fwrite(runtime, "results/runtime.csv")

# iterations
catboost_boosting = super_archives[[1]][, list(min_iterations = min(catboost.iterations, na.rm = TRUE), mean_iterations = median(catboost.iterations, na.rm = TRUE), max_iterations =  max(catboost.iterations, na.rm = TRUE)), by = .(learner, task, measure)]
lightgbm_boosting = super_archives[[4]][, list(min_iterations = min(lightgbm.num_iterations, na.rm = TRUE), mean_iterations = median(lightgbm.num_iterations, na.rm = TRUE), max_iterations =  max(lightgbm.num_iterations, na.rm = TRUE)), by = .(learner, task, measure)]
xgboost_boosting = super_archives[[7]][, list(min_iterations = min(xgboost.nrounds, na.rm = TRUE), mean_iterations = median(xgboost.nrounds, na.rm = TRUE), max_iterations =  max(xgboost.nrounds, na.rm = TRUE)), by = .(learner, task, measure)]

boosting = rbindlist(list(catboost_boosting, lightgbm_boosting, xgboost_boosting), fill = TRUE, use.names = TRUE)

fwrite(boosting, "results/boosting.csv")

# incumbent
super_archive = map_dtr(list.files("results", "super_archive", full.names = TRUE), function(path) {
  data = fread(path)
  data[, list(learner, task, measure, ce, bacc, logloss, auc, mcc)]
})

incumbent_ce = super_archive["ce", , on = "measure"][, best_ce := cummin(ce), by = .(learner, task, measure)]
incumbent_ce = incumbent_ce[, batch := seq(.N), by = .(learner, task, measure)][, list(learner, task, batch, best_ce)]

incumbent_bacc = super_archive["bacc", , on = "measure"][, best_bacc := cummax(bacc), by = .(learner, task, measure)]
incumbent_bacc = incumbent_bacc[, batch := seq(.N), by = .(learner, task, measure)][, list(learner, task, batch, best_bacc)]

incumbent_logloss = super_archive["logloss", , on = "measure"][, best_logloss := cummin(logloss), by = .(learner, task, measure)]
incumbent_logloss = incumbent_logloss[, batch := seq(.N), by = .(learner, task, measure)][, list(learner, task, batch, best_logloss)]

incumbent_auc = super_archive["auc", , on = "measure"][, best_auc := cummax(auc), by = .(learner, task, measure)]
incumbent_auc = incumbent_auc[, batch := seq(.N), by = .(learner, task, measure)][, list(learner, task, batch, best_auc)]

incumbent_mcc = super_archive["mcc", , on = "measure"][, best_mcc := cummax(mcc), by = .(learner, task, measure)]
incumbent_mcc = incumbent_mcc[, batch := seq(.N), by = .(learner, task, measure)][, list(learner, task, batch, best_mcc)]

incumbent = rbindlist(list(incumbent_ce, incumbent_bacc, incumbent_logloss, incumbent_auc, incumbent_mcc), fill = TRUE, use.names = TRUE)

incumbent = melt(incumbent, id.vars = c("learner", "task", "batch"), measure.vars = c("best_ce", "best_bacc", "best_logloss", "best_auc", "best_mcc"), variable.name = "measure", value.name = "score")
incumbent = incumbent[!is.na(score)]

library(ggplot2)


# incumbent over iterations
data = incumbent[task == "ilpd"]
data = data[!is.na(score)]


fwrite(incumbent, "results/incumbent.csv")


ggplot(data, aes(x = batch, y = score, color = learner)) + 
  geom_line() + 
  facet_wrap(~measure, scales = "free_y")
