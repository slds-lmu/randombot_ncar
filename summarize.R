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
job_table = unnest(job_table, c("algo.pars", "prob.pars"))[1:105]
job_table[, learner_id := map_chr(learner, "id")]
job_table[, measure_id := map_chr(measure, "id")]

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

    data[, task_id := job$instance$task$id]
    data[, measure_id := job$instance$measure$id]
    data[, job_id := job_id]
    data[, learner_id := name]
    data
  })
  super_archive = rbindlist(super_archive, fill = TRUE, use.names = TRUE)
  setcolorder(super_archive, c("job_id", "task_id", "measure_id", "batch_nr", "errors", "warnings", "classif.ce", "classif.bacc", "classif.logloss", "classif.mauc_aunu", "classif.mcc"))

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

    data[, task_id := job$instance$task$id]
    data[, measure_id := job$instance$measure$id]
    data[, job_id := job_id]
    data[, learner_id := name]
    data
  })
  best = rbindlist(best, fill = TRUE, use.names = TRUE)
  setcolorder(best, c("job_id", "task_id", "measure_id", "batch_nr", "errors", "warnings", "classif.ce", "classif.bacc", "classif.logloss", "classif.mauc_aunu", "classif.mcc"))

  fwrite(best, sprintf("results/best_%s.csv", name))
})

# runtime
job_table[, used_time := unclass(time.running / (3600 * 12))]
job_table = job_table[, list(job.id, time.running, used_time, learner_id, problem, measure_id)]
job_table[order(used_time)]
job_table_learner = split(job_table, job_table$learner_id)

# error rate
super_archive = map(list.files("results", "super_archive", full.names = TRUE), function(path) {
  data = fread(path)
  data[, sum(errors)]
})
