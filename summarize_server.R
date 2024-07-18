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
job_table = unnest(job_table, c("algo.pars", "prob.pars"))[1:420]
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

      if ("internal_tuned_values" %in% names(data)) {
        data = unnest(data, "internal_tuned_values")
        id = instance$archive$internal_search_space$ids()
        data[, paste0("x_domain_", id) := .SD[[id]]]
      }

      # fix parameter names
      param_ids = c(instance$archive$cols_x, instance$archive$internal_search_space$ids())
      setnames(data, param_ids, gsub(paste0(name, "\\."), "", param_ids))
      x_domain_ids = grep("x_domain_", names(data), value = TRUE)
      new_x_domain_ids = gsub(paste0(name, "\\."), "", x_domain_ids)
      setnames(data, x_domain_ids, new_x_domain_ids)
      data

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

      if ("internal_tuned_values" %in% names(data)) {
        data = unnest(data, "internal_tuned_values")
        id = instance$archive$internal_search_space$ids()
        data[, paste0("x_domain_", id) := .SD[[id]]]
      }

      # fix parameter names
      param_ids = c(instance$archive$cols_x, instance$archive$internal_search_space$ids())
      setnames(data, param_ids, gsub(paste0(name, "\\."), "", param_ids), skip_absent=TRUE)
      x_domain_ids = grep("x_domain_", names(data), value = TRUE)
      new_x_domain_ids = gsub(paste0(name, "\\."), "", x_domain_ids)
      setnames(data, x_domain_ids, new_x_domain_ids)
      data

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

