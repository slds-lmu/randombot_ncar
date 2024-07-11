library(brew)
library(uuid)
library(batchtools)
library(mlr3)
library(mlr3learners)
library(mlr3extralearners)
library(mlr3oml)
library(paradox)
library(mlr3misc)
library(mlr3batchmark)
library(mlr3pipelines)
library(data.table)

# unlink("/glade/derecho/scratch/marcbecker/randombot_ncar/registry", recursive = TRUE)

# reg = makeExperimentRegistry(
#   file.dir = "/glade/derecho/scratch/marcbecker/randombot_ncar/registry", 
#   conf.file = "batchtools.conf.R",
#   seed = 7832)

reg = loadRegistry(
  file.dir = "/glade/derecho/scratch/marcbecker/randombot_ncar/registry", 
  conf.file = "batchtools.conf.R",
  writeable = TRUE)

# source("problems.R")
# source("algorithms.R")
# source("experiments.R")

job_ids = setdiff(findNotDone()$job.id, findRunning()$job.id)

chunks = split(job_ids, ceiling(seq_along(job_ids) / 12))

time = format(Sys.time(), "%Y-%m-%d_%H-%M-%S")

walk(chunks[1], function(chunk) {
  env = new.env()
  set(reg$status, i = chunk, j = "started", value = NA_integer_)
  set(reg$status, i = chunk, j = "done", value = NA_integer_)
  set(reg$status, i = chunk, j = "error", value = NA)

  hash = sprintf("%s_%s", time, str_collapse(chunk, "_"))

  assign("job.name", sprintf("job_%s", hash), env = env)
  assign("log.file", sprintf("logs_nodes/job_%s.log", hash), env = env)

  iwalk(chunk, function(id, i) {
      jc = makeJobCollection(id)
      saveRDS(jc, jc$uri)
      assign(sprintf("uri_%i", i), jc$uri, env = env)
  })

  tmp = tempfile()
  brew("pbs_ncar.tmpl", output = tmp, envir = env)
  batch.id = system2("qsub", tmp, stdout = TRUE)

  message(batch.id)

  set(reg$status, i = chunk, j = "log.file", value = sprintf("%s.log", chunk))
  set(reg$status, i = chunk, j = "log.file", value = sprintf("%s.log", chunk))
  set(reg$status, i = chunk, j = "job.name", value = sprintf("job_%s", hash))
  set(reg$status, i = chunk, j = "submitted", value = Sys.time())
  set(reg$status, i = chunk, j = "batch.id", value = batch.id)
  saveRegistry(reg)
})
