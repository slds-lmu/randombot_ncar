library(brew)
library(batchtools)
library(mlr3misc)
library(uuid)

reg = loadRegistry(
  file.dir = "/glade/derecho/scratch/marcbecker/randombot_ncar/registry", 
  conf.file = "batchtools.conf.R",
  writeable = TRUE)

job_ids = findNotDone()$job.id

chunks = split(job_ids, ceiling(seq_along(job_ids) / 12))

time = format(Sys.time(), "%Y-%m-%d_%H-%M-%S")

walk(chunks[57], function(chunk) {
  env = new.env()

  hash = sprintf("%s_%s", time, str_collapse(chunk, "_"))

  assign("job.name", sprintf("job_%s", hash), env = env)
  assign("log.file", sprintf("job_%s.log", hash), env = env)

  iwalk(chunk, function(id, i) {
      jc = makeJobCollection(id)
      saveRDS(jc, jc$uri)
      assign(sprintf("uri_%i", i), jc$uri, env = env)
  })

  tmp = tempfile()
  brew("pbs_ncar.tmpl", output = tmp, envir = env)
  system2("qsub", tmp)
})

