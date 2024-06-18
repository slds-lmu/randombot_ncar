library(brew)
library(batchtools)

job_table = getJobTable()
job_ids = job_table$job.id

chunks = split(job_ids, ceiling(seq_along(job_ids) / 12))

walk(chunks[1], function(chunk) {
  env = new.env()

  assign("job.name", sprintf("job_%i_%i", chunk[1], chunk[length(chunk)]), env = env)
  assign("log.file", sprintf("job_%i_%i.log", chunk[1], chunk[length(chunk)]), env = env)

  walk(chunk, function(id) {
      jc = makeJobCollection(id)
      saveRDS(jc, jc$uri)
      assign(sprintf("uri_%i", id), jc$uri, env = env)
  })

  tmp = tempfile()
  brew("pbs_ncar.tmpl", output = tmp, envir = env)
  system2("qsub", tmp)
})

