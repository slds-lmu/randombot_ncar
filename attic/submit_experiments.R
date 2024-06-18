library(batchtools)

reg = loadRegistry(file.dir = "/glade/derecho/scratch/marcbecker/randombot_ncar/registry", writeable = TRUE)

# ids = getJobTable()[, list(job.id)]
# ids[, chunk := batchtools::chunk(job.id, chunk.size = 128, shuffle = FALSE)]

submitJobs(1)
