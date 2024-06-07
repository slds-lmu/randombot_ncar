cluster.functions = batchtools::makeClusterFunctionsTORQUE("/glade/u/home/marcbecker/randombot_ncar/pbs_ncar.tmpl")
cluster.functions$array.var = "PBS_ARRAY_INDEX"
default.resources = list(walltime = "1:00:00", chunks.as.arrayjobs = TRUE)
max.concurrent.jobs = 4864L


