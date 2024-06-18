library(batchtools)

reg = makeRegistry(
  file.dir = "test_2/registry", 
  conf.file = "test_2/batchtools.conf.R",
  seed = 1)

fun = function(n) {
  future::plan("multicore", workers = 128)
}

batchMap(fun = fun, n = 1)

submitJobs()
