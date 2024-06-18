library(batchtools)

reg = makeRegistry(
  file.dir = "test_1/registry", 
  conf.file = "test_1/batchtools.conf.R",
  seed = 1)

fun = function(n) {
  future::plan("multisession", workers = 128)
}

batchMap(fun = fun, n = 1)

submitJobs()
