library(batchtools)

reg = makeRegistry(
  file.dir = "test_2/registry", 
  conf.file = "test_2/batchtools.conf.R",
  seed = 1)

fun = function(n) {
  future::plan("multicore", workers = 128)

  fun = function(n) {
    start_time = Sys.time()
    Sys.sleep(2)
    list(n = n, pid = Sys.getpid(), timestamp = Sys.time(), start_time = start_time)
  }

  future.apply::future_lapply(seq(128), fun)
}

batchMap(fun = fun, n = 1)

submitJobs()
