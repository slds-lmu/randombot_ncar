job_table = getJobTable()
tab = unnest(job_table, "algo.pars")
tab
tab[, lid := map_chr(learner, "id")]
job_ids = tab[c("xgboost", "catboost", "lightgbm"), job.id, on = "lid"]
