job_table = getJobTable()
tab = unnest(job_table, "algo.pars")[1:35]
tab[, lid := map_chr(learner, "id")]
job_ids = tab[c("xgboost", "catboost", "lightgbm"), job.id, on = "lid"]
