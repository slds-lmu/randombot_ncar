job_table = getJobTable()
unnest(job_table, "algo.pars")[1:100]
