# add problems 
# tasks and resamplings

# collection_218 = ocl(218)
# collection_99 = ocl(99)

# tab_218 = list_oml_tasks(
#   task_id = collection_218$task_ids,
#   number_classes = c(2, 10),
#   number_instances = c(1, 100000)
# )

# tab_99 = list_oml_tasks(
#   task_id = collection_99$task_ids,
#   number_classes = c(2, 10),
#   number_instances = c(1, 100000)
# )

# tab_99 = tab_99[!tab_218$name, , on = "name"]
# tab = rbindlist(list(tab_218, tab_99))
# tab = tab[, c("task_id", "name", "NumberOfClasses", "NumberOfFeatures", "NumberOfInstances", "NumberOfInstancesWithMissingValues", "NumberOfNumericFeatures", "NumberOfSymbolicFeatures"), with = FALSE]
# tab[, Size := NumberOfClasses * NumberOfFeatures * NumberOfInstances]
# tab = tab[order(Size)]
# knitr::kable(tab)

# task_ids = tab$task_id
# # download of task albert fails
# task_ids = task_ids[task_ids != 189356]

# connection to openml often times out
# sorted by n*p*c
task_ids = c(10101L, 11L, 9971L, 125920L, 10093L, 37L, 15L, 49L, 146818L, 
  29L, 146819L, 3913L, 3560L, 9946L, 31L, 14954L, 23L, 146821L, 
  3918L, 146820L, 53L, 9952L, 2079L, 9957L, 3917L, 3902L, 3903L, 
  18L, 167141L, 168912L, 3021L, 3L, 3549L, 146822L, 9978L, 146817L, 
  3904L, 43L, 9960L, 45L, 34539L, 14952L, 146800L, 219L, 168911L, 
  167119L, 22L, 16L, 2074L, 7592L, 14965L, 14L, 14969L, 167140L, 
  32L, 9985L, 9976L, 28L, 146212L, 9964L, 167120L, 12L, 146824L, 
  146606L, 9977L, 9981L, 146195L, 167125L, 9910L, 168908L, 168330L, 
  3945L, 168868L, 14970L, 168910L, 168909L, 168331L, 168337L, 168338L, 
  146825L, 3573L, 168332L, 167124L)

walk(task_ids, function(id) {
  try({
    otask = otsk(id = id)
    task = as_task(otask)
    resampling = as_resampling(otask)

    addProblem(
      name = otask$data_name,
      data = list(task = task, resampling = resampling),
      fun = function(data, job, measure, ...) {
        c(data, list(measure = measure))
      })

    rm(otask)
    rm(task)
    rm(resampling)
    gc()
  })
})

