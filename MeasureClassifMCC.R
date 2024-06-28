# create multiclass mcc
MeasureClassifMCC = R6::R6Class("MeasureClassifMCC",
  inherit = MeasureClassif,
  public = list(
    #' @description
    #' Creates a new instance of this [R6][R6::R6Class] class.
    initialize = function() {
      super$initialize(
        id = "classif.mcc",
        param_set = ps(),
        range = c(-1, 1),
        minimize = FALSE,
        label = "Matthews Correlation Coefficient",
        man = "mlr3::mlr_measures_classif.mcc"
      )
    }
  ),

  private = list(
    .score = function(prediction, ...) {
      m = table(response = prediction$response, truth = prediction$truth)

      t_sum = rowSums(m)
      p_sum = colSums(m)
      n_correct = sum(diag(m))
      n_samples = sum(p_sum)

      cov_ytyp = n_correct * n_samples - sum(t_sum * p_sum)
      cov_ypyp = n_samples^2 - sum(p_sum^2)
      cov_ytyt = n_samples^2 - sum(t_sum^2)

      if (cov_ypyp * cov_ytyt == 0) return(0)
      cov_ytyp / sqrt(cov_ytyt * cov_ypyp)
    }
  )  
)