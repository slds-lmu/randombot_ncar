  install.packages("Rmpi", configure.args = c(
    paste0("--with-Rmpi-include=/opt/cray/pe/mpich/8.1.27/ofi/intel/2022.1/include"),
    paste0("--with-Rmpi-libpath=/opt/cray/pe/mpich/8.1.27/ofi/intel/2022.1/lib"),
    paste0("--with-Rmpi-type=MPICH")
  ))