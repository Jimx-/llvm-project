let TargetPrefix = "riscv" in {

  def int_riscv_gpu_tmask : Intrinsic<[llvm_anyint_ty], [], [IntrNoMem, IntrWillReturn]>;
  def int_riscv_gpu_tmc   : Intrinsic<[], [llvm_anyint_ty], []>;
  def int_riscv_gpu_pred  : Intrinsic<[], [llvm_anyint_ty], []>;
  def int_riscv_gpu_split : Intrinsic<[], [llvm_anyint_ty], []>;
  def int_riscv_gpu_join  : Intrinsic<[], [], []>;
  
}
