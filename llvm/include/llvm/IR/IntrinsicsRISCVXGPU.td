let TargetPrefix = "riscv" in {

  class GroomCSRIntrinsicsImpl<LLVMType itype>
      : Intrinsic<[itype], [], [IntrNoMem, IntrWillReturn]>;

  multiclass GroomCSRIntrinsics {
    def _i32 : GroomCSRIntrinsicsImpl<llvm_i32_ty>;
    def _i64 : GroomCSRIntrinsicsImpl<llvm_i64_ty>;
  }

  class GroomD0S1IntrinsicsImpl<LLVMType itype>
      : Intrinsic<[], [itype], []>;

  multiclass GroomD0S1Intrinsics {
    def _i32 : GroomD0S1IntrinsicsImpl<llvm_i32_ty>;
    def _i64 : GroomD0S1IntrinsicsImpl<llvm_i64_ty>;
  }

  defm int_riscv_gpu_tmask : GroomCSRIntrinsics;
  defm int_riscv_gpu_tmc   : GroomD0S1Intrinsics;
  defm int_riscv_gpu_pred  : GroomD0S1Intrinsics;
  defm int_riscv_gpu_split : GroomD0S1Intrinsics;
  def int_riscv_gpu_join  : Intrinsic<[], [], []>;

}
