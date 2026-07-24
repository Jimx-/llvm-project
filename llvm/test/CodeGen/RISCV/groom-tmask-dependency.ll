; RUN: llc -mtriple=riscv32 -mattr=+groom -stop-after=groom-tmask-dependency \
; RUN:   -o - %s | FileCheck %s --check-prefix=MIR
; RUN: llc -mtriple=riscv32 -mattr=+groom -verify-machineinstrs -o - < %s \
; RUN:   | FileCheck %s --check-prefix=ASM

define i32 @sfb(i1 %cond, ptr %ptr) {
entry:
  br i1 %cond, label %then, label %merge

then:
  %value = load i32, ptr %ptr
  br label %merge

merge:
  %result = phi i32 [ 0, %entry ], [ %value, %then ]
  ret i32 %result
}

; MIR-LABEL: name: sfb
; MIR: GPU_PRED
; MIR-SAME: implicit-def $tmask
; MIR-SAME: implicit $tmask
; MIR: LW {{.*}} implicit $tmask
; MIR: GPU_TMC
; MIR-SAME: implicit-def $tmask
; MIR-SAME: implicit $tmask

; ASM-LABEL: sfb:
; ASM-NOT: gpu_split
; ASM: gpu_pred
; ASM: lw
; ASM: gpu_tmc
