target datalayout = "e-p6:32:32-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x() nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.y() nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.y() nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.tid.z() nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.z() nounwind readnone
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.z() nounwind readnone

define i64 @_Z13get_global_idj(i32 %dim) nounwind readnone {
  switch i32 %dim, label %default [
    i32 0, label %dim0
    i32 1, label %dim1
    i32 2, label %dim2
  ]
dim0:
  %tid_x   = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %ntid_x  = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %ctaid_x = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %prod_x  = mul i32 %ctaid_x, %ntid_x
  %gid_x   = add i32 %prod_x, %tid_x
  %r0      = zext i32 %gid_x to i64
  ret i64 %r0
dim1:
  %tid_y   = call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
  %ntid_y  = call i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
  %ctaid_y = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  %prod_y  = mul i32 %ctaid_y, %ntid_y
  %gid_y   = add i32 %prod_y, %tid_y
  %r1      = zext i32 %gid_y to i64
  ret i64 %r1
dim2:
  %tid_z   = call i32 @llvm.nvvm.read.ptx.sreg.tid.z()
  %ntid_z  = call i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
  %ctaid_z = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.z()
  %prod_z  = mul i32 %ctaid_z, %ntid_z
  %gid_z   = add i32 %prod_z, %tid_z
  %r2      = zext i32 %gid_z to i64
  ret i64 %r2
default:
  ret i64 0
}
