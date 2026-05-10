; ModuleID = '/home/ambikas2/cs526/bench_results/bc/bounded_kernels.bc'
source_filename = "/home/ambikas2/cs526/kernels/bounded_kernels.cl"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spir64-unknown-unknown"

; Function Attrs: convergent noinline norecurse nounwind
define dso_local spir_kernel void @bounded_strided(ptr addrspace(1) noundef align 4 %A, ptr addrspace(1) noundef align 4 %B, i32 noundef %N) #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !5 !kernel_arg_type_qual !6 {
entry:
  %A.addr.i = alloca ptr addrspace(1), align 8
  %B.addr.i = alloca ptr addrspace(1), align 8
  %N.addr.i = alloca i32, align 4
  %tid.i = alloca i32, align 4
  %A.addr = alloca ptr addrspace(1), align 8
  %B.addr = alloca ptr addrspace(1), align 8
  %N.addr = alloca i32, align 4
  store ptr addrspace(1) %A, ptr %A.addr, align 8
  store ptr addrspace(1) %B, ptr %B.addr, align 8
  store i32 %N, ptr %N.addr, align 4
  %0 = load ptr addrspace(1), ptr %A.addr, align 8
  %1 = load ptr addrspace(1), ptr %B.addr, align 8
  %2 = load i32, ptr %N.addr, align 4
  store ptr addrspace(1) %0, ptr %A.addr.i, align 8
  store ptr addrspace(1) %1, ptr %B.addr.i, align 8
  store i32 %2, ptr %N.addr.i, align 4
  %call.i = call spir_func i64 @_Z13get_global_idj(i32 noundef 0) #3
  %conv.i = trunc i64 %call.i to i32
  store i32 %conv.i, ptr %tid.i, align 4
  %3 = load i32, ptr %tid.i, align 4
  %mul.i = mul nsw i32 4, %3
  %4 = load i32, ptr %N.addr.i, align 4
  %cmp.i = icmp slt i32 %mul.i, %4
  br i1 %cmp.i, label %if.then.i, label %__clang_ocl_kern_imp_bounded_strided.exit

if.then.i:                                        ; preds = %entry
  %5 = load ptr addrspace(1), ptr %A.addr.i, align 8
  %6 = load i32, ptr %tid.i, align 4
  %mul2.i = mul nsw i32 4, %6
  %idxprom.i = sext i32 %mul2.i to i64
  %arrayidx.i = getelementptr inbounds float, ptr addrspace(1) %5, i64 %idxprom.i
  %7 = load float, ptr addrspace(1) %arrayidx.i, align 4
  %8 = load ptr addrspace(1), ptr %B.addr.i, align 8
  %9 = load i32, ptr %tid.i, align 4
  %idxprom3.i = sext i32 %9 to i64
  %arrayidx4.i = getelementptr inbounds float, ptr addrspace(1) %8, i64 %idxprom3.i
  store float %7, ptr addrspace(1) %arrayidx4.i, align 4
  br label %__clang_ocl_kern_imp_bounded_strided.exit

__clang_ocl_kern_imp_bounded_strided.exit:        ; preds = %if.then.i, %entry
  ret void
}

; Function Attrs: alwaysinline convergent norecurse nounwind
define dso_local spir_func void @__clang_ocl_kern_imp_bounded_strided(ptr addrspace(1) noundef align 4 %A, ptr addrspace(1) noundef align 4 %B, i32 noundef %N) #1 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !5 !kernel_arg_type_qual !6 {
entry:
  %A.addr = alloca ptr addrspace(1), align 8
  %B.addr = alloca ptr addrspace(1), align 8
  %N.addr = alloca i32, align 4
  %tid = alloca i32, align 4
  store ptr addrspace(1) %A, ptr %A.addr, align 8
  store ptr addrspace(1) %B, ptr %B.addr, align 8
  store i32 %N, ptr %N.addr, align 4
  %call = call spir_func i64 @_Z13get_global_idj(i32 noundef 0) #3
  %conv = trunc i64 %call to i32
  store i32 %conv, ptr %tid, align 4
  %0 = load i32, ptr %tid, align 4
  %mul = mul nsw i32 4, %0
  %1 = load i32, ptr %N.addr, align 4
  %cmp = icmp slt i32 %mul, %1
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %2 = load ptr addrspace(1), ptr %A.addr, align 8
  %3 = load i32, ptr %tid, align 4
  %mul2 = mul nsw i32 4, %3
  %idxprom = sext i32 %mul2 to i64
  %arrayidx = getelementptr inbounds float, ptr addrspace(1) %2, i64 %idxprom
  %4 = load float, ptr addrspace(1) %arrayidx, align 4
  %5 = load ptr addrspace(1), ptr %B.addr, align 8
  %6 = load i32, ptr %tid, align 4
  %idxprom3 = sext i32 %6 to i64
  %arrayidx4 = getelementptr inbounds float, ptr addrspace(1) %5, i64 %idxprom3
  store float %4, ptr addrspace(1) %arrayidx4, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

; Function Attrs: convergent nounwind willreturn memory(none)
declare spir_func i64 @_Z13get_global_idj(i32 noundef) #2

attributes #0 = { convergent noinline norecurse nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "uniform-work-group-size"="false" }
attributes #1 = { alwaysinline convergent norecurse nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "uniform-work-group-size"="false" }
attributes #2 = { convergent nounwind willreturn memory(none) "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { convergent nounwind willreturn memory(none) }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
!2 = !{!"clang version 22.1.4 (https://github.com/conda-forge/clangdev-feedstock 8fb2e1c666e3daad00e02a3278e63348e3c9ffcb)"}
!3 = !{i32 1, i32 1, i32 0}
!4 = !{!"none", !"none", !"none"}
!5 = !{!"float*", !"float*", !"int"}
!6 = !{!"", !"", !""}
