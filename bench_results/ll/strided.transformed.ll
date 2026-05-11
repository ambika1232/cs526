; ModuleID = '/home/ambikas2/cs526/bench_results/bc/strided.bc'
source_filename = "/home/ambikas2/cs526/kernels/strided.cl"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spir64-unknown-unknown"

; Function Attrs: convergent noinline norecurse nounwind
define dso_local spir_kernel void @test(ptr addrspace(1) noundef align 4 %A, ptr addrspace(1) noundef align 4 %B) #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !5 !kernel_arg_type_qual !6 {
entry:
  %A.addr.i = alloca ptr addrspace(1), align 8
  %B.addr.i = alloca ptr addrspace(1), align 8
  %tid.i = alloca i32, align 4
  %A.addr = alloca ptr addrspace(1), align 8
  %B.addr = alloca ptr addrspace(1), align 8
  store ptr addrspace(1) %A, ptr %A.addr, align 8
  store ptr addrspace(1) %B, ptr %B.addr, align 8
  %0 = load ptr addrspace(1), ptr %A.addr, align 8
  %1 = load ptr addrspace(1), ptr %B.addr, align 8
  store ptr addrspace(1) %0, ptr %A.addr.i, align 8
  store ptr addrspace(1) %1, ptr %B.addr.i, align 8
  %call.i = call spir_func i64 @_Z13get_global_idj(i32 noundef 0) #3
  %conv.i = trunc i64 %call.i to i32
  store i32 %conv.i, ptr %tid.i, align 4
  %2 = load ptr addrspace(1), ptr %A.addr.i, align 8
  %3 = load i32, ptr %tid.i, align 4
  %mul.i = mul nsw i32 %3, 4
  %idxprom.i = sext i32 %mul.i to i64
  %arrayidx.i = getelementptr inbounds float, ptr addrspace(1) %2, i64 %idxprom.i
  %4 = load float, ptr addrspace(1) %arrayidx.i, align 4
  %5 = load ptr addrspace(1), ptr %B.addr.i, align 8
  %6 = load i32, ptr %tid.i, align 4
  %idxprom1.i = sext i32 %6 to i64
  %arrayidx2.i = getelementptr inbounds float, ptr addrspace(1) %5, i64 %idxprom1.i
  store float %4, ptr addrspace(1) %arrayidx2.i, align 4
  ret void
}

; Function Attrs: alwaysinline convergent norecurse nounwind
define dso_local spir_func void @__clang_ocl_kern_imp_test(ptr addrspace(1) noundef align 4 %A, ptr addrspace(1) noundef align 4 %B) #1 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !5 !kernel_arg_type_qual !6 {
entry:
  %A.addr = alloca ptr addrspace(1), align 8
  %B.addr = alloca ptr addrspace(1), align 8
  %tid = alloca i32, align 4
  store ptr addrspace(1) %A, ptr %A.addr, align 8
  store ptr addrspace(1) %B, ptr %B.addr, align 8
  %call = call spir_func i64 @_Z13get_global_idj(i32 noundef 0) #3
  %conv = trunc i64 %call to i32
  store i32 %conv, ptr %tid, align 4
  %0 = load ptr addrspace(1), ptr %A.addr, align 8
  %1 = load i32, ptr %tid, align 4
  %mul = mul nsw i32 %1, 4
  %idxprom = sext i32 %mul to i64
  %arrayidx = getelementptr inbounds float, ptr addrspace(1) %0, i64 %idxprom
  %2 = load float, ptr addrspace(1) %arrayidx, align 4
  %3 = load ptr addrspace(1), ptr %B.addr, align 8
  %4 = load i32, ptr %tid, align 4
  %idxprom1 = sext i32 %4 to i64
  %arrayidx2 = getelementptr inbounds float, ptr addrspace(1) %3, i64 %idxprom1
  store float %2, ptr addrspace(1) %arrayidx2, align 4
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
!3 = !{i32 1, i32 1}
!4 = !{!"none", !"none"}
!5 = !{!"float*", !"float*"}
!6 = !{!"", !""}
