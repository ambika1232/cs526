; ModuleID = '/home/ambikas2/cs526/bench_results/bc/2mm.bc'
source_filename = "/home/ambikas2/cs526/kernels/2mm.cl"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spir64-unknown-unknown"

; Function Attrs: convergent noinline norecurse nounwind
define dso_local spir_kernel void @mm2_kernel1(ptr addrspace(1) noundef align 4 %tmp, ptr addrspace(1) noundef align 4 %A, ptr addrspace(1) noundef align 4 %B, i32 noundef %ni, i32 noundef %nj, i32 noundef %nk, i32 noundef %nl, float noundef %alpha, float noundef %beta) #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !6 !kernel_arg_type_qual !7 {
entry:
  %tmp.addr.i = alloca ptr addrspace(1), align 8
  %A.addr.i = alloca ptr addrspace(1), align 8
  %B.addr.i = alloca ptr addrspace(1), align 8
  %ni.addr.i = alloca i32, align 4
  %nj.addr.i = alloca i32, align 4
  %nk.addr.i = alloca i32, align 4
  %nl.addr.i = alloca i32, align 4
  %alpha.addr.i = alloca float, align 4
  %beta.addr.i = alloca float, align 4
  %j.i = alloca i32, align 4
  %i.i = alloca i32, align 4
  %k.i = alloca i32, align 4
  %tmp.addr = alloca ptr addrspace(1), align 8
  %A.addr = alloca ptr addrspace(1), align 8
  %B.addr = alloca ptr addrspace(1), align 8
  %ni.addr = alloca i32, align 4
  %nj.addr = alloca i32, align 4
  %nk.addr = alloca i32, align 4
  %nl.addr = alloca i32, align 4
  %alpha.addr = alloca float, align 4
  %beta.addr = alloca float, align 4
  store ptr addrspace(1) %tmp, ptr %tmp.addr, align 8
  store ptr addrspace(1) %A, ptr %A.addr, align 8
  store ptr addrspace(1) %B, ptr %B.addr, align 8
  store i32 %ni, ptr %ni.addr, align 4
  store i32 %nj, ptr %nj.addr, align 4
  store i32 %nk, ptr %nk.addr, align 4
  store i32 %nl, ptr %nl.addr, align 4
  store float %alpha, ptr %alpha.addr, align 4
  store float %beta, ptr %beta.addr, align 4
  %0 = load ptr addrspace(1), ptr %tmp.addr, align 8
  %1 = load ptr addrspace(1), ptr %A.addr, align 8
  %2 = load ptr addrspace(1), ptr %B.addr, align 8
  %3 = load i32, ptr %ni.addr, align 4
  %4 = load i32, ptr %nj.addr, align 4
  %5 = load i32, ptr %nk.addr, align 4
  %6 = load i32, ptr %nl.addr, align 4
  %7 = load float, ptr %alpha.addr, align 4
  %8 = load float, ptr %beta.addr, align 4
  store ptr addrspace(1) %0, ptr %tmp.addr.i, align 8
  store ptr addrspace(1) %1, ptr %A.addr.i, align 8
  store ptr addrspace(1) %2, ptr %B.addr.i, align 8
  store i32 %3, ptr %ni.addr.i, align 4
  store i32 %4, ptr %nj.addr.i, align 4
  store i32 %5, ptr %nk.addr.i, align 4
  store i32 %6, ptr %nl.addr.i, align 4
  store float %7, ptr %alpha.addr.i, align 4
  store float %8, ptr %beta.addr.i, align 4
  %call.i = call spir_func i64 @_Z13get_global_idj(i32 noundef 0) #4
  %conv.i = trunc i64 %call.i to i32
  store i32 %conv.i, ptr %j.i, align 4
  %call1.i = call spir_func i64 @_Z13get_global_idj(i32 noundef 1) #4
  %conv2.i = trunc i64 %call1.i to i32
  store i32 %conv2.i, ptr %i.i, align 4
  %9 = load i32, ptr %i.i, align 4
  %10 = load i32, ptr %ni.addr.i, align 4
  %cmp.i = icmp slt i32 %9, %10
  br i1 %cmp.i, label %land.lhs.true.i, label %__clang_ocl_kern_imp_mm2_kernel1.exit

land.lhs.true.i:                                  ; preds = %entry
  %11 = load i32, ptr %j.i, align 4
  %12 = load i32, ptr %nj.addr.i, align 4
  %cmp4.i = icmp slt i32 %11, %12
  br i1 %cmp4.i, label %if.then.i, label %__clang_ocl_kern_imp_mm2_kernel1.exit

if.then.i:                                        ; preds = %land.lhs.true.i
  %13 = load ptr addrspace(1), ptr %tmp.addr.i, align 8
  %14 = load i32, ptr %i.i, align 4
  %15 = load i32, ptr %nj.addr.i, align 4
  %mul.i = mul nsw i32 %14, %15
  %16 = load i32, ptr %j.i, align 4
  %add.i = add nsw i32 %mul.i, %16
  %idxprom.i = sext i32 %add.i to i64
  %arrayidx.i = getelementptr inbounds float, ptr addrspace(1) %13, i64 %idxprom.i
  store float 0.000000e+00, ptr addrspace(1) %arrayidx.i, align 4
  store i32 0, ptr %k.i, align 4
  br label %for.cond.i

for.cond.i:                                       ; preds = %for.body.i, %if.then.i
  %17 = load i32, ptr %k.i, align 4
  %18 = load i32, ptr %nk.addr.i, align 4
  %cmp6.i = icmp slt i32 %17, %18
  br i1 %cmp6.i, label %for.body.i, label %for.end.i

for.body.i:                                       ; preds = %for.cond.i
  %19 = load float, ptr %alpha.addr.i, align 4
  %20 = load ptr addrspace(1), ptr %A.addr.i, align 8
  %21 = load i32, ptr %i.i, align 4
  %22 = load i32, ptr %nk.addr.i, align 4
  %mul8.i = mul nsw i32 %21, %22
  %23 = load i32, ptr %k.i, align 4
  %add9.i = add nsw i32 %mul8.i, %23
  %idxprom10.i = sext i32 %add9.i to i64
  %arrayidx11.i = getelementptr inbounds float, ptr addrspace(1) %20, i64 %idxprom10.i
  %24 = load float, ptr addrspace(1) %arrayidx11.i, align 4
  %mul12.i = fmul float %19, %24
  %25 = load ptr addrspace(1), ptr %B.addr.i, align 8
  %26 = load i32, ptr %k.i, align 4
  %27 = load i32, ptr %nj.addr.i, align 4
  %mul13.i = mul nsw i32 %26, %27
  %28 = load i32, ptr %j.i, align 4
  %add14.i = add nsw i32 %mul13.i, %28
  %idxprom15.i = sext i32 %add14.i to i64
  %arrayidx16.i = getelementptr inbounds float, ptr addrspace(1) %25, i64 %idxprom15.i
  %29 = load float, ptr addrspace(1) %arrayidx16.i, align 4
  %30 = load ptr addrspace(1), ptr %tmp.addr.i, align 8
  %31 = load i32, ptr %i.i, align 4
  %32 = load i32, ptr %nj.addr.i, align 4
  %mul18.i = mul nsw i32 %31, %32
  %33 = load i32, ptr %j.i, align 4
  %add19.i = add nsw i32 %mul18.i, %33
  %idxprom20.i = sext i32 %add19.i to i64
  %arrayidx21.i = getelementptr inbounds float, ptr addrspace(1) %30, i64 %idxprom20.i
  %34 = load float, ptr addrspace(1) %arrayidx21.i, align 4
  %35 = call float @llvm.fmuladd.f32(float %mul12.i, float %29, float %34)
  store float %35, ptr addrspace(1) %arrayidx21.i, align 4
  %36 = load i32, ptr %k.i, align 4
  %inc.i = add nsw i32 %36, 1
  store i32 %inc.i, ptr %k.i, align 4
  br label %for.cond.i

for.end.i:                                        ; preds = %for.cond.i
  br label %__clang_ocl_kern_imp_mm2_kernel1.exit

__clang_ocl_kern_imp_mm2_kernel1.exit:            ; preds = %for.end.i, %land.lhs.true.i, %entry
  ret void
}

; Function Attrs: alwaysinline convergent norecurse nounwind
define dso_local spir_func void @__clang_ocl_kern_imp_mm2_kernel1(ptr addrspace(1) noundef align 4 %tmp, ptr addrspace(1) noundef align 4 %A, ptr addrspace(1) noundef align 4 %B, i32 noundef %ni, i32 noundef %nj, i32 noundef %nk, i32 noundef %nl, float noundef %alpha, float noundef %beta) #1 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !6 !kernel_arg_type_qual !7 {
entry:
  %tmp.addr = alloca ptr addrspace(1), align 8
  %A.addr = alloca ptr addrspace(1), align 8
  %B.addr = alloca ptr addrspace(1), align 8
  %ni.addr = alloca i32, align 4
  %nj.addr = alloca i32, align 4
  %nk.addr = alloca i32, align 4
  %nl.addr = alloca i32, align 4
  %alpha.addr = alloca float, align 4
  %beta.addr = alloca float, align 4
  %j = alloca i32, align 4
  %i = alloca i32, align 4
  %k = alloca i32, align 4
  store ptr addrspace(1) %tmp, ptr %tmp.addr, align 8
  store ptr addrspace(1) %A, ptr %A.addr, align 8
  store ptr addrspace(1) %B, ptr %B.addr, align 8
  store i32 %ni, ptr %ni.addr, align 4
  store i32 %nj, ptr %nj.addr, align 4
  store i32 %nk, ptr %nk.addr, align 4
  store i32 %nl, ptr %nl.addr, align 4
  store float %alpha, ptr %alpha.addr, align 4
  store float %beta, ptr %beta.addr, align 4
  %call = call spir_func i64 @_Z13get_global_idj(i32 noundef 0) #4
  %conv = trunc i64 %call to i32
  store i32 %conv, ptr %j, align 4
  %call1 = call spir_func i64 @_Z13get_global_idj(i32 noundef 1) #4
  %conv2 = trunc i64 %call1 to i32
  store i32 %conv2, ptr %i, align 4
  %0 = load i32, ptr %i, align 4
  %1 = load i32, ptr %ni.addr, align 4
  %cmp = icmp slt i32 %0, %1
  br i1 %cmp, label %land.lhs.true, label %if.end

land.lhs.true:                                    ; preds = %entry
  %2 = load i32, ptr %j, align 4
  %3 = load i32, ptr %nj.addr, align 4
  %cmp4 = icmp slt i32 %2, %3
  br i1 %cmp4, label %if.then, label %if.end

if.then:                                          ; preds = %land.lhs.true
  %4 = load ptr addrspace(1), ptr %tmp.addr, align 8
  %5 = load i32, ptr %i, align 4
  %6 = load i32, ptr %nj.addr, align 4
  %mul = mul nsw i32 %5, %6
  %7 = load i32, ptr %j, align 4
  %add = add nsw i32 %mul, %7
  %idxprom = sext i32 %add to i64
  %arrayidx = getelementptr inbounds float, ptr addrspace(1) %4, i64 %idxprom
  store float 0.000000e+00, ptr addrspace(1) %arrayidx, align 4
  store i32 0, ptr %k, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %if.then
  %8 = load i32, ptr %k, align 4
  %9 = load i32, ptr %nk.addr, align 4
  %cmp6 = icmp slt i32 %8, %9
  br i1 %cmp6, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %10 = load float, ptr %alpha.addr, align 4
  %11 = load ptr addrspace(1), ptr %A.addr, align 8
  %12 = load i32, ptr %i, align 4
  %13 = load i32, ptr %nk.addr, align 4
  %mul8 = mul nsw i32 %12, %13
  %14 = load i32, ptr %k, align 4
  %add9 = add nsw i32 %mul8, %14
  %idxprom10 = sext i32 %add9 to i64
  %arrayidx11 = getelementptr inbounds float, ptr addrspace(1) %11, i64 %idxprom10
  %15 = load float, ptr addrspace(1) %arrayidx11, align 4
  %mul12 = fmul float %10, %15
  %16 = load ptr addrspace(1), ptr %B.addr, align 8
  %17 = load i32, ptr %k, align 4
  %18 = load i32, ptr %nj.addr, align 4
  %mul13 = mul nsw i32 %17, %18
  %19 = load i32, ptr %j, align 4
  %add14 = add nsw i32 %mul13, %19
  %idxprom15 = sext i32 %add14 to i64
  %arrayidx16 = getelementptr inbounds float, ptr addrspace(1) %16, i64 %idxprom15
  %20 = load float, ptr addrspace(1) %arrayidx16, align 4
  %21 = load ptr addrspace(1), ptr %tmp.addr, align 8
  %22 = load i32, ptr %i, align 4
  %23 = load i32, ptr %nj.addr, align 4
  %mul18 = mul nsw i32 %22, %23
  %24 = load i32, ptr %j, align 4
  %add19 = add nsw i32 %mul18, %24
  %idxprom20 = sext i32 %add19 to i64
  %arrayidx21 = getelementptr inbounds float, ptr addrspace(1) %21, i64 %idxprom20
  %25 = load float, ptr addrspace(1) %arrayidx21, align 4
  %26 = call float @llvm.fmuladd.f32(float %mul12, float %20, float %25)
  store float %26, ptr addrspace(1) %arrayidx21, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %27 = load i32, ptr %k, align 4
  %inc = add nsw i32 %27, 1
  store i32 %inc, ptr %k, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  br label %if.end

if.end:                                           ; preds = %for.end, %land.lhs.true, %entry
  ret void
}

; Function Attrs: convergent noinline norecurse nounwind
define dso_local spir_kernel void @mm2_kernel2(ptr addrspace(1) noundef align 4 %tmp, ptr addrspace(1) noundef align 4 %C, ptr addrspace(1) noundef align 4 %D, i32 noundef %ni, i32 noundef %nj, i32 noundef %nk, i32 noundef %nl, float noundef %alpha, float noundef %beta) #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !6 !kernel_arg_type_qual !7 {
entry:
  %tmp.addr.i = alloca ptr addrspace(1), align 8
  %C.addr.i = alloca ptr addrspace(1), align 8
  %D.addr.i = alloca ptr addrspace(1), align 8
  %ni.addr.i = alloca i32, align 4
  %nj.addr.i = alloca i32, align 4
  %nk.addr.i = alloca i32, align 4
  %nl.addr.i = alloca i32, align 4
  %alpha.addr.i = alloca float, align 4
  %beta.addr.i = alloca float, align 4
  %j.i = alloca i32, align 4
  %i.i = alloca i32, align 4
  %k.i = alloca i32, align 4
  %tmp.addr = alloca ptr addrspace(1), align 8
  %C.addr = alloca ptr addrspace(1), align 8
  %D.addr = alloca ptr addrspace(1), align 8
  %ni.addr = alloca i32, align 4
  %nj.addr = alloca i32, align 4
  %nk.addr = alloca i32, align 4
  %nl.addr = alloca i32, align 4
  %alpha.addr = alloca float, align 4
  %beta.addr = alloca float, align 4
  store ptr addrspace(1) %tmp, ptr %tmp.addr, align 8
  store ptr addrspace(1) %C, ptr %C.addr, align 8
  store ptr addrspace(1) %D, ptr %D.addr, align 8
  store i32 %ni, ptr %ni.addr, align 4
  store i32 %nj, ptr %nj.addr, align 4
  store i32 %nk, ptr %nk.addr, align 4
  store i32 %nl, ptr %nl.addr, align 4
  store float %alpha, ptr %alpha.addr, align 4
  store float %beta, ptr %beta.addr, align 4
  %0 = load ptr addrspace(1), ptr %tmp.addr, align 8
  %1 = load ptr addrspace(1), ptr %C.addr, align 8
  %2 = load ptr addrspace(1), ptr %D.addr, align 8
  %3 = load i32, ptr %ni.addr, align 4
  %4 = load i32, ptr %nj.addr, align 4
  %5 = load i32, ptr %nk.addr, align 4
  %6 = load i32, ptr %nl.addr, align 4
  %7 = load float, ptr %alpha.addr, align 4
  %8 = load float, ptr %beta.addr, align 4
  store ptr addrspace(1) %0, ptr %tmp.addr.i, align 8
  store ptr addrspace(1) %1, ptr %C.addr.i, align 8
  store ptr addrspace(1) %2, ptr %D.addr.i, align 8
  store i32 %3, ptr %ni.addr.i, align 4
  store i32 %4, ptr %nj.addr.i, align 4
  store i32 %5, ptr %nk.addr.i, align 4
  store i32 %6, ptr %nl.addr.i, align 4
  store float %7, ptr %alpha.addr.i, align 4
  store float %8, ptr %beta.addr.i, align 4
  %call.i = call spir_func i64 @_Z13get_global_idj(i32 noundef 0) #4
  %conv.i = trunc i64 %call.i to i32
  store i32 %conv.i, ptr %j.i, align 4
  %call1.i = call spir_func i64 @_Z13get_global_idj(i32 noundef 1) #4
  %conv2.i = trunc i64 %call1.i to i32
  store i32 %conv2.i, ptr %i.i, align 4
  %9 = load i32, ptr %i.i, align 4
  %10 = load i32, ptr %ni.addr.i, align 4
  %cmp.i = icmp slt i32 %9, %10
  br i1 %cmp.i, label %land.lhs.true.i, label %__clang_ocl_kern_imp_mm2_kernel2.exit

land.lhs.true.i:                                  ; preds = %entry
  %11 = load i32, ptr %j.i, align 4
  %12 = load i32, ptr %nl.addr.i, align 4
  %cmp4.i = icmp slt i32 %11, %12
  br i1 %cmp4.i, label %if.then.i, label %__clang_ocl_kern_imp_mm2_kernel2.exit

if.then.i:                                        ; preds = %land.lhs.true.i
  %13 = load float, ptr %beta.addr.i, align 4
  %14 = load ptr addrspace(1), ptr %D.addr.i, align 8
  %15 = load i32, ptr %i.i, align 4
  %16 = load i32, ptr %nl.addr.i, align 4
  %mul.i = mul nsw i32 %15, %16
  %17 = load i32, ptr %j.i, align 4
  %add.i = add nsw i32 %mul.i, %17
  %idxprom.i = sext i32 %add.i to i64
  %arrayidx.i = getelementptr inbounds float, ptr addrspace(1) %14, i64 %idxprom.i
  %18 = load float, ptr addrspace(1) %arrayidx.i, align 4
  %mul6.i = fmul float %18, %13
  store float %mul6.i, ptr addrspace(1) %arrayidx.i, align 4
  store i32 0, ptr %k.i, align 4
  br label %for.cond.i

for.cond.i:                                       ; preds = %for.body.i, %if.then.i
  %19 = load i32, ptr %k.i, align 4
  %20 = load i32, ptr %nj.addr.i, align 4
  %cmp7.i = icmp slt i32 %19, %20
  br i1 %cmp7.i, label %for.body.i, label %for.end.i

for.body.i:                                       ; preds = %for.cond.i
  %21 = load ptr addrspace(1), ptr %tmp.addr.i, align 8
  %22 = load i32, ptr %i.i, align 4
  %23 = load i32, ptr %nj.addr.i, align 4
  %mul9.i = mul nsw i32 %22, %23
  %24 = load i32, ptr %k.i, align 4
  %add10.i = add nsw i32 %mul9.i, %24
  %idxprom11.i = sext i32 %add10.i to i64
  %arrayidx12.i = getelementptr inbounds float, ptr addrspace(1) %21, i64 %idxprom11.i
  %25 = load float, ptr addrspace(1) %arrayidx12.i, align 4
  %26 = load ptr addrspace(1), ptr %C.addr.i, align 8
  %27 = load i32, ptr %k.i, align 4
  %28 = load i32, ptr %nl.addr.i, align 4
  %mul13.i = mul nsw i32 %27, %28
  %29 = load i32, ptr %j.i, align 4
  %add14.i = add nsw i32 %mul13.i, %29
  %idxprom15.i = sext i32 %add14.i to i64
  %arrayidx16.i = getelementptr inbounds float, ptr addrspace(1) %26, i64 %idxprom15.i
  %30 = load float, ptr addrspace(1) %arrayidx16.i, align 4
  %31 = load ptr addrspace(1), ptr %D.addr.i, align 8
  %32 = load i32, ptr %i.i, align 4
  %33 = load i32, ptr %nl.addr.i, align 4
  %mul18.i = mul nsw i32 %32, %33
  %34 = load i32, ptr %j.i, align 4
  %add19.i = add nsw i32 %mul18.i, %34
  %idxprom20.i = sext i32 %add19.i to i64
  %arrayidx21.i = getelementptr inbounds float, ptr addrspace(1) %31, i64 %idxprom20.i
  %35 = load float, ptr addrspace(1) %arrayidx21.i, align 4
  %36 = call float @llvm.fmuladd.f32(float %25, float %30, float %35)
  store float %36, ptr addrspace(1) %arrayidx21.i, align 4
  %37 = load i32, ptr %k.i, align 4
  %inc.i = add nsw i32 %37, 1
  store i32 %inc.i, ptr %k.i, align 4
  br label %for.cond.i

for.end.i:                                        ; preds = %for.cond.i
  br label %__clang_ocl_kern_imp_mm2_kernel2.exit

__clang_ocl_kern_imp_mm2_kernel2.exit:            ; preds = %for.end.i, %land.lhs.true.i, %entry
  ret void
}

; Function Attrs: alwaysinline convergent norecurse nounwind
define dso_local spir_func void @__clang_ocl_kern_imp_mm2_kernel2(ptr addrspace(1) noundef align 4 %tmp, ptr addrspace(1) noundef align 4 %C, ptr addrspace(1) noundef align 4 %D, i32 noundef %ni, i32 noundef %nj, i32 noundef %nk, i32 noundef %nl, float noundef %alpha, float noundef %beta) #1 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !6 !kernel_arg_type_qual !7 {
entry:
  %tmp.addr = alloca ptr addrspace(1), align 8
  %C.addr = alloca ptr addrspace(1), align 8
  %D.addr = alloca ptr addrspace(1), align 8
  %ni.addr = alloca i32, align 4
  %nj.addr = alloca i32, align 4
  %nk.addr = alloca i32, align 4
  %nl.addr = alloca i32, align 4
  %alpha.addr = alloca float, align 4
  %beta.addr = alloca float, align 4
  %j = alloca i32, align 4
  %i = alloca i32, align 4
  %k = alloca i32, align 4
  store ptr addrspace(1) %tmp, ptr %tmp.addr, align 8
  store ptr addrspace(1) %C, ptr %C.addr, align 8
  store ptr addrspace(1) %D, ptr %D.addr, align 8
  store i32 %ni, ptr %ni.addr, align 4
  store i32 %nj, ptr %nj.addr, align 4
  store i32 %nk, ptr %nk.addr, align 4
  store i32 %nl, ptr %nl.addr, align 4
  store float %alpha, ptr %alpha.addr, align 4
  store float %beta, ptr %beta.addr, align 4
  %call = call spir_func i64 @_Z13get_global_idj(i32 noundef 0) #4
  %conv = trunc i64 %call to i32
  store i32 %conv, ptr %j, align 4
  %call1 = call spir_func i64 @_Z13get_global_idj(i32 noundef 1) #4
  %conv2 = trunc i64 %call1 to i32
  store i32 %conv2, ptr %i, align 4
  %0 = load i32, ptr %i, align 4
  %1 = load i32, ptr %ni.addr, align 4
  %cmp = icmp slt i32 %0, %1
  br i1 %cmp, label %land.lhs.true, label %if.end

land.lhs.true:                                    ; preds = %entry
  %2 = load i32, ptr %j, align 4
  %3 = load i32, ptr %nl.addr, align 4
  %cmp4 = icmp slt i32 %2, %3
  br i1 %cmp4, label %if.then, label %if.end

if.then:                                          ; preds = %land.lhs.true
  %4 = load float, ptr %beta.addr, align 4
  %5 = load ptr addrspace(1), ptr %D.addr, align 8
  %6 = load i32, ptr %i, align 4
  %7 = load i32, ptr %nl.addr, align 4
  %mul = mul nsw i32 %6, %7
  %8 = load i32, ptr %j, align 4
  %add = add nsw i32 %mul, %8
  %idxprom = sext i32 %add to i64
  %arrayidx = getelementptr inbounds float, ptr addrspace(1) %5, i64 %idxprom
  %9 = load float, ptr addrspace(1) %arrayidx, align 4
  %mul6 = fmul float %9, %4
  store float %mul6, ptr addrspace(1) %arrayidx, align 4
  store i32 0, ptr %k, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %if.then
  %10 = load i32, ptr %k, align 4
  %11 = load i32, ptr %nj.addr, align 4
  %cmp7 = icmp slt i32 %10, %11
  br i1 %cmp7, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %12 = load ptr addrspace(1), ptr %tmp.addr, align 8
  %13 = load i32, ptr %i, align 4
  %14 = load i32, ptr %nj.addr, align 4
  %mul9 = mul nsw i32 %13, %14
  %15 = load i32, ptr %k, align 4
  %add10 = add nsw i32 %mul9, %15
  %idxprom11 = sext i32 %add10 to i64
  %arrayidx12 = getelementptr inbounds float, ptr addrspace(1) %12, i64 %idxprom11
  %16 = load float, ptr addrspace(1) %arrayidx12, align 4
  %17 = load ptr addrspace(1), ptr %C.addr, align 8
  %18 = load i32, ptr %k, align 4
  %19 = load i32, ptr %nl.addr, align 4
  %mul13 = mul nsw i32 %18, %19
  %20 = load i32, ptr %j, align 4
  %add14 = add nsw i32 %mul13, %20
  %idxprom15 = sext i32 %add14 to i64
  %arrayidx16 = getelementptr inbounds float, ptr addrspace(1) %17, i64 %idxprom15
  %21 = load float, ptr addrspace(1) %arrayidx16, align 4
  %22 = load ptr addrspace(1), ptr %D.addr, align 8
  %23 = load i32, ptr %i, align 4
  %24 = load i32, ptr %nl.addr, align 4
  %mul18 = mul nsw i32 %23, %24
  %25 = load i32, ptr %j, align 4
  %add19 = add nsw i32 %mul18, %25
  %idxprom20 = sext i32 %add19 to i64
  %arrayidx21 = getelementptr inbounds float, ptr addrspace(1) %22, i64 %idxprom20
  %26 = load float, ptr addrspace(1) %arrayidx21, align 4
  %27 = call float @llvm.fmuladd.f32(float %16, float %21, float %26)
  store float %27, ptr addrspace(1) %arrayidx21, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %28 = load i32, ptr %k, align 4
  %inc = add nsw i32 %28, 1
  store i32 %inc, ptr %k, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  br label %if.end

if.end:                                           ; preds = %for.end, %land.lhs.true, %entry
  ret void
}

; Function Attrs: convergent nounwind willreturn memory(none)
declare spir_func i64 @_Z13get_global_idj(i32 noundef) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fmuladd.f32(float, float, float) #3

attributes #0 = { convergent noinline norecurse nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "uniform-work-group-size"="false" }
attributes #1 = { alwaysinline convergent norecurse nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "uniform-work-group-size"="false" }
attributes #2 = { convergent nounwind willreturn memory(none) "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { convergent nounwind willreturn memory(none) }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
!2 = !{!"clang version 22.1.4 (https://github.com/conda-forge/clangdev-feedstock 8fb2e1c666e3daad00e02a3278e63348e3c9ffcb)"}
!3 = !{i32 1, i32 1, i32 1, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0}
!4 = !{!"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none", !"none"}
!5 = !{!"DATA_TYPE*", !"DATA_TYPE*", !"DATA_TYPE*", !"int", !"int", !"int", !"int", !"DATA_TYPE", !"DATA_TYPE"}
!6 = !{!"float*", !"float*", !"float*", !"int", !"int", !"int", !"int", !"float", !"float"}
!7 = !{!"", !"", !"", !"", !"", !"", !"", !"", !""}
