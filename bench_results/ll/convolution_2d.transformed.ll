; ModuleID = '/home/ambikas2/cs526/bench_results/bc/convolution_2d.bc'
source_filename = "/home/ambikas2/cs526/kernels/convolution_2d.cl"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spir64-unknown-unknown"

; Function Attrs: convergent noinline norecurse nounwind
define dso_local spir_kernel void @Convolution2D_kernel(ptr addrspace(1) noundef align 4 %A, ptr addrspace(1) noundef align 4 %B, i32 noundef %ni, i32 noundef %nj) #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !6 !kernel_arg_type_qual !7 {
entry:
  %A.addr.i = alloca ptr addrspace(1), align 8
  %B.addr.i = alloca ptr addrspace(1), align 8
  %ni.addr.i = alloca i32, align 4
  %nj.addr.i = alloca i32, align 4
  %j.i = alloca i32, align 4
  %i.i = alloca i32, align 4
  %c11.i = alloca float, align 4
  %c12.i = alloca float, align 4
  %c13.i = alloca float, align 4
  %c21.i = alloca float, align 4
  %c22.i = alloca float, align 4
  %c23.i = alloca float, align 4
  %c31.i = alloca float, align 4
  %c32.i = alloca float, align 4
  %c33.i = alloca float, align 4
  %A.addr = alloca ptr addrspace(1), align 8
  %B.addr = alloca ptr addrspace(1), align 8
  %ni.addr = alloca i32, align 4
  %nj.addr = alloca i32, align 4
  store ptr addrspace(1) %A, ptr %A.addr, align 8
  store ptr addrspace(1) %B, ptr %B.addr, align 8
  store i32 %ni, ptr %ni.addr, align 4
  store i32 %nj, ptr %nj.addr, align 4
  %0 = load ptr addrspace(1), ptr %A.addr, align 8
  %1 = load ptr addrspace(1), ptr %B.addr, align 8
  %2 = load i32, ptr %ni.addr, align 4
  %3 = load i32, ptr %nj.addr, align 4
  store ptr addrspace(1) %0, ptr %A.addr.i, align 8
  store ptr addrspace(1) %1, ptr %B.addr.i, align 8
  store i32 %2, ptr %ni.addr.i, align 4
  store i32 %3, ptr %nj.addr.i, align 4
  %call.i = call spir_func i64 @_Z13get_global_idj(i32 noundef 0) #4
  %conv.i = trunc i64 %call.i to i32
  store i32 %conv.i, ptr %j.i, align 4
  %call1.i = call spir_func i64 @_Z13get_global_idj(i32 noundef 1) #4
  %conv2.i = trunc i64 %call1.i to i32
  store i32 %conv2.i, ptr %i.i, align 4
  store float 0x3FC99999A0000000, ptr %c11.i, align 4
  store float 5.000000e-01, ptr %c21.i, align 4
  store float 0xBFE99999A0000000, ptr %c31.i, align 4
  store float 0xBFD3333340000000, ptr %c12.i, align 4
  store float 0x3FE3333340000000, ptr %c22.i, align 4
  store float 0xBFECCCCCC0000000, ptr %c32.i, align 4
  store float 0x3FD99999A0000000, ptr %c13.i, align 4
  store float 0x3FE6666660000000, ptr %c23.i, align 4
  store float 0x3FB99999A0000000, ptr %c33.i, align 4
  %4 = load i32, ptr %i.i, align 4
  %5 = load i32, ptr %ni.addr.i, align 4
  %sub.i = sub nsw i32 %5, 1
  %cmp.i = icmp slt i32 %4, %sub.i
  br i1 %cmp.i, label %land.lhs.true.i, label %__clang_ocl_kern_imp_Convolution2D_kernel.exit

land.lhs.true.i:                                  ; preds = %entry
  %6 = load i32, ptr %j.i, align 4
  %7 = load i32, ptr %nj.addr.i, align 4
  %sub4.i = sub nsw i32 %7, 1
  %cmp5.i = icmp slt i32 %6, %sub4.i
  br i1 %cmp5.i, label %land.lhs.true7.i, label %__clang_ocl_kern_imp_Convolution2D_kernel.exit

land.lhs.true7.i:                                 ; preds = %land.lhs.true.i
  %8 = load i32, ptr %i.i, align 4
  %cmp8.i = icmp sgt i32 %8, 0
  br i1 %cmp8.i, label %land.lhs.true10.i, label %__clang_ocl_kern_imp_Convolution2D_kernel.exit

land.lhs.true10.i:                                ; preds = %land.lhs.true7.i
  %9 = load i32, ptr %j.i, align 4
  %cmp11.i = icmp sgt i32 %9, 0
  br i1 %cmp11.i, label %if.then.i, label %__clang_ocl_kern_imp_Convolution2D_kernel.exit

if.then.i:                                        ; preds = %land.lhs.true10.i
  %10 = load float, ptr %c11.i, align 4
  %11 = load ptr addrspace(1), ptr %A.addr.i, align 8
  %12 = load i32, ptr %i.i, align 4
  %sub13.i = sub nsw i32 %12, 1
  %13 = load i32, ptr %nj.addr.i, align 4
  %mul.i = mul nsw i32 %sub13.i, %13
  %14 = load i32, ptr %j.i, align 4
  %sub14.i = sub nsw i32 %14, 1
  %add.i = add nsw i32 %mul.i, %sub14.i
  %idxprom.i = sext i32 %add.i to i64
  %arrayidx.i = getelementptr inbounds float, ptr addrspace(1) %11, i64 %idxprom.i
  %15 = load float, ptr addrspace(1) %arrayidx.i, align 4
  %16 = load float, ptr %c21.i, align 4
  %17 = load ptr addrspace(1), ptr %A.addr.i, align 8
  %18 = load i32, ptr %i.i, align 4
  %sub16.i = sub nsw i32 %18, 1
  %19 = load i32, ptr %nj.addr.i, align 4
  %mul17.i = mul nsw i32 %sub16.i, %19
  %20 = load i32, ptr %j.i, align 4
  %add19.i = add nsw i32 %mul17.i, %20
  %idxprom20.i = sext i32 %add19.i to i64
  %arrayidx21.i = getelementptr inbounds float, ptr addrspace(1) %17, i64 %idxprom20.i
  %21 = load float, ptr addrspace(1) %arrayidx21.i, align 4
  %mul22.i = fmul float %16, %21
  %22 = call float @llvm.fmuladd.f32(float %10, float %15, float %mul22.i)
  %23 = load float, ptr %c31.i, align 4
  %24 = load ptr addrspace(1), ptr %A.addr.i, align 8
  %25 = load i32, ptr %i.i, align 4
  %sub23.i = sub nsw i32 %25, 1
  %26 = load i32, ptr %nj.addr.i, align 4
  %mul24.i = mul nsw i32 %sub23.i, %26
  %27 = load i32, ptr %j.i, align 4
  %add25.i = add nsw i32 %27, 1
  %add26.i = add nsw i32 %mul24.i, %add25.i
  %idxprom27.i = sext i32 %add26.i to i64
  %arrayidx28.i = getelementptr inbounds float, ptr addrspace(1) %24, i64 %idxprom27.i
  %28 = load float, ptr addrspace(1) %arrayidx28.i, align 4
  %29 = call float @llvm.fmuladd.f32(float %23, float %28, float %22)
  %30 = load float, ptr %c12.i, align 4
  %31 = load ptr addrspace(1), ptr %A.addr.i, align 8
  %32 = load i32, ptr %i.i, align 4
  %33 = load i32, ptr %nj.addr.i, align 4
  %mul31.i = mul nsw i32 %32, %33
  %34 = load i32, ptr %j.i, align 4
  %sub32.i = sub nsw i32 %34, 1
  %add33.i = add nsw i32 %mul31.i, %sub32.i
  %idxprom34.i = sext i32 %add33.i to i64
  %arrayidx35.i = getelementptr inbounds float, ptr addrspace(1) %31, i64 %idxprom34.i
  %35 = load float, ptr addrspace(1) %arrayidx35.i, align 4
  %36 = call float @llvm.fmuladd.f32(float %30, float %35, float %29)
  %37 = load float, ptr %c22.i, align 4
  %38 = load ptr addrspace(1), ptr %A.addr.i, align 8
  %39 = load i32, ptr %i.i, align 4
  %40 = load i32, ptr %nj.addr.i, align 4
  %mul38.i = mul nsw i32 %39, %40
  %41 = load i32, ptr %j.i, align 4
  %add40.i = add nsw i32 %mul38.i, %41
  %idxprom41.i = sext i32 %add40.i to i64
  %arrayidx42.i = getelementptr inbounds float, ptr addrspace(1) %38, i64 %idxprom41.i
  %42 = load float, ptr addrspace(1) %arrayidx42.i, align 4
  %43 = call float @llvm.fmuladd.f32(float %37, float %42, float %36)
  %44 = load float, ptr %c32.i, align 4
  %45 = load ptr addrspace(1), ptr %A.addr.i, align 8
  %46 = load i32, ptr %i.i, align 4
  %47 = load i32, ptr %nj.addr.i, align 4
  %mul45.i = mul nsw i32 %46, %47
  %48 = load i32, ptr %j.i, align 4
  %add46.i = add nsw i32 %48, 1
  %add47.i = add nsw i32 %mul45.i, %add46.i
  %idxprom48.i = sext i32 %add47.i to i64
  %arrayidx49.i = getelementptr inbounds float, ptr addrspace(1) %45, i64 %idxprom48.i
  %49 = load float, ptr addrspace(1) %arrayidx49.i, align 4
  %50 = call float @llvm.fmuladd.f32(float %44, float %49, float %43)
  %51 = load float, ptr %c13.i, align 4
  %52 = load ptr addrspace(1), ptr %A.addr.i, align 8
  %53 = load i32, ptr %i.i, align 4
  %add51.i = add nsw i32 %53, 1
  %54 = load i32, ptr %nj.addr.i, align 4
  %mul52.i = mul nsw i32 %add51.i, %54
  %55 = load i32, ptr %j.i, align 4
  %sub53.i = sub nsw i32 %55, 1
  %add54.i = add nsw i32 %mul52.i, %sub53.i
  %idxprom55.i = sext i32 %add54.i to i64
  %arrayidx56.i = getelementptr inbounds float, ptr addrspace(1) %52, i64 %idxprom55.i
  %56 = load float, ptr addrspace(1) %arrayidx56.i, align 4
  %57 = call float @llvm.fmuladd.f32(float %51, float %56, float %50)
  %58 = load float, ptr %c23.i, align 4
  %59 = load ptr addrspace(1), ptr %A.addr.i, align 8
  %60 = load i32, ptr %i.i, align 4
  %add58.i = add nsw i32 %60, 1
  %61 = load i32, ptr %nj.addr.i, align 4
  %mul59.i = mul nsw i32 %add58.i, %61
  %62 = load i32, ptr %j.i, align 4
  %add61.i = add nsw i32 %mul59.i, %62
  %idxprom62.i = sext i32 %add61.i to i64
  %arrayidx63.i = getelementptr inbounds float, ptr addrspace(1) %59, i64 %idxprom62.i
  %63 = load float, ptr addrspace(1) %arrayidx63.i, align 4
  %64 = call float @llvm.fmuladd.f32(float %58, float %63, float %57)
  %65 = load float, ptr %c33.i, align 4
  %66 = load ptr addrspace(1), ptr %A.addr.i, align 8
  %67 = load i32, ptr %i.i, align 4
  %add65.i = add nsw i32 %67, 1
  %68 = load i32, ptr %nj.addr.i, align 4
  %mul66.i = mul nsw i32 %add65.i, %68
  %69 = load i32, ptr %j.i, align 4
  %add67.i = add nsw i32 %69, 1
  %add68.i = add nsw i32 %mul66.i, %add67.i
  %idxprom69.i = sext i32 %add68.i to i64
  %arrayidx70.i = getelementptr inbounds float, ptr addrspace(1) %66, i64 %idxprom69.i
  %70 = load float, ptr addrspace(1) %arrayidx70.i, align 4
  %71 = call float @llvm.fmuladd.f32(float %65, float %70, float %64)
  %72 = load ptr addrspace(1), ptr %B.addr.i, align 8
  %73 = load i32, ptr %i.i, align 4
  %74 = load i32, ptr %nj.addr.i, align 4
  %mul72.i = mul nsw i32 %73, %74
  %75 = load i32, ptr %j.i, align 4
  %add73.i = add nsw i32 %mul72.i, %75
  %idxprom74.i = sext i32 %add73.i to i64
  %arrayidx75.i = getelementptr inbounds float, ptr addrspace(1) %72, i64 %idxprom74.i
  store float %71, ptr addrspace(1) %arrayidx75.i, align 4
  br label %__clang_ocl_kern_imp_Convolution2D_kernel.exit

__clang_ocl_kern_imp_Convolution2D_kernel.exit:   ; preds = %if.then.i, %land.lhs.true10.i, %land.lhs.true7.i, %land.lhs.true.i, %entry
  ret void
}

; Function Attrs: alwaysinline convergent norecurse nounwind
define dso_local spir_func void @__clang_ocl_kern_imp_Convolution2D_kernel(ptr addrspace(1) noundef align 4 %A, ptr addrspace(1) noundef align 4 %B, i32 noundef %ni, i32 noundef %nj) #1 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !6 !kernel_arg_type_qual !7 {
entry:
  %A.addr = alloca ptr addrspace(1), align 8
  %B.addr = alloca ptr addrspace(1), align 8
  %ni.addr = alloca i32, align 4
  %nj.addr = alloca i32, align 4
  %j = alloca i32, align 4
  %i = alloca i32, align 4
  %c11 = alloca float, align 4
  %c12 = alloca float, align 4
  %c13 = alloca float, align 4
  %c21 = alloca float, align 4
  %c22 = alloca float, align 4
  %c23 = alloca float, align 4
  %c31 = alloca float, align 4
  %c32 = alloca float, align 4
  %c33 = alloca float, align 4
  store ptr addrspace(1) %A, ptr %A.addr, align 8
  store ptr addrspace(1) %B, ptr %B.addr, align 8
  store i32 %ni, ptr %ni.addr, align 4
  store i32 %nj, ptr %nj.addr, align 4
  %call = call spir_func i64 @_Z13get_global_idj(i32 noundef 0) #4
  %conv = trunc i64 %call to i32
  store i32 %conv, ptr %j, align 4
  %call1 = call spir_func i64 @_Z13get_global_idj(i32 noundef 1) #4
  %conv2 = trunc i64 %call1 to i32
  store i32 %conv2, ptr %i, align 4
  store float 0x3FC99999A0000000, ptr %c11, align 4
  store float 5.000000e-01, ptr %c21, align 4
  store float 0xBFE99999A0000000, ptr %c31, align 4
  store float 0xBFD3333340000000, ptr %c12, align 4
  store float 0x3FE3333340000000, ptr %c22, align 4
  store float 0xBFECCCCCC0000000, ptr %c32, align 4
  store float 0x3FD99999A0000000, ptr %c13, align 4
  store float 0x3FE6666660000000, ptr %c23, align 4
  store float 0x3FB99999A0000000, ptr %c33, align 4
  %0 = load i32, ptr %i, align 4
  %1 = load i32, ptr %ni.addr, align 4
  %sub = sub nsw i32 %1, 1
  %cmp = icmp slt i32 %0, %sub
  br i1 %cmp, label %land.lhs.true, label %if.end

land.lhs.true:                                    ; preds = %entry
  %2 = load i32, ptr %j, align 4
  %3 = load i32, ptr %nj.addr, align 4
  %sub4 = sub nsw i32 %3, 1
  %cmp5 = icmp slt i32 %2, %sub4
  br i1 %cmp5, label %land.lhs.true7, label %if.end

land.lhs.true7:                                   ; preds = %land.lhs.true
  %4 = load i32, ptr %i, align 4
  %cmp8 = icmp sgt i32 %4, 0
  br i1 %cmp8, label %land.lhs.true10, label %if.end

land.lhs.true10:                                  ; preds = %land.lhs.true7
  %5 = load i32, ptr %j, align 4
  %cmp11 = icmp sgt i32 %5, 0
  br i1 %cmp11, label %if.then, label %if.end

if.then:                                          ; preds = %land.lhs.true10
  %6 = load float, ptr %c11, align 4
  %7 = load ptr addrspace(1), ptr %A.addr, align 8
  %8 = load i32, ptr %i, align 4
  %sub13 = sub nsw i32 %8, 1
  %9 = load i32, ptr %nj.addr, align 4
  %mul = mul nsw i32 %sub13, %9
  %10 = load i32, ptr %j, align 4
  %sub14 = sub nsw i32 %10, 1
  %add = add nsw i32 %mul, %sub14
  %idxprom = sext i32 %add to i64
  %arrayidx = getelementptr inbounds float, ptr addrspace(1) %7, i64 %idxprom
  %11 = load float, ptr addrspace(1) %arrayidx, align 4
  %12 = load float, ptr %c21, align 4
  %13 = load ptr addrspace(1), ptr %A.addr, align 8
  %14 = load i32, ptr %i, align 4
  %sub16 = sub nsw i32 %14, 1
  %15 = load i32, ptr %nj.addr, align 4
  %mul17 = mul nsw i32 %sub16, %15
  %16 = load i32, ptr %j, align 4
  %add18 = add nsw i32 %16, 0
  %add19 = add nsw i32 %mul17, %add18
  %idxprom20 = sext i32 %add19 to i64
  %arrayidx21 = getelementptr inbounds float, ptr addrspace(1) %13, i64 %idxprom20
  %17 = load float, ptr addrspace(1) %arrayidx21, align 4
  %mul22 = fmul float %12, %17
  %18 = call float @llvm.fmuladd.f32(float %6, float %11, float %mul22)
  %19 = load float, ptr %c31, align 4
  %20 = load ptr addrspace(1), ptr %A.addr, align 8
  %21 = load i32, ptr %i, align 4
  %sub23 = sub nsw i32 %21, 1
  %22 = load i32, ptr %nj.addr, align 4
  %mul24 = mul nsw i32 %sub23, %22
  %23 = load i32, ptr %j, align 4
  %add25 = add nsw i32 %23, 1
  %add26 = add nsw i32 %mul24, %add25
  %idxprom27 = sext i32 %add26 to i64
  %arrayidx28 = getelementptr inbounds float, ptr addrspace(1) %20, i64 %idxprom27
  %24 = load float, ptr addrspace(1) %arrayidx28, align 4
  %25 = call float @llvm.fmuladd.f32(float %19, float %24, float %18)
  %26 = load float, ptr %c12, align 4
  %27 = load ptr addrspace(1), ptr %A.addr, align 8
  %28 = load i32, ptr %i, align 4
  %add30 = add nsw i32 %28, 0
  %29 = load i32, ptr %nj.addr, align 4
  %mul31 = mul nsw i32 %add30, %29
  %30 = load i32, ptr %j, align 4
  %sub32 = sub nsw i32 %30, 1
  %add33 = add nsw i32 %mul31, %sub32
  %idxprom34 = sext i32 %add33 to i64
  %arrayidx35 = getelementptr inbounds float, ptr addrspace(1) %27, i64 %idxprom34
  %31 = load float, ptr addrspace(1) %arrayidx35, align 4
  %32 = call float @llvm.fmuladd.f32(float %26, float %31, float %25)
  %33 = load float, ptr %c22, align 4
  %34 = load ptr addrspace(1), ptr %A.addr, align 8
  %35 = load i32, ptr %i, align 4
  %add37 = add nsw i32 %35, 0
  %36 = load i32, ptr %nj.addr, align 4
  %mul38 = mul nsw i32 %add37, %36
  %37 = load i32, ptr %j, align 4
  %add39 = add nsw i32 %37, 0
  %add40 = add nsw i32 %mul38, %add39
  %idxprom41 = sext i32 %add40 to i64
  %arrayidx42 = getelementptr inbounds float, ptr addrspace(1) %34, i64 %idxprom41
  %38 = load float, ptr addrspace(1) %arrayidx42, align 4
  %39 = call float @llvm.fmuladd.f32(float %33, float %38, float %32)
  %40 = load float, ptr %c32, align 4
  %41 = load ptr addrspace(1), ptr %A.addr, align 8
  %42 = load i32, ptr %i, align 4
  %add44 = add nsw i32 %42, 0
  %43 = load i32, ptr %nj.addr, align 4
  %mul45 = mul nsw i32 %add44, %43
  %44 = load i32, ptr %j, align 4
  %add46 = add nsw i32 %44, 1
  %add47 = add nsw i32 %mul45, %add46
  %idxprom48 = sext i32 %add47 to i64
  %arrayidx49 = getelementptr inbounds float, ptr addrspace(1) %41, i64 %idxprom48
  %45 = load float, ptr addrspace(1) %arrayidx49, align 4
  %46 = call float @llvm.fmuladd.f32(float %40, float %45, float %39)
  %47 = load float, ptr %c13, align 4
  %48 = load ptr addrspace(1), ptr %A.addr, align 8
  %49 = load i32, ptr %i, align 4
  %add51 = add nsw i32 %49, 1
  %50 = load i32, ptr %nj.addr, align 4
  %mul52 = mul nsw i32 %add51, %50
  %51 = load i32, ptr %j, align 4
  %sub53 = sub nsw i32 %51, 1
  %add54 = add nsw i32 %mul52, %sub53
  %idxprom55 = sext i32 %add54 to i64
  %arrayidx56 = getelementptr inbounds float, ptr addrspace(1) %48, i64 %idxprom55
  %52 = load float, ptr addrspace(1) %arrayidx56, align 4
  %53 = call float @llvm.fmuladd.f32(float %47, float %52, float %46)
  %54 = load float, ptr %c23, align 4
  %55 = load ptr addrspace(1), ptr %A.addr, align 8
  %56 = load i32, ptr %i, align 4
  %add58 = add nsw i32 %56, 1
  %57 = load i32, ptr %nj.addr, align 4
  %mul59 = mul nsw i32 %add58, %57
  %58 = load i32, ptr %j, align 4
  %add60 = add nsw i32 %58, 0
  %add61 = add nsw i32 %mul59, %add60
  %idxprom62 = sext i32 %add61 to i64
  %arrayidx63 = getelementptr inbounds float, ptr addrspace(1) %55, i64 %idxprom62
  %59 = load float, ptr addrspace(1) %arrayidx63, align 4
  %60 = call float @llvm.fmuladd.f32(float %54, float %59, float %53)
  %61 = load float, ptr %c33, align 4
  %62 = load ptr addrspace(1), ptr %A.addr, align 8
  %63 = load i32, ptr %i, align 4
  %add65 = add nsw i32 %63, 1
  %64 = load i32, ptr %nj.addr, align 4
  %mul66 = mul nsw i32 %add65, %64
  %65 = load i32, ptr %j, align 4
  %add67 = add nsw i32 %65, 1
  %add68 = add nsw i32 %mul66, %add67
  %idxprom69 = sext i32 %add68 to i64
  %arrayidx70 = getelementptr inbounds float, ptr addrspace(1) %62, i64 %idxprom69
  %66 = load float, ptr addrspace(1) %arrayidx70, align 4
  %67 = call float @llvm.fmuladd.f32(float %61, float %66, float %60)
  %68 = load ptr addrspace(1), ptr %B.addr, align 8
  %69 = load i32, ptr %i, align 4
  %70 = load i32, ptr %nj.addr, align 4
  %mul72 = mul nsw i32 %69, %70
  %71 = load i32, ptr %j, align 4
  %add73 = add nsw i32 %mul72, %71
  %idxprom74 = sext i32 %add73 to i64
  %arrayidx75 = getelementptr inbounds float, ptr addrspace(1) %68, i64 %idxprom74
  store float %67, ptr addrspace(1) %arrayidx75, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %land.lhs.true10, %land.lhs.true7, %land.lhs.true, %entry
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
!3 = !{i32 1, i32 1, i32 0, i32 0}
!4 = !{!"none", !"none", !"none", !"none"}
!5 = !{!"DATA_TYPE*", !"DATA_TYPE*", !"int", !"int"}
!6 = !{!"float*", !"float*", !"int", !"int"}
!7 = !{!"", !"", !"", !""}
