; ModuleID = '/home/ambikas2/cs526/bench_results/bc/bounded_kernels.bc'
source_filename = "/home/ambikas2/cs526/kernels/bounded_kernels.cl"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spir64-unknown-unknown"

; Function Attrs: convergent noinline norecurse nounwind
define dso_local spir_kernel void @bounded_strided(ptr addrspace(1) noundef align 4 %A, ptr addrspace(1) noundef align 4 %B, i32 noundef %N) #0 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !5 !kernel_arg_type_qual !6 {
entry:
  %tile.local = alloca float, i64 1052, align 16, addrspace(3)
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
  %lid0 = call i64 @_Z12get_local_idj(i32 0)
  %lsize0 = call i64 @_Z14get_local_sizej(i32 0)
  %gid0 = call i64 @_Z13get_global_idj(i32 0)
  %group.base = sub i64 %gid0, %lid0
  %dense.mul = mul i64 %group.base, 4
  %dense.base = add i64 %dense.mul, 0
  %valid.lsize.minus1 = sub i64 %lsize0, 1
  %valid.tile.span = mul i64 4, %valid.lsize.minus1
  %valid.tile.elems = add i64 %valid.tile.span, 1
  %off.init.vec = mul i64 %lid0, 4
  %off.step.vec = mul i64 %lsize0, 4
  %groupid0 = call i64 @_Z12get_group_idj(i32 0)
  br label %tile.preload.header

tile.preload.header:                              ; preds = %tile.preload.latch, %if.then.i
  %off = phi i64 [ %off.init.vec, %if.then.i ], [ %off.next, %tile.preload.latch ]
  %tile.cond = icmp ult i64 %off, %valid.tile.elems
  br i1 %tile.cond, label %tile.preload.body, label %tile.preload.exit

tile.preload.body:                                ; preds = %tile.preload.header
  %global.idx = add i64 %dense.base, %off
  %bound64 = sext i32 %N to i64
  %vec.end.global = add i64 %global.idx, 4
  %vec.end.tile = add i64 %off, 4
  %vec.global.in.bounds = icmp ule i64 %vec.end.global, %bound64
  %vec.tile.in.bounds = icmp ule i64 %vec.end.tile, %valid.tile.elems
  %can.vectorize.preload = and i1 %vec.global.in.bounds, %vec.tile.in.bounds
  br i1 %can.vectorize.preload, label %tile.preload.vec, label %tile.preload.scalar

tile.preload.vec:                                 ; preds = %tile.preload.body
  %vec.global.ptr = getelementptr float, ptr addrspace(1) %5, i64 %global.idx
  %tile.vec.ld = load <4 x float>, ptr addrspace(1) %vec.global.ptr, align 16
  %tile.pad.groups = udiv i64 %off, 32
  %tile.pad = mul i64 %tile.pad.groups, 1
  %tile.physical.idx = add i64 %off, %tile.pad
  %vec.tile.ptr = getelementptr float, ptr addrspace(3) %tile.local, i64 %tile.physical.idx
  store <4 x float> %tile.vec.ld, ptr addrspace(3) %vec.tile.ptr, align 16
  br label %tile.preload.latch

tile.preload.scalar:                              ; preds = %tile.preload.body
  %lane.global.idx = add i64 %dense.base, %off
  %lane.in.tile = icmp ult i64 %off, %valid.tile.elems
  %lane.in.global = icmp ult i64 %lane.global.idx, %bound64
  %lane.in.bounds = and i1 %lane.in.tile, %lane.in.global
  %lane.safe.global.idx = select i1 %lane.in.bounds, i64 %lane.global.idx, i64 0
  %lane.safe.global.ptr = getelementptr float, ptr addrspace(1) %5, i64 %lane.safe.global.idx
  %tile.scalar.ld.raw = load float, ptr addrspace(1) %lane.safe.global.ptr, align 4
  %tile.scalar.ld = select i1 %lane.in.bounds, float %tile.scalar.ld.raw, float 0.000000e+00
  %tile.pad.groups1 = udiv i64 %off, 32
  %tile.pad2 = mul i64 %tile.pad.groups1, 1
  %tile.physical.idx3 = add i64 %off, %tile.pad2
  %tile.scalar.ptr = getelementptr float, ptr addrspace(3) %tile.local, i64 %tile.physical.idx3
  store float %tile.scalar.ld, ptr addrspace(3) %tile.scalar.ptr, align 4
  %lane.off = add i64 %off, 1
  %lane.global.idx4 = add i64 %dense.base, %lane.off
  %lane.in.tile5 = icmp ult i64 %lane.off, %valid.tile.elems
  %lane.in.global6 = icmp ult i64 %lane.global.idx4, %bound64
  %lane.in.bounds7 = and i1 %lane.in.tile5, %lane.in.global6
  %lane.safe.global.idx8 = select i1 %lane.in.bounds7, i64 %lane.global.idx4, i64 0
  %lane.safe.global.ptr9 = getelementptr float, ptr addrspace(1) %5, i64 %lane.safe.global.idx8
  %tile.scalar.ld.raw10 = load float, ptr addrspace(1) %lane.safe.global.ptr9, align 4
  %tile.scalar.ld11 = select i1 %lane.in.bounds7, float %tile.scalar.ld.raw10, float 0.000000e+00
  %tile.pad.groups12 = udiv i64 %lane.off, 32
  %tile.pad13 = mul i64 %tile.pad.groups12, 1
  %tile.physical.idx14 = add i64 %lane.off, %tile.pad13
  %tile.scalar.ptr15 = getelementptr float, ptr addrspace(3) %tile.local, i64 %tile.physical.idx14
  store float %tile.scalar.ld11, ptr addrspace(3) %tile.scalar.ptr15, align 4
  %lane.off16 = add i64 %off, 2
  %lane.global.idx17 = add i64 %dense.base, %lane.off16
  %lane.in.tile18 = icmp ult i64 %lane.off16, %valid.tile.elems
  %lane.in.global19 = icmp ult i64 %lane.global.idx17, %bound64
  %lane.in.bounds20 = and i1 %lane.in.tile18, %lane.in.global19
  %lane.safe.global.idx21 = select i1 %lane.in.bounds20, i64 %lane.global.idx17, i64 0
  %lane.safe.global.ptr22 = getelementptr float, ptr addrspace(1) %5, i64 %lane.safe.global.idx21
  %tile.scalar.ld.raw23 = load float, ptr addrspace(1) %lane.safe.global.ptr22, align 4
  %tile.scalar.ld24 = select i1 %lane.in.bounds20, float %tile.scalar.ld.raw23, float 0.000000e+00
  %tile.pad.groups25 = udiv i64 %lane.off16, 32
  %tile.pad26 = mul i64 %tile.pad.groups25, 1
  %tile.physical.idx27 = add i64 %lane.off16, %tile.pad26
  %tile.scalar.ptr28 = getelementptr float, ptr addrspace(3) %tile.local, i64 %tile.physical.idx27
  store float %tile.scalar.ld24, ptr addrspace(3) %tile.scalar.ptr28, align 4
  %lane.off29 = add i64 %off, 3
  %lane.global.idx30 = add i64 %dense.base, %lane.off29
  %lane.in.tile31 = icmp ult i64 %lane.off29, %valid.tile.elems
  %lane.in.global32 = icmp ult i64 %lane.global.idx30, %bound64
  %lane.in.bounds33 = and i1 %lane.in.tile31, %lane.in.global32
  %lane.safe.global.idx34 = select i1 %lane.in.bounds33, i64 %lane.global.idx30, i64 0
  %lane.safe.global.ptr35 = getelementptr float, ptr addrspace(1) %5, i64 %lane.safe.global.idx34
  %tile.scalar.ld.raw36 = load float, ptr addrspace(1) %lane.safe.global.ptr35, align 4
  %tile.scalar.ld37 = select i1 %lane.in.bounds33, float %tile.scalar.ld.raw36, float 0.000000e+00
  %tile.pad.groups38 = udiv i64 %lane.off29, 32
  %tile.pad39 = mul i64 %tile.pad.groups38, 1
  %tile.physical.idx40 = add i64 %lane.off29, %tile.pad39
  %tile.scalar.ptr41 = getelementptr float, ptr addrspace(3) %tile.local, i64 %tile.physical.idx40
  store float %tile.scalar.ld37, ptr addrspace(3) %tile.scalar.ptr41, align 4
  br label %tile.preload.latch

tile.preload.latch:                               ; preds = %tile.preload.scalar, %tile.preload.vec
  %off.next = add i64 %off, %off.step.vec
  br label %tile.preload.header

tile.preload.exit:                                ; preds = %tile.preload.header
  call void @_Z7barrierj(i32 1)
  br label %tile.cont

tile.cont:                                        ; preds = %tile.preload.exit
  %shared.logical.idx = mul i64 %lid0, 4
  %shared.in.bounds = icmp ult i64 %shared.logical.idx, %valid.tile.elems
  %shared.safe.logical.idx = select i1 %shared.in.bounds, i64 %shared.logical.idx, i64 0
  %tile.pad.groups42 = udiv i64 %shared.safe.logical.idx, 32
  %tile.pad43 = mul i64 %tile.pad.groups42, 1
  %tile.physical.idx44 = add i64 %shared.safe.logical.idx, %tile.pad43
  %shared.ptr = getelementptr float, ptr addrspace(3) %tile.local, i64 %tile.physical.idx44
  %shared.load = load float, ptr addrspace(3) %shared.ptr, align 4
  %7 = load ptr addrspace(1), ptr %B.addr.i, align 8
  %8 = load i32, ptr %tid.i, align 4
  %idxprom3.i = sext i32 %8 to i64
  %arrayidx4.i = getelementptr inbounds float, ptr addrspace(1) %7, i64 %idxprom3.i
  store float %shared.load, ptr addrspace(1) %arrayidx4.i, align 4
  br label %__clang_ocl_kern_imp_bounded_strided.exit

__clang_ocl_kern_imp_bounded_strided.exit:        ; preds = %tile.cont, %entry
  ret void
}

; Function Attrs: alwaysinline convergent norecurse nounwind
define dso_local spir_func void @__clang_ocl_kern_imp_bounded_strided(ptr addrspace(1) noundef align 4 %A, ptr addrspace(1) noundef align 4 %B, i32 noundef %N) #1 !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !5 !kernel_arg_type_qual !6 {
entry:
  %tile.local = alloca float, i64 1052, align 16, addrspace(3)
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
  %lid0 = call i64 @_Z12get_local_idj(i32 0)
  %lsize0 = call i64 @_Z14get_local_sizej(i32 0)
  %gid0 = call i64 @_Z13get_global_idj(i32 0)
  %group.base = sub i64 %gid0, %lid0
  %dense.mul = mul i64 %group.base, 4
  %dense.base = add i64 %dense.mul, 0
  %valid.lsize.minus1 = sub i64 %lsize0, 1
  %valid.tile.span = mul i64 4, %valid.lsize.minus1
  %valid.tile.elems = add i64 %valid.tile.span, 1
  %off.init.vec = mul i64 %lid0, 4
  %off.step.vec = mul i64 %lsize0, 4
  %groupid0 = call i64 @_Z12get_group_idj(i32 0)
  br label %tile.preload.header

tile.preload.header:                              ; preds = %tile.preload.latch, %if.then
  %off = phi i64 [ %off.init.vec, %if.then ], [ %off.next, %tile.preload.latch ]
  %tile.cond = icmp ult i64 %off, %valid.tile.elems
  br i1 %tile.cond, label %tile.preload.body, label %tile.preload.exit

tile.preload.body:                                ; preds = %tile.preload.header
  %global.idx = add i64 %dense.base, %off
  %bound64 = sext i32 %N to i64
  %vec.end.global = add i64 %global.idx, 4
  %vec.end.tile = add i64 %off, 4
  %vec.global.in.bounds = icmp ule i64 %vec.end.global, %bound64
  %vec.tile.in.bounds = icmp ule i64 %vec.end.tile, %valid.tile.elems
  %can.vectorize.preload = and i1 %vec.global.in.bounds, %vec.tile.in.bounds
  br i1 %can.vectorize.preload, label %tile.preload.vec, label %tile.preload.scalar

tile.preload.vec:                                 ; preds = %tile.preload.body
  %vec.global.ptr = getelementptr float, ptr addrspace(1) %2, i64 %global.idx
  %tile.vec.ld = load <4 x float>, ptr addrspace(1) %vec.global.ptr, align 16
  %tile.pad.groups = udiv i64 %off, 32
  %tile.pad = mul i64 %tile.pad.groups, 1
  %tile.physical.idx = add i64 %off, %tile.pad
  %vec.tile.ptr = getelementptr float, ptr addrspace(3) %tile.local, i64 %tile.physical.idx
  store <4 x float> %tile.vec.ld, ptr addrspace(3) %vec.tile.ptr, align 16
  br label %tile.preload.latch

tile.preload.scalar:                              ; preds = %tile.preload.body
  %lane.global.idx = add i64 %dense.base, %off
  %lane.in.tile = icmp ult i64 %off, %valid.tile.elems
  %lane.in.global = icmp ult i64 %lane.global.idx, %bound64
  %lane.in.bounds = and i1 %lane.in.tile, %lane.in.global
  %lane.safe.global.idx = select i1 %lane.in.bounds, i64 %lane.global.idx, i64 0
  %lane.safe.global.ptr = getelementptr float, ptr addrspace(1) %2, i64 %lane.safe.global.idx
  %tile.scalar.ld.raw = load float, ptr addrspace(1) %lane.safe.global.ptr, align 4
  %tile.scalar.ld = select i1 %lane.in.bounds, float %tile.scalar.ld.raw, float 0.000000e+00
  %tile.pad.groups1 = udiv i64 %off, 32
  %tile.pad2 = mul i64 %tile.pad.groups1, 1
  %tile.physical.idx3 = add i64 %off, %tile.pad2
  %tile.scalar.ptr = getelementptr float, ptr addrspace(3) %tile.local, i64 %tile.physical.idx3
  store float %tile.scalar.ld, ptr addrspace(3) %tile.scalar.ptr, align 4
  %lane.off = add i64 %off, 1
  %lane.global.idx4 = add i64 %dense.base, %lane.off
  %lane.in.tile5 = icmp ult i64 %lane.off, %valid.tile.elems
  %lane.in.global6 = icmp ult i64 %lane.global.idx4, %bound64
  %lane.in.bounds7 = and i1 %lane.in.tile5, %lane.in.global6
  %lane.safe.global.idx8 = select i1 %lane.in.bounds7, i64 %lane.global.idx4, i64 0
  %lane.safe.global.ptr9 = getelementptr float, ptr addrspace(1) %2, i64 %lane.safe.global.idx8
  %tile.scalar.ld.raw10 = load float, ptr addrspace(1) %lane.safe.global.ptr9, align 4
  %tile.scalar.ld11 = select i1 %lane.in.bounds7, float %tile.scalar.ld.raw10, float 0.000000e+00
  %tile.pad.groups12 = udiv i64 %lane.off, 32
  %tile.pad13 = mul i64 %tile.pad.groups12, 1
  %tile.physical.idx14 = add i64 %lane.off, %tile.pad13
  %tile.scalar.ptr15 = getelementptr float, ptr addrspace(3) %tile.local, i64 %tile.physical.idx14
  store float %tile.scalar.ld11, ptr addrspace(3) %tile.scalar.ptr15, align 4
  %lane.off16 = add i64 %off, 2
  %lane.global.idx17 = add i64 %dense.base, %lane.off16
  %lane.in.tile18 = icmp ult i64 %lane.off16, %valid.tile.elems
  %lane.in.global19 = icmp ult i64 %lane.global.idx17, %bound64
  %lane.in.bounds20 = and i1 %lane.in.tile18, %lane.in.global19
  %lane.safe.global.idx21 = select i1 %lane.in.bounds20, i64 %lane.global.idx17, i64 0
  %lane.safe.global.ptr22 = getelementptr float, ptr addrspace(1) %2, i64 %lane.safe.global.idx21
  %tile.scalar.ld.raw23 = load float, ptr addrspace(1) %lane.safe.global.ptr22, align 4
  %tile.scalar.ld24 = select i1 %lane.in.bounds20, float %tile.scalar.ld.raw23, float 0.000000e+00
  %tile.pad.groups25 = udiv i64 %lane.off16, 32
  %tile.pad26 = mul i64 %tile.pad.groups25, 1
  %tile.physical.idx27 = add i64 %lane.off16, %tile.pad26
  %tile.scalar.ptr28 = getelementptr float, ptr addrspace(3) %tile.local, i64 %tile.physical.idx27
  store float %tile.scalar.ld24, ptr addrspace(3) %tile.scalar.ptr28, align 4
  %lane.off29 = add i64 %off, 3
  %lane.global.idx30 = add i64 %dense.base, %lane.off29
  %lane.in.tile31 = icmp ult i64 %lane.off29, %valid.tile.elems
  %lane.in.global32 = icmp ult i64 %lane.global.idx30, %bound64
  %lane.in.bounds33 = and i1 %lane.in.tile31, %lane.in.global32
  %lane.safe.global.idx34 = select i1 %lane.in.bounds33, i64 %lane.global.idx30, i64 0
  %lane.safe.global.ptr35 = getelementptr float, ptr addrspace(1) %2, i64 %lane.safe.global.idx34
  %tile.scalar.ld.raw36 = load float, ptr addrspace(1) %lane.safe.global.ptr35, align 4
  %tile.scalar.ld37 = select i1 %lane.in.bounds33, float %tile.scalar.ld.raw36, float 0.000000e+00
  %tile.pad.groups38 = udiv i64 %lane.off29, 32
  %tile.pad39 = mul i64 %tile.pad.groups38, 1
  %tile.physical.idx40 = add i64 %lane.off29, %tile.pad39
  %tile.scalar.ptr41 = getelementptr float, ptr addrspace(3) %tile.local, i64 %tile.physical.idx40
  store float %tile.scalar.ld37, ptr addrspace(3) %tile.scalar.ptr41, align 4
  br label %tile.preload.latch

tile.preload.latch:                               ; preds = %tile.preload.scalar, %tile.preload.vec
  %off.next = add i64 %off, %off.step.vec
  br label %tile.preload.header

tile.preload.exit:                                ; preds = %tile.preload.header
  call void @_Z7barrierj(i32 1)
  br label %tile.cont

tile.cont:                                        ; preds = %tile.preload.exit
  %shared.logical.idx = mul i64 %lid0, 4
  %shared.in.bounds = icmp ult i64 %shared.logical.idx, %valid.tile.elems
  %shared.safe.logical.idx = select i1 %shared.in.bounds, i64 %shared.logical.idx, i64 0
  %tile.pad.groups42 = udiv i64 %shared.safe.logical.idx, 32
  %tile.pad43 = mul i64 %tile.pad.groups42, 1
  %tile.physical.idx44 = add i64 %shared.safe.logical.idx, %tile.pad43
  %shared.ptr = getelementptr float, ptr addrspace(3) %tile.local, i64 %tile.physical.idx44
  %shared.load = load float, ptr addrspace(3) %shared.ptr, align 4
  %4 = load ptr addrspace(1), ptr %B.addr, align 8
  %5 = load i32, ptr %tid, align 4
  %idxprom3 = sext i32 %5 to i64
  %arrayidx4 = getelementptr inbounds float, ptr addrspace(1) %4, i64 %idxprom3
  store float %shared.load, ptr addrspace(1) %arrayidx4, align 4
  br label %if.end

if.end:                                           ; preds = %tile.cont, %entry
  ret void
}

; Function Attrs: convergent nounwind willreturn memory(none)
declare spir_func i64 @_Z13get_global_idj(i32 noundef) #2

declare i64 @_Z12get_local_idj(i32)

declare i64 @_Z14get_local_sizej(i32)

declare i64 @_Z12get_group_idj(i32)

declare void @_Z7barrierj(i32)

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
