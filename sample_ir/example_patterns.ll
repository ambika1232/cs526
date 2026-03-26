; Minimal illustrative LLVM IR snippets for report discussion only.
;
; Coalesced pattern:
;   %tid = call i64 @_Z13get_global_idj(i32 0)
;   %ptr = getelementptr float, ptr %A, i64 %tid
;
; Strided pattern:
;   %tid = call i64 @_Z13get_global_idj(i32 0)
;   %idx = mul i64 %tid, 4
;   %ptr = getelementptr float, ptr %A, i64 %idx
