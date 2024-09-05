; RUN: llc %s -o - -mtriple=x86_64-unknown-linux-gnu | FileCheck %s

;; Test that, even though there are no source locations attached to the foo
;; function, we still give it the start-of-function source location of the
;; definition line. Otherwise, this function would have no entry in the
;; line table at all.

; CHECK-LABEL: foo:
; CHECK-NEXT:   .Lfunc_begin0:
; CHECK-NEXT:   .file   0 "." "foobar.c"
; CHECK-NEXT:   .cfi_startproc
; CHECK-NEXT:   # %bb.0:
; CHECK-NEXT:   .loc 0 1 0 prologue_end

define dso_local noundef i32 @foo(ptr nocapture noundef writeonly %bar) local_unnamed_addr !dbg !10 {
entry:
  store i32 0, ptr %bar, align 4
  ret i32 0
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "foobar.c", directory: ".")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"debug-info-assignment-tracking", i1 true}
!9 = !{!"clang"}
!10 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !11, scopeLine: 1, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !15)
!11 = !DISubroutineType(types: !12)
!12 = !{!13, !14}
!13 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !13, size: 64)
!15 = !{!16}
!16 = !DILocalVariable(name: "bar", arg: 1, scope: !10, file: !1, line: 1, type: !14)
!17 = !DILocation(line: 0, scope: !10)
!18 = !DILocation(line: 2, column: 8, scope: !10)
!19 = !{!20, !20, i64 0}
!20 = !{!"int", !21, i64 0}
!21 = !{!"omnipotent char", !22, i64 0}
!22 = !{!"Simple C/C++ TBAA"}
!23 = !DILocation(line: 3, column: 3, scope: !10)
