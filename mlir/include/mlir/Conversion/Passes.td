//===-- Passes.td - Conversion pass definition file --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_PASSES
#define MLIR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// AffineToStandard
//===----------------------------------------------------------------------===//

def ConvertAffineToStandard : Pass<"lower-affine"> {
  let summary = "Lower Affine operations to a combination of Standard and SCF "
                "operations";
  let description = [{

    Convert operations from the affine dialect into operations from the SCF and
    standard dialects.

    `affine.for` operations are converted to `scf.for` operations that are free
    of certain structural restrictions (on their bounds and step). `affine.if`
    is similarly converted to the `scf.if` operation. `affine.apply` operations
    are converted into sequences of primitive arithmetic operations from the
    standard dialect that have the same effect, using operands of the `index`
    type. Consequently, named maps and sets thare are no longer in use may be
    removed from the module.

    For example, `%r = affine.apply affine_map<(d0, d1)[s0] -> (d0 + 2*d1 +
    s0)>(%d0, %d1)[%s0]`
    can be converted into:

    ```mlir
    %d0 = <...>
    %d1 = <...>
    %s0 = <...>
    %0 = constant 2 : index
    %1 = muli %0, %d1
    %2 = addi %d0, %1
    %r = addi %2, %s0
    ```

    #### Input invariant

    -   no `Tensor` types;

    These restrictions may be lifted in the future.

    #### Output IR

    Functions with `affine.for` and `affine.if` operations eliminated. These
    functions may contain operations from the Standard dialect in addition to
    those already present before the pass.

    #### Invariants

    -   Functions without a body are not modified.
    -   The semantics of the other functions is preserved.
    -   Individual operations other than those mentioned above are not modified
        if they do not depend on the loop iterator value or on the result of
        `affine.apply`.
  }];
  let constructor = "mlir::createLowerAffinePass()";
  let dependentDialects = [
    "memref::MemRefDialect",
    "scf::SCFDialect",
    "StandardOpsDialect",
    "vector::VectorDialect"
  ];
}

//===----------------------------------------------------------------------===//
// AsyncToLLVM
//===----------------------------------------------------------------------===//

def ConvertAsyncToLLVM : Pass<"convert-async-to-llvm", "ModuleOp"> {
  let summary = "Convert the operations from the async dialect into the LLVM "
                "dialect";
  let description = [{
    Convert `async.execute` operations to LLVM coroutines and use async runtime
    API to execute them.
  }];
  let constructor = "mlir::createConvertAsyncToLLVMPass()";
  let dependentDialects = ["LLVM::LLVMDialect"];
}

//===----------------------------------------------------------------------===//
// ComplexToLLVM
//===----------------------------------------------------------------------===//

def ConvertComplexToLLVM : Pass<"convert-complex-to-llvm", "ModuleOp"> {
  let summary = "Convert Complex dialect to LLVM dialect";
  let constructor = "mlir::createConvertComplexToLLVMPass()";
  let dependentDialects = ["LLVM::LLVMDialect"];
}

//===----------------------------------------------------------------------===//
// ComplexToStandard
//===----------------------------------------------------------------------===//

def ConvertComplexToStandard : FunctionPass<"convert-complex-to-standard"> {
  let summary = "Convert Complex dialect to standard dialect";
  let constructor = "mlir::createConvertComplexToStandardPass()";
  let dependentDialects = [
    "complex::ComplexDialect",
    "math::MathDialect",
    "StandardOpsDialect"
  ];
}

//===----------------------------------------------------------------------===//
// GPUCommon
//===----------------------------------------------------------------------===//

def GpuToLLVMConversionPass : Pass<"gpu-to-llvm", "ModuleOp"> {
  let summary = "Convert GPU dialect to LLVM dialect with GPU runtime calls";
  let constructor = "mlir::createGpuToLLVMConversionPass()";
  let dependentDialects = ["LLVM::LLVMDialect"];
}

def LowerHostCodeToLLVM : Pass<"lower-host-to-llvm", "ModuleOp"> {
  let summary = "Lowers the host module code and `gpu.launch_func` to LLVM";
  let constructor = "mlir::createLowerHostCodeToLLVMPass()";
  let dependentDialects = ["LLVM::LLVMDialect"];
}

//===----------------------------------------------------------------------===//
// GPUToNVVM
//===----------------------------------------------------------------------===//

def ConvertGpuOpsToNVVMOps : Pass<"convert-gpu-to-nvvm", "gpu::GPUModuleOp"> {
  let summary = "Generate NVVM operations for gpu operations";
  let constructor = "mlir::createLowerGpuOpsToNVVMOpsPass()";
  let dependentDialects = ["NVVM::NVVMDialect", "memref::MemRefDialect"];
  let options = [
    Option<"indexBitwidth", "index-bitwidth", "unsigned",
           /*default=kDeriveIndexBitwidthFromDataLayout*/"0",
           "Bitwidth of the index type, 0 to use size of machine word">
  ];
}

//===----------------------------------------------------------------------===//
// GPUToROCDL
//===----------------------------------------------------------------------===//

def ConvertGpuOpsToROCDLOps : Pass<"convert-gpu-to-rocdl", "gpu::GPUModuleOp"> {
  let summary = "Generate ROCDL operations for gpu operations";
  let constructor = "mlir::createLowerGpuOpsToROCDLOpsPass()";
  let dependentDialects = ["ROCDL::ROCDLDialect"];
  let options = [
    Option<"indexBitwidth", "index-bitwidth", "unsigned",
           /*default=kDeriveIndexBitwidthFromDataLayout*/"0",
           "Bitwidth of the index type, 0 to use size of machine word">
  ];
}

//===----------------------------------------------------------------------===//
// GPUToSPIRV
//===----------------------------------------------------------------------===//

def ConvertGPUToSPIRV : Pass<"convert-gpu-to-spirv", "ModuleOp"> {
  let summary = "Convert GPU dialect to SPIR-V dialect";
  let description = [{
    This pass converts supported GPU device ops to SPIR-V ops. It does not
    handle GPU host ops.

    A `gpu.func` op can have parameters to pass in resources. But in SPIR-V
    entry functions cannot take parameters; they use descriptors to access
    resources. By default, parameters to a `gpu.func` op will be converted to
    global variables. These global variables will be assigned sequential binding
    numbers following their order in the original `gpu.func` op, starting from
    0, in set 0. One can attach `spv.interface_var_abi` to those parameters
    to control the set and binding if wanted.
  }];
  let constructor = "mlir::createConvertGPUToSPIRVPass()";
  let dependentDialects = ["spirv::SPIRVDialect"];
}

//===----------------------------------------------------------------------===//
// GPUToVulkan
//===----------------------------------------------------------------------===//

def ConvertGpuLaunchFuncToVulkanLaunchFunc
    : Pass<"convert-gpu-launch-to-vulkan-launch", "ModuleOp"> {
  let summary = "Convert gpu.launch_func to vulkanLaunch external call";
  let description = [{
    This pass is only intended for the mlir-vulkan-runner.
  }];
  let constructor = "mlir::createConvertGpuLaunchFuncToVulkanLaunchFuncPass()";
  let dependentDialects = ["spirv::SPIRVDialect"];
}

def ConvertVulkanLaunchFuncToVulkanCalls
    : Pass<"launch-func-to-vulkan", "ModuleOp"> {
  let summary = "Convert vulkanLaunch external call to Vulkan runtime external "
                "calls";
  let description = [{
    This pass is only intended for the mlir-vulkan-runner.
  }];
  let constructor = "mlir::createConvertVulkanLaunchFuncToVulkanCallsPass()";
  let dependentDialects = ["LLVM::LLVMDialect"];
}

//===----------------------------------------------------------------------===//
// LinalgToLLVM
//===----------------------------------------------------------------------===//

def ConvertLinalgToLLVM : Pass<"convert-linalg-to-llvm", "ModuleOp"> {
  let summary = "Convert the operations from the linalg dialect into the LLVM "
                "dialect";
  let constructor = "mlir::createConvertLinalgToLLVMPass()";
  let dependentDialects = ["scf::SCFDialect", "LLVM::LLVMDialect"];
}

//===----------------------------------------------------------------------===//
// LinalgToStandard
//===----------------------------------------------------------------------===//

def ConvertLinalgToStandard : Pass<"convert-linalg-to-std", "ModuleOp"> {
  let summary = "Convert the operations from the linalg dialect into the "
                "Standard dialect";
  let constructor = "mlir::createConvertLinalgToStandardPass()";
  let dependentDialects = ["memref::MemRefDialect", "StandardOpsDialect"];
}

//===----------------------------------------------------------------------===//
// LinalgToSPIRV
//===----------------------------------------------------------------------===//

def ConvertLinalgToSPIRV : Pass<"convert-linalg-to-spirv", "ModuleOp"> {
  let summary = "Convert Linalg dialect to SPIR-V dialect";
  let description = [{
    This pass converts supported Linalg ops to SPIR-V ops. It's quite
    experimental and are expected to migrate to other proper conversions.
  }];
  let constructor = "mlir::createLinalgToSPIRVPass()";
  let dependentDialects = ["spirv::SPIRVDialect"];
}

//===----------------------------------------------------------------------===//
// MathToLibm
//===----------------------------------------------------------------------===//

def ConvertMathToLibm : Pass<"convert-math-to-libm", "ModuleOp"> {
  let summary = "Convert Math dialect to libm calls";
  let description = [{
    This pass converts supported Math ops to libm calls.
  }];
  let constructor = "mlir::createConvertMathToLibmPass()";
  let dependentDialects = ["StandardOpsDialect", "vector::VectorDialect"];
}

//===----------------------------------------------------------------------===//
// MemRefToLLVM
//===----------------------------------------------------------------------===//

def ConvertMemRefToLLVM : Pass<"convert-memref-to-llvm", "ModuleOp"> {
  let summary = "Convert operations from the MemRef dialect to the LLVM "
                "dialect";
  let constructor = "mlir::createMemRefToLLVMPass()";
  let dependentDialects = ["LLVM::LLVMDialect"];
  let options = [
    Option<"useAlignedAlloc", "use-aligned-alloc", "bool", /*default=*/"false",
           "Use aligned_alloc in place of malloc for heap allocations">,
    Option<"indexBitwidth", "index-bitwidth", "unsigned",
           /*default=kDeriveIndexBitwidthFromDataLayout*/"0",
           "Bitwidth of the index type, 0 to use size of machine word">,
  ];
}

//===----------------------------------------------------------------------===//
// OpenACCToSCF
//===----------------------------------------------------------------------===//

def ConvertOpenACCToSCF : Pass<"convert-openacc-to-scf", "ModuleOp"> {
  let summary = "Convert the OpenACC ops to OpenACC with SCF dialect";
  let constructor = "mlir::createConvertOpenACCToSCFPass()";
  let dependentDialects = ["scf::SCFDialect", "acc::OpenACCDialect"];
}

//===----------------------------------------------------------------------===//
// OpenACCToLLVM
//===----------------------------------------------------------------------===//

def ConvertOpenACCToLLVM : Pass<"convert-openacc-to-llvm", "ModuleOp"> {
  let summary = "Convert the OpenACC ops to LLVM dialect";
  let constructor = "mlir::createConvertOpenACCToLLVMPass()";
  let dependentDialects = ["LLVM::LLVMDialect"];
}

//===----------------------------------------------------------------------===//
// OpenMPToLLVM
//===----------------------------------------------------------------------===//

def ConvertOpenMPToLLVM : Pass<"convert-openmp-to-llvm", "ModuleOp"> {
  let summary = "Convert the OpenMP ops to OpenMP ops with LLVM dialect";
  let constructor = "mlir::createConvertOpenMPToLLVMPass()";
  let dependentDialects = ["LLVM::LLVMDialect"];
}

//===----------------------------------------------------------------------===//
// PDLToPDLInterp
//===----------------------------------------------------------------------===//

def ConvertPDLToPDLInterp : Pass<"convert-pdl-to-pdl-interp", "ModuleOp"> {
  let summary = "Convert PDL ops to PDL interpreter ops";
  let constructor = "mlir::createPDLToPDLInterpPass()";
  let dependentDialects = ["pdl_interp::PDLInterpDialect"];
}

//===----------------------------------------------------------------------===//
// SCFToOpenMP
//===----------------------------------------------------------------------===//

def ConvertSCFToOpenMP : FunctionPass<"convert-scf-to-openmp"> {
  let summary = "Convert SCF parallel loop to OpenMP parallel + workshare "
                "constructs.";
  let constructor = "mlir::createConvertSCFToOpenMPPass()";
  let dependentDialects = ["omp::OpenMPDialect"];
}

//===----------------------------------------------------------------------===//
// SCFToSPIRV
//===----------------------------------------------------------------------===//

def SCFToSPIRV : Pass<"convert-scf-to-spirv", "ModuleOp"> {
  let summary = "Convert SCF dialect to SPIR-V dialect.";
  let description = [{
    This pass converts SCF ops into SPIR-V structured control flow ops.
    SPIR-V structured control flow ops does not support yielding values.
    So for SCF ops yielding values, SPIR-V variables are created for
    holding the values and load/store operations are emitted for updating
    them.
  }];
  let constructor = "mlir::createConvertSCFToSPIRVPass()";
  let dependentDialects = ["spirv::SPIRVDialect"];
}

//===----------------------------------------------------------------------===//
// SCFToStandard
//===----------------------------------------------------------------------===//

def SCFToStandard : Pass<"convert-scf-to-std"> {
  let summary = "Convert SCF dialect to Standard dialect, replacing structured"
                " control flow with a CFG";
  let constructor = "mlir::createLowerToCFGPass()";
  let dependentDialects = ["StandardOpsDialect"];
}

//===----------------------------------------------------------------------===//
// SCFToGPU
//===----------------------------------------------------------------------===//

def ConvertAffineForToGPU : FunctionPass<"convert-affine-for-to-gpu"> {
  let summary = "Convert top-level AffineFor Ops to GPU kernels";
  let constructor = "mlir::createAffineForToGPUPass()";
  let dependentDialects = ["gpu::GPUDialect"];
  let options = [
    Option<"numBlockDims", "gpu-block-dims", "unsigned", /*default=*/"1u",
           "Number of GPU block dimensions for mapping">,
    Option<"numThreadDims", "gpu-thread-dims", "unsigned", /*default=*/"1u",
           "Number of GPU thread dimensions for mapping">
  ];
}

def ConvertParallelLoopToGpu : Pass<"convert-parallel-loops-to-gpu"> {
  let summary = "Convert mapped scf.parallel ops to gpu launch operations";
  let constructor = "mlir::createParallelLoopToGpuPass()";
  let dependentDialects = ["AffineDialect", "gpu::GPUDialect"];
}

//===----------------------------------------------------------------------===//
// ShapeToStandard
//===----------------------------------------------------------------------===//

def ConvertShapeToStandard : Pass<"convert-shape-to-std", "ModuleOp"> {
  let summary = "Convert operations from the shape dialect into the standard "
                "dialect";
  let constructor = "mlir::createConvertShapeToStandardPass()";
  let dependentDialects = [
    "StandardOpsDialect",
    "scf::SCFDialect",
    "tensor::TensorDialect"
  ];
}

def ConvertShapeConstraints: Pass<"convert-shape-constraints", "FuncOp"> {
  let summary = "Convert shape constraint operations to the standard dialect";
  let description = [{
    This pass eliminates shape constraints from the program, converting them to
    eager (side-effecting) error handling code.

    This pass is separate from the regular convert-shape-to-standard, despite
    converting between the same dialects, because converting shape constraints
    can happen at a different part of the program than general shape
    computation lowering.
  }];
  let constructor = "mlir::createConvertShapeConstraintsPass()";
  let dependentDialects = ["StandardOpsDialect", "scf::SCFDialect"];
}

//===----------------------------------------------------------------------===//
// SPIRVToLLVM
//===----------------------------------------------------------------------===//

def ConvertSPIRVToLLVM : Pass<"convert-spirv-to-llvm", "ModuleOp"> {
  let summary = "Convert SPIR-V dialect to LLVM dialect";
  let description = [{
    See https://mlir.llvm.org/docs/SPIRVToLLVMDialectConversion/
    for more details.
  }];
  let constructor = "mlir::createConvertSPIRVToLLVMPass()";
  let dependentDialects = ["LLVM::LLVMDialect"];
}

//===----------------------------------------------------------------------===//
// StandardToLLVM
//===----------------------------------------------------------------------===//

def ConvertStandardToLLVM : Pass<"convert-std-to-llvm", "ModuleOp"> {
  let summary = "Convert scalar and vector operations from the Standard to the "
                "LLVM dialect";
  let description = [{
    Convert standard operations into the LLVM IR dialect operations.

    #### Input invariant

    -   operations including: arithmetic on integers and floats, constants,
        direct calls, returns and branches;
    -   no `tensor` types;
    -   all `vector` are one-dimensional;
    -   all blocks are reachable by following the successors of the first basic
        block;

    If other operations are present and their results are required by the LLVM
    IR dialect operations, the pass will fail.  Any LLVM IR operations or types
    already present in the IR will be kept as is.

    #### Output IR

    Functions converted to LLVM IR. Function arguments types are converted
    one-to-one. Function results are converted one-to-one and, in case more than
    1 value is returned, packed into an LLVM IR struct type. Function calls and
    returns are updated accordingly. Block argument types are updated to use
    LLVM IR types.
  }];
  let constructor = "mlir::createLowerToLLVMPass()";
  let dependentDialects = ["LLVM::LLVMDialect"];
  let options = [
    Option<"useBarePtrCallConv", "use-bare-ptr-memref-call-conv", "bool",
           /*default=*/"false",
           "Replace FuncOp's MemRef arguments with bare pointers to the MemRef "
           "element types">,
    Option<"emitCWrappers", "emit-c-wrappers", "bool", /*default=*/"false",
           "Emit wrappers for C-compatible pointer-to-struct memref "
           "descriptors">,
    Option<"indexBitwidth", "index-bitwidth", "unsigned",
           /*default=kDeriveIndexBitwidthFromDataLayout*/"0",
           "Bitwidth of the index type, 0 to use size of machine word">,
    Option<"dataLayout", "data-layout", "std::string",
           /*default=*/"\"\"",
           "String description (LLVM format) of the data layout that is "
           "expected on the produced module">
  ];
}

//===----------------------------------------------------------------------===//
// StandardToSPIRV
//===----------------------------------------------------------------------===//

def ConvertStandardToSPIRV : Pass<"convert-std-to-spirv", "ModuleOp"> {
  let summary = "Convert Standard dialect to SPIR-V dialect";
  let constructor = "mlir::createConvertStandardToSPIRVPass()";
  let dependentDialects = ["spirv::SPIRVDialect"];
  let options = [
    Option<"emulateNon32BitScalarTypes", "emulate-non-32-bit-scalar-types",
           "bool", /*default=*/"true",
           "Emulate non-32-bit scalar types with 32-bit ones if "
           "missing native support">
  ];
}

//===----------------------------------------------------------------------===//
// TosaToLinalg
//===----------------------------------------------------------------------===//

def TosaToLinalgOnTensors : FunctionPass<"tosa-to-linalg-on-tensors"> {
  let summary = "Lower TOSA to LinAlg on tensors";
  let description = [{
    Pass that converts TOSA operations to the equivalent operations using the
    tensor operations in LinAlg.
  }];

  let constructor = "tosa::createTosaToLinalgOnTensors()";
}

//===----------------------------------------------------------------------===//
// TosaToSCF
//===----------------------------------------------------------------------===//

def TosaToSCF : Pass<"tosa-to-scf"> {
  let summary = "Lower TOSA to the SCF dialect";
  let dependentDialects = ["tensor::TensorDialect, scf::SCFDialect"];
  let description = [{
    Pass that converts TOSA's control flow operations to the equivalent SCF
    operations.
  }];

  let constructor = "tosa::createTosaToSCF()";
}

//===----------------------------------------------------------------------===//
// TosaToStandard
//===----------------------------------------------------------------------===//

def TosaToStandard : Pass<"tosa-to-standard"> {
  let summary = "Lower TOSA to the Standard dialect";
  let dependentDialects = ["StandardOpsDialect", "tensor::TensorDialect"];
  let description = [{
    Pass that converts TOSA operations to the equivalent operations using the
    operations in the Standard dialect.
  }];

  let constructor = "tosa::createTosaToStandard()";
}

//===----------------------------------------------------------------------===//
// VectorToGPU
//===----------------------------------------------------------------------===//

def ConvertVectorToGPU : FunctionPass<"convert-vector-to-gpu"> {
  let summary = "Lower the operations from the vector dialect into the GPU "
                "dialect";
  let constructor = "mlir::createConvertVectorToGPUPass()";
  let dependentDialects = [
    "memref::MemRefDialect",
    "gpu::GPUDialect"
  ];
}

//===----------------------------------------------------------------------===//
// VectorToSCF
//===----------------------------------------------------------------------===//

def ConvertVectorToSCF : FunctionPass<"convert-vector-to-scf"> {
  let summary = "Lower the operations from the vector dialect into the SCF "
                "dialect";
  let constructor = "mlir::createConvertVectorToSCFPass()";
  let dependentDialects = [
    "AffineDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect"
  ];
  let options = [
    Option<"fullUnroll", "full-unroll", "bool", /*default=*/"false",
           "Perform full unrolling when converting vector transfers to SCF">,
    Option<"targetRank", "target-rank", "unsigned", /*default=*/"1",
           "Target vector rank to which transfer ops should be lowered">,
    Option<"lowerPermutationMaps", "lower-permutation-maps", "bool",
           /*default=*/"false", "Replace permutation maps with vector "
           "transposes/broadcasts before lowering transfer ops">,
    Option<"lowerTensors", "lower-tensors", "bool", /*default=*/"false",
           "Lower transfer ops that operate on tensors">
  ];
}

//===----------------------------------------------------------------------===//
// VectorToLLVM
//===----------------------------------------------------------------------===//

def ConvertVectorToLLVM : Pass<"convert-vector-to-llvm", "ModuleOp"> {
  let summary = "Lower the operations from the vector dialect into the LLVM "
                "dialect";
  let description = [{

    Convert operations from the vector dialect into the LLVM IR dialect
    operations. The lowering pass provides several options to control
    the kinds of optimizations that are allowed. It also provides options
    that enable the use of one or more architectural-specific dialects
    (AMX, X86Vector, ArmNeon, ArmSVE, etc.) in combination with the
    architectural-neutral vector dialect lowering.

  }];
  let constructor = "mlir::createConvertVectorToLLVMPass()";
  // Override explicitly in C++ to allow conditional dialect dependence.
  // let dependentDialects;
  let options = [
    Option<"reassociateFPReductions", "reassociate-fp-reductions",
           "bool", /*default=*/"false",
           "Allows llvm to reassociate floating-point reductions for speed">,
    Option<"enableIndexOptimizations", "enable-index-optimizations",
           "bool", /*default=*/"true",
           "Allows compiler to assume indices fit in 32-bit if that yields "
	   "faster code">,
    Option<"enableAMX", "enable-amx",
           "bool", /*default=*/"false",
           "Enables the use of AMX dialect while lowering the vector "
	   "dialect.">,
    Option<"enableArmNeon", "enable-arm-neon",
           "bool", /*default=*/"false",
           "Enables the use of ArmNeon dialect while lowering the vector "
	   "dialect.">,
    Option<"enableArmSVE", "enable-arm-sve",
           "bool", /*default=*/"false",
           "Enables the use of ArmSVE dialect while lowering the vector "
       "dialect.">,
    Option<"enableX86Vector", "enable-x86vector",
           "bool", /*default=*/"false",
           "Enables the use of X86Vector dialect while lowering the vector "
	   "dialect.">
  ];
}

//===----------------------------------------------------------------------===//
// VectorToROCDL
//===----------------------------------------------------------------------===//

def ConvertVectorToROCDL : Pass<"convert-vector-to-rocdl", "ModuleOp"> {
  let summary = "Lower the operations from the vector dialect into the ROCDL "
                "dialect";
  let constructor = "mlir::createConvertVectorToROCDLPass()";
  let dependentDialects = ["ROCDL::ROCDLDialect"];
}

//===----------------------------------------------------------------------===//
// VectorToSPIRV
//===----------------------------------------------------------------------===//

def ConvertVectorToSPIRV : Pass<"convert-vector-to-spirv", "ModuleOp"> {
  let summary = "Convert Vector dialect to SPIR-V dialect";
  let constructor = "mlir::createConvertVectorToSPIRVPass()";
  let dependentDialects = ["spirv::SPIRVDialect"];
}

//===----------------------------------------------------------------------===//
// ArmNeon2dToIntr
//===----------------------------------------------------------------------===//

def ConvertArmNeon2dToIntr : Pass<"arm-neon-2d-to-intr", "FuncOp"> {
  let summary = "Convert Arm NEON structured ops to intrinsics";
  let constructor = "mlir::createConvertArmNeon2dToIntrPass()";
  let dependentDialects = ["arm_neon::ArmNeonDialect", "vector::VectorDialect"];
}


#endif // MLIR_CONVERSION_PASSES
