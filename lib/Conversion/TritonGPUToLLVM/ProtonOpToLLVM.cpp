#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace {

using namespace mlir;
using namespace mlir::triton::gpu;

struct ProtonFinalizeOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::ProtonFinalizeOp> {
  explicit ProtonFinalizeOpConversion(LLVMTypeConverter &typeConverter,
                                      const TargetInfoBase &targetInfo,
                                      PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::gpu::ProtonFinalizeOp>(typeConverter,
                                                              benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::ProtonFinalizeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Value indexPtr = adaptor.getIndexPtr();
    Value dataStruct = adaptor.getData();

    auto loc = op.getLoc();
    auto mod = op.getOperation()->getParentOfType<ModuleOp>();
    const int warpsPerGroup = triton::gpu::getWarpGroupSize();
    const int wordsPerEntry = triton::gpu::getWordsPerProtonEntry();
    // Hack: adapt the hacking from Warp Specialization.
    int wgSpecNum = 1;
    if (Attribute attr = mod->getAttr("triton_gpu.num-warp-groups-per-cta")) {
      wgSpecNum = cast<IntegerAttr>(attr).getInt();
    }
    const int numWarp =
        triton::gpu::lookupNumWarps(mod) * wgSpecNum;
    Value threadId = getThreadId(rewriter, loc);
    Value warpId =
        udiv(threadId,
             i32_val(triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod)));
    Value isFirstThread = icmp_eq(threadId, i32_val(0));

    const int slots =
        cast<IntegerAttr>(mod->getAttr("triton_gpu.proton-slots")).getInt();
    // scratch: preample (1), block id (1), [hwid, index] (2 * numWarp), data
    // (slots * wordsPerEntry)
    const int scratchWordSize = 1 + 1 + 2 * numWarp + slots * wordsPerEntry;

    // Note: we compute use i64 data type to compute and then truncate to i32
    // because the amd backend has __ockl_get_num_groups(i32) -> i64.
    // This is a workaround.
    Value pidX = sext(i64_ty, targetInfo.programId(rewriter, loc, mod, 0));
    Value pidY = sext(i64_ty, targetInfo.programId(rewriter, loc, mod, 1));
    Value pidZ = sext(i64_ty, targetInfo.programId(rewriter, loc, mod, 2));
    Value hwid = targetInfo.hardwareId(rewriter, loc);
    Value gridDimX = rewriter.create<arith::IndexCastOp>(
        loc, i64_ty,
        rewriter.create<::mlir::gpu::GridDimOp>(loc, mlir::gpu::Dimension::x));
    Value gridDimY = rewriter.create<arith::IndexCastOp>(
        loc, i64_ty,
        rewriter.create<::mlir::gpu::GridDimOp>(loc, mlir::gpu::Dimension::y));
    Value pid = trunc(i32_ty, add(add(pidX, mul(pidY, gridDimX)),
                                  mul(pidZ, mul(gridDimX, gridDimY))));
    Value programOffset = mul(i32_val(scratchWordSize), pid);

    auto gmemPtrTy = ptr_ty(rewriter.getContext(), 1);
    Value gmemBasePtr = adaptor.getPtr();
    auto smemPtrTy = ptr_ty(rewriter.getContext(), 3);

    // Add the [hwid, index] section.
    Value warpHwidOffset =
        add(programOffset, add(mul(warpId, i32_val(2)), i32_val(2)));
    Value warpIndexOffset = add(warpHwidOffset, i32_val(1));
    Value gmemWarpHwidPtr = gep(gmemPtrTy, i32_ty, gmemBasePtr, warpHwidOffset);
    store(hwid, gmemWarpHwidPtr);
    Value gmemWarpIndexPtr =
        gep(gmemPtrTy, i32_ty, gmemBasePtr, warpIndexOffset);
    Value index = load(i32_ty, indexPtr);
    store(index, gmemWarpIndexPtr);

    Block *prevBlock = op->getBlock();
    // Add the 'if' block.
    Block *ifBlock = rewriter.splitBlock(prevBlock, op->getIterator());
    rewriter.setInsertionPointToStart(ifBlock);

    // Lambda function to load a word from smem and store it to gmem.
    auto copyWord = [&](Value smemStruct, Value smemOffset, Value gmemOffset) {
      Value smemBasePtr = extract_val(smemPtrTy, smemStruct, 0);
      // Load the value from smem
      Value ptr = gep(smemPtrTy, i32_ty, smemBasePtr, smemOffset);
      Value smemLoad =
          targetInfo.loadShared(rewriter, loc, ptr, i32_ty, true_val());
      // Store the value to global memory
      Value gmemPtr = gep(gmemPtrTy, i32_ty, gmemBasePtr, gmemOffset);
      store(smemLoad, gmemPtr);
    };

    // Write back 'preample'.
    Value preample = i32_val(0xdeadbeef);
    Value gmemPreampleOffset = programOffset;
    Value gmemPreamplePtr =
        gep(gmemPtrTy, i32_ty, gmemBasePtr, gmemPreampleOffset);
    store(preample, gmemPreamplePtr);

    // Write back 'program id'.
    Value gmemPidOffset = add(programOffset, i32_val(1));
    Value gmemPidPtr = gep(gmemPtrTy, i32_ty, gmemBasePtr, gmemPidOffset);
    store(pid, gmemPidPtr);

    int offset = 2 + 2 * numWarp;
    // Add the 'else' block and the condition.
    Block *thenBlock = rewriter.splitBlock(ifBlock, op->getIterator());
    rewriter.setInsertionPointToEnd(prevBlock);
    rewriter.create<cf::CondBranchOp>(loc, isFirstThread, ifBlock, thenBlock);

    // Write back the data.
    const int upper = wordsPerEntry * (slots - 1);
    rewriter.setInsertionPointToEnd(ifBlock);
    Value initIdx = rewriter.create<LLVM::ConstantOp>(loc, i32_ty, 0);
    Value wbBaseOffset = add(programOffset, i32_val(offset));

    Block *writeBackBlock = rewriter.createBlock(
        op->getParentRegion(), std::next(Region::iterator(ifBlock)), {i32_ty},
        {loc});
    rewriter.setInsertionPointToStart(writeBackBlock);
    BlockArgument idx = writeBackBlock->getArgument(0);
    Value gmemWbTagOffset = add(wbBaseOffset, idx);
    Value smemTagOffset = idx;
    Value gmemWbCycleOffset = add(gmemWbTagOffset, i32_val(1));
    Value smemCycleOffset = add(smemTagOffset, i32_val(1));
    copyWord(dataStruct, smemTagOffset, gmemWbTagOffset);
    copyWord(dataStruct, smemCycleOffset, gmemWbCycleOffset);
    Value pred = icmp_slt(idx, i32_val(upper));
    Value updatedIdx = add(idx, i32_val(wordsPerEntry));
    rewriter.create<cf::CondBranchOp>(loc, pred, writeBackBlock, updatedIdx,
                                      thenBlock, ArrayRef<Value>());

    rewriter.setInsertionPointToEnd(ifBlock);
    rewriter.create<cf::BranchOp>(loc, writeBackBlock, initIdx);
    rewriter.eraseOp(op);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

struct ProtonInitOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::ProtonInitOp> {
  explicit ProtonInitOpConversion(LLVMTypeConverter &typeConverter,
                                  const TargetInfoBase &targetInfo,
                                  PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::gpu::ProtonInitOp>(typeConverter,
                                                          benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::ProtonInitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto mod = op.getOperation()->getParentOfType<ModuleOp>();
    auto ptrTy = ptr_ty(rewriter.getContext(), 1);
    auto indexPtr = rewriter.create<LLVM::AllocaOp>(
        loc, ptrTy, i32_ty, i32_val(1), /*alignment=*/0);
    store(i32_val(0), indexPtr);
    rewriter.replaceOp(op, indexPtr);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

} // namespace

void mlir::triton::populateProtonOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<ProtonFinalizeOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<ProtonInitOpConversion>(typeConverter, targetInfo, benefit);
}