#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "PatternTritonGPUOpToLLVM.h"

namespace {

using namespace mlir;
using namespace mlir::triton::gpu;

struct LocalRecordOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::LocalRecordOp> {
  explicit LocalRecordOpConversion(LLVMTypeConverter &typeConverter,
                                   const TargetInfoBase &targetInfo,
                                   PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::gpu::LocalRecordOp>(typeConverter,
                                                           benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::LocalRecordOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    assert(op.getMetric() == triton::ProtonMetric::CYCLE);
    const bool isWarpLevel =
        op.getGranularity() == triton::ProtonGranularity::WARP;

    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto mod = op.getOperation()->getParentOfType<ModuleOp>();

    // Hack: adapt the hacking from Warp Specialization.
    int wgSpecNum = 1;
    if (Attribute attr = mod->getAttr("triton_gpu.num-warp-groups-per-cta")) {
      wgSpecNum = cast<IntegerAttr>(attr).getInt();
    }

    const int warpsPerGroup = triton::gpu::getWarpGroupSize();
    const int wordsPerEntry = triton::gpu::getWordsPerProtonEntry();
    const int warpSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    const int warpgroupSize = warpsPerGroup * warpSize;
    const int slots =
        cast<IntegerAttr>(mod->getAttr("triton_gpu.proton-slots")).getInt();
    const int numWarpgroup = triton::gpu::lookupNumWarps(mod) *
                             wgSpecNum / warpsPerGroup;
    const int step =
        isWarpLevel ? (warpsPerGroup * wordsPerEntry) : wordsPerEntry;

    Value indexPtr = adaptor.getIndexPtr();
    Value dataStruct = adaptor.getData();

    auto smemPtrTy = ptr_ty(rewriter.getContext(), 3);
    Value smemDataBasePtr = b.extract_val(smemPtrTy, dataStruct, 0);

    Value threadId = getThreadId(rewriter, loc);
    Value warpId = b.udiv(threadId, b.i32_val(warpSize));
    Value warpgroupId = b.udiv(threadId, b.i32_val(warpgroupSize));
    Value isWarp = b.icmp_eq(b.urem(threadId, b.i32_val(warpSize)), b.i32_val(0));
    Value isWarpgroup = b.icmp_eq(b.urem(threadId, b.i32_val(warpgroupSize)), b.i32_val(0));

    // Load the index from gmem.
    Value curIdx = b.load(i32_ty, indexPtr);
    Value newIdx = b.add(curIdx, b.i32_val(step));
    b.store(newIdx, indexPtr);

    // Compute the offset in smem.
    int numWgSlot = slots / numWarpgroup;
    Value warpOffset = isWarpLevel ? b.mul(b.urem(warpId, b.i32_val(warpsPerGroup)), b.i32_val(wordsPerEntry))
                                   : b.i32_val(0);
    Value wgSlotOffset = b.add(warpOffset, b.mul(warpgroupId, b.i32_val(wordsPerEntry * numWgSlot)));
    Value smemTagOffset = b.add(wgSlotOffset, b.urem(curIdx, b.i32_val(wordsPerEntry * numWgSlot)));

    // Record the entry and vectorized store to smem.
    Value vecPtr = b.gep(smemPtrTy, i32_ty, smemDataBasePtr, smemTagOffset);
    Value tag = op.getIsStart() ? i32_val(op.getRegionId())
                                : i32_val(1 << 31 | op.getRegionId());
    Value clock = targetInfo.clock(rewriter, loc, false);

    Value valsVec = packLLVector(loc, {tag, clock}, rewriter);

    if (isWarpLevel)
      targetInfo.storeDShared(rewriter, loc, vecPtr, std::nullopt, valsVec,
                              /*pred=*/isWarp);
    else
      targetInfo.storeDShared(rewriter, loc, vecPtr, std::nullopt, valsVec,
                              /*pred=*/isWarpgroup);

    rewriter.eraseOp(op);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

} // namespace

void mlir::triton::NVIDIA::populateProtonOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<LocalRecordOpConversion>(typeConverter, targetInfo, benefit);
}