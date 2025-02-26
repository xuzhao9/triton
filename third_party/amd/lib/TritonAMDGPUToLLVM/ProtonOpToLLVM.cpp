#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"

#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "Utility.h"

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
    auto mod = op.getOperation()->getParentOfType<ModuleOp>();
    const int warpsPerGroup = triton::gpu::getWarpGroupSize();
    const int wordsPerEntry = triton::gpu::getWordsPerProtonEntry();
    const int slots =
        cast<IntegerAttr>(mod->getAttr("triton_gpu.proton-slots")).getInt();
    const int numWarpgroup =
        triton::gpu::TritonGPUDialect::getNumWarps(mod) / warpsPerGroup;
    const int step =
        isWarpLevel ? (warpsPerGroup * wordsPerEntry) : wordsPerEntry;

    Value indexPtr = adaptor.getIndexPtr();
    Value dataStruct = adaptor.getData();

    auto smemPtrTy = ptr_ty(rewriter.getContext(), 3);
    Value smemDataBasePtr = extract_val(smemPtrTy, dataStruct, 0);

    Value threadId = getThreadId(rewriter, loc);
    Value warpId =
        udiv(threadId,
             i32_val(triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod)));
    Value warpgroupSize = i32_val(
        warpsPerGroup * triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod));
    Value warpgroupId = udiv(threadId, warpgroupSize);

    // Load the index from smem.
    Value curIdx = load(i32_ty, indexPtr);
    Value newIdx = add(curIdx, i32_val(step));
    store(newIdx, indexPtr);

    // Compute the offset in smem.
    int numWgSlot = slots / numWarpgroup;
    Value warpOffset = isWarpLevel ? mul(urem(warpId, i32_val(warpsPerGroup)),
                                         i32_val(wordsPerEntry))
                                   : i32_val(0);
    Value wgSlotOffset =
        add(warpOffset, mul(warpgroupId, i32_val(wordsPerEntry * numWgSlot)));
    Value smemTagOffset =
        add(wgSlotOffset, urem(curIdx, i32_val(wordsPerEntry * numWgSlot)));

    // Record the entry and vectorized store to smem.
    Value vecPtr = gep(smemPtrTy, i32_ty, smemDataBasePtr, smemTagOffset);
    Value tag = op.getIsStart() ? i32_val(op.getRegionId())
                                : i32_val(1 << 31 | op.getRegionId());

    // Insert the LSB 12-bit HW_ID to the tag as a signature.
    GCNBuilder hwBuilder;
    auto &gethwid = *hwBuilder.create("s_getreg_b32");
    auto hwreg = hwBuilder.newOperand("=s");
    auto src = hwBuilder.newConstantOperand("hwreg(HW_REG_HW_ID, 0, 12)");
    gethwid(hwreg, src);
    auto hwid = hwBuilder.launch(rewriter, loc, i32_ty, false);
    Value tagWithHwid = or_(tag, shl(hwid, i32_val(16)));

    // Get the clock with region hint as comment.
    GCNBuilder clkBuilder;
    auto &rdclk = *clkBuilder.create("s_memtime");
    auto sreg = clkBuilder.newOperand("=s");
    rdclk(sreg);
    llvm::Twine startEnd =
        op.getIsStart() ? llvm::Twine("start") : llvm::Twine("end");
    llvm::Twine inst = isWarpLevel ? llvm::Twine("s_nop 0x6")
                                   : llvm::Twine("s_waitcnt lgkmcnt(0)");
    std::string instStr =
        llvm::Twine(inst + "    ;;#proton_record_region_" +
                    llvm::Twine(op.getRegionId()) + llvm::Twine("_") + startEnd)
            .str();
    clkBuilder.create<>(instStr)->operator()();
    auto clk64 = clkBuilder.launch(rewriter, loc, i64_ty, true);
    auto i32x2VecTy = vec_ty(i32_ty, 2);
    auto i32x2Vec = bitcast(clk64, i32x2VecTy);
    Value clock = extract_element(i32_ty, i32x2Vec, i32_val(0));

    Value valsVec = isWarpLevel
                        ? packLLVector(loc, {tag, clock}, rewriter)
                        : packLLVector(loc, {tagWithHwid, clock}, rewriter);

    targetInfo.storeDShared(rewriter, loc, vecPtr, std::nullopt, valsVec,
                            /*pred=*/true_val());

    rewriter.eraseOp(op);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

} // namespace

void mlir::triton::AMD::populateProtonOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfoBase &targetInfo, PatternBenefit benefit) {
  patterns.add<LocalRecordOpConversion>(typeConverter, targetInfo, benefit);
}