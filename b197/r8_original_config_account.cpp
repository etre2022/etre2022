#include "config.hpp"
#include <cstdint>
#include <cstdio>
#include <cstddef>
#include <cassert>
using namespace gdn;
int main(){
  constexpr std::size_t words=(std::size_t)kPackedTotalBytes/kPackedWordBytes;
  constexpr std::size_t beats=words*kBeatsPerWord;
  constexpr std::size_t read_embedding_words=kDimGroups*2;
  constexpr std::size_t layer_words=kPackedQKVGWords+kPackedProjWords+2*kPackedW13Words+kPackedW2Words;
  constexpr std::size_t expected= kPackedTokWords+std::size_t(kNumLayers)*layer_words;
  constexpr std::size_t state_s_bytes=kSStateCount*sizeof(float);
  constexpr std::size_t state_conv_bytes=(kQConvStateCount+kKConvStateCount+kVConvStateCount)*sizeof(float);
  constexpr std::size_t total_model_bytes=kPackedTotalBytes+kSideFloatCount*sizeof(float)+kCheckpointHeaderBytes;
  static_assert(kPackedWordBytes==64&&kBeatsPerWord==4,"unexpected packed bus");
  static_assert(kPackedRows==16&&kQuantGroupSize==32,"unexpected layout");
  static_assert(kStateBanks==16&&kStateGroups==2,"unexpected state banking");
  static_assert(words==expected,"packed breakdown inconsistent");
  static_assert(kPackedTotalBytes==17473536,"unpinned model contract");
  static_assert(kSideFloatCount==62080,"unpinned side");
  static_assert(kPackedTokWords==144000,"unexpected LM head");
  static_assert(layer_words==16128,"unexpected layer words");
  static_assert(total_model_bytes==17722112,"unexpected checkpoint size");
  static_assert(state_s_bytes==262144,"unexpected GDN state");
  static_assert(state_conv_bytes==98304,"unexpected conv state");
  std::printf("B197_R8_COMPILED_CONFIG_PASS\n");
  std::printf("{\"source\":\"compile_original_header\",\"packed_weight_words\":%zu,\"packed_weight_bytes\":%zu,\"weight_beats_128b\":%zu,\"embedding_extra_weight_words\":%zu,\"embedding_extra_weight_bytes\":%zu,\"lm_head_words\":%d,\"per_layer_words\":%zu,\"layers\":%d,\"heads_per_layer\":%d,\"recurrent_FP32_bytes\":%zu,\"conv_FP32_bytes\":%zu,\"total_FP32_state_bytes\":%zu,\"side_float_bytes\":%zu,\"checkpoint_bytes\":%zu}\n",
  words,(std::size_t)kPackedTotalBytes,beats,read_embedding_words,read_embedding_words*kPackedWordBytes,kPackedTokWords,layer_words,kNumLayers,kNumHeads,state_s_bytes,state_conv_bytes,state_s_bytes+state_conv_bytes,kSideFloatCount*sizeof(float),total_model_bytes);
  return 0;
}
