// B197 independently authored black-box testbench.
// Original upstream kernel/reference are linked but never altered.
// The local ap_int shim is native C++ only, not hardware simulation.
#include <ap_int.h>
#include "decode.hpp"
#include "weight.hpp"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
extern "C" void decode(int token,int reset_state,const ap_uint<128>* packed_params,
                       const float* side,std::uint32_t* next_token);
static std::vector<ap_uint<128>> pack_beats(const std::vector<std::uint8_t>& bytes){
  if(bytes.size()%16)std::abort();
  std::vector<ap_uint<128>> out(bytes.size()/16);
  for(std::size_t i=0;i<out.size();i++){
    ap_uint<128> value=0;
    for(int j=0;j<16;j++)value.range(j*8+7,j*8)=bytes[16*i+j];
    out[i]=value;
  }
  return out;
}
int main(int argc,char** argv){
  if(argc!=2){std::fprintf(stderr,"usage: %s checkpoint\n",argv[0]);return 2;}
  gdn::Weights weights;gdn::LoadWeights(weights,argv[1]);
  auto blob=gdn::PackParameters(weights);
  auto side=gdn::BuildFp32Side(weights);
  auto beats=pack_beats(blob);
  std::printf("B197_PACK bytes=%zu beats=%zu side=%zu\n",blob.size(),beats.size(),side.size());
  if(blob.size()!=gdn::kPackedTotalBytes||side.size()!=gdn::kSideFloatCount)return 3;
  std::array<int,4> counts{{64,64,64,256}};
  std::vector<int> first;
  unsigned total=0;
  for(std::size_t scenario=0;scenario<counts.size();scenario++){
    gdn::RunState cpu;
    std::vector<float> logits(gdn::kVocabSize);
    std::vector<int> outputs;
    for(int t=0;t<counts[scenario];t++){
      const int tok=(scenario==1)?((197+71*t+17*t*t)%gdn::kVocabSize):
                    (scenario==3)?((1024+23*t+13*t*t)%gdn::kVocabSize):
                                  ((1+97*t+3*t*t)%gdn::kVocabSize);
      std::uint32_t native_out=0;
      decode(tok,t==0?1:0,beats.data(),side.data(),&native_out);
      gdn::CpuForward(cpu,weights,tok,logits.data());
      const int cpu_out=gdn::ArgmaxLogits(logits.data(),gdn::kVocabSize);
      if(cpu_out!=int(native_out)){
        std::fprintf(stderr,"B197_KERNEL_MISMATCH scenario=%zu step=%d input=%d native=%u cpu=%d\n",scenario,t,tok,native_out,cpu_out);
        return 5;
      }
      outputs.push_back(int(native_out));total++;
    }
    if(scenario==0) first=outputs;
    if(scenario==2 && outputs!=first){
      std::fprintf(stderr,"B197_RESET_DID_NOT_REPLAY_IDENTICALLY\n");return 6;
    }
    std::printf("B197_SCENARIO_PASS %zu steps=%d repeated_A=%s\n",scenario,counts[scenario],scenario==2?"yes":"n/a");
  }
  std::printf("B197_GDN_NATIVE_RESET_4_SESSIONS_PASS total_compare_points=%u\n",total);
  return 0;
}
