// B197 independently authored GDN native-only numeric diversity diagnostic.
// This does NOT represent real Vitis C-simulation or hardware.
#include <ap_int.h>
#include "decode.hpp"
#include "weight.hpp"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
extern "C" void decode(int token,int reset_state,const ap_uint<128>* params,const float* side,std::uint32_t* output);
static std::vector<ap_uint<128>> beats(const std::vector<std::uint8_t>& b){
  std::vector<ap_uint<128>> x(b.size()/16);
  for(std::size_t i=0;i<x.size();++i){ap_uint<128> v=0;
    for(int j=0;j<16;++j)v.range(j*8+7,j*8)=b[16*i+j];
    x[i]=v;}
  return x;
}
int main(int argc,char**argv){
  if(argc!=2)return 2;
  gdn::Weights w;gdn::LoadWeights(w,argv[1]);
  auto b=beats(gdn::PackParameters(w));
  auto s=gdn::BuildFp32Side(w);
  std::array<int,4> lens{{64,64,64,256}};
  std::vector<int> first_native,first_cpu;
  int total=0,mismatches=0,replayed_diff_native=-1,replayed_diff_cpu=-1;
  float max_cpu_gap=0.0f;
  for(int seq=0;seq<4;++seq){
    gdn::RunState state;
    std::vector<float> logits(gdn::kVocabSize);
    std::vector<int> native_ids,cpu_ids;
    int m=0;
    for(int t=0;t<lens[seq];++t){
      int token=seq==1?(197+71*t+17*t*t)%gdn::kVocabSize:
                seq==3?(1024+23*t+13*t*t)%gdn::kVocabSize:
                       (1+97*t+3*t*t)%gdn::kVocabSize;
      std::uint32_t hw=0;
      decode(token,t==0, b.data(),s.data(),&hw);
      gdn::CpuForward(state,w,token,logits.data());
      int cpu=gdn::ArgmaxLogits(logits.data(),gdn::kVocabSize);
      native_ids.push_back(int(hw));cpu_ids.push_back(cpu);
      if(int(hw)!=cpu){
        float gap=hw<gdn::kVocabSize?logits[cpu]-logits[hw]:999999.f;
        if(gap>max_cpu_gap)max_cpu_gap=gap;
        if(mismatches<20)std::printf("B197_DIVERGENCE seq=%d step=%d input=%d native=%u cpu=%d cpu_margin_over_native=%.9g cpu_top=%.9g\n",seq,t,token,hw,cpu,double(gap),double(logits[cpu]));
        ++m;++mismatches;
      }
      ++total;
    }
    if(seq==0){first_native=native_ids;first_cpu=cpu_ids;}
    if(seq==2){
      replayed_diff_native=0;replayed_diff_cpu=0;
      for(int i=0;i<64;i++){
        if(first_native[i]!=native_ids[i])replayed_diff_native++;
        if(first_cpu[i]!=cpu_ids[i])replayed_diff_cpu++;
      }
    }
    std::printf("B197_SEQ_REPORT seq=%d steps=%d mismatches=%d\n",seq,lens[seq],m);
  }
  std::printf("B197_DIVERSITY_FINAL compared=%d mismatches=%d repeated_native_difference=%d repeated_cpu_difference=%d max_cpu_margin_over_native=%.9g\n",total,mismatches,replayed_diff_native,replayed_diff_cpu,double(max_cpu_gap));
  return replayed_diff_native==0&&replayed_diff_cpu==0?0:7;
}
