// B197 R7 independent driver for pinned GDN. Diagnostics only.
#include "decode.hpp"
#include "weight.hpp"
#include <ap_int.h>
#include <cstdint>
#include <cstdio>
#include <vector>
extern "C" void decode(int,int,const ap_uint<128>*,const float*,std::uint32_t*);
static std::vector<ap_uint<128>> mkbeats(const std::vector<std::uint8_t>& b){
  std::vector<ap_uint<128>> v(b.size()/16);
  for(std::size_t i=0;i<v.size();i++){
    ap_uint<128> x=0;
    for(int j=0;j<16;j++) x.range(j*8+7,j*8)=b[16*i+j];
    v[i]=x;
  }
  return v;
}
int main(int argc,char** argv){
  if(argc!=2) return 2;
  gdn::Weights w;
  gdn::LoadWeights(w,argv[1]);
  auto b=mkbeats(gdn::PackParameters(w));
  auto side=gdn::BuildFp32Side(w);
  gdn::RunState cpu;
  std::vector<float> logits(gdn::kVocabSize,0.0f);
  int tokens[]={1,101,207};
  for(int step=0;step<3;step++){
    std::uint32_t ans=0;
    std::printf("B197_BEGIN_TOKEN index=%d token=%d\n",step,tokens[step]);
    decode(tokens[step],step==0,b.data(),side.data(),&ans);
    gdn::CpuForward(cpu,w,tokens[step],logits.data());
    int expect=gdn::ArgmaxLogits(logits.data(),gdn::kVocabSize);
    std::printf("B197_END_TOKEN index=%d token=%d kernel=%u cpu=%d\n",step,tokens[step],ans,expect);
  }
}
