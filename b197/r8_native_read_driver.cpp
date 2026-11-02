// B197 R8 native-only read-count driver; no hardware emulation claims.
#include "decode.hpp"
#include "weight.hpp"
#include <ap_int.h>
#include <cstdint>
#include <cstdio>
#include <vector>
extern "C" void decode(int,int,const ap_uint<128>*,const float*,std::uint32_t*);
static std::vector<ap_uint<128>> beats(const std::vector<std::uint8_t>& b){
 if(b.size()%16)return {};
 std::vector<ap_uint<128>> v(b.size()/16);
 for(std::size_t i=0;i<v.size();i++){
   ap_uint<128> x=0;
   for(int j=0;j<16;j++)x.range(j*8+7,j*8)=b[i*16+j];
   v[i]=x;
 }
 return v;
}
int main(int argc,char**argv){
 if(argc!=2)return 2;
 gdn::Weights w;gdn::LoadWeights(w,argv[1]);
 auto packed=beats(gdn::PackParameters(w));
 auto side=gdn::BuildFp32Side(w);
 if(packed.size()!=gdn::kPackedTotalBytes/16)return 3;
 const int drive[]={1,9038,207,1900};
 for(int t=0;t<4;t++){
   std::uint32_t out=0;
   decode(drive[t],t==0,packed.data(),side.data(),&out);
   std::printf("B197_R8_NATIVE_RESULT step=%d token=%d next=%u\n",t,drive[t],out);
 }
 return 0;
}
