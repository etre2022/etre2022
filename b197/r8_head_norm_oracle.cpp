// B197 R8 native-only *oracle* norm-screen probe; NOT an exact-FP32
// certificate, NOT a hardware implementation, NOT evidence of PPA.
// Third-party GDN upstream is fetched at pinned SHA at CI runtime.
#include "decode.hpp"
#include "weight.hpp"
#include <array>
#include <vector>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <numeric>
#include <limits>
#include <stdexcept>
using namespace gdn;
struct HeadNorms {
 std::vector<double> full;
 std::vector<std::array<double,kDimGroups>> groups;
 explicit HeadNorms(const QuantizedTensor& w){
   if(w.rows!=kVocabSize||w.cols!=kDim||w.groups!=kDimGroups)throw std::runtime_error("bad dims");
   full.resize(kVocabSize);
   groups.resize(kVocabSize);
   for(int row=0;row<kVocabSize;row++){
     double row_sum2=0;
     for(int g=0;g<kDimGroups;g++){
       double ss=0;
       double scale=(double)w.s[row*kDimGroups+g];
       for(int j=0;j<kQuantGroupSize;j++){
         int v=w.q[row*kDim+g*kQuantGroupSize+j];
         ss+=(double)v*v;
       }
       double gn=std::sqrt(ss)*std::abs(scale);
       groups[row][g]=gn;
       row_sum2+=ss*scale*scale;
     }
     full[row]=std::sqrt(row_sum2);
   }
 }
};
void evaluate(int seq,int step,int token,RunState& state,Weights& w,HeadNorms& norm,
              std::vector<float>& logits,int& done,long long& sumglobal,long long& sumgroup){
 CpuForward(state,w,token,logits.data());
 int arg=ArgmaxLogits(logits.data(),kVocabSize);
 const double best=logits[arg];
 double asum=0;
 std::array<double,kDimGroups> ag{};
 for(int g=0;g<kDimGroups;g++){
   double sums=0,scale=(double)state.xq_s[g];
   for(int j=0;j<kQuantGroupSize;j++){
     double val=state.xq_q[g*kQuantGroupSize+j];
     sums+=val*val;
   }
   ag[g]=std::sqrt(sums)*std::abs(scale);
   asum+=sums*scale*scale;
 }
 double anorm=std::sqrt(asum);
 int prune_global=0,prune_group=0,oracle_misses=0;
 // generous 0.01 absolute margin because this is a preliminary FP32 probe.
 const double margin=0.01;
 for(int row=0;row<kVocabSize;row++){
   double upper_global=norm.full[row]*anorm;
   double upper_group=0;
   for(int g=0;g<kDimGroups;g++)upper_group+=norm.groups[row][g]*ag[g];
   if(upper_global < best-margin) prune_global++;
   if(upper_group < best-margin) prune_group++;
   if((upper_global<best-margin || upper_group<best-margin) && logits[row]>=best-margin)
      oracle_misses++;
 }
 if(oracle_misses)throw std::runtime_error("unsound empirical bound on this trace");
 if(prune_global>prune_group)throw std::runtime_error("group bound unexpectedly looser");
 std::printf("B197_R8_NORM_ORACLE seq=%d step=%d token=%d top1=%d best=%.9g global_skippable=%d group_skippable=%d rows=%d margin=0.01\n",
             seq,step,token,arg,best,prune_global,prune_group,kVocabSize);
 done++;sumglobal+=prune_global;sumgroup+=prune_group;
}
int main(int argc,char**argv){
 if(argc!=2)return 2;
 Weights w;LoadWeights(w,argv[1]);
 HeadNorms norms(w.tok_emb);
 std::vector<float> logits(kVocabSize);
 int cnt=0;long long global=0,group=0;
 int first[]={1,9038,2501,263,931,29892,297,263,2319,4726,4257,4726,29892,10600,1023,1900};
 RunState s;
 for(int i=0;i<16;i++) evaluate(0,i,first[i],s,w,norms,logits,cnt,global,group);
 RunState d;
 int seq[]={1,101,207};
 for(int i=0;i<3;i++)evaluate(1,i,seq[i],d,w,norms,logits,cnt,global,group);
 std::printf("B197_R8_NORM_ORACLE_SUMMARY queries=%d rows=%d rowglobal_metadata_bytes=%zu groupnorm_metadata_bytes=%zu average_global_screen_fraction=%.9g average_group_screen_fraction=%.9g\n",
   cnt,kVocabSize,norms.full.size()*sizeof(float),
   norms.groups.size()*kDimGroups*sizeof(float),
   double(global)/(double(cnt)*kVocabSize),double(group)/(double(cnt)*kVocabSize));
 return 0;
}
