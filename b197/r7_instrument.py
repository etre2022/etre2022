#!/usr/bin/env python3
"""B197 R7: Add read-only native C++ instrumentation to pinned GDN source.
No mathematical code is rewritten. Not a Vitis CSim or RTL test."""
from pathlib import Path
import hashlib
import json

upstream=Path("upstream/src/decode.cpp")
original=upstream.read_text()
traced=original
helper=r"""
#include <cstdio>
#include <cstring>
static void b197_trace_f(const char* side,const char* stage,int token,int layer,
                         const float* a,int n) {
  std::printf("B197_FLOAT|%s|%s|%d|%d|",side,stage,token,layer);
  for(int i=0;i<n;i++){uint32_t bits=0;std::memcpy(&bits,a+i,4);std::printf("%08x",bits);}
  std::printf("\n");
}
static void b197_trace_q(const char* side,const char* stage,int token,int layer,
                         const int8_t* a,int n) {
  std::printf("B197_Q8|%s|%s|%d|%d|",side,stage,token,layer);
  for(int i=0;i<n;i++)std::printf("%02x",(unsigned)(uint8_t)a[i]);
  std::printf("\n");
}
"""
needle="namespace gdn {"
assert traced.count(needle)==3, "expected three upstream namespaces"
# Emit the helper in the two mutually exclusive BUILD_DECODE_KERNEL and
# CPU-reference translation units. The third namespace is host glue.
traced=traced.replace(needle,helper+"\n"+needle,2)

def before(needle,addition):
    global traced
    n=traced.count(needle)
    if n!=1: raise ValueError("before count="+str(n)+" "+repr(needle[:110]))
    traced=traced.replace(needle,addition+"\n"+needle,1)

def after(needle,addition):
    global traced
    n=traced.count(needle)
    if n!=1: raise ValueError("after count="+str(n)+" "+repr(needle[:110]))
    traced=traced.replace(needle,needle+"\n"+addition,1)

after('  load_embedding(x, packed_params, token);',
      '  b197_trace_f("K","embed",token,-1,x,kDim);')
after('    rmsnorm<kDim>(attn_norm, x, rms_att);',
      '    b197_trace_f("K","rms_att",token,layer,attn_norm,kDim);')
after('    quantize<kDim>(act_q8, act_scales, attn_norm);',
      '    b197_trace_q("K","q8_att",token,layer,act_q8,kDim);')
before('      conv_step(q, q_conv_state[layer] + h * kHeadKDim,',
       '      if(h==0){b197_trace_f("K","proj_q",token,layer,q,32);b197_trace_f("K","proj_k",token,layer,k,32);b197_trace_f("K","proj_v",token,layer,v,32);}')
before('      float qacc[kAccStages];',
       '      if(h==0){b197_trace_f("K","conv_q",token,layer,q,32);b197_trace_f("K","conv_k",token,layer,k,32);b197_trace_f("K","conv_v",token,layer,v,32);}')
before('      recurrence_head(S[layer][h], q, k, v, beta, decay, valid[layer][h],',
       '      if(h==0){b197_trace_f("K","norm_q",token,layer,q,32);b197_trace_f("K","norm_k",token,layer,k,32);}')
before('      head_rmsnorm_gated(head_out, gate, o_norm);',
       '      if(h==0)b197_trace_f("K","recur_out",token,layer,head_out,32);')
before('    rmsnorm<kDim>(ffn_norm, x, rms_ffn);',
       '    b197_trace_f("K","att_residual",token,layer,x,kDim);')
before('  rmsnorm<kDim>(attn_norm, x, side + kSideRmsFinalOffset);',
       '  b197_trace_f("K","last_layer",token,7,x,kDim);')
after('  rmsnorm<kDim>(attn_norm, x, side + kSideRmsFinalOffset);',
      '  b197_trace_f("K","final_rms",token,-1,attn_norm,kDim);')
# Marker at end of kernel loop, immediately before final RMSNorm
# CPU source
before('  const float q_scale = 1.0f / std::sqrt(static_cast<float>(kHeadKDim));',
       '  b197_trace_f("C","embed",token,-1,s.x.data(),kDim);')
after('    RmsNorm(s.xb.data(), s.x.data(), l.attn_norm.data(), kDim);',
      '    b197_trace_f("C","rms_att",token,layer,s.xb.data(),kDim);')
before('    Matmul(s.q.data(), s.xq_q.data(), s.xq_s.data(), l.q_proj);',
       '    b197_trace_q("C","q8_att",token,layer,s.xq_q.data(),kDim);')
before('    ConvStep(s.q.data(),',
       '    b197_trace_f("C","proj_q",token,layer,s.q.data(),32);b197_trace_f("C","proj_k",token,layer,s.k.data(),32);b197_trace_f("C","proj_v",token,layer,s.v.data(),32);')
before('    for (int h = 0; h < kNumHeads; ++h) {\n      float* qh = s.q.data() + h * kHeadKDim;',
       '    b197_trace_f("C","conv_q",token,layer,s.q.data(),32);b197_trace_f("C","conv_k",token,layer,s.k.data(),32);b197_trace_f("C","conv_v",token,layer,s.v.data(),32);')
before('    float* S_layer =',
       '    b197_trace_f("C","norm_q",token,layer,s.q.data(),32);b197_trace_f("C","norm_k",token,layer,s.k.data(),32);')
before('    // FFN.',
       '    b197_trace_f("C","att_residual",token,layer,s.x.data(),kDim);')
before('  RmsNorm(s.x.data(), s.x.data(), w.rms_final.data(), kDim);',
       '  b197_trace_f("C","last_layer",token,7,s.x.data(),kDim);')
after('  RmsNorm(s.x.data(), s.x.data(), w.rms_final.data(), kDim);',
      '  b197_trace_f("C","final_rms",token,-1,s.x.data(),kDim);')
# Recurrent output for head zero: insert after inner j loop, before h closes.
cpu_recur='''        out[j] = value;
      }
    }

    for (int h = 0; h < kNumHeads; ++h) {'''
replacement='''        out[j] = value;
      }
      if(h==0)b197_trace_f("C","recur_out",token,layer,out,32);
    }

    for (int h = 0; h < kNumHeads; ++h) {'''
assert traced.count(cpu_recur)==1
traced=traced.replace(cpu_recur,replacement,1)
# Capture after the two FFN residual additions near respective final norm calls.
kernel_end='''      x[i] += q8_out[i];
    }
  }

  b197_trace_f("K","last_layer",token,7,x,kDim);'''
assert traced.count(kernel_end)==1
traced=traced.replace(kernel_end,'''      x[i] += q8_out[i];
    }
    b197_trace_f("K","layer_x",token,layer,x,kDim);
  }

  b197_trace_f("K","last_layer",token,7,x,kDim);''',1)
cpu_end='''    for (int i = 0; i < kDim; ++i) s.x[i] += s.xb[i];
  }

  b197_trace_f("C","last_layer",token,7,s.x.data(),kDim);'''
assert traced.count(cpu_end)==1
traced=traced.replace(cpu_end,'''    for (int i = 0; i < kDim; ++i) s.x[i] += s.xb[i];
    b197_trace_f("C","layer_x",token,layer,s.x.data(),kDim);
  }

  b197_trace_f("C","last_layer",token,7,s.x.data(),kDim);''',1)
out=Path("build/instrumented_decode.cpp")
out.write_text(traced)
info=dict(source_commit="45a7e14315d78e2c38b93e740cb0e4fb3f032c4d",
          upstream_sha256=hashlib.sha256(original.encode()).hexdigest(),
          instrumented_sha256=hashlib.sha256(traced.encode()).hexdigest(),
          native_only=True,no_hls_rtl_or_ppa=True)
Path("B197_logs/instrumentation.json").write_text(json.dumps(info,indent=2))
print("B197_R7_INSTRUMENTATION_PASS",json.dumps(info))
