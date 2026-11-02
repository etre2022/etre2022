#!/usr/bin/env python3
"""B197-R7b additional read-only FFN/quantization trace on original GDN C++.
Requires b197/r7_instrument.py; neither stage changes arithmetic.
Not Vitis or RTL simulation."""
from pathlib import Path
import runpy,hashlib,json
runpy.run_path("b197/r7_instrument.py",run_name="__main__")
p=Path("build/instrumented_decode.cpp")
t=p.read_text()
def before(needle,addition):
    global t
    n=t.count(needle)
    if n!=1:raise ValueError("nonunique before="+str(n)+" "+repr(needle[:120]))
    t=t.replace(needle,addition+"\n"+needle,1)
def after(needle,addition):
    global t
    n=t.count(needle)
    if n!=1:raise ValueError("nonunique after="+str(n)+" "+repr(needle[:120]))
    t=t.replace(needle,needle+"\n"+addition,1)
# attention RMSNorm scales are FP32 even when activations are identical
after('    b197_trace_q("K","q8_att",token,layer,act_q8,kDim);',
      '    b197_trace_f("K","att_scale",token,layer,act_scales,kDimGroups);')
after('    b197_trace_q("C","q8_att",token,layer,s.xq_q.data(),kDim);',
      '    b197_trace_f("C","att_scale",token,layer,s.xq_s.data(),kDimGroups);')
# pre-O projection activation and quantization
before('    quantize<kValueDim>(act_q8, act_scales, linear_out);',
       '    b197_trace_f("K","o_input",token,layer,linear_out,kValueDim);')
after('    quantize<kValueDim>(act_q8, act_scales, linear_out);',
      '    b197_trace_q("K","o_q8",token,layer,act_q8,kValueDim);b197_trace_f("K","o_scale",token,layer,act_scales,kValueGroups);')
before('    Quantize(s.xq_q.data(), s.xq_s.data(), s.linear_out.data(), kValueDim);',
       '    b197_trace_f("C","o_input",token,layer,s.linear_out.data(),kValueDim);')
after('    Quantize(s.xq_q.data(), s.xq_s.data(), s.linear_out.data(), kValueDim);',
      '    b197_trace_q("C","o_q8",token,layer,s.xq_q.data(),kValueDim);b197_trace_f("C","o_scale",token,layer,s.xq_s.data(),kValueGroups);')
# O GEMV output still live in q8_out / s.xb after residual addition
before('    b197_trace_f("K","att_residual",token,layer,x,kDim);',
       '    b197_trace_f("K","o_proj_out",token,layer,q8_out,kDim);')
before('    b197_trace_f("C","att_residual",token,layer,s.x.data(),kDim);',
       '    b197_trace_f("C","o_proj_out",token,layer,s.xb.data(),kDim);')
# FFN normalization and input quantization
after('    rmsnorm<kDim>(ffn_norm, x, rms_ffn);',
      '    b197_trace_f("K","ffn_norm",token,layer,ffn_norm,kDim);')
after('    RmsNorm(s.xb.data(), s.x.data(), l.ffn_norm.data(), kDim);',
      '    b197_trace_f("C","ffn_norm",token,layer,s.xb.data(),kDim);')
after('    quantize<kDim>(act_q8, act_scales, ffn_norm);',
      '    b197_trace_q("K","ffn_q8",token,layer,act_q8,kDim);b197_trace_f("K","ffn_scale",token,layer,act_scales,kDimGroups);')
before('    Matmul(s.hb.data(), s.xq_q.data(), s.xq_s.data(), l.w1);',
       '    b197_trace_q("C","ffn_q8",token,layer,s.xq_q.data(),kDim);b197_trace_f("C","ffn_scale",token,layer,s.xq_s.data(),kDimGroups);')
# Paired W1/W3 output and hidden int8 input to W2
before('    quantize<kHiddenDim>(act_q8, act_scales, q8_out);',
       '    b197_trace_f("K","w13_out",token,layer,q8_out,kHiddenDim);')
before('    Quantize(s.xq_q.data(), s.xq_s.data(), s.hb.data(), kHiddenDim);',
       '    b197_trace_f("C","w13_out",token,layer,s.hb.data(),kHiddenDim);')
after('    quantize<kHiddenDim>(act_q8, act_scales, q8_out);',
      '    b197_trace_q("K","w13_q8",token,layer,act_q8,kHiddenDim);b197_trace_f("K","w13_scale",token,layer,act_scales,kHiddenGroups);')
after('    Quantize(s.xq_q.data(), s.xq_s.data(), s.hb.data(), kHiddenDim);',
      '    b197_trace_q("C","w13_q8",token,layer,s.xq_q.data(),kHiddenDim);b197_trace_f("C","w13_scale",token,layer,s.xq_s.data(),kHiddenGroups);')
# W2 GEMV floats remain live after FFN residual addition
before('    b197_trace_f("K","layer_x",token,layer,x,kDim);',
       '    b197_trace_f("K","w2_out",token,layer,q8_out,kDim);')
before('    b197_trace_f("C","layer_x",token,layer,s.x.data(),kDim);',
       '    b197_trace_f("C","w2_out",token,layer,s.xb.data(),kDim);')
p.write_text(t)
orig=Path("upstream/src/decode.cpp").read_bytes()
meta=dict(stage="extended_FFN_and_dynamic_quantization",
          original_sha256=hashlib.sha256(orig).hexdigest(),
          instrumented_sha256=hashlib.sha256(t.encode()).hexdigest(),
          native_only=True,not_fpga_or_vitis=True)
Path("B197_logs/r7b_manifest.json").write_text(json.dumps(meta,indent=2))
print("B197_R7B_INSTRUMENTATION_PASS",json.dumps(meta))
