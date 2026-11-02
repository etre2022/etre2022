#!/usr/bin/env python3
"""Native-only read-count instrumentation for pinned third-party GDN HLS C++.
Do not synthesize or redistribute the instrumented upstream source."""
from pathlib import Path
import json,hashlib
p=Path('upstream/src/decode.cpp')
original=p.read_text()
t=original
def anchor(old,new):
    global t
    assert t.count(old)==1,('unexpected GDN source: ',old[:80],t.count(old))
    t=t.replace(old,new,1)
anchor('namespace gdn {\nnamespace {',
       '#include <cstdio>\n'
       'static unsigned long long b197_q8words=0,b197_embedwords=0,b197_q8calls=0,b197_reccalls=0;\n'
       'namespace gdn {\nnamespace {')
anchor('#pragma HLS INLINE off\n#pragma HLS ARRAY_PARTITION variable = activation cyclic factor = 32 dim = 1',
       '#pragma HLS INLINE off\n  ++b197_q8calls;\n#pragma HLS ARRAY_PARTITION variable = activation cyclic factor = 32 dim = 1')
anchor('    const int beat = (word_offset + it) * kBeatsPerWord;',
       '    ++b197_q8words;\n    const int beat = (word_offset + it) * kBeatsPerWord;')
anchor('  const int beat = offset * kBeatsPerWord;',
       '  ++b197_embedwords;\n  const int beat = offset * kBeatsPerWord;')
anchor('#pragma HLS INLINE off\n  float prediction[kHeadVDim];',
       '#pragma HLS INLINE off\n  ++b197_reccalls;\n  float prediction[kHeadVDim];')
anchor('  forward(token, reset_state, side, packed_params, q_conv_state, k_conv_state,',
       '  b197_q8words=0; b197_embedwords=0; b197_q8calls=0; b197_reccalls=0;\n'
       '  forward(token, reset_state, side, packed_params, q_conv_state, k_conv_state,')
anchor('          v_conv_state, S, valid, next_token);\n}',
       '          v_conv_state, S, valid, next_token);\n'
       '  std::printf("B197_ACTUAL_NATIVE_WEIGHT_CALLS token=%d q8_words=%llu embed_words=%llu q8_calls=%llu recurrence_calls=%llu\\n",\n'
       '    token,b197_q8words,b197_embedwords,b197_q8calls,b197_reccalls);\n}')
out=Path('build/instrumented_decode.cpp')
out.write_text(t)
record={'commit':'45a7e14315d78e2c38b93e740cb0e4fb3f032c4d','original_sha256':hashlib.sha256(original.encode()).hexdigest(),'instrumented_sha256':hashlib.sha256(t.encode()).hexdigest(),'instrument':'read-only native counter; not HLS and not FPGA','original_algorithm_unchanged':True}
Path('B197_R8_logs/instrumentation.json').write_text(json.dumps(record,indent=2))
print('B197_R8_INSTRUMENTATION_PASS',json.dumps(record))
