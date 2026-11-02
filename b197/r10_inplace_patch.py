#!/usr/bin/env python3
"""B197-R10. Derive an *ephemeral* in-place Mamba-3 HLS C++ variant, plus a
dual-output native testbench. Upstream is fetched at a pinned commit at CI time;
neither the original nor derivative HLS code is redistributable via artifact.
Source instrumentation/proof only; no Vivado/Vitis physical result.
"""
from pathlib import Path
import hashlib,json,re
from textwrap import dedent
p=Path("upstream/src/hls/mamba3_step.cpp")
t=p.read_text()
base=t
def change(a,b,n=1):
 global t
 c=t.count(a)
 if c!=n:raise AssertionError(("upstream source changed",a[:110],n,c))
 t=t.replace(a,b,n)
change("    data_t state_out[STATE_SIZE];\n","")
change("#pragma HLS BIND_STORAGE variable=state_out  type=ram_2p impl=bram\n","")
change("state_out[IDX_BHDN(", "state[IDX_BHDN(",2)
change("""        // Carry state forward: state_out → state for next step.
        for (int i = 0; i < STATE_SIZE; ++i) {
#pragma HLS PIPELINE II=1
            state[i] = state_out[i];
        }
""","")
change("    //  Phase 4: Write final state_out\n    WRITE_ARRAY(state_out, STATE_SIZE);",
       "    //  Phase 4: Write final in-place state\n    WRITE_ARRAY(state, STATE_SIZE);")
assert "state_out" not in t.replace("state_out → state","")
Path("work/inplace.cpp").write_text(t)
test_path=Path("upstream/src/hls/mamba3_step_test.cpp")
tb=test_path.read_text()
orig_tb=tb
def patch_tb(a,b,n=1):
 global tb
 c=tb.count(a)
 if c!=n:raise AssertionError(("upstream TB changed",a[:100],n,c))
 tb=tb.replace(a,b,n)
patch_tb('const float threshold = 5e-1f;', 'const float threshold = 1e-4f;')
patch_tb('using namespace std;', '''using namespace std;
void dut_original(hls::stream<bit32_t>&,hls::stream<bit32_t>&);
void dut_inplace(hls::stream<bit32_t>&,hls::stream<bit32_t>&);
static unsigned long long b197_bit_comparisons=0;
''')
patch_tb('        hls::stream<bit32_t> strm_out;',
         '        hls::stream<bit32_t> strm_out;\n        hls::stream<bit32_t> strm_in2, strm_out2;')
patch_tb('            strm_in.write((bit32_t)conv.u);             \\',
         '            strm_in.write((bit32_t)conv.u); strm_in2.write((bit32_t)conv.u); \\')
patch_tb('        dut(strm_in, strm_out);',
         '        dut_original(strm_in, strm_out);\n        dut_inplace(strm_in2, strm_out2);')
# In both output and final-state comparisons, read and compare identical bits before interpreting float.
patch_tb('            conv.u = (uint32_t)strm_out.read();',
         '''            conv.u = (uint32_t)strm_out.read();
            const uint32_t b197_variant_word = (uint32_t)strm_out2.read();
            if (conv.u != b197_variant_word) {
                fprintf(stderr,"B197_INPLACE_MISMATCH seq=%d index=%llu original=%08x variant=%08x\\n", seq,
                        b197_bit_comparisons, conv.u, b197_variant_word);
                return 88;
            }
            ++b197_bit_comparisons;''',2)
patch_tb('    return all_pass ? 0 : 1;',
         '''    if (b197_bit_comparisons != 245760) {
        fprintf(stderr,"B197_WRONG_OUTPUT_COMPARE_COUNT got=%llu expected=245760\\n",b197_bit_comparisons);
        return 90;
    }
    printf("B197_R10_INPLACE_STATE_NATIVE_BIT_EXACT_PASS words=%llu sequences=%d steps=%d\\n",
           b197_bit_comparisons, num_sequences, NSTEPS);
    return all_pass ? 0 : 1;''')
Path("work/dual_test.cpp").write_text(tb)
manifest={"baseline_commit":"a380d1b8ccc503fe42459056d3627201d101927d",
          "original_hls_sha256":hashlib.sha256(base.encode()).hexdigest(),
          "inplace_hls_sha256":hashlib.sha256(t.encode()).hexdigest(),
          "original_testbench_sha256":hashlib.sha256(orig_tb.encode()).hexdigest(),
          "dual_native_testbench_sha256":hashlib.sha256(tb.encode()).hexdigest(),
          "fixed_state_scalars":8192,"redundant_state_copies_per_64step_invocation":64*8192,
          "bit_compare_words_expected":245760,
          "math_expression_changes":0,"external_interface_changes":0,
          "state_output_storage_removed":True,
          "FPGA_synthesis_verified":False,"warning":"native software proof of same outputs, not HLS scheduling/power/energy or RTL equivalence"}
Path("logs/inplace_provenance.json").write_text(json.dumps(manifest,indent=2))
print("B197_R10_INPLACE_PATCH_PASS",json.dumps(manifest))
