#!/usr/bin/env python3
"""B197 R10 purely combinational address-decoding rewrite in MIT S4D bank.
Original and variant are isolated; no change to arithmetic/state equations.
The output files are artifacts of MIT-licensed source, not new architecture.
"""
from pathlib import Path
import hashlib,json
orig=Path('original/s4d/new/s4d_biquad_bank.v').read_text()
needle="""    integer jj;
    always @(posedge clk) begin
        if (res_ev)
            for (jj = 0; jj < NUNITS; jj = jj + 1)
                res[(jj*NCH + slot_cnt)*W_ACC +: W_ACC] <=
                    u_oacc[jj*W_ACC +: W_ACC];
    end"""
replacement="""    // B197 R10 counterfactual: static decoded per-channel register writes.
    // No datapath arithmetic, state dependence, or cycle schedule changed.
    genvar b197_ui, b197_ch;
    generate
      for (b197_ui=0; b197_ui<NUNITS; b197_ui=b197_ui+1) begin : b197_static_bank
        for (b197_ch=0; b197_ch<NCH; b197_ch=b197_ch+1) begin : b197_static_ch
          always @(posedge clk) begin
            if (res_ev && slot_cnt == b197_ch)
              res[(b197_ui*NCH+b197_ch)*W_ACC +: W_ACC] <=
                  u_oacc[b197_ui*W_ACC +: W_ACC];
          end
        end
      end
    endgenerate"""
assert orig.count(needle)==1,('source mismatch',orig.count(needle))
variant=orig.replace(needle,replacement)
assert variant.count(replacement)==1
assert variant.replace(replacement,needle)==orig
assert orig.count('module s4d_biquad_bank #(')==1
original_renamed=orig.replace('module s4d_biquad_bank #(', 'module s4d_biquad_bank_original #(',1)
work=Path('work');work.mkdir(exist_ok=True)
(work/'variant_bank.v').write_text(variant)
(work/'original_bank.v').write_text(original_renamed)
metadata={'source':'D1nDo/S4_Silicon_ZCU104',
          'source_commit':'bf6c7f0344349803a2ff197a23bbac9dab6669dd',
          'license':'MIT',
          'upstream_bank_sha256':hashlib.sha256(orig.encode()).hexdigest(),
          'variant_bank_sha256':hashlib.sha256(variant.encode()).hexdigest(),
          'logic_change':'one dynamic variable part-select write replaced with generate static channel comparator and constant part-select; same cycle',
          'arithmetic_changes':0,'control_latency_changes':0,
          'native_rtl_equivalence':'pending','formal_equivalence':False,
          'target_fpga_PPA':'pending'}
Path('B197_R10_logs/static_demux_patch_provenance.json').write_text(json.dumps(metadata,indent=2))
print('B197_R10_STATIC_DEMUX_SOURCE_PATCH_PASS',json.dumps(metadata))
