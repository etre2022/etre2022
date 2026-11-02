// B197 R10 native Icarus RTL miter. Same clock, cfg, input, backpressure.
// Original S4D bank and static-demux variant have identical ports/cycles.
`timescale 1ns/1ps
module tb_s4d_bank_static_demux;
  reg clk=0,rst_n=0;
  reg cfg_we=0,cfgd_we=0;
  reg [0:0] cfg_unit=0;
  reg [6:0] cfg_addr=0;
  reg [79:0] cfg_data=0;
  reg [2:0] cfgd_ch=0;
  reg signed [15:0] cfgd_data=0;
  reg [4:0] cfgd_bsh=7;
  reg in_valid=0;
  wire ready0,ready1;
  reg signed [15:0] in_data=0;
  reg in_last_seq=0;
  wire valid0,valid1;
  reg out_ready=1;
  wire signed [15:0] data0,data1;
  wire [2:0] ch0,ch1;
  wire last0,last1,ovf0,ovf1;
  integer cycle=0,seen=0,last_seen=0,unit,addr,ch,i;
  s4d_biquad_bank_original #(
    .H(8),.LOG2_H(3),.NUNITS(2),.LOG2_NU(1),.NCH(4),.LOG2_NCH(2),
    .MODES(32),.LOG2_MOD(5),.LOG2_NB(7)
  ) original (
    .clk(clk),.rst_n(rst_n),.cfg_we(cfg_we),.cfg_unit(cfg_unit),.cfg_addr(cfg_addr),.cfg_data(cfg_data),
    .cfgd_we(cfgd_we),.cfgd_ch(cfgd_ch),.cfgd_data(cfgd_data),.cfgd_bsh(cfgd_bsh),
    .in_valid(in_valid),.in_ready(ready0),.in_data(in_data),.in_last_seq(in_last_seq),
    .out_valid(valid0),.out_ready(out_ready),.out_data(data0),.out_ch(ch0),.out_last(last0),.ovf(ovf0)
  );
  s4d_biquad_bank #(
    .H(8),.LOG2_H(3),.NUNITS(2),.LOG2_NU(1),.NCH(4),.LOG2_NCH(2),
    .MODES(32),.LOG2_MOD(5),.LOG2_NB(7)
  ) modified (
    .clk(clk),.rst_n(rst_n),.cfg_we(cfg_we),.cfg_unit(cfg_unit),.cfg_addr(cfg_addr),.cfg_data(cfg_data),
    .cfgd_we(cfgd_we),.cfgd_ch(cfgd_ch),.cfgd_data(cfgd_data),.cfgd_bsh(cfgd_bsh),
    .in_valid(in_valid),.in_ready(ready1),.in_data(in_data),.in_last_seq(in_last_seq),
    .out_valid(valid1),.out_ready(out_ready),.out_data(data1),.out_ch(ch1),.out_last(last1),.ovf(ovf1)
  );
  always #5 clk=~clk;
  always @(posedge clk) begin
    if(rst_n) begin
      cycle <= cycle+1;
      out_ready <= (cycle%7 != 0 && cycle%7 != 1);
      if(ready0 !== ready1) $fatal(1,"INPUT_READY_MISMATCH cycle=%0d",cycle);
      if(valid0 !== valid1) $fatal(1,"OUTPUT_VALID_MISMATCH cycle=%0d",cycle);
      if(ovf0 !== ovf1) $fatal(1,"OVERFLOW_MISMATCH cycle=%0d",cycle);
      if(valid0) begin
        if({data0,ch0,last0} !== {data1,ch1,last1})
          $fatal(1,"OUTPUT_BUS_MISMATCH cycle=%0d old=%0d new=%0d",cycle,data0,data1);
        if((^data0) === 1'bx) $fatal(1,"UNKNOWN_OUTPUT cycle=%0d",cycle);
      end
      if(valid0 && out_ready) begin
        seen <= seen+1;
        if(last0) last_seen <= last_seen+1;
      end
    end
  end
  initial begin
    repeat(5) @(negedge clk);
    rst_n=1;
    for(unit=0;unit<2;unit=unit+1) begin
      for(addr=0;addr<128;addr=addr+1) begin
        @(negedge clk);
        cfg_we=1;cfg_unit=unit;cfg_addr=addr;
        cfg_data=(addr%32==0)?(80'd128 << 48):80'd0;
      end
    end
    @(negedge clk);cfg_we=0;
    for(ch=0;ch<8;ch=ch+1) begin
      @(negedge clk);
      cfgd_we=1;cfgd_ch=ch;cfgd_data=0;cfgd_bsh=7;
    end
    @(negedge clk);cfgd_we=0;
    repeat(8) @(negedge clk);
    for(i=0;i<16;i=i+1) begin
      @(negedge clk);
      while(!ready0) @(negedge clk);
      in_valid=1;
      in_data=(i+1)*128;
      in_last_seq=(i==15);
      @(negedge clk);in_valid=0;in_last_seq=0;
    end
    repeat(1200) @(negedge clk);
    if(seen!=16) $fatal(1,"OUTPUT_COUNT_MISMATCH seen=%0d expected16",seen);
    if(last_seen!=2) $fatal(1,"LAST_COUNT_MISMATCH seen=%0d expected2",last_seen);
    $display("B197_R10_STATIC_DEMUX_EQUIV_PASS words=%0d last_flags=%0d",seen,last_seen);
    $finish;
  end
endmodule
