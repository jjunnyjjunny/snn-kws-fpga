// snn_core.v — M3k3 spiking-TCN keyword-spotting core (bit-exact with snn/golden/snn_int_ref.py).
// Frame interface: 40 features streamed in (feat_valid / feat_data / feat_last), then the core
// processes the frame (busy=1) and pulses frame_done with class_idx. clip_start clears all state.
// Parameters are frozen for M3k3: K=3, dilations 1/2/4/8, 64 ch x 4 layers, 8-bit weights,
// LEAK_K=3, IN_SHIFT=6, 16-bit membranes, 24-bit output integrator. Only the .mem prefix changes
// when new weights are trained.
`timescale 1ns/1ps
module snn_core #(
    parameter N_IN       = 40,
    parameter N_HID      = 64,
    parameter N_CLASS    = 12,
    parameter K          = 3,
    parameter D2         = 2,            // dilation of layer 2 (reads layer-1 spikes)
    parameter D3         = 4,
    parameter D4         = 8,
    parameter IN_SHIFT   = 6,
    parameter LEAK_K     = 3,
    parameter ZERO_SKIP  = 1,
    parameter WIN_SUM    = 98,
    parameter MARGIN     = 0,
    parameter N_CONSEC   = 1,
    parameter REFRACT    = 0,
    parameter N_KEYWORD  = 10,
    parameter MEM_PREFIX = "../rom/m3k3"
) (
    input  wire        clk,
    input  wire        rst,           // synchronous, active high
    input  wire        clip_start,    // clear all state (clip evaluation); not used in streaming
    input  wire        feat_valid,
    input  wire [7:0]  feat_data,
    input  wire        feat_last,     // asserted with the 40th feature of a frame
    output wire        frame_done,    // one-cycle pulse, class_idx / vo_sum valid
    output wire [3:0]  class_idx,
    output wire [N_CLASS*32-1:0] vo_sum,
    output wire        det_valid,     // streaming detection flag (see decision.v)
    output wire        busy,
    output wire [N_CLASS*24-1:0] vo_cur  // current output membranes (segment-mode decision / debug)
);
    localparam ACC_W = 24;
    // ------------------------------------------------------------------ control
    wire clr_all, vclr_en, hist_shift, feat_wr, issue, acc_clr, upd_en, dec_start;
    wire [7:0] vclr_idx; wire [5:0] feat_wr_idx, ch, post, upd_idx; wire [2:0] layer; wire [1:0] tap;
    wire [13:0] row_base;
    wire act_zero, acc_busy, dec_done, dec_busy;
    wire [31:0] cyc_frame, cyc_acc_l1, cyc_acc_hid, cyc_acc_out, cyc_upd, cyc_dec;
    wire [15:0] rows_skip_l1, rows_run_l1, rows_run_hid, rows_run_out;
    snn_ctrl #(.N_IN(N_IN), .N_HID(N_HID), .N_CLASS(N_CLASS), .K(K), .ZERO_SKIP(ZERO_SKIP)) u_ctrl (
        .clk(clk), .rst(rst), .clip_start(clip_start), .feat_valid(feat_valid), .feat_last(feat_last),
        .act_zero(act_zero), .acc_busy(acc_busy), .dec_done(dec_done),
        .clr_all(clr_all), .vclr_en(vclr_en), .vclr_idx(vclr_idx), .hist_shift(hist_shift),
        .feat_wr(feat_wr), .feat_wr_idx(feat_wr_idx), .layer(layer), .tap(tap), .ch(ch), .row_base(row_base),
        .post(post), .issue(issue), .acc_clr(acc_clr), .upd_en(upd_en), .upd_idx(upd_idx),
        .dec_start(dec_start), .frame_done(frame_done), .busy(busy),
        .cyc_frame(cyc_frame), .cyc_acc_l1(cyc_acc_l1), .cyc_acc_hid(cyc_acc_hid), .cyc_acc_out(cyc_acc_out),
        .cyc_upd(cyc_upd), .cyc_dec(cyc_dec), .rows_skip_l1(rows_skip_l1), .rows_run_l1(rows_run_l1),
        .rows_run_hid(rows_run_hid), .rows_run_out(rows_run_out));
    // ------------------------------------------------------------------ histories
    wire [7:0] fh_rd;
    feat_hist #(.N_IN(N_IN), .DEPTH(K), .IW(6), .FW(2)) u_fh (
        .clk(clk), .clr(clr_all), .shift(hist_shift), .wr_en(feat_wr), .wr_idx(feat_wr_idx), .wr_data(feat_data),
        .rd_frame(tap), .rd_idx(ch), .rd_data(fh_rd));
    wire sh1_rd, sh2_rd, sh3_rd, sh4_rd;
    wire [N_HID-1:0] sh1_cur, sh2_cur, sh3_cur, sh4_cur;
    wire upd_spk;
    wire wr1 = upd_en && (layer == 3'd0), wr2 = upd_en && (layer == 3'd1),
         wr3 = upd_en && (layer == 3'd2), wr4 = upd_en && (layer == 3'd3);
    wire [4:0] rf2 = {3'd0, tap} * D2;   // frame select for the reading layer's taps
    wire [4:0] rf3 = {3'd0, tap} * D3;
    wire [4:0] rf4 = {3'd0, tap} * D4;
    spike_hist #(.N(N_HID), .DEPTH(D2*(K-1)+1), .IW(6), .FW(5)) u_sh1 (.clk(clk), .clr(clr_all), .shift(hist_shift),
        .wr_en(wr1), .wr_idx(upd_idx), .wr_bit(upd_spk), .rd_frame(rf2), .rd_idx(ch), .rd_bit(sh1_rd), .cur(sh1_cur));
    spike_hist #(.N(N_HID), .DEPTH(D3*(K-1)+1), .IW(6), .FW(5)) u_sh2 (.clk(clk), .clr(clr_all), .shift(hist_shift),
        .wr_en(wr2), .wr_idx(upd_idx), .wr_bit(upd_spk), .rd_frame(rf3), .rd_idx(ch), .rd_bit(sh2_rd), .cur(sh2_cur));
    spike_hist #(.N(N_HID), .DEPTH(D4*(K-1)+1), .IW(6), .FW(5)) u_sh3 (.clk(clk), .clr(clr_all), .shift(hist_shift),
        .wr_en(wr3), .wr_idx(upd_idx), .wr_bit(upd_spk), .rd_frame(rf4), .rd_idx(ch), .rd_bit(sh3_rd), .cur(sh3_cur));
    spike_hist #(.N(N_HID), .DEPTH(1), .IW(6), .FW(5)) u_sh4 (.clk(clk), .clr(clr_all), .shift(hist_shift),
        .wr_en(wr4), .wr_idx(upd_idx), .wr_bit(upd_spk), .rd_frame(5'd0), .rd_idx(ch), .rd_bit(sh4_rd), .cur(sh4_cur));
    // activation of the current row (tap, ch) for the layer being accumulated
    reg [7:0] act;
    always @* begin
        case (layer)
            3'd0: act = fh_rd;
            3'd1: act = {7'd0, sh1_rd};
            3'd2: act = {7'd0, sh2_rd};
            3'd3: act = {7'd0, sh3_rd};
            default: act = {7'd0, sh4_rd};
        endcase
    end
    assign act_zero = (act == 8'd0);
    // ------------------------------------------------------------------ weight ROMs (one per layer, enable-gated)
    wire [13:0] rom_addr = row_base + {8'd0, post};
    wire [7:0] w1, w2, w3, w4, wo;
    wrom #(.WIDTH(8), .DEPTH(K*N_IN*N_HID),  .AW(14), .MEMFILE({MEM_PREFIX, "_tcn1_w.mem"})) u_w1 (.clk(clk), .en(issue && layer==3'd0), .addr(rom_addr), .dout(w1));
    wrom #(.WIDTH(8), .DEPTH(K*N_HID*N_HID), .AW(14), .MEMFILE({MEM_PREFIX, "_tcn2_w.mem"})) u_w2 (.clk(clk), .en(issue && layer==3'd1), .addr(rom_addr), .dout(w2));
    wrom #(.WIDTH(8), .DEPTH(K*N_HID*N_HID), .AW(14), .MEMFILE({MEM_PREFIX, "_tcn3_w.mem"})) u_w3 (.clk(clk), .en(issue && layer==3'd2), .addr(rom_addr), .dout(w3));
    wrom #(.WIDTH(8), .DEPTH(K*N_HID*N_HID), .AW(14), .MEMFILE({MEM_PREFIX, "_tcn4_w.mem"})) u_w4 (.clk(clk), .en(issue && layer==3'd3), .addr(rom_addr), .dout(w4));
    wrom #(.WIDTH(8), .DEPTH(N_HID*N_CLASS), .AW(14), .MEMFILE({MEM_PREFIX, "_out_w.mem"}))  u_wo (.clk(clk), .en(issue && layer==3'd4), .addr(rom_addr), .dout(wo));
    reg [7:0] w_sel;
    always @* begin
        case (layer)
            3'd0: w_sel = w1; 3'd1: w_sel = w2; 3'd2: w_sel = w3; 3'd3: w_sel = w4; default: w_sel = wo;
        endcase
    end
    // ------------------------------------------------------------------ accumulation engine
    wire signed [ACC_W-1:0] acc_rd;
    row_acc #(.N_POST(N_HID), .PW(6), .ACC_W(ACC_W)) u_acc (
        .clk(clk), .clr(acc_clr), .issue(issue), .is_input(layer == 3'd0), .act(act), .post(post),
        .rom_dout($signed(w_sel)), .rd_post(upd_idx), .rd_acc(acc_rd), .busy(acc_busy));
    // ------------------------------------------------------------------ per-neuron parameters (theta, bias)
    wire [15:0] th1, th2, th3, th4, b1, b2, b3, b4, bo;
    prom #(.WIDTH(16), .DEPTH(N_HID), .AW(6), .MEMFILE({MEM_PREFIX, "_tcn1_theta.mem"})) u_t1 (.addr(upd_idx), .dout(th1));
    prom #(.WIDTH(16), .DEPTH(N_HID), .AW(6), .MEMFILE({MEM_PREFIX, "_tcn2_theta.mem"})) u_t2 (.addr(upd_idx), .dout(th2));
    prom #(.WIDTH(16), .DEPTH(N_HID), .AW(6), .MEMFILE({MEM_PREFIX, "_tcn3_theta.mem"})) u_t3 (.addr(upd_idx), .dout(th3));
    prom #(.WIDTH(16), .DEPTH(N_HID), .AW(6), .MEMFILE({MEM_PREFIX, "_tcn4_theta.mem"})) u_t4 (.addr(upd_idx), .dout(th4));
    prom #(.WIDTH(16), .DEPTH(N_HID), .AW(6), .MEMFILE({MEM_PREFIX, "_tcn1_b.mem"}))     u_b1 (.addr(upd_idx), .dout(b1));
    prom #(.WIDTH(16), .DEPTH(N_HID), .AW(6), .MEMFILE({MEM_PREFIX, "_tcn2_b.mem"}))     u_b2 (.addr(upd_idx), .dout(b2));
    prom #(.WIDTH(16), .DEPTH(N_HID), .AW(6), .MEMFILE({MEM_PREFIX, "_tcn3_b.mem"}))     u_b3 (.addr(upd_idx), .dout(b3));
    prom #(.WIDTH(16), .DEPTH(N_HID), .AW(6), .MEMFILE({MEM_PREFIX, "_tcn4_b.mem"}))     u_b4 (.addr(upd_idx), .dout(b4));
    prom #(.WIDTH(16), .DEPTH(N_CLASS), .AW(6), .MEMFILE({MEM_PREFIX, "_out_b.mem"}))    u_bo (.addr(upd_idx), .dout(bo));
    reg [15:0] th_sel, b_sel;
    always @* begin
        case (layer)
            3'd0: begin th_sel = th1; b_sel = b1; end
            3'd1: begin th_sel = th2; b_sel = b2; end
            3'd2: begin th_sel = th3; b_sel = b3; end
            3'd3: begin th_sel = th4; b_sel = b4; end
            default: begin th_sel = 16'd0; b_sel = bo; end
        endcase
    end
`ifdef ACTIVITY
    integer act_cyc = 0, act_rom_rd = 0, act_mult = 0, act_vmem_wr = 0;
    always @(posedge clk) begin if (busy) act_cyc = act_cyc + 1; if (issue) act_rom_rd = act_rom_rd + 1; if (issue && layer == 3'd0) act_mult = act_mult + 1;
                                if (upd_en && layer != 3'd4) act_vmem_wr = act_vmem_wr + 1; end
`endif
    // ------------------------------------------------------------------ membrane state (4 x 64 x 16, LUTRAM)
    reg signed [15:0] v_mem [0:4*N_HID-1];
    wire [7:0] v_addr = {layer[1:0], upd_idx};
    wire signed [15:0] v_cur = v_mem[v_addr];
    wire signed [15:0] v_nxt;
    lif_update #(.ACC_W(ACC_W), .IN_SHIFT(IN_SHIFT), .LEAK_K(LEAK_K)) u_lif (
        .is_input(layer == 3'd0), .acc(acc_rd), .b($signed(b_sel)), .theta(th_sel), .v(v_cur),
        .v_next(v_nxt), .spike(upd_spk), .cur());
    always @(posedge clk) begin
        if (vclr_en) v_mem[vclr_idx] <= 16'sd0;
        else if (upd_en && layer != 3'd4) v_mem[v_addr] <= v_nxt;
    end
    // ------------------------------------------------------------------ output integrators (12 x 24)
    reg signed [23:0] vo [0:N_CLASS-1];
    wire signed [23:0] vo_nxt;
    wire [3:0] oc = upd_idx[3:0];
    out_integrator #(.ACC_W(ACC_W), .LEAK_K(LEAK_K)) u_out (.acc(acc_rd), .b($signed(b_sel)), .vo(vo[oc]), .vo_next(vo_nxt));
    integer i;
    always @(posedge clk) begin
        if (clr_all) begin for (i = 0; i < N_CLASS; i = i + 1) vo[i] <= 24'sd0; end
        else if (upd_en && layer == 3'd4) vo[oc] <= vo_nxt;
    end
    wire [N_CLASS*24-1:0] vo_flat;
    genvar g;
    generate for (g = 0; g < N_CLASS; g = g + 1) begin : GV
        assign vo_flat[g*24 +: 24] = vo[g];
    end endgenerate
    assign vo_cur = vo_flat;
    // ------------------------------------------------------------------ decision
    decision #(.N_CLASS(N_CLASS), .CW(4), .WIN_SUM(WIN_SUM), .WIN_DEPTH(128), .MARGIN(MARGIN), .N_CONSEC(N_CONSEC),
               .REFRACT(REFRACT), .N_KEYWORD(N_KEYWORD)) u_dec (
        .clk(clk), .clr(clr_all), .start(dec_start), .vo_flat(vo_flat), .done(dec_done), .class_idx(class_idx),
        .sum_flat(vo_sum), .det_valid(det_valid), .busy(dec_busy));
endmodule
