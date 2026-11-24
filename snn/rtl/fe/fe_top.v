// fe_top.v — integer front-end top: new_sample/pcm24 -> [in_scale] -> [ring] -> per frame: [window] -> [fft512]
// -> [mel_log] -> feature stream (feat_valid/feat_data/feat_last, same protocol as snn_core) + frame energies.
// Frame processing (window 512 + fft ~2.3k x SLOT + mel ~1k cycles) must finish before the next frame_start
// (160 samples = 480k cycles at 16 kHz; accelerated test benches must keep new_sample spacing >= cycles/160).
`timescale 1ns/1ps
module fe_top #(parameter DB = 18, MULT_SERIAL = 0, ROM_DIR = "../rom", T0 = 6694, CLK_GATE = 1) (   // T0: proven lossless gating threshold on sum|x18| (0 = off); CLK_GATE: BUFGCE on the processing blocks
    input  wire clk, input wire rst,
    input  wire new_sample, input wire signed [23:0] pcm,
    input  wire gate_off,              // 1 = keep the processing clock running (clock gating disabled)
    output wire clk_ce,                // processing-clock enable (duty)
    output wire [31:0] cnt_ce,         // cycles with the processing clock enabled
    output wire feat_valid, output wire [7:0] feat_data, output wire feat_last,
    output wire [13:0] e_sum, output wire [9:0] e_log,
    output reg  frame_done,            // one cycle after the last feature
    output wire busy,
    output reg  gated,                 // last frame was skipped (E0 < T0): features forced to 0
    output wire [27:0] e0_out,
    output reg  [31:0] cyc_frame, cyc_win, cyc_fft, cyc_mel,  // debug: cycles of the last frame
    output reg  [31:0] cnt_frames, cnt_gated                   // cumulative
);
    wire signed [DB-1:0] x18;
    fe_in_scale #(.DB(DB)) u_sc (.pcm(pcm), .x(x18));
    wire frame_start; wire [8:0] rd_base; wire [8:0] win_raddr; wire signed [DB-1:0] ring_q; wire [27:0] e0;
    fe_ring #(.DB(DB)) u_ring (.clk(clk), .rst(rst), .new_sample(new_sample), .x(x18), .frame_start(frame_start), .rd_base(rd_base), .e0(e0),
                              .rd_addr(rd_base + win_raddr), .rd_data(ring_q));
    assign e0_out = e0;
    wire gate_now = (T0 != 0) && (e0 < T0);
    // ---- gated processing clock (window / FFT / power / mel / buffer); in_scale, ring, E0, this FSM stay on clk
    reg [2:0] st; localparam S_IDLE=0, S_WIN=1, S_FFT=2, S_MEL=3, S_GATE=4, S_GATE_DONE=5, S_WAKE=6;
    wire proc_req = (st == S_WAKE) || (st == S_WIN) || (st == S_FFT) || (st == S_MEL);
    wire ce_ready, clk_p;
    clk_gate_ctrl u_cg (.clk(clk), .rst(rst), .req(proc_req), .force_on(gate_off), .ce(clk_ce), .ready(ce_ready), .cnt_ce(cnt_ce));
    generate if (CLK_GATE) begin : G_CG
        clk_gate_buf u_cb (.clk_in(clk), .ce(clk_ce), .clk_out(clk_p));
    end else begin : G_NOCG
        assign clk_p = clk;
    end endgenerate
    // buffer + muxed ports
    wire w_we; wire [8:0] w_addr; wire [2*DB-1:0] w_wd; wire win_done, win_busy;
    wire f_en_a, f_we_a, f_en_b, f_we_b; wire [8:0] f_addr_a, f_addr_b; wire [2*DB-1:0] f_wd_a, f_wd_b, rd_a, rd_b; wire fft_done, fft_busy;
    wire m_en_a; wire [8:0] m_addr_a; wire mel_done, mel_busy;
    reg [1:0] owner;  localparam O_WIN=1, O_FFT=2, O_MEL=3;
    wire en_a = (owner == O_WIN) ? w_we : (owner == O_FFT) ? f_en_a : (owner == O_MEL) ? m_en_a : 1'b0;
    wire we_a = (owner == O_WIN) ? w_we : (owner == O_FFT) ? f_we_a : 1'b0;
    wire [8:0] addr_a = (owner == O_WIN) ? w_addr : (owner == O_FFT) ? f_addr_a : m_addr_a;
    wire [2*DB-1:0] wd_a = (owner == O_WIN) ? w_wd : f_wd_a;
    fe_buf #(.DB(DB)) u_buf (.clk(clk_p), .en_a(en_a), .we_a(we_a), .addr_a(addr_a), .wd_a(wd_a), .rd_a(rd_a),
                              .en_b(f_en_b && owner == O_FFT), .we_b(f_we_b), .addr_b(f_addr_b), .wd_b(f_wd_b), .rd_b(rd_b));
    reg win_start, fft_start, mel_start;
    fe_window #(.DB(DB), .HANN_MEM({ROM_DIR, "/hann_w480.mem"})) u_win (.clk(clk_p), .rst(rst), .start(win_start), .done(win_done),
        .ring_addr(win_raddr), .ring_data(ring_q), .buf_we(w_we), .buf_addr(w_addr), .buf_wd(w_wd), .busy(win_busy));
    fe_fft512 #(.DB(DB), .MULT_SERIAL(MULT_SERIAL), .TW_RE_MEM({ROM_DIR, "/twiddle_re.mem"}), .TW_IM_MEM({ROM_DIR, "/twiddle_im.mem"})) u_fft (
        .clk(clk_p), .rst(rst), .start(fft_start), .done(fft_done), .busy(fft_busy),
        .en_a(f_en_a), .we_a(f_we_a), .addr_a(f_addr_a), .wd_a(f_wd_a), .rd_a(rd_a), .en_b(f_en_b), .we_b(f_we_b), .addr_b(f_addr_b), .wd_b(f_wd_b), .rd_b(rd_b));
    wire m_fv, m_fl; wire [7:0] m_fd; wire [13:0] m_es; wire [9:0] m_el;
    fe_mel_log #(.DB(DB), .TABLE_MEM({ROM_DIR, "/mel40_table.mem"}), .WEIGHT_MEM({ROM_DIR, "/mel40_weights.mem"}), .LUT_MEM({ROM_DIR, "/log2_lut.mem"})) u_mel (
        .clk(clk_p), .rst(rst), .start(mel_start), .done(mel_done), .busy(mel_busy), .en_a(m_en_a), .addr_a(m_addr_a), .rd_a(rd_a),
        .feat_valid(m_fv), .feat_data(m_fd), .feat_last(m_fl), .e_sum(m_es), .e_log(m_el), .dbg_valid(), .dbg_acc(), .dbg_log2q());
    reg g_fv, g_fl; reg [5:0] g_idx;                         // gated frame: 40 zero features streamed out
    assign feat_valid = m_fv | g_fv; assign feat_data = m_fv ? m_fd : 8'd0; assign feat_last = m_fl | g_fl;
    assign e_sum = gated ? 14'd0 : m_es; assign e_log = gated ? 10'd0 : m_el;
    assign busy = (st != S_IDLE);
    always @(posedge clk) begin
        win_start <= 0; fft_start <= 0; mel_start <= 0; frame_done <= 0; g_fv <= 0; g_fl <= 0;
        if (rst) begin st <= S_IDLE; owner <= 0; cyc_frame <= 0; gated <= 0; cnt_frames <= 0; cnt_gated <= 0; end
        else begin
            if (st != S_IDLE) cyc_frame <= cyc_frame + 1;
            case (st)
            S_IDLE: if (frame_start) begin
                cyc_frame <= 0; cyc_win <= 0; cyc_fft <= 0; cyc_mel <= 0; cnt_frames <= cnt_frames + 1;
                if (gate_now) begin st <= S_GATE; gated <= 1; g_idx <= 0; cnt_gated <= cnt_gated + 1; end     // lossless skip: all features are provably 0
                else begin st <= S_WAKE; owner <= O_WIN; gated <= 0; end                           // wake the processing clock first
            end
            S_WAKE: if (ce_ready) begin st <= S_WIN; win_start <= 1; end
            S_GATE: begin
                g_fv <= 1; g_fl <= (g_idx == 6'd39);
                if (g_idx == 6'd39) st <= S_GATE_DONE; else g_idx <= g_idx + 1'b1;
            end
            S_GATE_DONE: begin st <= S_IDLE; frame_done <= 1; end          // frame_done one cycle after the last feature (same as the mel path)
            S_WIN: begin cyc_win <= cyc_win + 1; if (win_done) begin st <= S_FFT; owner <= O_FFT; fft_start <= 1; end end
            S_FFT: begin cyc_fft <= cyc_fft + 1; if (fft_done) begin st <= S_MEL; owner <= O_MEL; mel_start <= 1; end end
            S_MEL: begin cyc_mel <= cyc_mel + 1; if (mel_done) begin st <= S_IDLE; owner <= 0; frame_done <= 1; end end
            endcase
        end
    end
endmodule
