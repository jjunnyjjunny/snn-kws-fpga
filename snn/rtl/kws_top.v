// kws_top.v — live pipeline: fe_top -> VAD -> pre-roll buffer -> snn_core -> segment decision.
// Clock gating (CLK_GATE=1): snn_core runs on a BUFGCE clock enabled only while a frame is being fed/processed; the
// front-end processing blocks have their own gated clock inside fe_top. gate_off = 1 forces both clocks on (A/B test).
// The SNN only runs inside VAD segments (pre-roll replay + live frames); snn_active / duty counters are exposed.
// Per frame (fe frame_done): VAD step with E -> start: clip_start + replay n_pre frames ; active: replay 1 frame.
// After every replayed frame the output membranes are accumulated; at seg_end (after the frame) the decision is taken.
`timescale 1ns/1ps
module kws_top #(
    parameter N_CLASS = 12, ENERGY_SEL = 1,            // 1 = E1 (sum of features), 2 = E2 (log2q of the mel-acc sum)
    parameter T_ON = 600, T_OFF = 360, N_ON = 4, H = 30, P = 20, L_MIN = 25, L_MAX = 130, MARGIN = 0, N_KEYWORD = 10,   // B-2 val-selected
    parameter MARGIN1 = 2838, MARGIN2 = 6212, MARGIN3 = 9633, T0 = 6694,   // T0: proven lossless front-end gating threshold (0 = off)
    parameter MULT_SERIAL = 0, ZERO_SKIP = 1, ROM_DIR = "../rom", MEM_PREFIX = "../rom/m3k3",
    parameter CLK_GATE = 1                               // BUFGCE clock gating of snn_core and the front-end processing blocks
) (
    input  wire clk, input wire rst,
    input  wire new_sample, input wire signed [23:0] pcm,
    input  wire [1:0] margin_sel,                        // decision operating point (A-3)
    input  wire gate_off,                                // 1 = clock gating disabled (clocks forced on), for A/B comparison
    // external feature injection (board mode 1): drives snn_core directly in clip mode, sequencer idle
    input  wire ext_en, input wire ext_clip_start, input wire ext_valid, input wire [7:0] ext_data, input wire ext_last,
    output wire snn_frame_done, output wire [3:0] snn_class,
    output wire fe_frame_done, output wire [15:0] energy,
    output wire vad_active,
    output wire det_valid, output wire det_any, output wire [3:0] det_class,
    output wire snn_active,                              // SNN clock-enable request (segment or replay in progress)
    output reg  [31:0] cnt_frames, cnt_snn_frames, cnt_snn_busy_cycles, cnt_total_cycles,   // duty counters
    output wire [N_CLASS*32-1:0] seg_sum,
    output reg  overrun,                                 // sticky: a VAD event arrived while the previous frame was still being replayed
    output wire fe_gated, output wire [31:0] cnt_fe_frames, cnt_fe_gated,
    output wire fe_clk_ce, snn_clk_ce, output wire [31:0] cnt_fe_ce_cycles, cnt_snn_ce_cycles   // clock-enable duty
);
    // ---------------- front-end
    wire fv, fl, fe_busy; wire [7:0] fd; wire [13:0] e_sum; wire [9:0] e_log; wire [31:0] c0, c1, c2, c3;
    fe_top #(.MULT_SERIAL(MULT_SERIAL), .ROM_DIR(ROM_DIR), .T0(T0), .CLK_GATE(CLK_GATE)) u_fe (.clk(clk), .rst(rst), .new_sample(new_sample), .pcm(pcm),
        .gate_off(gate_off), .clk_ce(fe_clk_ce), .cnt_ce(cnt_fe_ce_cycles),
        .feat_valid(fv), .feat_data(fd), .feat_last(fl), .e_sum(e_sum), .e_log(e_log), .frame_done(fe_frame_done), .busy(fe_busy),
        .gated(fe_gated), .e0_out(), .cyc_frame(c0), .cyc_win(c1), .cyc_fft(c2), .cyc_mel(c3), .cnt_frames(cnt_fe_frames), .cnt_gated(cnt_fe_gated));
    assign energy = (ENERGY_SEL == 1) ? {2'b0, e_sum} : {6'b0, e_log};
    // ---------------- VAD
    wire seg_start, seg_end, seg_keep, seg_forced; wire [7:0] n_pre, n_fed;
    vad #(.EW(16), .T_ON(T_ON), .T_OFF(T_OFF), .N_ON(N_ON), .H(H), .P(P), .L_MIN(L_MIN), .L_MAX(L_MAX)) u_vad (
        .clk(clk), .rst(rst), .strobe(fe_frame_done), .E(energy), .seg_start(seg_start), .n_pre(n_pre), .active(vad_active),
        .seg_end(seg_end), .seg_keep(seg_keep), .seg_forced(seg_forced), .n_fed(n_fed));
    // ---------------- pre-roll buffer -> SNN
    reg replay; reg [7:0] replay_n; wire pb_valid, pb_last, pb_frame_done, pb_busy; wire [7:0] pb_data;
    wire snn_busy;
    preroll_buf #(.DEPTH(32), .N_IN(40), .DW(5)) u_pb (.clk(clk), .rst(rst), .in_valid(fv), .in_data(fd), .in_last(fl),
        .replay(replay), .n_frames(replay_n), .snn_busy(snn_busy), .out_valid(pb_valid), .out_data(pb_data), .out_last(pb_last),
        .out_frame_done(pb_frame_done), .busy(pb_busy));
    reg clip_start; wire snn_det_valid; wire [N_CLASS*32-1:0] snn_vo_sum; wire [N_CLASS*24-1:0] vo_cur;
    wire s_clip = ext_en ? ext_clip_start : clip_start, s_valid = ext_en ? ext_valid : pb_valid, s_last = ext_en ? ext_last : pb_last;
    wire [7:0] s_data = ext_en ? ext_data : pb_data;
    // ---- gated SNN clock: requested while the sequencer / pre-roll replay / core are busy (ext mode keeps it on)
    reg [2:0] st; localparam S_IDLE=0, S_CLR=1, S_REPLAY=2, S_END=3, S_WAKE=4;
    wire snn_req = (st != S_IDLE) | pb_busy | snn_busy | ext_en;
    wire snn_ready, clk_snn;
    clk_gate_ctrl u_scg (.clk(clk), .rst(rst), .req(snn_req), .force_on(gate_off | ext_en), .ce(snn_clk_ce), .ready(snn_ready), .cnt_ce(cnt_snn_ce_cycles));
    generate if (CLK_GATE) begin : G_CG
        clk_gate_buf u_scb (.clk_in(clk), .ce(snn_clk_ce), .clk_out(clk_snn));
    end else begin : G_NOCG
        assign clk_snn = clk;
    end endgenerate
    snn_core #(.N_CLASS(N_CLASS), .ZERO_SKIP(ZERO_SKIP), .N_KEYWORD(N_KEYWORD), .MEM_PREFIX(MEM_PREFIX)) u_snn (
        .clk(clk_snn), .rst(rst), .clip_start(s_clip), .feat_valid(s_valid), .feat_data(s_data), .feat_last(s_last),
        .frame_done(snn_frame_done), .class_idx(snn_class), .vo_sum(snn_vo_sum), .det_valid(snn_det_valid), .busy(snn_busy), .vo_cur(vo_cur));
    // (ext_en: the sequencer below never leaves S_IDLE because VAD events are ignored)
    // ---------------- segment decision
    reg seg_begin, finalize; wire dec_busy;
    decision_seg #(.N_CLASS(N_CLASS), .CW(4), .N_KEYWORD(N_KEYWORD), .MARGIN(MARGIN), .MARGIN1(MARGIN1), .MARGIN2(MARGIN2), .MARGIN3(MARGIN3)) u_dec (
        .clk(clk), .rst(rst), .margin_sel(margin_sel), .seg_begin(seg_begin),
        .frame_acc(pb_frame_done), .vo_cur(vo_cur), .finalize(finalize), .det_valid(det_valid), .det_any(det_any), .det_class(det_class),
        .seg_sum_flat(seg_sum), .busy(dec_busy));
`ifdef ACTIVITY
    integer act_pb_cyc = 0, act_dec_cyc = 0, act_fe_cyc = 0, act_total = 0;
    always @(posedge clk) begin act_total = act_total + 1; if (pb_busy) act_pb_cyc = act_pb_cyc + 1; if (dec_busy) act_dec_cyc = act_dec_cyc + 1; if (fe_busy) act_fe_cyc = act_fe_cyc + 1; end
`endif
    // ---------------- sequencer
    reg pend_end, pend_keep, wake_clip;
    reg strobe_d;                                        // fe_frame_done delayed 1 cycle = cycle in which the VAD outputs are valid
    assign snn_active = vad_active | (st != S_IDLE) | pb_busy | snn_busy;
    always @(posedge clk) begin
        replay <= 0; clip_start <= 0; seg_begin <= 0; finalize <= 0;
        if (rst) begin st <= S_IDLE; cnt_frames <= 0; cnt_snn_frames <= 0; cnt_snn_busy_cycles <= 0; cnt_total_cycles <= 0; pend_end <= 0; overrun <= 0; strobe_d <= 0; wake_clip <= 0; end
        else begin
            strobe_d <= fe_frame_done;
            if ((seg_start || seg_end || (vad_active && strobe_d)) && st != S_IDLE && !ext_en) overrun <= 1;
            cnt_total_cycles <= cnt_total_cycles + 1;
            if (snn_busy | pb_busy) cnt_snn_busy_cycles <= cnt_snn_busy_cycles + 1;
            if (fe_frame_done) cnt_frames <= cnt_frames + 1;
            if (pb_frame_done) cnt_snn_frames <= cnt_snn_frames + 1;
            case (st)
            S_IDLE: if (!ext_en) begin                                   // every event first wakes the SNN clock (S_WAKE)
                if (seg_start) begin wake_clip <= 1; replay_n <= n_pre; st <= S_WAKE; end
                else if (seg_end) begin wake_clip <= 0; replay_n <= 1; pend_end <= 1; pend_keep <= seg_keep; st <= S_WAKE; end   // frame that ends the segment is fed too
                else if (vad_active && strobe_d) begin wake_clip <= 0; replay_n <= 1; st <= S_WAKE; end   // continuing frame
            end
            S_WAKE: if (snn_ready) begin
                if (wake_clip) begin clip_start <= 1; seg_begin <= 1; st <= S_CLR; end
                else begin replay <= 1; st <= S_REPLAY; end
            end
            S_CLR: if (!snn_busy && !clip_start) begin replay <= 1; st <= S_REPLAY; end          // SNN cleared -> feed pre-roll + current
            S_REPLAY: if (!pb_busy && !replay && !dec_busy) begin
                if (pend_end) begin pend_end <= 0; if (pend_keep) finalize <= 1; st <= S_END; end else st <= S_IDLE;
            end
            S_END: if (!dec_busy && !finalize) st <= S_IDLE;
            default: st <= S_IDLE;
            endcase
        end
    end
endmodule
