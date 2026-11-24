// user_logic.v — KWS user logic for the ZSK reference design (replaces ref/user_logic.v; same ports + new_sample).
// sw is already inverted by ZSK_Design_Top (~sw): a switch ON reads as 1.
//   sw[1:0] mode : 0 pass-through (reference bypass)   1 golden-feature injection (feat_sets.mem, set = sw[4:2])
//                  2 live KWS (mic -> fe_top -> VAD -> SNN -> beep)   3 audio capture (ILA, new_sample qualifier)
//   sw[5]        : mode 1 fast frame rate (sim) ; mode 2 clock gating OFF (BUFGCE CE forced 1) ; sw[6] : mute beep ; sw[7] : hold
//   mode 2       : sw[3:2] = decision operating point (0 normal / 1 strict / 2 stricter / 3 very strict margins)
// ILA (instantiated IP ila_kws, probes = dbg_* registers at the end of this file); led[7:0] is internal only (no LED pins yet).
`timescale 1ns / 100ps
`include "rom_paths.vh"
module user_logic #(
    parameter N_CLASS = 12, ROM_DIR = `ROM_DIR_DEF,
    parameter T_ON = 600, T_OFF = 360, N_ON = 4, H = 30, P = 20, L_MIN = 25, L_MAX = 130, MARGIN = 0, ENERGY_SEL = 1,   // B-2 val-selected
    parameter FRAME_CYC = 480000, FRAME_CYC_FAST = 24000, GAP_CYC = 24000000
) (
    input  wire          clk,
    input  wire          rst,
    input  wire [07:0]   sw,
    input  wire [23:0]   audio_l_in,
    input  wire [23:0]   audio_r_in,
    input  wire          new_sample,
    input  wire          i2s_lr,        // ILA taps only (from adau1761_interface)
    input  wire          i2s_bclk,
    output reg  [23:0]   audio_l_out,
    output reg  [23:0]   audio_r_out
);
    localparam T = 98, N_IN = 40;
    wire [1:0] mode = sw[1:0];
    wire gate_off = (mode == 2'd2) & sw[5];       // live mode: switch 5 disables the clock gating for A/B power comparison
    // ---------------------------------------------------------------- pipeline (always running: front-end + VAD; SNN only in segments)
    wire fe_frame_done, vad_active, det_valid, det_any, snn_active, overrun, snn_frame_done, fe_gated;
    wire [31:0] cnt_fe_frames, cnt_fe_gated, cnt_fe_ce, cnt_snn_ce; wire fe_clk_ce, snn_clk_ce;
    wire [3:0] det_class; wire [3:0] snn_class; wire [15:0] energy;
    wire [31:0] cnt_frames, cnt_snn_frames, cnt_snn_busy, cnt_total; wire [N_CLASS*32-1:0] seg_sum;
    reg ext_en, ext_clip, ext_valid, ext_last; reg [7:0] ext_data;
`ifdef NO_CLK_GATE
    localparam CLK_GATE = 0;                      // build variant without BUFGCE gating (A/B reference)
`else
    localparam CLK_GATE = 1;
`endif
    kws_top #(.N_CLASS(N_CLASS), .ENERGY_SEL(ENERGY_SEL), .CLK_GATE(CLK_GATE), .T_ON(T_ON), .T_OFF(T_OFF), .N_ON(N_ON), .H(H), .P(P), .L_MIN(L_MIN), .L_MAX(L_MAX), .MARGIN(MARGIN),
              .ROM_DIR(ROM_DIR), .MEM_PREFIX({ROM_DIR, "/m3k3"})) u_kws (
        .clk(clk), .rst(rst), .new_sample(new_sample && mode == 2'd2), .pcm(audio_l_in), .margin_sel(sw[3:2]), .gate_off(gate_off),
        .ext_en(ext_en), .ext_clip_start(ext_clip), .ext_valid(ext_valid), .ext_data(ext_data), .ext_last(ext_last),
        .snn_frame_done(snn_frame_done), .snn_class(snn_class),
        .fe_frame_done(fe_frame_done), .energy(energy), .vad_active(vad_active), .det_valid(det_valid), .det_any(det_any), .det_class(det_class),
        .snn_active(snn_active), .cnt_frames(cnt_frames), .cnt_snn_frames(cnt_snn_frames), .cnt_snn_busy_cycles(cnt_snn_busy), .cnt_total_cycles(cnt_total),
        .seg_sum(seg_sum), .overrun(overrun), .fe_gated(fe_gated), .cnt_fe_frames(cnt_fe_frames), .cnt_fe_gated(cnt_fe_gated),
        .fe_clk_ce(fe_clk_ce), .snn_clk_ce(snn_clk_ce), .cnt_fe_ce_cycles(cnt_fe_ce), .cnt_snn_ce_cycles(cnt_snn_ce));
    // ---------------------------------------------------------------- mode 1: golden feature injection (clip mode)
    reg [7:0] feat_rom [0:5*T*N_IN-1];
    initial $readmemh({ROM_DIR, "/feat_sets.mem"}, feat_rom);
    reg [2:0] st; localparam S_START=0, S_WAITCLR=1, S_FRAME=2, S_SEND=3, S_WAITDONE=4, S_RESULT=5, S_GAP=6;
    reg [6:0] t; reg [5:0] m; reg [31:0] cyc; reg [2:0] set_sel;
    reg [3:0] result_class; reg result_valid; reg [15:0] run_count; reg result_pulse;
    wire [31:0] frame_cyc = sw[5] ? FRAME_CYC_FAST : FRAME_CYC;
    wire [14:0] rom_addr = set_sel * (T*N_IN) + t * N_IN + m;
    always @(posedge clk) begin
        ext_clip <= 0; ext_valid <= 0; ext_last <= 0; result_pulse <= 0;
        if (rst || mode != 2'd1) begin st <= S_START; t <= 0; m <= 0; cyc <= 0; ext_en <= (mode == 2'd1); if (rst) begin result_valid <= 0; run_count <= 0; end end
        else begin
            ext_en <= 1;
            case (st)
            S_START:   begin set_sel <= (sw[4:2] > 3'd4) ? 3'd4 : sw[4:2]; ext_clip <= 1; st <= S_WAITCLR; end
            S_WAITCLR: if (!u_kws.snn_busy && !ext_clip) begin t <= 0; cyc <= 0; st <= S_FRAME; end
            S_FRAME:   begin cyc <= cyc + 1; if (cyc >= frame_cyc) begin cyc <= 0; m <= 0; st <= S_SEND; end end
            S_SEND:    begin ext_valid <= 1; ext_data <= feat_rom[rom_addr]; ext_last <= (m == N_IN-1);
                             if (m == N_IN-1) st <= S_WAITDONE; else m <= m + 1'b1; end
            S_WAITDONE: if (snn_frame_done) begin if (t == T-1) st <= S_RESULT; else begin t <= t + 1'b1; st <= S_FRAME; end end
            S_RESULT:  begin if (!sw[7]) begin result_class <= snn_class; result_valid <= 1; run_count <= run_count + 1'b1; end result_pulse <= 1; cyc <= 0; st <= S_GAP; end
            S_GAP:     begin cyc <= cyc + 1; if (cyc >= GAP_CYC) st <= S_START; end
            default:   st <= S_START;
            endcase
        end
    end
    // ---------------------------------------------------------------- live result registers (mode 2)
    reg [3:0] live_class; reg live_valid; reg [15:0] live_count;
    always @(posedge clk) begin
        if (rst) begin live_class <= 0; live_valid <= 0; live_count <= 0; end
        else if (det_valid && !sw[7]) begin live_class <= det_class; live_valid <= 1; live_count <= live_count + 1'b1; end
    end
    // ---------------------------------------------------------------- audio capture taps (mode 3): 18-bit sample + qualifier for the ILA
    wire [17:0] cap_sample = audio_l_in[23:6];
    wire        cap_valid  = new_sample;
    // ---------------------------------------------------------------- beep: 300 ms tone, pitch by class 0..9 (silence/unknown: none)
    wire beep_trig = (mode == 2'd1) ? result_pulse : (mode == 2'd2) ? det_valid : 1'b0;
    wire [3:0] beep_class = (mode == 2'd1) ? snn_class : det_class;
    reg [23:0] beep_cnt; reg [15:0] tone_div, tone_cnt; reg tone;
    function [15:0] half_period; input [3:0] c; begin   // 48e6 / (2 * 440 * 2^(c/12))
        case (c) 0: half_period = 54545; 1: half_period = 51484; 2: half_period = 48595; 3: half_period = 45867; 4: half_period = 43293;
                 5: half_period = 40863; 6: half_period = 38570; 7: half_period = 36405; 8: half_period = 34362; 9: half_period = 32433; default: half_period = 0; endcase end endfunction
    always @(posedge clk) begin
        if (rst) begin beep_cnt <= 0; tone <= 0; tone_cnt <= 0; tone_div <= 0; end
        else begin
            if (beep_trig) begin beep_cnt <= 24'd14400000; tone_div <= half_period(beep_class); tone_cnt <= 0; end
            else if (beep_cnt != 0) beep_cnt <= beep_cnt - 1'b1;
            if (beep_cnt != 0 && tone_div != 0) begin if (tone_cnt == tone_div - 1) begin tone_cnt <= 0; tone <= ~tone; end else tone_cnt <= tone_cnt + 1'b1; end
            else tone <= 0;
        end
    end
    wire [23:0] beep_sample = (beep_cnt != 0 && tone_div != 0 && !sw[6]) ? (tone ? 24'sh200000 : -24'sh200000) : 24'd0;   // -12 dBFS square
    always @(posedge clk) begin
        if (rst) begin audio_l_out <= 0; audio_r_out <= 0; end
        else if (mode == 2'd0 || mode == 2'd3) begin audio_l_out <= audio_l_in; audio_r_out <= audio_r_in; end   // pass-through (reference behaviour)
        else begin audio_l_out <= beep_sample; audio_r_out <= beep_sample; end
    end
    // ---------------------------------------------------------------- led (internal only; add pins to the xdc when known)
    wire [7:0] led = {det_valid | result_pulse, vad_active, mode, (mode == 2'd1) ? result_class : live_class};
    // ---------------------------------------------------------------- ILA: one-cycle delayed copies of the probed signals feed an
    // instantiated ILA IP (ila_kws, created by build_bitstream.tcl). No XDC net-name lookup (that failed in opt_design).
    // Probe map (also in ila_kws.xdc): 0 mode[1:0] 1 cap_valid 2 cap_sample[17:0] 3 vad_active 4 det_valid 5 det_class[3:0]
    //   6 fe_frame_done 7 snn_active 8 cnt_snn_frames[23:0] 9 cnt_frames[23:0] 10 energy[15:0] 11 result_class[3:0]
    //   12 result_valid 13 run_count[15:0] 14 overrun 15 led[7:0] 16 live_class[3:0] 17 live_valid 18 live_count[15:0]
    //   19 i2s_lr 20 i2s_bclk (driven from ZSK_Design_Top via the i2s_lr/i2s_bclk ports) 21 fe_clk_ce 22 snn_clk_ce (gated clock enables).
    (* DONT_TOUCH="true" *) reg [1:0]  dbg_mode;
    (* DONT_TOUCH="true" *) reg        dbg_cap_valid;
    (* DONT_TOUCH="true" *) reg [17:0] dbg_cap_sample;
    (* DONT_TOUCH="true" *) reg        dbg_vad_active, dbg_det_valid, dbg_fe_frame_done, dbg_snn_active, dbg_result_valid, dbg_overrun, dbg_live_valid;
    (* DONT_TOUCH="true" *) reg [3:0]  dbg_det_class, dbg_result_class, dbg_live_class;
    (* DONT_TOUCH="true" *) reg [23:0] dbg_cnt_snn_frames, dbg_cnt_frames;
    (* DONT_TOUCH="true" *) reg [15:0] dbg_energy, dbg_run_count, dbg_live_count;
    (* DONT_TOUCH="true" *) reg [7:0]  dbg_led;
    (* DONT_TOUCH="true" *) reg        dbg_i2s_lr, dbg_i2s_bclk, dbg_fe_ce, dbg_snn_ce;
    always @(posedge clk) begin
        dbg_mode <= mode; dbg_cap_valid <= cap_valid; dbg_cap_sample <= cap_sample;
        dbg_vad_active <= vad_active; dbg_det_valid <= det_valid; dbg_det_class <= det_class; dbg_fe_frame_done <= fe_frame_done;
        dbg_snn_active <= snn_active; dbg_cnt_snn_frames <= cnt_snn_frames[23:0]; dbg_cnt_frames <= cnt_frames[23:0]; dbg_energy <= energy;
        dbg_result_class <= result_class; dbg_result_valid <= result_valid; dbg_run_count <= run_count; dbg_overrun <= overrun; dbg_led <= led;
        dbg_live_class <= live_class; dbg_live_valid <= live_valid; dbg_live_count <= live_count;
        dbg_i2s_lr <= i2s_lr; dbg_i2s_bclk <= i2s_bclk; dbg_fe_ce <= fe_clk_ce; dbg_snn_ce <= snn_clk_ce;
    end
`ifdef SYNTHESIS
    ila_kws u_ila (.clk(clk),
        .probe0(dbg_mode), .probe1(dbg_cap_valid), .probe2(dbg_cap_sample), .probe3(dbg_vad_active), .probe4(dbg_det_valid),
        .probe5(dbg_det_class), .probe6(dbg_fe_frame_done), .probe7(dbg_snn_active), .probe8(dbg_cnt_snn_frames), .probe9(dbg_cnt_frames),
        .probe10(dbg_energy), .probe11(dbg_result_class), .probe12(dbg_result_valid), .probe13(dbg_run_count), .probe14(dbg_overrun),
        .probe15(dbg_led), .probe16(dbg_live_class), .probe17(dbg_live_valid), .probe18(dbg_live_count), .probe19(dbg_i2s_lr), .probe20(dbg_i2s_bclk),
        .probe21(dbg_fe_ce), .probe22(dbg_snn_ce));
`endif
endmodule
