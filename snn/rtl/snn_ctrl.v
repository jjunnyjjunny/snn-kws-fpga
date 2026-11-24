// snn_ctrl.v — frame-level FSM of the SNN core (see SPEC_snn_core.md section 4).
//   RESET_CLR -> IDLE -(feat_valid)-> LOAD -(feat_last)-> [per layer: INIT, ROWCHK/ROWRUN, DRAIN, UPD]
//   -> DEC (decision module) -> DONE -> IDLE. busy=0 only in IDLE (clock enable may be dropped).
// Zero-skip: a row whose activation is 0 costs one ROWCHK cycle and is never issued
// (input layer: feature == 0, controlled by ZERO_SKIP; spike layers: always event-driven).
`timescale 1ns/1ps
module snn_ctrl #(
    parameter N_IN      = 40,
    parameter N_HID     = 64,
    parameter N_CLASS   = 12,
    parameter K         = 3,
    parameter ZERO_SKIP = 1
) (
    input  wire        clk,
    input  wire        rst,
    input  wire        clip_start,
    input  wire        feat_valid,
    input  wire        feat_last,
    input  wire        act_zero,      // activation of the current row is 0
    input  wire        acc_busy,      // row_acc pipeline non-empty
    input  wire        dec_done,
    // state / control outputs
    output reg         clr_all,       // one-cycle clear of histories, acc, vo, decision
    output reg         vclr_en,       // clearing v_mem
    output reg  [7:0]  vclr_idx,
    output wire        hist_shift,    // combinational: first feature of a frame (same cycle)
    output wire        feat_wr,       // combinational: feature write enable (same cycle as feat_valid)
    output wire [5:0]  feat_wr_idx,
    output reg  [2:0]  layer,         // 0..3 hidden, 4 = output layer
    output reg  [1:0]  tap,
    output reg  [5:0]  ch,
    output reg  [13:0] row_base,
    output reg  [5:0]  post,
    output reg         issue,
    output reg         acc_clr,
    output reg         upd_en,
    output reg  [5:0]  upd_idx,
    output reg         dec_start,
    output reg         frame_done,
    output wire        busy,
    // cycle counters (debug, reset at frame start)
    output reg  [31:0] cyc_frame, cyc_acc_l1, cyc_acc_hid, cyc_acc_out, cyc_upd, cyc_dec,
    output reg  [15:0] rows_skip_l1, rows_run_l1, rows_run_hid, rows_run_out
);
    localparam S_CLR=0, S_IDLE=1, S_LOAD=2, S_INIT=3, S_ROWCHK=4, S_ROWRUN=5, S_DRAIN=6, S_UPD=7,
               S_DEC=8, S_DECW=9, S_DONE=10;
    reg [3:0] st;
    reg [1:0] drain;
    reg [5:0] load_cnt;
    assign hist_shift  = (st == S_IDLE) && feat_valid && !clip_start;
    assign feat_wr     = feat_valid && (((st == S_IDLE) && !clip_start) || (st == S_LOAD));
    assign feat_wr_idx = (st == S_IDLE) ? 6'd0 : load_cnt;
    // geometry of the current layer
    wire [5:0] n_ch   = (layer == 3'd0) ? N_IN - 1 : N_HID - 1;             // last channel index per tap
    wire [1:0] n_tap  = (layer == 3'd4) ? 2'd0 : K - 1;                     // last tap index
    wire [5:0] n_post = (layer == 3'd4) ? N_CLASS - 1 : N_HID - 1;           // last post index
    wire       last_row = (ch == n_ch) && (tap == n_tap);
    wire       skip = act_zero && ((layer != 3'd0) || (ZERO_SKIP != 0));
    assign busy = (st != S_IDLE);
    task advance_row;  // next (tap, ch), row_base += N_POST
        begin
            row_base <= row_base + {8'd0, n_post} + 14'd1;
            if (ch == n_ch) begin ch <= 6'd0; tap <= tap + 2'd1; end
            else ch <= ch + 6'd1;
        end
    endtask
    always @(posedge clk) begin
        // defaults (pulses)
        clr_all <= 1'b0; issue <= 1'b0; acc_clr <= 1'b0;
        upd_en <= 1'b0; dec_start <= 1'b0; frame_done <= 1'b0; vclr_en <= 1'b0;
        if (rst) begin
            st <= S_CLR; vclr_idx <= 8'd0; clr_all <= 1'b1; layer <= 3'd0;
            cyc_frame <= 0; cyc_acc_l1 <= 0; cyc_acc_hid <= 0; cyc_acc_out <= 0; cyc_upd <= 0; cyc_dec <= 0;
            rows_skip_l1 <= 0; rows_run_l1 <= 0; rows_run_hid <= 0; rows_run_out <= 0;
        end else begin
            if (st != S_IDLE && st != S_LOAD && st != S_CLR) cyc_frame <= cyc_frame + 1;
            case (st)
            S_CLR: begin                               // zero v_mem (256 entries), histories already cleared
                vclr_en <= 1'b1; vclr_idx <= vclr_idx + 8'd1;
                if (vclr_idx == 8'd255) st <= S_IDLE;
            end
            S_IDLE: begin
                if (clip_start) begin st <= S_CLR; vclr_idx <= 8'd0; clr_all <= 1'b1; end
                else if (feat_valid) begin
                    load_cnt <= 6'd1;
                    cyc_frame <= 0; cyc_acc_l1 <= 0; cyc_acc_hid <= 0; cyc_acc_out <= 0; cyc_upd <= 0; cyc_dec <= 0;
                    rows_skip_l1 <= 0; rows_run_l1 <= 0; rows_run_hid <= 0; rows_run_out <= 0;
                    if (feat_last) begin st <= S_INIT; layer <= 3'd0; end
                    else st <= S_LOAD;
                end
            end
            S_LOAD: begin
                if (feat_valid) begin
                    load_cnt <= load_cnt + 6'd1;
                    if (feat_last) begin st <= S_INIT; layer <= 3'd0; end
                end
            end
            S_INIT: begin
                acc_clr <= 1'b1; tap <= 2'd0; ch <= 6'd0; row_base <= 14'd0; post <= 6'd0;
                st <= S_ROWCHK;
            end
            S_ROWCHK: begin
                if (layer == 3'd0) cyc_acc_l1 <= cyc_acc_l1 + 1;
                else if (layer == 3'd4) cyc_acc_out <= cyc_acc_out + 1;
                else cyc_acc_hid <= cyc_acc_hid + 1;
                if (skip) begin
                    if (layer == 3'd0) rows_skip_l1 <= rows_skip_l1 + 1;
                    if (last_row) begin st <= S_DRAIN; drain <= 2'd0; end
                    else advance_row;
                end else begin
                    if (layer == 3'd0) rows_run_l1 <= rows_run_l1 + 1;
                    else if (layer == 3'd4) rows_run_out <= rows_run_out + 1;
                    else rows_run_hid <= rows_run_hid + 1;
                    post <= 6'd0; issue <= 1'b1; st <= S_ROWRUN;
                end
            end
            S_ROWRUN: begin                            // `issue` was set for post in the previous cycle
                if (layer == 3'd0) cyc_acc_l1 <= cyc_acc_l1 + 1;
                else if (layer == 3'd4) cyc_acc_out <= cyc_acc_out + 1;
                else cyc_acc_hid <= cyc_acc_hid + 1;
                if (post == n_post) begin
                    if (last_row) begin st <= S_DRAIN; drain <= 2'd0; end
                    else begin advance_row; st <= S_ROWCHK; end
                end else begin
                    post <= post + 6'd1; issue <= 1'b1;
                end
            end
            S_DRAIN: begin
                drain <= drain + 2'd1;
                if (drain == 2'd2 && !acc_busy) begin st <= S_UPD; upd_idx <= 6'd0; upd_en <= 1'b1; end
            end
            S_UPD: begin                               // upd_en/upd_idx for index upd_idx were set last cycle
                cyc_upd <= cyc_upd + 1;
                if (upd_idx == n_post) begin
                    if (layer == 3'd4) begin st <= S_DEC; dec_start <= 1'b1; end
                    else begin layer <= layer + 3'd1; st <= S_INIT; end
                end else begin
                    upd_idx <= upd_idx + 6'd1; upd_en <= 1'b1;
                end
            end
            S_DEC: begin cyc_dec <= cyc_dec + 1; st <= S_DECW; end
            S_DECW: begin
                cyc_dec <= cyc_dec + 1;
                if (dec_done) st <= S_DONE;
            end
            S_DONE: begin frame_done <= 1'b1; st <= S_IDLE; end
            default: st <= S_IDLE;
            endcase
        end
    end
endmodule
