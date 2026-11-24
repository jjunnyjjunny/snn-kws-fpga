// preroll_buf.v — ring of the last DEPTH frames of features (DEPTH x 40 x 8 bit). Every frame's feature
// stream is written; `replay(n)` streams the newest n frames (oldest first) to the SNN core with the
// feat_valid/feat_data/feat_last protocol, waiting for `snn_busy` to drop between frames.
`timescale 1ns/1ps
module preroll_buf #(parameter DEPTH = 32, N_IN = 40, DW = 5) (
    input  wire clk, input wire rst,
    input  wire in_valid, input wire [7:0] in_data, input wire in_last,     // from fe_top
    input  wire replay, input wire [7:0] n_frames,                            // pulse: feed the newest n frames
    input  wire snn_busy,
    output reg  out_valid, output reg [7:0] out_data, output reg out_last,
    output reg  out_frame_done,     // pulse after each replayed frame has been handed to the SNN (its busy went 1->0)
    output reg  busy
);
    reg [7:0] mem [0:DEPTH*N_IN-1];
    reg [DW-1:0] wslot; reg [5:0] widx;
    always @(posedge clk) begin
        if (rst) begin wslot <= 0; widx <= 0; end
        else if (in_valid) begin
            mem[wslot * N_IN + widx] <= in_data;
            if (in_last) begin widx <= 0; wslot <= wslot + 1'b1; end else widx <= widx + 1'b1;
        end
    end
    reg [2:0] st; localparam S_IDLE=0, S_FEED=1, S_WAIT_BUSY=2, S_WAIT_DONE=3;
    reg [7:0] left; reg [DW-1:0] rslot; reg [5:0] ridx;
    always @(posedge clk) begin
        out_valid <= 0; out_last <= 0; out_frame_done <= 0;
        if (rst) begin st <= S_IDLE; busy <= 0; end
        else case (st)
            S_IDLE: if (replay && n_frames != 0) begin left <= n_frames; rslot <= wslot - n_frames; ridx <= 0; st <= S_FEED; busy <= 1; end
            S_FEED: begin
                out_valid <= 1; out_data <= mem[rslot * N_IN + ridx]; out_last <= (ridx == N_IN - 1);
                if (ridx == N_IN - 1) begin ridx <= 0; st <= S_WAIT_BUSY; end else ridx <= ridx + 1'b1;
            end
            S_WAIT_BUSY: if (snn_busy) st <= S_WAIT_DONE;                    // core accepted the frame
            S_WAIT_DONE: if (!snn_busy) begin
                out_frame_done <= 1; rslot <= rslot + 1'b1; left <= left - 1'b1;
                if (left == 1) begin st <= S_IDLE; busy <= 0; end else st <= S_FEED;
            end
            default: st <= S_IDLE;
        endcase
    end
endmodule
