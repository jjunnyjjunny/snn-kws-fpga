// spike_hist.v — 1-bit spike history as a frame shift register: slot 0 = current frame,
// slot i = i frames ago. `shift` is pulsed once at the start of every frame (slot0 <- 0, slot i <- slot i-1),
// the owning layer then writes its spikes into slot 0 bit by bit (wr_en / wr_idx / wr_bit).
// Read: rd_bit = slot[rd_frame][rd_idx] (combinational). `clr` zeroes everything (clip_start / rst).
`timescale 1ns/1ps
module spike_hist #(
    parameter N     = 64,     // neurons per frame
    parameter DEPTH = 17,     // frames kept (= dilation*(K-1)+1 of the consuming layer)
    parameter IW    = 6,      // index width
    parameter FW    = 5       // frame-select width
) (
    input  wire          clk,
    input  wire          clr,
    input  wire          shift,
    input  wire          wr_en,
    input  wire [IW-1:0] wr_idx,
    input  wire          wr_bit,
    input  wire [FW-1:0] rd_frame,
    input  wire [IW-1:0] rd_idx,
    output wire          rd_bit,
    output wire [N-1:0]  cur      // slot 0 (debug / test bench)
);
    reg [N-1:0] slot [0:DEPTH-1];
    integer i;
    always @(posedge clk) begin
        if (clr) begin
            for (i = 0; i < DEPTH; i = i + 1) slot[i] <= {N{1'b0}};
        end else begin
            if (shift) begin
                for (i = DEPTH-1; i > 0; i = i - 1) slot[i] <= slot[i-1];
                slot[0] <= {N{1'b0}};
            end
            if (wr_en) slot[0][wr_idx] <= wr_bit;
        end
    end
    assign rd_bit = slot[rd_frame][rd_idx];
    assign cur    = slot[0];
endmodule
