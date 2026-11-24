// feat_hist.v — 8-bit feature history (K frames x N_IN) with the same frame-shift scheme as spike_hist.
`timescale 1ns/1ps
module feat_hist #(
    parameter N_IN  = 40,
    parameter DEPTH = 3,
    parameter IW    = 6,
    parameter FW    = 2
) (
    input  wire          clk,
    input  wire          clr,
    input  wire          shift,
    input  wire          wr_en,
    input  wire [IW-1:0] wr_idx,
    input  wire [7:0]    wr_data,
    input  wire [FW-1:0] rd_frame,
    input  wire [IW-1:0] rd_idx,
    output wire [7:0]    rd_data
);
    reg [7:0] mem [0:DEPTH*N_IN-1];
    integer i;
    always @(posedge clk) begin
        if (clr) begin
            for (i = 0; i < DEPTH*N_IN; i = i + 1) mem[i] <= 8'd0;
        end else begin
            if (shift) begin
                for (i = DEPTH*N_IN-1; i >= N_IN; i = i - 1) mem[i] <= mem[i-N_IN];
                for (i = 0; i < N_IN; i = i + 1) mem[i] <= 8'd0;
            end
            if (wr_en) mem[wr_idx] <= wr_data;      // a write in the shift cycle lands in the new slot 0
        end
    end
    assign rd_data = mem[rd_frame*N_IN + rd_idx];
endmodule
