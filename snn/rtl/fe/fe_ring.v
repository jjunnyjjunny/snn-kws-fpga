// fe_ring.v — block 2: 18-bit sample ring (512 deep, BRAM) + framing. Every sample arrives with `new_sample`
// (already scaled to 18 bit). frame_start pulses when 480 samples have been collected and then every 160
// samples (frame t = x[160t : 160t+480]); rd_base = ring address of the frame's first sample.
// The consumer reads 480 samples (addr = rd_base + n) through the read port; it must finish before 32 new
// samples arrive (512 - 480 slack) — 480 cycles at 48 MHz vs 32 x 3000 cycles at 16 kHz.
`timescale 1ns/1ps
module fe_ring #(parameter DB = 18, WIN = 480, HOP = 160, DEPTH = 512, AW = 9) (
    input  wire          clk,
    input  wire          rst,
    input  wire          new_sample,
    input  wire signed [DB-1:0] x,
    output reg           frame_start,
    output reg  [AW-1:0] rd_base,
    output reg  [27:0]   e0,            // sum |x| over the frame's 480 samples (3 hops), valid with frame_start
    input  wire [AW-1:0] rd_addr,
    output reg  signed [DB-1:0] rd_data
);
    reg signed [DB-1:0] mem [0:DEPTH-1];
    reg [AW-1:0] wp;
    reg [15:0]   cnt;       // samples since last frame start (saturating at WIN before the first frame)
    reg          primed;    // >= WIN samples seen
    reg [25:0]   hop_acc; reg [25:0] hop1, hop2;        // |x| sum of the current hop and the two previous hops
    wire [DB-1:0] ax = x[DB-1] ? (~x + 1'b1) : x;
    wire [25:0]  hop_now = hop_acc + {{(26-DB){1'b0}}, ax};
    reg  [15:0]  hcnt;
    always @(posedge clk) begin
        frame_start <= 1'b0;
        if (rst) begin wp <= 0; cnt <= 0; primed <= 0; hop_acc <= 0; hop1 <= 0; hop2 <= 0; hcnt <= 0; end
        else if (new_sample) begin
            mem[wp] <= x; wp <= wp + 1'b1;
            if (hcnt == HOP - 1) begin hcnt <= 0; hop_acc <= 0; hop2 <= hop1; hop1 <= hop_now; e0 <= hop_now + hop1 + hop2; end
            else begin hcnt <= hcnt + 1'b1; hop_acc <= hop_now; end
            if (!primed) begin
                if (cnt == WIN - 1) begin primed <= 1'b1; cnt <= 0; frame_start <= 1'b1; rd_base <= wp + 1'b1 - WIN; end
                else cnt <= cnt + 1'b1;
            end else begin
                if (cnt == HOP - 1) begin cnt <= 0; frame_start <= 1'b1; rd_base <= wp + 1'b1 - WIN; end
                else cnt <= cnt + 1'b1;
            end
        end
        rd_data <= mem[rd_addr];
    end
`ifdef ACTIVITY
    integer act_wr = 0;
    always @(posedge clk) if (new_sample) act_wr = act_wr + 1;
`endif
endmodule
