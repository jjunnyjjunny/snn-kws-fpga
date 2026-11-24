// tb_fe_blocks.v — block-level golden tests of the front-end on one vector dir (all 98 frames):
//   WINDOW : x_db.hex frames -> fe_window -> buffer (un-bit-reversed) == windowed.hex (+ zeros 480..511)
//   FFT    : buffer <- windowed.hex (bit-reversed) -> fe_fft512 -> buffer == fft_re.hex / fft_im.hex
//   POWER  : fe_power(fft_re, fft_im) == power.hex (bins 0..256)
//   MEL    : buffer <- fft_re/im.hex -> fe_mel_log -> mel_acc.hex / log2q.hex / feat.hex
// Config via tb_cfg.vh: `VEC_DIR, `ROM_DIR, `MULT_SERIAL, `T_FRAMES, `STOP_ON_MISMATCH
`timescale 1ns/1ps
`include "tb_cfg.vh"
module tb_fe_blocks;
    localparam DB = 18, N = 512, WIN = 480, T = `T_FRAMES, NB = 257, NM = 40;
    reg clk = 0; always #5 clk = ~clk;
    reg rst = 1;
    // ------------------------------------------------------------------ expected data
    reg [17:0] xdb [0:16000-1];
    reg [17:0] win_e [0:T*WIN-1];
    reg [17:0] fre [0:T*N-1]; reg [17:0] fim [0:T*N-1];
    reg [35:0] pow_e [0:T*NB-1];
    reg [49:0] mel_e [0:T*NM-1]; reg [9:0] lq_e [0:T*NM-1]; reg [7:0] feat_e [0:T*NM-1];
    integer t, n, mism_win, mism_fft, mism_pow, mism_mel, mism_lq, mism_feat, cyc_fft_tot, cyc_mel_tot, k;
    function [8:0] bitrev; input [8:0] a; integer i; begin for (i = 0; i < 9; i = i + 1) bitrev[i] = a[8-i]; end endfunction
    // ------------------------------------------------------------------ DUT: buffer + window + fft + mel (shared buffer, TB drives the mux)
    reg  en_a, we_a; reg [8:0] addr_a; reg [2*DB-1:0] wd_a; wire [2*DB-1:0] rd_a;
    reg  en_b, we_b; reg [8:0] addr_b; reg [2*DB-1:0] wd_b; wire [2*DB-1:0] rd_b;
    fe_buf #(.DB(DB)) u_buf (.clk(clk), .en_a(en_a), .we_a(we_a), .addr_a(addr_a), .wd_a(wd_a), .rd_a(rd_a),
                              .en_b(en_b), .we_b(we_b), .addr_b(addr_b), .wd_b(wd_b), .rd_b(rd_b));
    // window: ring emulated by the TB (registered read with 1-cycle latency like fe_ring)
    reg win_start; wire win_done, win_busy; wire [8:0] w_raddr; reg signed [DB-1:0] ring_q; wire w_we; wire [8:0] w_addr; wire [2*DB-1:0] w_wd;
    reg [8:0] ring_q_addr;
    always @(posedge clk) ring_q <= (w_raddr < WIN) ? $signed(xdb[t*160 + w_raddr]) : 18'sd0;   // 1-cycle latency like fe_ring
    fe_window #(.DB(DB), .HANN_MEM({`ROM_DIR, "/hann_w480.mem"})) u_win (.clk(clk), .rst(rst), .start(win_start), .done(win_done),
        .ring_addr(w_raddr), .ring_data(ring_q), .buf_we(w_we), .buf_addr(w_addr), .buf_wd(w_wd), .busy(win_busy));
    reg fft_start; wire fft_done, fft_busy; wire f_en_a, f_we_a, f_en_b, f_we_b; wire [8:0] f_addr_a, f_addr_b; wire [2*DB-1:0] f_wd_a, f_wd_b;
    fe_fft512 #(.DB(DB), .MULT_SERIAL(`MULT_SERIAL), .TW_RE_MEM({`ROM_DIR, "/twiddle_re.mem"}), .TW_IM_MEM({`ROM_DIR, "/twiddle_im.mem"})) u_fft (
        .clk(clk), .rst(rst), .start(fft_start), .done(fft_done), .busy(fft_busy),
        .en_a(f_en_a), .we_a(f_we_a), .addr_a(f_addr_a), .wd_a(f_wd_a), .rd_a(rd_a), .en_b(f_en_b), .we_b(f_we_b), .addr_b(f_addr_b), .wd_b(f_wd_b), .rd_b(rd_b));
    reg mel_start; wire mel_done, mel_busy, m_en_a; wire [8:0] m_addr_a; wire fv, fl; wire [7:0] fd; wire [13:0] es; wire [9:0] el; wire dv; wire [49:0] dacc; wire [9:0] dlq;
    fe_mel_log #(.DB(DB), .TABLE_MEM({`ROM_DIR, "/mel40_table.mem"}), .WEIGHT_MEM({`ROM_DIR, "/mel40_weights.mem"}), .LUT_MEM({`ROM_DIR, "/log2_lut.mem"})) u_mel (
        .clk(clk), .rst(rst), .start(mel_start), .done(mel_done), .busy(mel_busy), .en_a(m_en_a), .addr_a(m_addr_a), .rd_a(rd_a),
        .feat_valid(fv), .feat_data(fd), .feat_last(fl), .e_sum(es), .e_log(el), .dbg_valid(dv), .dbg_acc(dacc), .dbg_log2q(dlq));
    wire [35:0] pw; reg signed [DB-1:0] pre, pim;
    fe_power #(.DB(DB)) u_pow (.re(pre), .im(pim), .p(pw));
    // mux owner: 0 = TB, 1 = window, 2 = fft, 3 = mel
    reg [1:0] owner; reg tb_en_a, tb_we_a; reg [8:0] tb_addr_a; reg [2*DB-1:0] tb_wd_a;
    always @* begin
        case (owner)
            2'd1: begin en_a = w_we; we_a = w_we; addr_a = w_addr; wd_a = w_wd; en_b = 0; we_b = 0; addr_b = 0; wd_b = 0; end
            2'd2: begin en_a = f_en_a; we_a = f_we_a; addr_a = f_addr_a; wd_a = f_wd_a; en_b = f_en_b; we_b = f_we_b; addr_b = f_addr_b; wd_b = f_wd_b; end
            2'd3: begin en_a = m_en_a; we_a = 0; addr_a = m_addr_a; wd_a = 0; en_b = 0; we_b = 0; addr_b = 0; wd_b = 0; end
            default: begin en_a = tb_en_a; we_a = tb_we_a; addr_a = tb_addr_a; wd_a = tb_wd_a; en_b = 0; we_b = 0; addr_b = 0; wd_b = 0; end
        endcase
    end
    task buf_write(input [8:0] a, input [2*DB-1:0] d); begin tb_en_a = 1; tb_we_a = 1; tb_addr_a = a; tb_wd_a = d; @(posedge clk); tb_en_a = 0; tb_we_a = 0; end endtask
    task buf_read(input [8:0] a, output [2*DB-1:0] d); begin tb_en_a = 1; tb_we_a = 0; tb_addr_a = a; @(posedge clk); tb_en_a = 0; @(posedge clk); d = rd_a; end endtask
    reg [2*DB-1:0] rd; integer band, mcount; reg [49:0] mel_got [0:NM-1]; reg [9:0] lq_got [0:NM-1]; reg [7:0] feat_got [0:NM-1];
    always @(posedge clk) if (dv) begin mel_got[mcount] <= dacc; lq_got[mcount] <= dlq; mcount <= mcount + 1; end
    integer fcount; always @(posedge clk) if (fv) begin feat_got[fcount] <= fd; fcount <= fcount + 1; end
    initial begin
        $readmemh({`VEC_DIR, "/x_db.hex"}, xdb); $readmemh({`VEC_DIR, "/windowed.hex"}, win_e);
        $readmemh({`VEC_DIR, "/fft_re.hex"}, fre); $readmemh({`VEC_DIR, "/fft_im.hex"}, fim); $readmemh({`VEC_DIR, "/power.hex"}, pow_e);
        $readmemh({`VEC_DIR, "/mel_acc.hex"}, mel_e); $readmemh({`VEC_DIR, "/log2q.hex"}, lq_e); $readmemh({`VEC_DIR, "/feat.hex"}, feat_e);
        mism_win = 0; mism_fft = 0; mism_pow = 0; mism_mel = 0; mism_lq = 0; mism_feat = 0; cyc_fft_tot = 0; cyc_mel_tot = 0;
        owner = 0; tb_en_a = 0; tb_we_a = 0; win_start = 0; fft_start = 0; mel_start = 0; mcount = 0; fcount = 0;
        repeat (3) @(posedge clk); rst <= 0; @(posedge clk);
        for (t = 0; t < T; t = t + 1) begin
            // ---------------- WINDOW
            owner = 1; win_start <= 1; @(posedge clk); win_start <= 0; while (!win_done) @(posedge clk); @(posedge clk); owner = 0;
            for (n = 0; n < N; n = n + 1) begin
                buf_read(bitrev(n), rd);
                if (n < WIN) begin if ($signed(rd[2*DB-1:DB]) !== $signed(win_e[t*WIN+n]) || rd[DB-1:0] !== 0) begin mism_win = mism_win + 1;
                    if (mism_win <= 3) $display("WIN mismatch frame %0d n %0d exp %0d got %0d", t, n, $signed(win_e[t*WIN+n]), $signed(rd[2*DB-1:DB])); end end
                else if (rd !== 0) mism_win = mism_win + 1;
            end
            // ---------------- FFT (load expected windowed, bit-reversed)
            for (n = 0; n < N; n = n + 1) buf_write(bitrev(n), {(n < WIN) ? win_e[t*WIN+n] : 18'd0, 18'd0});
            owner = 2; fft_start <= 1; @(posedge clk); fft_start <= 0; k = 0;
            while (!fft_done) begin @(posedge clk); k = k + 1; end
            @(posedge clk); owner = 0; cyc_fft_tot = cyc_fft_tot + k;
            for (n = 0; n < N; n = n + 1) begin
                buf_read(n, rd);
                if ($signed(rd[2*DB-1:DB]) !== $signed(fre[t*N+n]) || $signed(rd[DB-1:0]) !== $signed(fim[t*N+n])) begin mism_fft = mism_fft + 1;
                    if (mism_fft <= 3) $display("FFT mismatch frame %0d bin %0d exp %0d,%0d got %0d,%0d", t, n, $signed(fre[t*N+n]), $signed(fim[t*N+n]), $signed(rd[2*DB-1:DB]), $signed(rd[DB-1:0])); end
            end
            // ---------------- POWER (combinational)
            for (n = 0; n < NB; n = n + 1) begin
                pre = $signed(fre[t*N+n]); pim = $signed(fim[t*N+n]); #1;
                if (pw !== pow_e[t*NB+n]) begin mism_pow = mism_pow + 1; if (mism_pow <= 3) $display("POW mismatch frame %0d bin %0d exp %0d got %0d", t, n, pow_e[t*NB+n], pw); end
            end
            // ---------------- MEL (load expected fft)
            for (n = 0; n < N; n = n + 1) buf_write(n, {fre[t*N+n], fim[t*N+n]});
            mcount = 0; fcount = 0; owner = 3; mel_start <= 1; @(posedge clk); mel_start <= 0; k = 0;
            while (!mel_done) begin @(posedge clk); k = k + 1; end
            @(posedge clk); owner = 0; cyc_mel_tot = cyc_mel_tot + k;
            for (band = 0; band < NM; band = band + 1) begin
                if (mel_got[band] !== mel_e[t*NM+band]) begin mism_mel = mism_mel + 1; if (mism_mel <= 3) $display("MEL mismatch frame %0d band %0d exp %0d got %0d", t, band, mel_e[t*NM+band], mel_got[band]); end
                if (lq_got[band] !== lq_e[t*NM+band]) begin mism_lq = mism_lq + 1; if (mism_lq <= 3) $display("LOG2 mismatch frame %0d band %0d exp %0d got %0d", t, band, lq_e[t*NM+band], lq_got[band]); end
                if (feat_got[band] !== feat_e[t*NM+band]) begin mism_feat = mism_feat + 1; if (mism_feat <= 3) $display("FEAT mismatch frame %0d band %0d exp %0d got %0d", t, band, feat_e[t*NM+band], feat_got[band]); end
            end
            if (`STOP_ON_MISMATCH && (mism_win + mism_fft + mism_pow + mism_mel + mism_lq + mism_feat) > 0) begin $display("STOP at frame %0d", t); $finish; end
        end
        $display("FE_BLOCKS vec=%s win=%0d fft=%0d pow=%0d mel=%0d log2=%0d feat=%0d %s cyc_fft_avg=%0d cyc_mel_avg=%0d", `VEC_DIR,
                 mism_win, mism_fft, mism_pow, mism_mel, mism_lq, mism_feat, (mism_win + mism_fft + mism_pow + mism_mel + mism_lq + mism_feat) == 0 ? "PASS" : "FAIL", cyc_fft_tot / T, cyc_mel_tot / T);
        $finish;
    end
endmodule
