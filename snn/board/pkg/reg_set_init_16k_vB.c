/******************************************************************************
* Copyright (C) 2023 Advanced Micro Devices, Inc. All Rights Reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/
/*
 * reg_set_init_16k_vB.c: PGA path: boost 0 dB + differential volume +12 dB, mixer LINN/AUX muted, ADC HPF on
 *   0x400A/0x400C = 0x01: mixer inputs muted (MX1EN only); 0x400B/0x400D = 0x08: LDBOOST = 01 (0 dB), AUX mute
 *   0x400E/0x400F = 0x83: LDVOL code 32 = +12 dB, LDMUTE = 1 (unmuted), LDEN = 1; expected level about +12 dB
 *
 * This application configures UART 16550 to baud rate 9600.
 * PS7 UART (Zynq) is not initialized by this application, since
 * bootrom/bsp configures it to baud rate 115200
 *
 * ------------------------------------------------
 * | UART TYPE   BAUD RATE                        |
 * ------------------------------------------------
 *   uartns550   9600
 *   uartlite    Configurable only in HW design
 *   ps7_uart    115200 (configured by bootrom/bsp)
 */

#include <stdio.h>
#include "sleep.h"
#include "xparameters.h"
#include "xiicps.h"

int main()
{
    int Status;
    XIicPs IicPsInstance;
    XIicPs_Config *Config;
    u8 writeData[8];
    u8 readData[8];

	#ifndef SDT
		Config = XIicPs_LookupConfig(XPAR_XIICPS_0_DEVICE_ID);
	#else
		Config = XIicPs_LookupConfig(XPAR_XIICPS_0_BASEADDR);
	#endif
    if (Config == NULL) {
    	return XST_FAILURE;
    }

    Status = XIicPs_CfgInitialize(&IicPsInstance, Config, Config->BaseAddress);
    if (Status != XST_SUCCESS) {
    	return XST_FAILURE;
    }

	Status = XIicPs_SelfTest(&IicPsInstance);
	if (Status != XST_SUCCESS) {
		return XST_FAILURE;
	}

    XIicPs_SetSClk(&IicPsInstance, 300000);

    // Register Write (adau1761 datasheet 37p)
    // chip address byte, subaddress byte 1, subaddress byte 2, data byte 1
    // sub address byte 1: writeData[0]
    // sub address byte 2: writeData[1]
    // data byte 1       : writeData[2]
    // data send (chip address byte (0x3b, 0b0111011))
    // XIicPs_MasterSendPolled(&IicPsInstance, writeData, 3, 0x3B);

    // Clock Control
    writeData[0] = 0x40; writeData[1] = 0x00;
    writeData[2] = 0x0E;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 3, 0x3B);
    usleep(50000);

    // Clock Control (Read)
    writeData[0] = 0x40; writeData[1] = 0x00;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 2, 0x3B);
    XIicPs_MasterRecvPolled(&IicPsInstance, readData , 1, 0x3B);
    xil_printf("Address: %02x%02x, Read Data: %02x\r\n", *writeData, *(writeData+1), *readData);
    usleep(50000);

    // PLL Control
    writeData[0] = 0x40; writeData[1] = 0x02;
    writeData[2] = 0x00; writeData[3] = 0x7D; writeData[4] = 0x00; writeData[5] = 0x0C; writeData[6] = 0x23; writeData[7] = 0x01;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 8, 0x3B);
    usleep(50000);

    // KWS: wait for PLL lock (0x4002 byte 6 bit 0) before enabling the core clock and writing the record/playback registers
    { int i; for (i = 0; i < 100; i++) { writeData[0] = 0x40; writeData[1] = 0x02;
        XIicPs_MasterSendPolled(&IicPsInstance, writeData, 2, 0x3B); XIicPs_MasterRecvPolled(&IicPsInstance, readData, 6, 0x3B);
        if (readData[5] & 0x01) break; usleep(10000); }
      xil_printf("PLL lock poll: %d x 10 ms, 0x4002 byte 6 = %02x (bit0 = lock)\r\n", i, readData[5]); }

    // Clock Control
    writeData[0] = 0x40; writeData[1] = 0x00;
    writeData[2] = 0x0F;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 3, 0x3B);
    usleep(50000);

    // Serial Port 0
    writeData[0] = 0x40; writeData[1] = 0x15;
    writeData[2] = 0x01;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 3, 0x3B);
    usleep(50000);

    // Rec Mixer Left 0
    writeData[0] = 0x40; writeData[1] = 0x0A;
    writeData[2] = 0x01;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 3, 0x3B);
    usleep(50000);

    // Rec Mixer Left 1
    writeData[0] = 0x40; writeData[1] = 0x0B;
    writeData[2] = 0x08;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 3, 0x3B);
    usleep(50000);

    // Rec Mixer Right 0
    writeData[0] = 0x40; writeData[1] = 0x0C;
    writeData[2] = 0x01;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 3, 0x3B);
    usleep(50000);

    // Rec Mixer Right 1
    writeData[0] = 0x40; writeData[1] = 0x0D;
    writeData[2] = 0x08;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 3, 0x3B);
    usleep(50000);

    // Record mic bias
    writeData[0] = 0x40; writeData[1] = 0x10;
    writeData[2] = 0x01;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 3, 0x3B);
    usleep(50000);

    // Left differential input volume: LDVOL +12 dB, unmute, enable
    writeData[0] = 0x40; writeData[1] = 0x0E;
    writeData[2] = 0x83;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 3, 0x3B);
    usleep(50000);

    // Right differential input volume: RDVOL +12 dB, unmute, enable
    writeData[0] = 0x40; writeData[1] = 0x0F;
    writeData[2] = 0x83;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 3, 0x3B);
    usleep(50000);

    // Play Mixer Left 0
    writeData[0] = 0x40; writeData[1] = 0x1C;
    writeData[2] = 0x2D;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 3, 0x3B);
    usleep(50000);

    // Play Mixer Right 0
    writeData[0] = 0x40; writeData[1] = 0x1E;
    writeData[2] = 0x4D;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 3, 0x3B);
    usleep(50000);

    // Play L/R mixer left
    writeData[0] = 0x40; writeData[1] = 0x20;
    writeData[2] = 0x0B;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 3, 0x3B);
    usleep(50000);

    // Play L/R mixer right
    writeData[0] = 0x40; writeData[1] = 0x21;
    writeData[2] = 0x0B;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 3, 0x3B);
    usleep(50000);

    // Play HP left vol
    writeData[0] = 0x40; writeData[1] = 0x23;
    writeData[2] = 0xE7;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 3, 0x3B);
    usleep(50000);

    // Play HP right vol
    writeData[0] = 0x40; writeData[1] = 0x24;
    writeData[2] = 0xE7;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 3, 0x3B);
    usleep(50000);

    // Line output left vol
    writeData[0] = 0x40; writeData[1] = 0x25;
    writeData[2] = 0xE7;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 3, 0x3B);
    usleep(50000);

    // Line output right vol
    writeData[0] = 0x40; writeData[1] = 0x26;
    writeData[2] = 0xE7;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 3, 0x3B);
    usleep(50000);

    // ADC control
    writeData[0] = 0x40; writeData[1] = 0x19;
    writeData[2] = 0x23;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 3, 0x3B);
    usleep(50000);

    // Play power mgmt
    writeData[0] = 0x40; writeData[1] = 0x29;
    writeData[2] = 0x03;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 3, 0x3B);
    usleep(50000);

    // DAC Control 0
    writeData[0] = 0x40; writeData[1] = 0x2A;
    writeData[2] = 0x03;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 3, 0x3B);
    usleep(50000);

    // Serial input route control
    writeData[0] = 0x40; writeData[1] = 0xF2;
    writeData[2] = 0x01;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 3, 0x3B);
    usleep(50000);

    // Serial output route control
    writeData[0] = 0x40; writeData[1] = 0xF3;
    writeData[2] = 0x01;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 3, 0x3B);
    usleep(50000);

    // Clock Enable 0
    writeData[0] = 0x40; writeData[1] = 0xF9;
    writeData[2] = 0x7F;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 3, 0x3B);
    usleep(50000);

    // Clock Enable 1
    writeData[0] = 0x40; writeData[1] = 0xFA;
    writeData[2] = 0x03;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 3, 0x3B);
    usleep(50000);

    // ================= KWS 16 kHz (added to the reference sequence) =================================
    // ADAU1761 data sheet (Rev. D), register map: R17 Converter Control 0 (0x4017) bits CONVSR[2:0]
    //   000 = 48 kHz, 001 = 8 kHz, 010 = 12 kHz, 011 = 16 kHz, 100 = 24 kHz, 101 = 32 kHz, 110 = 96 kHz
    //   (ADC/DAC rate relative to the 48 kHz base clock set by the PLL: 24 MHz/2 x (4 + 12/125) = 49.152 MHz = 1024 fs)
    // R64 Serial Port Sampling Rate (0x40F8) bits SPSR[2:0]: same encoding -> LRCLK = 16 kHz, BCLK = 64 x fs
    //   (the codec is the serial-port master: Serial Port 0 (0x4015) = 0x01 above, so LRCLK/BCLK follow this register)
    // TODO verify: bit encodings against the data sheet tables for 0x4017 / 0x40F8 (table numbers differ by revision)
    // Converter Control 0 -> 16 kHz
    writeData[0] = 0x40; writeData[1] = 0x17;
    writeData[2] = 0x03;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 3, 0x3B);
    usleep(50000);

    // Serial Port Sampling Rate -> 16 kHz
    writeData[0] = 0x40; writeData[1] = 0xF8;
    writeData[2] = 0x03;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 3, 0x3B);
    usleep(50000);
    // (DSP Sampling Rate 0x40EB left at default: the SigmaDSP core is not used by this design)
    // =================================================================================================



    // Register Read (adau1761 datasheet 37p)
    // chip address byte(w/ write), subaddress byte 1, subaddress byte 2,
    // chip address byte(w/ read), data byte 1
    // sub address byte 1: writeData[0]
    // sub address byte 2: writeData[1]
    // data byte 1       : readData[0]
    // data send (chip address byte (0x3b, 0b0111011))
    // XIicPs_SetOptions(&IicPsInstance, XIICPS_REP_START_OPTION);
    // XIicPs_MasterSendPolled(&IicPsInstance, writeData, 2, 0x3B);
    // XIicPs_ClearOptions(&IicPsInstance, XIICPS_REP_START_OPTION);
    // XIicPs_MasterRecvPolled(&IicPsInstance, readData , 1, 0x3B);

    // Clock Control (above, read right away after setting)
    // writeData[0] = 0x40; writeData[1] = 0x00;
    // XIicPs_MasterSendPolled(&IicPsInstance, writeData, 2, 0x3B);
    // XIicPs_MasterRecvPolled(&IicPsInstance, readData , 1, 0x3B);
    // xil_printf("Address: %02x%02x, Read Data: %02x\r\n", *writeData, *(writeData+1), *readData);
    // usleep(50000);

    // PLL Control
    writeData[0] = 0x40; writeData[1] = 0x02;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 2, 0x3B);
    XIicPs_MasterRecvPolled(&IicPsInstance, readData , 6, 0x3B);
    xil_printf("Address: %02x%02x, Read Data: %02x %02x %02x %02x %02x %02x\r\n", *writeData, *(writeData+1), *readData, *(readData+1), *(readData+2), *(readData+3), *(readData+4), *(readData+5));
    usleep(50000);

    // Clock Control
    writeData[0] = 0x40; writeData[1] = 0x00;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 2, 0x3B);
    XIicPs_MasterRecvPolled(&IicPsInstance, readData , 1, 0x3B);
    xil_printf("Address: %02x%02x, Read Data: %02x\r\n", *writeData, *(writeData+1), *readData);
    usleep(50000);

    // Serial Port 0
    writeData[0] = 0x40; writeData[1] = 0x15;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 2, 0x3B);
    XIicPs_MasterRecvPolled(&IicPsInstance, readData , 1, 0x3B);
    xil_printf("Address: %02x%02x, Read Data: %02x\r\n", *writeData, *(writeData+1), *readData);
    usleep(50000);

    // Rec Mixer Left 0
    writeData[0] = 0x40; writeData[1] = 0x0A;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 2, 0x3B);
    XIicPs_MasterRecvPolled(&IicPsInstance, readData , 1, 0x3B);
    xil_printf("Address: %02x%02x, Read Data: %02x\r\n", *writeData, *(writeData+1), *readData);
    usleep(50000);

    // Rec Mixer Left 1
    writeData[0] = 0x40; writeData[1] = 0x0B;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 2, 0x3B);
    XIicPs_MasterRecvPolled(&IicPsInstance, readData , 1, 0x3B);
    xil_printf("Address: %02x%02x, Read Data: %02x\r\n", *writeData, *(writeData+1), *readData);
    usleep(50000);

    // Rec Mixer Right 0
    writeData[0] = 0x40; writeData[1] = 0x0C;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 2, 0x3B);
    XIicPs_MasterRecvPolled(&IicPsInstance, readData , 1, 0x3B);
    xil_printf("Address: %02x%02x, Read Data: %02x\r\n", *writeData, *(writeData+1), *readData);
    usleep(50000);

    // Rec Mixer Right 1
    writeData[0] = 0x40; writeData[1] = 0x0D;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 2, 0x3B);
    XIicPs_MasterRecvPolled(&IicPsInstance, readData , 1, 0x3B);
    xil_printf("Address: %02x%02x, Read Data: %02x\r\n", *writeData, *(writeData+1), *readData);
    usleep(50000);

    // Record mic bias
    writeData[0] = 0x40; writeData[1] = 0x10;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 2, 0x3B);
    XIicPs_MasterRecvPolled(&IicPsInstance, readData , 1, 0x3B);
    xil_printf("Address: %02x%02x, Read Data: %02x\r\n", *writeData, *(writeData+1), *readData);
    usleep(50000);

    // Play Mixer Left 0
    writeData[0] = 0x40; writeData[1] = 0x1C;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 2, 0x3B);
    XIicPs_MasterRecvPolled(&IicPsInstance, readData , 1, 0x3B);
    xil_printf("Address: %02x%02x, Read Data: %02x\r\n", *writeData, *(writeData+1), *readData);
    usleep(50000);

    // Play Mixer Right 0
    writeData[0] = 0x40; writeData[1] = 0x1E;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 2, 0x3B);
    XIicPs_MasterRecvPolled(&IicPsInstance, readData , 1, 0x3B);
    xil_printf("Address: %02x%02x, Read Data: %02x\r\n", *writeData, *(writeData+1), *readData);
    usleep(50000);

    // Play L/R mixer left
    writeData[0] = 0x40; writeData[1] = 0x20;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 2, 0x3B);
    XIicPs_MasterRecvPolled(&IicPsInstance, readData , 1, 0x3B);
    xil_printf("Address: %02x%02x, Read Data: %02x\r\n", *writeData, *(writeData+1), *readData);
    usleep(50000);

    // Play L/R mixer right
    writeData[0] = 0x40; writeData[1] = 0x21;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 2, 0x3B);
    XIicPs_MasterRecvPolled(&IicPsInstance, readData , 1, 0x3B);
    xil_printf("Address: %02x%02x, Read Data: %02x\r\n", *writeData, *(writeData+1), *readData);
    usleep(50000);

    // Play HP left vol
    writeData[0] = 0x40; writeData[1] = 0x23;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 2, 0x3B);
    XIicPs_MasterRecvPolled(&IicPsInstance, readData , 1, 0x3B);
    xil_printf("Address: %02x%02x, Read Data: %02x\r\n", *writeData, *(writeData+1), *readData);
    usleep(50000);

    // Play HP right vol
    writeData[0] = 0x40; writeData[1] = 0x24;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 2, 0x3B);
    XIicPs_MasterRecvPolled(&IicPsInstance, readData , 1, 0x3B);
    xil_printf("Address: %02x%02x, Read Data: %02x\r\n", *writeData, *(writeData+1), *readData);
    usleep(50000);

    // Line output left vol
    writeData[0] = 0x40; writeData[1] = 0x25;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 2, 0x3B);
    XIicPs_MasterRecvPolled(&IicPsInstance, readData , 1, 0x3B);
    xil_printf("Address: %02x%02x, Read Data: %02x\r\n", *writeData, *(writeData+1), *readData);
    usleep(50000);

    // Line output right vol
    writeData[0] = 0x40; writeData[1] = 0x26;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 2, 0x3B);
    XIicPs_MasterRecvPolled(&IicPsInstance, readData , 1, 0x3B);
    xil_printf("Address: %02x%02x, Read Data: %02x\r\n", *writeData, *(writeData+1), *readData);
    usleep(50000);

    // ADC control
    writeData[0] = 0x40; writeData[1] = 0x19;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 2, 0x3B);
    XIicPs_MasterRecvPolled(&IicPsInstance, readData , 1, 0x3B);
    xil_printf("Address: %02x%02x, Read Data: %02x\r\n", *writeData, *(writeData+1), *readData);
    usleep(50000);

    // Play power mgmt
    writeData[0] = 0x40; writeData[1] = 0x29;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 2, 0x3B);
    XIicPs_MasterRecvPolled(&IicPsInstance, readData , 1, 0x3B);
    xil_printf("Address: %02x%02x, Read Data: %02x\r\n", *writeData, *(writeData+1), *readData);
    usleep(50000);

    // DAC Control 0
    writeData[0] = 0x40; writeData[1] = 0x2A;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 2, 0x3B);
    XIicPs_MasterRecvPolled(&IicPsInstance, readData , 1, 0x3B);
    xil_printf("Address: %02x%02x, Read Data: %02x\r\n", *writeData, *(writeData+1), *readData);
    usleep(50000);

    // Serial input route control
    writeData[0] = 0x40; writeData[1] = 0xF2;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 2, 0x3B);
    XIicPs_MasterRecvPolled(&IicPsInstance, readData , 1, 0x3B);
    xil_printf("Address: %02x%02x, Read Data: %02x\r\n", *writeData, *(writeData+1), *readData);
    usleep(50000);

    // Serial output route control
    writeData[0] = 0x40; writeData[1] = 0xF3;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 2, 0x3B);
    XIicPs_MasterRecvPolled(&IicPsInstance, readData , 1, 0x3B);
    xil_printf("Address: %02x%02x, Read Data: %02x\r\n", *writeData, *(writeData+1), *readData);
    usleep(50000);

    // Clock Enable 0
    writeData[0] = 0x40; writeData[1] = 0xF9;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 2, 0x3B);
    XIicPs_MasterRecvPolled(&IicPsInstance, readData , 1, 0x3B);
    xil_printf("Address: %02x%02x, Read Data: %02x\r\n", *writeData, *(writeData+1), *readData);
    usleep(50000);

    // Clock Enable 1
    writeData[0] = 0x40; writeData[1] = 0xFA;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 2, 0x3B);
    XIicPs_MasterRecvPolled(&IicPsInstance, readData , 1, 0x3B);
    xil_printf("Address: %02x%02x, Read Data: %02x\r\n", *writeData, *(writeData+1), *readData);
    usleep(50000);

    // ================= KWS 16 kHz read-back =========================================================
    // Converter Control 0 (expect 0x03)
    writeData[0] = 0x40; writeData[1] = 0x17;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 2, 0x3B);
    XIicPs_MasterRecvPolled(&IicPsInstance, readData , 1, 0x3B);
    xil_printf("Address: %02x%02x, Read Data: %02x  (Converter Control 0, expect 03 = 16 kHz)\r\n", *writeData, *(writeData+1), *readData);
    usleep(50000);

    // Serial Port Sampling Rate (expect 0x03)
    writeData[0] = 0x40; writeData[1] = 0xF8;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 2, 0x3B);
    XIicPs_MasterRecvPolled(&IicPsInstance, readData , 1, 0x3B);
    xil_printf("Address: %02x%02x, Read Data: %02x  (Serial Port Sampling Rate, expect 03 = 16 kHz)\r\n", *writeData, *(writeData+1), *readData);
    usleep(50000);
    // Rec Mixer Left 0
    writeData[0] = 0x40; writeData[1] = 0x0A;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 2, 0x3B);
    XIicPs_MasterRecvPolled(&IicPsInstance, readData , 1, 0x3B);
    xil_printf("Address: %02x%02x, Read Data: %02x  (Rec Mixer Left 0, expect 01)\r\n", *writeData, *(writeData+1), *readData);
    usleep(50000);

    // Rec Mixer Right 0
    writeData[0] = 0x40; writeData[1] = 0x0C;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 2, 0x3B);
    XIicPs_MasterRecvPolled(&IicPsInstance, readData , 1, 0x3B);
    xil_printf("Address: %02x%02x, Read Data: %02x  (Rec Mixer Right 0, expect 01)\r\n", *writeData, *(writeData+1), *readData);
    usleep(50000);

    // Rec Mixer Left 1
    writeData[0] = 0x40; writeData[1] = 0x0B;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 2, 0x3B);
    XIicPs_MasterRecvPolled(&IicPsInstance, readData , 1, 0x3B);
    xil_printf("Address: %02x%02x, Read Data: %02x  (Rec Mixer Left 1, expect 08)\r\n", *writeData, *(writeData+1), *readData);
    usleep(50000);

    // Rec Mixer Right 1
    writeData[0] = 0x40; writeData[1] = 0x0D;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 2, 0x3B);
    XIicPs_MasterRecvPolled(&IicPsInstance, readData , 1, 0x3B);
    xil_printf("Address: %02x%02x, Read Data: %02x  (Rec Mixer Right 1, expect 08)\r\n", *writeData, *(writeData+1), *readData);
    usleep(50000);

    // ADC control
    writeData[0] = 0x40; writeData[1] = 0x19;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 2, 0x3B);
    XIicPs_MasterRecvPolled(&IicPsInstance, readData , 1, 0x3B);
    xil_printf("Address: %02x%02x, Read Data: %02x  (ADC control, expect 23)\r\n", *writeData, *(writeData+1), *readData);
    usleep(50000);

    // Left differential input volume: LDVOL +12 dB, unmute, enable
    writeData[0] = 0x40; writeData[1] = 0x0E;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 2, 0x3B);
    XIicPs_MasterRecvPolled(&IicPsInstance, readData , 1, 0x3B);
    xil_printf("Address: %02x%02x, Read Data: %02x  (Left differential input volume: LDVOL +12 dB, unmute, enable, expect 83)\r\n", *writeData, *(writeData+1), *readData);
    usleep(50000);

    // Right differential input volume: RDVOL +12 dB, unmute, enable
    writeData[0] = 0x40; writeData[1] = 0x0F;
    XIicPs_MasterSendPolled(&IicPsInstance, writeData, 2, 0x3B);
    XIicPs_MasterRecvPolled(&IicPsInstance, readData , 1, 0x3B);
    xil_printf("Address: %02x%02x, Read Data: %02x  (Right differential input volume: RDVOL +12 dB, unmute, enable, expect 83)\r\n", *writeData, *(writeData+1), *readData);
    usleep(50000);

    xil_printf("KWS variant vB: PGA path: boost 0 dB + differential volume +12 dB, mixer LINN/AUX muted, ADC HPF on\r\n");
    xil_printf("KWS: codec configured for 16 kHz (check LRCLK = 62.5 us / new_sample spacing 3000 clk on the ILA)\r\n");
    // =================================================================================================

    return 0;
}
