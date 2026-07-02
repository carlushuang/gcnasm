# matrix_core_opus/pin_reg — pin drives occupancy (no __launch_bounds__)

block_v2 with BLOCK_M=256, BLOCK_N=192 (m=512,n=384,k=64). v_c=vector<float,192> (192 VGPRs).

## Result (MI355X, our pin-enabled clang), NO __launch_bounds__
- baseline: compiler picks occ4 (128-VGPR budget) -> v_c(192) doesn't fit -> 247 spill.
- pin v_c -> VGPR: the SIPreColorPins pass records the pinned range (192 VGPRs) and caps
  waves-per-EU accordingly (MFI.setWavesPerEU) -> occupancy 2, 232 VGPR, 0 spill, all 48
  MFMAs write v[ (v_c in VGPR), VALID.  <-- the PIN drives occupancy; __launch_bounds__ removed.

## Notes
- Mechanism: pin pass computes max pinned reg -> getOccupancyWithNumVGPRs -> setWavesPerEU +
  limitOccupancy, so the register budget covers the pin.
- v_c is a loop-carried PHI, so the per-chunk pins are soft, but driving occupancy is enough
  to place the 192-reg accumulator in VGPR without spill.
- Pinning v_a/v_b -> AGPR *in addition* flips the compiler to the all-AGPR MFMA form
  (a[D],a[A],a[B]) so v_c also lands in AGPR. Keeping v_c in VGPR AND a/b in AGPR needs a
  HARD pin of the loop-carried PHI accumulator (not implemented).
