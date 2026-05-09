"""Update _batchfold_forward to use fused kernel with SKIP_LIF."""
with open('catfuse/sparseflow/ops/st_fusion_conv_bn_lif.py', 'r') as f:
    code = f.read()

# Replace entire _batchfold_forward
old = """    def _batchfold_forward(self, x):"""
# Find the end of the method (next def or end)
idx_start = code.index(old)
# Find the next method definition after _batchfold_forward
idx_next = code.index("    def forward(", idx_start + 1)

old_method = code[idx_start:idx_next]

new_method = '''    def _batchfold_forward(self, x):
        """BatchFolded NCHW fused kernel (SKIP_LIF) + sequential LIF.

        1. BatchFold: [T,B,C,H,W] -> [T*B,C,H,W]
        2. 1x prescan_v2 (NCHW, 1 launch)
        3. 1x fused kernel with SKIP_LIF=True (NCHW, 1 launch) -> z[T*B,...]
        4. Reshape -> [T,B,C_out,H,W]
        5. LIF loop over T (element-wise, fast)

        No NHWC permute needed!
        """
        T, B = x.shape[0], x.shape[1]
        c = self._lean_cache
        device = x.device

        # 1. BatchFold
        x_flat = x.reshape(T * B, self.in_channels, x.shape[3], x.shape[4])

        # Recompute grid for T*B batch
        H_out, W_out = c['H_out'], c['W_out']
        BH, BW = c['BH'], c['BW']
        GH = triton.cdiv(H_out, BH)
        GW = triton.cdiv(W_out, BW)
        N_TILES = T * B * GH * GW

        # Ensure buffers
        if self._ag_mask_buf is None or self._ag_mask_buf.numel() < N_TILES:
            self._ag_mask_buf = torch.empty(N_TILES, dtype=torch.int32, device=device)
        if self._tile_class_buf is None or self._tile_class_buf.numel() < N_TILES:
            self._tile_class_buf = torch.empty(N_TILES, dtype=torch.int32, device=device)

        # 2. Prescan (1 launch, NCHW)
        fast_spike_prescan_2d_v2(
            x_flat, H_out, W_out,
            kernel_size=self.kernel_size,
            stride=self.stride, padding=self.padding,
            block_h=BH, block_w=BW,
            group_size_c=c['GSC'],
            ag_mask_out=self._ag_mask_buf,
            tile_class_out=self._tile_class_buf,
        )

        # 3. Fused kernel with SKIP_LIF (1 launch, NCHW, writes z not spikes)
        x_f16 = x_flat.half()
        z_buf = torch.empty(T * B, self.out_channels, H_out, W_out,
                            dtype=torch.float32, device=device)
        v_dummy = torch.empty(1, dtype=torch.float32, device=device)

        def _grid(META):
            return (N_TILES, triton.cdiv(self.out_channels, META["BLOCK_N"]))

        c['kernel'][_grid](
            x_f16,
            self._w_cl,
            c['bias_arg'],
            c['bn_scale_arg'],
            c['bn_bias_arg'],
            self._ag_mask_buf,
            v_dummy,        # v_prev not used when SKIP_LIF
            z_buf,          # spike_ptr becomes z output
            v_dummy,        # v_next not used when SKIP_LIF
            T * B,
            self.in_channels, self.out_channels,
            H_out, W_out, GH, GW,
            HAS_BIAS=c['has_bias'],
            HAS_BN=c['has_bn'],
            SKIP_LIF=True,
            DECAY=c['decay'],
            RECIP_TAU=c['recip_tau'],
            V_TH=float(self.v_threshold),
            HAS_V_RESET=c['has_v_reset'],
            V_RESET=c['v_reset_val'],
            GROUP_SIZE_C=c['GSC'],
            NUM_GROUPS=c['NUM_GROUPS'],
        )

        # 4. Reshape
        z = z_buf.reshape(T, B, self.out_channels, H_out, W_out)

        # 5. LIF loop (element-wise, very fast)
        v = self.v
        v_reset_val = 0.0 if self.v_reset is None else float(self.v_reset)
        decay = c['decay']
        recip_tau = c['recip_tau']

        spikes = []
        for t in range(T):
            v = v * decay + z[t] * recip_tau + v_reset_val * recip_tau
            spike = (v >= self.v_threshold).float()
            if self.v_reset is not None:
                v = v * (1.0 - spike) + v_reset_val * spike
            else:
                v = v - spike * self.v_threshold
            spikes.append(spike)

        self.v = v
        return torch.stack(spikes, dim=0)

'''

code = code[:idx_start] + new_method + code[idx_next:]

with open('catfuse/sparseflow/ops/st_fusion_conv_bn_lif.py', 'w') as f:
    f.write(code)
print("Done. _batchfold_forward updated to use fused kernel + SKIP_LIF")
