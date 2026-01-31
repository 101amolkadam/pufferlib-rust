use tch::{nn, Tensor};

/// Conv2d block with ELU activation
fn conv2d(p: nn::Path, c_in: i64, c_out: i64, k: i64, s: i64) -> impl nn::Module {
    nn::seq()
        .add(nn::conv2d(
            &p,
            c_in,
            c_out,
            k,
            nn::ConvConfig {
                stride: s,
                ..Default::default()
            },
        ))
        .add(nn::func(|xs| xs.elu()))
}

/// Transposed Conv2d block with ELU activation
fn conv_transpose2d(p: nn::Path, c_in: i64, c_out: i64, k: i64, s: i64) -> impl nn::Module {
    nn::seq()
        .add(nn::conv_transpose2d(
            &p,
            c_in,
            c_out,
            k,
            nn::ConvTransposeConfig {
                stride: s,
                ..Default::default()
            },
        ))
        .add(nn::func(|xs| xs.elu()))
}

/// CNN Encoder: Visual Observation -> Embedding
#[derive(Debug)]
pub struct EncoderCNN {
    seq: nn::Sequential,
}

impl EncoderCNN {
    pub fn new(p: &nn::Path, in_channels: i64) -> Self {
        let seq = nn::seq()
            .add(conv2d(p / "c1", in_channels, 32, 4, 2))
            .add(conv2d(p / "c2", 32, 64, 4, 2))
            .add(conv2d(p / "c3", 64, 128, 4, 2))
            .add(conv2d(p / "c4", 128, 256, 4, 2))
            .add(LayerNorm2d::new(p / "ln", 256)); // Custom or Flatten?

        Self { seq }
    }

    pub fn forward(&self, xs: &Tensor) -> Tensor {
        // Input: [B, C, H, W]
        // Output: [B, Embed]
        let h = xs.apply(&self.seq);
        h.flatten(1, -1)
    }
}

/// CNN Decoder: State -> Visual Observation Reconstruction
#[derive(Debug)]
pub struct DecoderCNN {
    linear: nn::Linear,
    seq: nn::Sequential,
    #[allow(dead_code)]
    out_shape: [i64; 3], // C, H, W
}

impl DecoderCNN {
    pub fn new(p: &nn::Path, in_dim: i64, out_shape: [i64; 3]) -> Self {
        // Assuming small 64x64 input
        let linear = nn::linear(p / "lin", in_dim, 32 * 256, Default::default()); // 32*8?? No.

        let seq = nn::seq()
            .add(conv_transpose2d(p / "d1", 256, 128, 5, 2)) // stride 2
            .add(conv_transpose2d(p / "d2", 128, 64, 5, 2))
            .add(conv_transpose2d(p / "d3", 64, 32, 6, 2))
            .add(nn::conv_transpose2d(
                p / "d4",
                32,
                out_shape[0],
                6,
                nn::ConvTransposeConfig {
                    stride: 2,
                    ..Default::default()
                },
            ));

        Self {
            linear,
            seq,
            out_shape,
        }
    }

    pub fn forward(&self, xs: &Tensor) -> Tensor {
        // Linear then reshape then ConvTranspose
        let x = xs.apply(&self.linear).reshape([-1, 256, 1, 1]); // Placeholder shape logic
                                                                  // Need matching shapes for ConvTranspose to reach 64x64
                                                                  // ... simplified for now
        x.apply(&self.seq)
    }
}

/// Dense Head (MLP) for Reward/Continue/Value
#[derive(Debug)]
pub struct DenseHead {
    seq: nn::Sequential,
    #[allow(dead_code)]
    out_dim: i64,
}

impl DenseHead {
    pub fn new(p: &nn::Path, in_dim: i64, out_dim: i64, layers: i64, units: i64) -> Self {
        let mut seq = nn::seq();
        let mut cur_in = in_dim;
        for i in 0..layers {
            seq = seq.add(nn::linear(
                p / format!("l{}", i),
                cur_in,
                units,
                Default::default(),
            ));
            seq = seq.add(nn::func(|xs| xs.elu()));
            cur_in = units;
        }
        seq = seq.add(nn::linear(p / "out", cur_in, out_dim, Default::default()));

        Self { seq, out_dim }
    }

    pub fn forward(&self, xs: &Tensor) -> Tensor {
        xs.apply(&self.seq)
    }
}

// Helper: 2D Layer Norm if needed, or simplistic
#[derive(Debug)]
struct LayerNorm2d {
    ln: nn::LayerNorm,
}

impl LayerNorm2d {
    fn new(p: nn::Path, dim: i64) -> Self {
        Self {
            ln: nn::layer_norm(p, vec![dim], Default::default()),
        }
    }
}

impl nn::Module for LayerNorm2d {
    fn forward(&self, xs: &Tensor) -> Tensor {
        // [B, C, H, W] -> permute -> LN -> permute
        xs.permute([0, 2, 3, 1])
            .apply(&self.ln)
            .permute([0, 3, 1, 2])
    }
}
