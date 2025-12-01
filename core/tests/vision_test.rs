use xla_rs::models::resnet::ResNet;
use xla_rs::models::vit::ViT;
use xla_rs::nn::{Conv2d, MaxPool2d};
use xla_rs::tensor::{Cpu, Tensor};

#[test]
fn test_conv2d_shapes() {
    // Input: [B, 3, 32, 32]
    let input = Tensor::<f32, 4, Cpu>::zeros([1, 3, 32, 32]);

    // Conv: 3->16, 3x3, stride 1, pad 1
    let conv = Conv2d::new(3, 16, 3, 1, 1);
    let out = conv.forward(&input).unwrap();

    assert_eq!(out.shape(), &[1, 16, 32, 32]);

    // Conv: 3->16, 3x3, stride 2, pad 1
    let conv_strided = Conv2d::new(3, 16, 3, 2, 1);
    let out_strided = conv_strided.forward(&input).unwrap();

    assert_eq!(out_strided.shape(), &[1, 16, 16, 16]);
}

#[test]
fn test_maxpool2d_shapes() {
    // Input: [B, 16, 32, 32]
    let input = Tensor::<f32, 4, Cpu>::zeros([1, 16, 32, 32]);

    // Pool: 2x2, stride 2
    let pool = MaxPool2d::new(2, 2, 0);
    let out = pool.forward(&input).unwrap();

    assert_eq!(out.shape(), &[1, 16, 16, 16]);
}

#[test]
fn test_resnet18_forward() {
    // ResNet18 for ImageNet (224x224)
    let resnet = ResNet::<f32>::new(1000); // 1000 classes

    // Input: [1, 3, 224, 224]
    let input = Tensor::<f32, 4, Cpu>::zeros([1, 3, 224, 224]);
    let out = resnet.forward(&input).unwrap();

    assert_eq!(out.shape(), &[1, 1000]);
}

#[test]
fn test_vit_base_forward() {
    // ViT Base for ImageNet (224x224)
    // Use a smaller depth/embed_dim for speed in test if needed, but let's try full base.
    // ViT Base: Embed 768, Depth 12, Heads 12.
    // Might be slow on CPU debug build.
    // Let's use a "ViT Tiny" for testing shapes.

    let vit = ViT::<f32>::new(
        224,  // img_size
        16,   // patch_size
        3,    // in_channels
        1000, // num_classes
        64,   // embed_dim (small)
        2,    // depth (small)
        2,    // heads
        2,    // mlp_ratio
    );

    let input = Tensor::<f32, 4, Cpu>::zeros([1, 3, 224, 224]);
    let out = vit.forward(&input).unwrap();

    assert_eq!(out.shape(), &[1, 1000]);
}
