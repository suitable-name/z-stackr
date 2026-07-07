use stacker_core::image::PlanarImage;

#[test]
fn test_planar_image_new() {
    let img = PlanarImage::<f32>::new(10, 20);
    assert_eq!(img.width, 10);
    assert_eq!(img.height, 20);
    assert_eq!(img.luma.len(), 200);
    assert_eq!(img.chroma_a.len(), 200);
    assert_eq!(img.chroma_b.len(), 200);
}

// #[test]
// fn test_luma_chunks_mut() {
//     let mut img = PlanarImage::<f32>::new(4, 4);

//     // Test mutability and parallel chunks
//     img.luma_chunks_mut(4).enumerate().for_each(|(i, chunk)| {
//         for (j, v) in chunk.iter_mut().enumerate() {
//             *v = (i * 4 + j) as f32;
//         }
//     });

//     for i in 0..16 {
//         assert_eq!(img.luma[i], i as f32);
//     }
// }
