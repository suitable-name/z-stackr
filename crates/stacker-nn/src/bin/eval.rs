//! `stacker-nn-eval` — the inference side of `scripts/04_eval_alignment.py`.
//!
//! The script judges a trained alignment network against simulated misalignment
//! (`scripts/03_simulate_misalignment.py`) and shells out to this binary. Two modes:
//!
//! ## 1. Request on stdin (what the script calls)
//!
//! No arguments: read one JSON request from stdin, write one JSON response to stdout.
//!
//! ```json
//! { "model_path": "models/batchalign_final.mpk", "ref_index": 0,
//!   "output_dims": [3000, 2000], "refine": false,
//!   "frames": ["/abs/focus_00.png", "/abs/focus_01.png"] }
//! ```
//!
//! ```json
//! { "status": "ok", "matrices": [[[1,0,0],[0,1,0],[0,0,1]], [[...]], null],
//!   "seconds": 1.9, "refinement_fallbacks": 0 }
//! ```
//!
//! `matrices[k]` maps reference pixels to frame `k` pixels in the frames' own pixel
//! grid (`[x_k, y_k, 1]^T = M_k [x_ref, y_ref, 1]^T`, the convention of
//! `alignment_gt`), `null` for a frame that could not be aligned. With
//! `"refine": true` the network's matrices seed the classical registration refiner
//! (`stacker_align::refine::refine_alignment_registration`); with `false` they are
//! returned as predicted. Omit `model_path` (or send `null`) and the classical
//! baseline runs alone, seeded from the identity. `seconds` times the alignment
//! only, not the image decoding. A fatal failure answers
//! `{"status": "error", "message": "..."}` and exits 0.
//!
//! `model_path` is a weights file (`.mpk`), its manifest (`.json`), a directory
//! (its best alignment model) or the name of a model found in `models/`.
//!
//! ## 2. Direct evaluation
//!
//! ```text
//! stacker-nn-eval --scenes-dir DIR [--model PATH_OR_NAME] [--classical]
//!                 [--device cpu|gpu] [--json]
//! ```
//!
//! evaluates every scene under `DIR` and prints, per scene and method, the mean
//! corner error in pixels (the four image corners moved by the predicted and by
//! the ground-truth matrix, averaged over the non-reference frames), plus the same
//! number scaled to a 3000 px wide image, then mean / median / worst per method.
//! Methods: `neural` and `neural_refine` (with `--model`), `classical` (with
//! `--classical`). `--json` prints one JSON document instead of the table.

use std::{
    error::Error,
    path::{Path, PathBuf},
    time::Instant,
};

use serde::{Deserialize, Serialize};
use serde_json::json;
use stacker_align::{
    Matrix3,
    refine::{BoundedRefineOptions, refine_alignment_registration},
};
use stacker_core::{
    color::{linear_rgb_to_planar, srgb_decode_u8_table},
    image::PlanarImage,
};
use stacker_nn::{
    data::{AlignSequence, discover_align_scenes},
    discovery::{ModelEntry, default_model_dirs, discover_models},
    runtime::{InferDevice, LoadedAlignModel, available_devices},
};

/// The help text.
const USAGE: &str = "\
stacker-nn-eval
    read a JSON request on stdin, answer on stdout (see scripts/04_eval_alignment.py)
stacker-nn-eval --scenes-dir DIR [--model PATH_OR_NAME] [--classical]
                [--device cpu|gpu] [--json]
    evaluate every scene under DIR and print the corner errors";

/// A 3x3 matrix as it travels in JSON.
type Mat = [[f64; 3]; 3];

const IDENTITY: Mat = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// The request the Python script sends.
#[derive(Debug, Deserialize)]
struct Request {
    /// Weights file, manifest, directory or model name; absent = classical only.
    model_path: Option<String>,
    #[serde(default)]
    ref_index: usize,
    /// `[width, height]` the matrices are expressed in; checked against the frames.
    output_dims: Option<[u32; 2]>,
    #[serde(default)]
    refine: bool,
    frames: Vec<PathBuf>,
}

/// The response written to stdout.
#[derive(Debug, Serialize)]
struct Response {
    status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    matrices: Option<Vec<Option<Mat>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seconds: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    refinement_fallbacks: Option<usize>,
}

/// The three ways to produce matrices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Method {
    /// The network alone.
    Neural,
    /// The network's matrices seed the classical refiner.
    NeuralRefine,
    /// The classical refiner seeded from the identity.
    Classical,
}

impl Method {
    const fn name(self) -> &'static str {
        match self {
            Self::Neural => "neural",
            Self::NeuralRefine => "neural_refine",
            Self::Classical => "classical",
        }
    }
}

// ---------------------------------------------------------------------------
// Images and matrices
// ---------------------------------------------------------------------------

/// Decode an image file to the pipeline's linear-light planar representation.
fn load_planar(path: &Path) -> Result<PlanarImage<f32>, Box<dyn Error>> {
    let rgb = image::open(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?
        .into_rgb8();
    let table = srgb_decode_u8_table();
    let mut planar = PlanarImage::<f32>::new(rgb.width() as usize, rgb.height() as usize);
    for (i, p) in rgb.pixels().enumerate() {
        let linear = [
            table[usize::from(p[0])],
            table[usize::from(p[1])],
            table[usize::from(p[2])],
        ];
        [planar.luma[i], planar.chroma_a[i], planar.chroma_b[i]] = linear_rgb_to_planar(linear);
    }
    Ok(planar)
}

fn to_array(m: &Matrix3<f32>) -> Mat {
    let mut out = [[0.0; 3]; 3];
    for (r, row) in out.iter_mut().enumerate() {
        for (c, cell) in row.iter_mut().enumerate() {
            *cell = f64::from(m[(r, c)]);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Models and alignment
// ---------------------------------------------------------------------------

/// Find the alignment model named by `spec`: a weights or manifest file, a
/// directory (its best alignment model), or a name in the default model directories.
fn resolve_model(spec: &str) -> Result<ModelEntry, Box<dyn Error>> {
    let path = Path::new(spec);
    let alignment = |entries: Vec<ModelEntry>| -> Vec<ModelEntry> {
        entries
            .into_iter()
            .filter(ModelEntry::is_alignment)
            .collect()
    };

    if path.is_dir() {
        return alignment(discover_models(path)?)
            .into_iter()
            .next()
            .ok_or_else(|| format!("no alignment model in {spec}").into());
    }
    if path.is_file() {
        let dir = path.parent().unwrap_or_else(|| Path::new("."));
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or("model file name is not valid UTF-8")?;
        return alignment(discover_models(dir)?)
            .into_iter()
            .find(|e| e.weights_path.file_stem().and_then(|s| s.to_str()) == Some(stem))
            .ok_or_else(|| format!("{spec} is not a usable alignment model (manifest?)").into());
    }
    for dir in default_model_dirs() {
        if let Some(entry) = alignment(discover_models(&dir).unwrap_or_default())
            .into_iter()
            .find(|e| e.name == spec)
        {
            return Ok(entry);
        }
    }
    Err(format!("no alignment model '{spec}': not a file, a directory or a name in models/").into())
}

/// Parse `--device`; without it the first compiled-in device (CPU when present).
fn pick_device(name: Option<&str>) -> Result<InferDevice, Box<dyn Error>> {
    let available = available_devices();
    name.map_or_else(
        || {
            available
                .first()
                .copied()
                .ok_or_else(|| "no inference backend compiled in".into())
        },
        |label| {
            available
                .iter()
                .copied()
                .find(|d| d.label().eq_ignore_ascii_case(label))
                .ok_or_else(|| {
                    format!(
                        "device '{label}' is not available (have: {})",
                        available
                            .iter()
                            .map(|d| d.label())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                    .into()
                })
        },
    )
}

/// The matrices of one method for `frames` (frame 0 is the reference), and the
/// number of frames whose refinement failed and fell back to its seed.
fn align_frames(
    model: Option<&LoadedAlignModel>,
    method: Method,
    frames: &[PlanarImage<f32>],
) -> Result<(Vec<Mat>, usize), Box<dyn Error>> {
    let seeds: Vec<Matrix3<f32>> = match (method, model) {
        (Method::Classical, _) => vec![Matrix3::identity(); frames.len()],
        (_, Some(model)) => model
            .align(frames)?
            .into_iter()
            .map(|m| Matrix3::from_fn(|r, c| m[r][c]))
            .collect(),
        (_, None) => return Err("this method needs a model".into()),
    };
    if seeds.len() != frames.len() {
        return Err(format!(
            "the model returned {} matrices for {} frames",
            seeds.len(),
            frames.len()
        )
        .into());
    }
    if method == Method::Neural {
        let mut out: Vec<Mat> = seeds.iter().map(to_array).collect();
        out[0] = IDENTITY;
        return Ok((out, 0));
    }

    let mut fallbacks = 0;
    let mut out = vec![IDENTITY];
    for (frame, seed) in frames.iter().zip(&seeds).skip(1) {
        let refined = refine_alignment_registration(
            &frames[0],
            frame,
            seed,
            &BoundedRefineOptions::default(),
        )
        .unwrap_or_else(|_| {
            fallbacks += 1;
            *seed
        });
        out.push(to_array(&refined));
    }
    Ok((out, fallbacks))
}

// ---------------------------------------------------------------------------
// Mode 1: JSON request on stdin
// ---------------------------------------------------------------------------

/// Answer one request.
fn handle(request: &Request, device: Option<&str>) -> Result<Response, Box<dyn Error>> {
    if request.frames.len() < 2 {
        return Err("need at least a reference and one more frame".into());
    }
    if request.ref_index >= request.frames.len() {
        return Err(format!("ref_index {} is out of range", request.ref_index).into());
    }
    // The network treats the first frame as the reference: move the requested one to
    // the front, then undo that when the matrices are returned.
    let mut order: Vec<usize> = (0..request.frames.len()).collect();
    order.swap(0, request.ref_index);
    let frames = order
        .iter()
        .map(|&i| load_planar(&request.frames[i]))
        .collect::<Result<Vec<_>, _>>()?;
    if let Some([w, h]) = request.output_dims
        && (frames[0].width, frames[0].height) != (w as usize, h as usize)
    {
        return Err(format!(
            "output_dims {w} x {h} does not match the frames ({} x {})",
            frames[0].width, frames[0].height
        )
        .into());
    }

    let (model, method) = match (&request.model_path, request.refine) {
        (None, _) => (None, Method::Classical),
        (Some(spec), refine) => {
            let entry = resolve_model(spec)?;
            let loaded = LoadedAlignModel::load(&entry, pick_device(device)?)?;
            (
                Some(loaded),
                if refine {
                    Method::NeuralRefine
                } else {
                    Method::Neural
                },
            )
        }
    };

    let start = Instant::now();
    let (aligned, fallbacks) = align_frames(model.as_ref(), method, &frames)?;
    let seconds = start.elapsed().as_secs_f64();

    let mut matrices: Vec<Option<Mat>> = vec![None; request.frames.len()];
    for (position, matrix) in order.into_iter().zip(aligned) {
        matrices[position] = Some(matrix);
    }
    Ok(Response {
        status: "ok",
        message: None,
        matrices: Some(matrices),
        seconds: Some(seconds),
        refinement_fallbacks: Some(fallbacks),
    })
}

fn stdin_mode(device: Option<&str>) -> Result<(), Box<dyn Error>> {
    let request: Request = serde_json::from_reader(std::io::stdin().lock())
        .map_err(|e| format!("cannot parse the request on stdin: {e}"))?;
    let response = handle(&request, device).unwrap_or_else(|e| Response {
        status: "error",
        message: Some(e.to_string()),
        matrices: None,
        seconds: None,
        refinement_fallbacks: None,
    });
    println!("{}", serde_json::to_string(&response)?);
    Ok(())
}

// ---------------------------------------------------------------------------
// Mode 2: direct evaluation
// ---------------------------------------------------------------------------

/// Mean distance between the four image corners moved by `pred` and by `gt`.
fn corner_error(pred: &Mat, gt: &[[f32; 3]; 3], width: u32, height: u32) -> f64 {
    let (w, h) = (f64::from(width) - 1.0, f64::from(height) - 1.0);
    let apply = |m: &Mat, x: f64, y: f64| {
        let z = m[2][0].mul_add(x, m[2][1].mul_add(y, m[2][2]));
        (
            m[0][0].mul_add(x, m[0][1].mul_add(y, m[0][2])) / z,
            m[1][0].mul_add(x, m[1][1].mul_add(y, m[1][2])) / z,
        )
    };
    let gt_mat: Mat = gt.map(|row| row.map(f64::from));
    let corners = [(0.0, 0.0), (w, 0.0), (0.0, h), (w, h)];
    let total: f64 = corners
        .iter()
        .map(|&(x, y)| {
            let (px, py) = apply(pred, x, y);
            let (gx, gy) = apply(&gt_mat, x, y);
            (px - gx).hypot(py - gy)
        })
        .sum();
    total / 4.0
}

/// Mean over the non-reference frames of a scene.
fn scene_error(scene: &AlignSequence, matrices: &[Mat]) -> f64 {
    let [w, h] = scene.meta.cropped_dims;
    let errors: Vec<f64> = matrices
        .iter()
        .zip(&scene.meta.alignment_gt)
        .skip(1)
        .map(|(pred, gt)| corner_error(pred, gt, w, h))
        .collect();
    #[expect(
        clippy::cast_precision_loss,
        reason = "a frame count is far below 2^52, so the conversion to f64 is exact"
    )]
    let n = errors.len() as f64;
    errors.iter().sum::<f64>() / n.max(1.0)
}

/// One scene's result for one method.
struct SceneResult {
    scene: String,
    method: Method,
    raw_px: f64,
    norm_3000_px: f64,
    seconds: f64,
    fallbacks: usize,
}

const fn median(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n.is_multiple_of(2) {
        sorted[n / 2 - 1].midpoint(sorted[n / 2])
    } else {
        sorted[n / 2]
    }
}

struct DirectArgs {
    scenes_dir: PathBuf,
    model: Option<String>,
    classical: bool,
    device: Option<String>,
    json: bool,
}

fn parse_direct(words: &[String]) -> Result<DirectArgs, Box<dyn Error>> {
    let mut parsed = DirectArgs {
        scenes_dir: PathBuf::new(),
        model: None,
        classical: false,
        device: None,
        json: false,
    };
    let mut have_dir = false;
    let mut i = 0;
    while i < words.len() {
        let value = |i: usize| -> Result<String, Box<dyn Error>> {
            words
                .get(i + 1)
                .cloned()
                .ok_or_else(|| format!("missing value for {}", words[i]).into())
        };
        match words[i].as_str() {
            "--classical" => parsed.classical = true,
            "--json" => parsed.json = true,
            "--scenes-dir" => {
                parsed.scenes_dir = PathBuf::from(value(i)?);
                have_dir = true;
                i += 1;
            }
            "--model" => {
                parsed.model = Some(value(i)?);
                i += 1;
            }
            "--device" => {
                parsed.device = Some(value(i)?);
                i += 1;
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
        i += 1;
    }
    if !have_dir {
        return Err("--scenes-dir <dir> is required".into());
    }
    if parsed.model.is_none() && !parsed.classical {
        return Err("give --model <path_or_name>, --classical, or both".into());
    }
    Ok(parsed)
}

fn direct_mode(words: &[String]) -> Result<(), Box<dyn Error>> {
    let args = parse_direct(words)?;
    let scenes = discover_align_scenes(&args.scenes_dir)?;
    if scenes.is_empty() {
        return Err(format!("no scenes under {}", args.scenes_dir.display()).into());
    }
    let model = match &args.model {
        Some(spec) => Some(LoadedAlignModel::load(
            &resolve_model(spec)?,
            pick_device(args.device.as_deref())?,
        )?),
        None => None,
    };
    let mut methods = Vec::new();
    if model.is_some() {
        methods.extend([Method::Neural, Method::NeuralRefine]);
    }
    if args.classical {
        methods.push(Method::Classical);
    }

    let mut results = Vec::new();
    for scene in &scenes {
        let frames = scene
            .meta
            .stack
            .iter()
            .map(|name| load_planar(&scene.scene_dir.join(name)))
            .collect::<Result<Vec<_>, _>>()?;
        if frames.len() < 2 {
            continue;
        }
        let name = scene
            .scene_dir
            .file_name()
            .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
        for &method in &methods {
            let start = Instant::now();
            let (matrices, fallbacks) = align_frames(model.as_ref(), method, &frames)?;
            let seconds = start.elapsed().as_secs_f64();
            let raw_px = scene_error(scene, &matrices);
            results.push(SceneResult {
                scene: name.clone(),
                method,
                raw_px,
                norm_3000_px: raw_px * 3000.0 / f64::from(scene.meta.cropped_dims[0]),
                seconds,
                fallbacks,
            });
        }
    }
    report(&results, &methods, args.json)
}

fn report(
    results: &[SceneResult],
    methods: &[Method],
    as_json: bool,
) -> Result<(), Box<dyn Error>> {
    let summary: Vec<(Method, f64, f64, f64)> = methods
        .iter()
        .map(|&m| {
            let mut norm: Vec<f64> = results
                .iter()
                .filter(|r| r.method == m)
                .map(|r| r.norm_3000_px)
                .collect();
            norm.sort_by(f64::total_cmp);
            #[expect(
                clippy::cast_precision_loss,
                reason = "a scene count is far below 2^52, so the conversion to f64 is exact"
            )]
            let mean = norm.iter().sum::<f64>() / (norm.len().max(1) as f64);
            (
                m,
                mean,
                median(&norm),
                norm.last().copied().unwrap_or(f64::NAN),
            )
        })
        .collect();

    if as_json {
        let doc = json!({
            "scenes": results.iter().map(|r| json!({
                "scene": r.scene,
                "method": r.method.name(),
                "corner_error_px": r.raw_px,
                "corner_error_3000px": r.norm_3000_px,
                "seconds": r.seconds,
                "refinement_fallbacks": r.fallbacks,
            })).collect::<Vec<_>>(),
            "summary": summary.iter().map(|(m, mean, med, worst)| json!({
                "method": m.name(),
                "mean_3000px": mean,
                "median_3000px": med,
                "worst_3000px": worst,
            })).collect::<Vec<_>>(),
        });
        println!("{}", serde_json::to_string_pretty(&doc)?);
        return Ok(());
    }

    println!(
        "{:<28} {:<14} {:>10} {:>12} {:>9}",
        "scene", "method", "px", "px@3000", "seconds"
    );
    for r in results {
        println!(
            "{:<28} {:<14} {:>10.3} {:>12.3} {:>9.2}",
            r.scene,
            r.method.name(),
            r.raw_px,
            r.norm_3000_px,
            r.seconds
        );
    }
    println!();
    println!(
        "{:<14} {:>10} {:>10} {:>10}   (px at 3000 px width)",
        "method", "mean", "median", "worst"
    );
    for (m, mean, med, worst) in summary {
        println!("{:<14} {mean:>10.3} {med:>10.3} {worst:>10.3}", m.name());
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.iter().any(|a| a == "-h" || a == "--help") {
        eprintln!("{USAGE}");
        return Ok(());
    }
    if argv.is_empty() {
        return stdin_mode(None);
    }
    direct_mode(&argv)
}
