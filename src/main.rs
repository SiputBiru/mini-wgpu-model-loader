use anyhow::{Context, Result};
use bytemuck::checked::pod_read_unaligned;
use std::{collections::HashMap, path::Path, sync::Arc, time::Instant};
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::ActiveEventLoop,
    window::{Window, WindowId},
};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
}

impl Vertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 3] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2];

    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

struct MeshData {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    uvs: Vec<[f32; 2]>,
    indices: Vec<u32>,
}

impl MeshData {
    fn vertex_count(&self) -> u32 {
        self.positions.len() as u32
    }
    fn index_count(&self) -> u32 {
        self.indices.len() as u32
    }

    fn to_vertices(&self) -> Vec<Vertex> {
        (0..self.positions.len())
            .map(|i| Vertex {
                position: self.positions[i],
                normal: self.normals.get(i).copied().unwrap_or([0.0, 1.0, 0.0]),
                uv: self.uvs.get(i).copied().unwrap_or([0.0, 0.0]),
            })
            .collect()
    }

    fn cube(size: f32) -> MeshData {
        let h = size * 0.5;

        // (position, normal) pairs: Uvs omitted for brevity -> defaulted
        let faces: [([f32; 3], [f32; 3]); 6] = [
            ([0.0, 0.0, h], [0.0, 0.0, 1.0]),
            ([0.0, 0.0, -h], [0.0, 0.0, -1.0]),
            ([h, 0.0, 0.0], [1.0, 0.0, 0.0]),
            ([-h, 0.0, 0.0], [-1.0, 0.0, 0.0]),
            ([0.0, h, 0.0], [0.0, 1.0, 0.0]),
            ([0.0, -h, 0.0], [0.0, -1.0, 0.0]),
        ];
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut indices = Vec::new();

        for (center, normal) in faces {
            let base = positions.len() as u32;

            // two tangen vectors perpendicular to the face normal
            let t = if normal[1].abs() > 0.5 {
                [1.0, 0.0, 0.0]
            } else {
                [0.0, 1.0, 0.0]
            };
            let b = cross(normal, t);

            for (su, sv) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                positions.push([
                    center[0] + (t[0] * su + b[0] * sv) * h,
                    center[1] + (t[1] * su + b[1] * sv) * h,
                    center[2] + (t[2] * su + b[2] * sv) * h,
                ]);
                normals.push(normal);
            }

            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }

        Self {
            positions,
            normals,
            uvs: Vec::new(),
            indices,
        }
    }
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[derive(Clone, Debug)]
struct MaterialData {
    name: Option<String>,
    base_color: [f32; 4],
    base_color_texture: Option<usize>,
    metallic: f32,
    roughness: f32,
    double_sided: bool,
}

#[derive(Clone, Debug)]
struct ImageData {
    name: Option<String>,
    rgba: Vec<u8>,
    width: u32,
    height: u32,
}

struct MeshPrimitive {
    mesh: MeshData,
    material: Option<usize>,
}

struct LoadedModel {
    meshes: Vec<MeshPrimitive>,
    materials: Vec<MaterialData>,
    images: Vec<ImageData>,

    // gltf texture index -> image_index, build from 'doc.textures()'
    texture_to_image: Vec<usize>,
}

fn face_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let n = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-8);
    [n[0] / len, n[1] / len, n[2] / len]
}

fn compute_flat_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut normals = vec![[0.0f32; 3]; positions.len()];

    for tri in indices.chunks_exact(3) {
        let (a, b, c) = (
            positions[tri[0] as usize],
            positions[tri[1] as usize],
            positions[tri[2] as usize],
        );
        let n = face_normal(a, b, c);
        for &i in tri {
            let slot = &mut normals[i as usize];
            slot[0] += n[0];
            slot[1] += n[1];
            slot[2] += n[2];
        }
    }

    for n in &mut normals {
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-8);
        n[0] /= len;
        n[1] /= len;
        n[2] /= len;
    }
    normals
}

fn read_primitive(
    primitive: &gltf::Primitive<'_>,
    buffers: &[gltf::buffer::Data],
) -> Result<Option<MeshData>> {
    if primitive.mode() != gltf::mesh::Mode::Triangles {
        eprintln!(
            "skipping primitive with mode {:?} (only TriangleList supported)",
            primitive.mode()
        );
        return Ok(None);
    };

    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

    let positions: Vec<[f32; 3]> = reader
        .read_positions()
        .context("primitive has no POSITION accessor")?
        .collect();

    let vertex_count = positions.len() as u32;

    let normals: Vec<[f32; 3]> = match reader.read_normals() {
        Some(iter) => iter.collect(),

        // no normals: compule flat normals per triangle below.
        None => Vec::new(),
    };

    let uvs: Vec<[f32; 2]> = match reader.read_tex_coords(0) {
        Some(iter) => iter.into_f32().collect(),
        None => Vec::new(),
    };

    let indices: Vec<u32> = match reader.read_indices() {
        Some(iter) => iter.into_u32().collect(),
        None => (0..vertex_count).collect(),
    };

    let normals = if normals.len() == positions.len() {
        normals
    } else {
        compute_flat_normals(&positions, &indices)
    };

    Ok(Some(MeshData {
        positions,
        normals,
        uvs,
        indices,
    }))
}

fn load_gltf(path: impl AsRef<Path>) -> Result<LoadedModel> {
    let path = path.as_ref();
    let (doc, buffers, _gltf_images) =
        gltf::import(path).with_context(|| format!("importing {}", path.display()))?;

    let mut materials = Vec::new();
    for material in doc.materials() {
        let pbr = material.pbr_metallic_roughness();
        let base = pbr.base_color_factor();

        materials.push(MaterialData {
            name: material.name().map(str::to_owned),
            base_color: base,
            base_color_texture: pbr.base_color_texture().map(|info| info.texture().index()),
            metallic: pbr.metallic_factor(),
            roughness: pbr.roughness_factor(),
            double_sided: material.double_sided(),
        });
    }
    // let images = gltf_images.iter().map(convert_image).collect();

    let mut meshes = Vec::new();

    for gltf_mesh in doc.meshes() {
        for primitive in gltf_mesh.primitives() {
            if let Some(mesh) = read_primitive(&primitive, &buffers)? {
                meshes.push(MeshPrimitive {
                    mesh,
                    material: primitive.material().index(),
                });
            }
        }
    }

    let texture_to_image = doc.textures().map(|t| t.source().index()).collect();

    Ok(LoadedModel {
        meshes,
        materials,
        images: Vec::new(),
        texture_to_image,
    })
}

#[derive(Clone)]
struct DrawItem {
    /// index into LoadedModel::meshes
    mesh_index: usize,
    /// World transform for this primitive
    model: Mat4,
    /// index into LoadedModel::materials, if the primitive had one.
    material: Option<usize>,
    /// Copied from the gltf material so the draw loop needs no CPU lookup
    double_sided: bool,
}

#[derive(Clone)]
struct Bounds {
    min: Vec3,
    max: Vec3,
}

impl Bounds {
    fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }
    fn radius(&self) -> f32 {
        (self.max - self.min).length() * 0.5
    }
}

struct LoadedScene {
    model: LoadedModel,
    draws: Vec<DrawItem>,
    bounds: Bounds,
}

fn load_scene(path: impl AsRef<std::path::Path>) -> anyhow::Result<LoadedScene> {
    let path = path.as_ref();
    let (doc, buffers, gltf_images) = gltf::import(path)?;

    let (model, lookup) = flatten(&doc, &buffers, &gltf_images)?;

    let mut draws = Vec::new();

    for scene in doc.scenes() {
        for node in scene.nodes() {
            visit_node(&node, Mat4::IDENTITY, &lookup, &mut draws);
        }
    }

    if draws.is_empty() {
        // no scene listed: fall back to every mesh with an identitiy transform
        for (i, prim) in model.meshes.iter().enumerate() {
            let double_sided = prim
                .material
                .and_then(|m| model.materials.get(m))
                .map(|m| m.double_sided)
                .unwrap_or(false);

            draws.push(DrawItem {
                mesh_index: i,
                model: Mat4::IDENTITY,
                material: prim.material,
                double_sided,
            });
        }
    }

    let bounds = compute_bounds(&model.meshes, &draws);

    Ok(LoadedScene {
        model,
        draws,
        bounds,
    })
}

fn build_materials(doc: &gltf::Document) -> Vec<MaterialData> {
    doc.materials()
        .map(|material| {
            let pbr = material.pbr_metallic_roughness();
            MaterialData {
                name: material.name().map(str::to_owned),
                base_color: pbr.base_color_factor(),
                base_color_texture: pbr.base_color_texture().map(|info| info.texture().index()),
                metallic: pbr.metallic_factor(),
                roughness: pbr.roughness_factor(),
                double_sided: material.double_sided(),
            }
        })
        .collect()
}

fn compute_bounds(primitives: &[MeshPrimitive], draws: &[DrawItem]) -> Bounds {
    let mut min = glam::Vec3::splat(f32::INFINITY);
    let mut max = glam::Vec3::splat(f32::NEG_INFINITY);

    for draw in draws {
        let prim = &primitives[draw.mesh_index];
        for p in &prim.mesh.positions {
            let w = draw.model.transform_point3(glam::Vec3::from_array(*p));
            min = min.min(w);
            max = max.max(w);
        }
    }

    if !min.is_finite() || !max.is_finite() {
        min = glam::Vec3::splat(-1.0);
        max = glam::Vec3::splat(1.0);
    }
    Bounds { min, max }
}

fn visit_node(
    node: &gltf::Node<'_>,
    parent: Mat4,
    lookup: &std::collections::HashMap<(usize, usize), usize>,
    draws: &mut Vec<DrawItem>,
) {
    let local = Mat4::from_cols_array_2d(&node.transform().matrix());
    let world = parent * local;

    if let Some(mesh) = node.mesh() {
        for primitive in mesh.primitives() {
            let Some(&flat) = lookup.get(&(mesh.index(), primitive.index())) else {
                continue; // primitive was skipped (not TriangleList, or no
                // POSITION)
            };

            draws.push(DrawItem {
                mesh_index: flat,
                model: world,
                material: primitive.material().index(),
                double_sided: primitive.material().double_sided(),
            });
        }
    }

    for child in node.children() {
        visit_node(&child, world, lookup, draws);
    }
}

fn flatten(
    doc: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    gltf_images: &[gltf::image::Data],
) -> anyhow::Result<(LoadedModel, HashMap<(usize, usize), usize>)> {
    let mut model = LoadedModel {
        meshes: Vec::new(),
        materials: build_materials(doc),
        images: gltf_images.iter().map(convert_image).collect(),
        texture_to_image: doc.textures().map(|t| t.source().index()).collect(),
    };

    let mut lookup = HashMap::new();

    for gltf_mesh in doc.meshes() {
        for primitive in gltf_mesh.primitives() {
            if let Some(mesh) = read_primitive(&primitive, buffers)? {
                let flat = model.meshes.len();
                lookup.insert((gltf_mesh.index(), primitive.index()), flat);
                model.meshes.push(MeshPrimitive {
                    mesh,
                    material: primitive.material().index(),
                });
            }
        }
    }

    Ok((model, lookup))
}

struct GpuMesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    index_format: wgpu::IndexFormat,
}

impl GpuMesh {
    fn upload(device: &wgpu::Device, data: &MeshData) -> Self {
        let vertices = data.to_vertices();

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        // pad so the u32 slice is non-empty even for degenerate meshes
        let indices = if data.indices.is_empty() {
            vec![0u32, 0, 0]
        } else {
            data.indices.clone()
        };

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Self {
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
            index_format: wgpu::IndexFormat::Uint32,
        }
    }

    fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_index_buffer(self.index_buffer.slice(..), self.index_format);
        pass.draw_indexed(0..self.index_count, 0, 0..1);
    }
}

// gpu model stuff
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuMaterialUniform {
    base_color: [f32; 4],
    metallic: f32,
    roughness: f32,
    has_texture: u32,
    _pad: f32,
}

struct DrawBinding {
    buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

struct GpuModel {
    meshes: Vec<GpuMesh>,
    materials: Vec<wgpu::BindGroup>,
    material_uniforms: Vec<wgpu::Buffer>,
    textures: Vec<wgpu::Texture>, // kept alive
    draws: Vec<DrawItem>,
    bounds: Bounds,
    draw_bindings: Vec<DrawBinding>,
    default_material: usize,
}

fn white_texture(device: &wgpu::Device, queue: &wgpu::Queue) -> (wgpu::Texture, wgpu::TextureView) {
    let image = ImageData {
        name: Some("white".into()),
        rgba: vec![255, 255, 255, 255],
        width: 1,
        height: 1,
    };
    upload_texture(device, queue, &image, true, "white 1x1")
}

fn texture_index_to_image(model: &LoadedModel, texture_index: usize) -> Option<usize> {
    model.texture_to_image.get(texture_index).copied()
}

fn upload_scene(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    scene: &LoadedScene,
    material_layout: &wgpu::BindGroupLayout,
    object_layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
) -> GpuModel {
    // geometry
    let meshes: Vec<GpuMesh> = scene
        .model
        .meshes
        .iter()
        .map(|p| GpuMesh::upload(device, &p.mesh))
        .collect();

    // Texture (one per image: sRGB assumed for base colour)
    let mut textures = Vec::new();
    let mut views = Vec::new();

    for (i, image) in scene.model.images.iter().enumerate() {
        let (tex, view) = upload_texture(device, queue, image, true, &format!("texture {i}"));
        textures.push(tex);
        views.push(view);
    }

    let (white_tex, white_view) = white_texture(device, queue);
    textures.push(white_tex);

    // Materials
    let mut material_uniforms = Vec::new();
    let mut materials = Vec::new();

    for (i, mat) in scene.model.materials.iter().enumerate() {
        let uniform = GpuMaterialUniform {
            base_color: mat.base_color,
            metallic: mat.metallic,
            roughness: mat.roughness,
            has_texture: if mat.base_color_texture.is_some() {
                1
            } else {
                0
            },
            _pad: 0.0,
        };
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("material uniform"),
            size: std::mem::size_of::<GpuMaterialUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&[uniform]));

        // gltf texture index is not necessarily image index; resolve it here.
        let view = mat
            .base_color_texture
            .and_then(|tex_index| texture_index_to_image(&scene.model, tex_index))
            .and_then(|image_index| views.get(image_index))
            .unwrap_or(&white_view);

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("material {i}")),
            layout: material_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });

        material_uniforms.push(buffer);
        materials.push(bind_group);
    }

    // a default material so draws iwhtout one still work
    if materials.is_empty() {
        let uniform = GpuMaterialUniform {
            base_color: [1.0, 1.0, 1.0, 1.0],
            metallic: 0.0,
            roughness: 1.0,
            has_texture: 0,
            _pad: 0.0,
        };
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("default material uniform"),
            size: std::mem::size_of::<GpuMaterialUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&[uniform]));

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("default material"),
            layout: material_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&white_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });

        material_uniforms.push(buffer);
        materials.push(bind_group);
    }
    let draw_bindings = (0..scene.draws.len())
        .map(|_| create_draw_binding(device, object_layout))
        .collect();

    let default_material = materials.len();

    GpuModel {
        meshes,
        materials,
        material_uniforms,
        textures,
        draws: scene.draws.clone(),
        bounds: scene.bounds.clone(),
        draw_bindings,
        default_material,
    }
}

// Camera
use glam::{Mat4, Vec3};

struct Camera {
    target: Vec3,
    yaw: f32,
    pitch: f32,
    distance: f32,
    fov_y: f32,
    znear: f32,
    zfar: f32,
    aspect: f32,
}

impl Camera {
    fn new(aspect: f32) -> Self {
        Self {
            target: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            distance: 20.0,
            fov_y: 45f32.to_radians(),
            znear: 0.05,
            zfar: 100.0,
            aspect,
        }
    }

    fn eye(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        self.target
            + Vec3::new(
                self.distance * cp * sy,
                self.distance * sp,
                self.distance * cp * cy,
            )
    }

    fn view(&self) -> Mat4 {
        glam::camera::rh::view::look_at_mat4(self.eye(), self.target, Vec3::Y)
    }

    fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * 0.005;
        self.pitch = (self.pitch + dy * 0.005).clamp(-1.5, 1.5);
    }

    fn zoom(&mut self, scroll: f32) {
        self.distance = (self.distance * (1.0 - scroll * 0.1)).clamp(0.05, 10_000.0);
    }

    fn frame(&mut self, center: Vec3, radius: f32) {
        self.target = center;
        let r = radius.max(0.001);
        self.distance = r / (self.fov_y * 0.5).sin() * 1.2;
        self.znear = (self.distance * 0.01).max(0.01);
        self.zfar = self.distance + r * 8.0;
    }

    fn projection(&self) -> Mat4 {
        glam::camera::rh::proj::directx::perspective(self.fov_y, self.aspect, self.znear, self.zfar)
    }

    fn view_projection(&self) -> Mat4 {
        self.projection() * self.view()
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    fn from_camera(camera: &Camera) -> Self {
        Self {
            view_proj: camera.view_projection().to_cols_array_2d(),
        }
    }
}

// models/object stuff
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ObjectUniform {
    model: [[f32; 4]; 4],
    normal_matrix: [[f32; 4]; 4],
}

impl ObjectUniform {
    fn from_model(model: glam::Mat4) -> Self {
        Self {
            model: model.to_cols_array_2d(),
            normal_matrix: model.inverse().transpose().to_cols_array_2d(),
        }
    }
}

// Renderer
struct Renderer {
    // surface // adapter
    window: Arc<Window>,
    instance: wgpu::Instance,
    surface: wgpu::Surface<'static>,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,

    // depth
    depth_view: wgpu::TextureView,

    // camera
    camera: Camera,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,

    // geometry + pipeline
    pipeline: wgpu::RenderPipeline,
    gpu_model: GpuModel,

    // input state
    dragging: bool,
    last_cursor: Option<winit::dpi::PhysicalPosition<f64>>,
    dirty: bool,

    // Timing
    start: Instant,
}

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

fn create_depth_texture(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth texture"),
        size: wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });

    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

fn create_draw_binding(device: &wgpu::Device, layout: &wgpu::BindGroupLayout) -> DrawBinding {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("object uniform"),
        size: std::mem::size_of::<ObjectUniform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("object bind group"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });

    DrawBinding { buffer, bind_group }
}

struct Data {
    pixels: Vec<u8>,
    format: gltf::image::Format, // R8, R8G8, R8G8B8, ..
    width: u32,
    height: u32,
}

use gltf::image::Format as GltfFormat;
fn convert_image(data: &gltf::image::Data) -> ImageData {
    let (w, h) = (data.width as usize, data.height as usize);
    let rgba: Vec<u8> = match data.format {
        GltfFormat::R8 => data.pixels.iter().flat_map(|&r| [r, r, r, 255]).collect(),
        GltfFormat::R8G8 => data
            .pixels
            .chunks_exact(2)
            .flat_map(|p| [p[0], p[1], 0, 255])
            .collect(),
        GltfFormat::R8G8B8 => data
            .pixels
            .chunks_exact(3)
            .flat_map(|p| [p[0], p[1], p[2], 255])
            .collect(),
        GltfFormat::R8G8B8A8 => data.pixels.clone(),
        GltfFormat::R16G16B16A16 => data
            .pixels
            .chunks_exact(8)
            .flat_map(|p| {
                // Take the high byte of each little-endian u16.
                [p[1], p[3], p[5], p[7]]
            })
            .collect(),
        other => {
            eprintln!("unsupported image format {other:?}; using magenta");
            vec![255, 0, 255, 255].repeat(w * h)
        }
    };

    ImageData {
        name: None,
        rgba,
        width: w as u32,
        height: h as u32,
    }
}

impl Renderer {
    async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();
        let size = PhysicalSize::new(size.width.max(1), size.height.max(1));

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());

        let surface = instance.create_surface(Arc::clone(&window)).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
                ..Default::default()
            })
            .await
            .expect("no compatibe GPU adapter");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::defaults(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("failed to create device");

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            color_space: wgpu::SurfaceColorSpace::Auto,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &config);

        let depth_view = create_depth_texture(&device, &config);

        // let mesh = GpuMesh::upload(&device, &MeshData::cube(1.0));

        // let shader = device.create_shader_module(wgpu::include_wgsl!("shaders/triangle.wgsl"));
        let shader = device.create_shader_module(wgpu::include_wgsl!("shaders/model.wgsl"));

        // camera
        let aspect = size.width as f32 / size.height as f32;
        let mut camera = Camera::new(aspect);
        let camera_uniform = CameraUniform::from_camera(&camera);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera uniform"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("camera layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera bind group"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        // model/object
        let object_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("object layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("material sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        let material_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("material layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let scene =
            load_scene("assets/models/Duck.glb").expect("failed to load assets/models/Duck.glb");

        // let scene = load_scene("assets/models/BoxTextured.glb")
        //     .expect("failed to load assets/models/Duck.glb");
        // let gpu_model = upload_scene(&device, &queue, &scene, &material_layout, &sampler);
        let gpu_model = upload_scene(
            &device,
            &queue,
            &scene,
            &material_layout,
            &object_layout,
            &sampler,
        );

        camera.frame(gpu_model.bounds.center(), gpu_model.bounds.radius());

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("model pipeline layout"),
            bind_group_layouts: &[
                Some(&camera_bind_group_layout),
                Some(&object_layout),
                Some(&material_layout),
            ],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("model pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(Vertex::layout())], // now will have 3 attributes
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        Self {
            window,
            instance,
            surface,
            adapter,
            device,
            queue,
            config,
            depth_view,
            camera,
            camera_buffer,
            camera_bind_group,
            pipeline,
            // mesh,
            gpu_model,
            // input state
            dirty: true,
            dragging: false,
            last_cursor: None,
            start: Instant::now(),
        }
    }

    fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            return;
        }
        self.config.width = size.width;
        self.config.height = size.height;
        self.surface.configure(&self.device, &self.config);

        self.depth_view = create_depth_texture(&self.device, &self.config);
        self.camera.aspect = size.width as f32 / size.height as f32;
        self.dirty = true;
    }

    fn update(&mut self) {
        let time = self.start.elapsed().as_secs_f32();

        for (i, draw) in self.gpu_model.draws.iter().enumerate() {
            let model = glam::Mat4::from_rotation_y(time * 0.8) * draw.model;
            let object = ObjectUniform::from_model(model);

            self.queue.write_buffer(
                &self.gpu_model.draw_bindings[i].buffer,
                0,
                bytemuck::cast_slice(&[object]),
            );
        }

        // let model = glam::Mat4::from_rotation_y(time * 0.8);
        // let object = ObjectUniform::from_model(model);
        // self.queue
        //     .write_buffer(&self.object_buffer, 0, bytemuck::cast_slice(&[object]));

        let camera = CameraUniform::from_camera(&self.camera);
        self.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[camera]));
    }

    fn request_redraw(&self) {
        self.window.request_redraw();
    }

    fn render(&mut self) {
        use wgpu::CurrentSurfaceTexture as Acquired;

        let frame = match self.surface.get_current_texture() {
            Acquired::Success(frame) => frame,
            Acquired::Suboptimal(frame) => {
                self.surface.configure(&self.device, &self.config);
                frame
            }
            Acquired::Timeout | Acquired::Occluded => return,
            Acquired::Outdated => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
            Acquired::Lost => {
                self.surface = self
                    .instance
                    .create_surface(Arc::clone(&self.window))
                    .unwrap();
                self.surface.configure(&self.device, &self.config);
                return;
            }
            Acquired::Validation => return,
        };

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame encoder"),
            });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("clear pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.05,
                            g: 0.06,
                            b: 0.09,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.camera_bind_group, &[]);
            // pass.set_bind_group(1, &self.object_bind_group, &[]);
            for (i, draw) in self.gpu_model.draws.iter().enumerate() {
                pass.set_bind_group(1, &self.gpu_model.draw_bindings[i].bind_group, &[]);
                pass.set_bind_group(
                    2,
                    &self.gpu_model.materials
                        [draw.material.unwrap_or(self.gpu_model.default_material)],
                    &[],
                );

                self.gpu_model.meshes[draw.mesh_index].draw(&mut pass);
            }
            // self.meshses.draw(&mut pass);
            drop(pass); // we can drop this?
        }

        self.queue.submit(Some(encoder.finish()));
        self.queue.present(frame);
    }

    fn handle_window_event(&mut self, event: &WindowEvent) -> bool {
        use winit::event::{ElementState, MouseButton, MouseScrollDelta};

        match event {
            WindowEvent::MouseInput {
                device_id,
                state,
                button,
            } => {
                self.dragging = *state == ElementState::Pressed;
                if !self.dragging {
                    self.last_cursor = None;
                }
                self.dirty = true;
                true
            }

            WindowEvent::CursorMoved { position, .. } => {
                if self.dragging {
                    if let Some(last) = self.last_cursor {
                        let dx = (position.x - last.x) as f32;
                        let dy = (position.y - last.y) as f32;
                        self.camera.orbit(dx, dy);
                        self.dirty = true;
                    }
                }
                self.last_cursor = Some(*position);
                true
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let amount = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 50.0,
                };
                self.camera.zoom(amount);
                self.dirty = true;
                true
            }

            _ => false,
        }
    }
}

const ALIGN: u32 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

fn pad_rows(rgba: &[u8], width: u32, height: u32) -> (Vec<u8>, u32) {
    let unpadded = width * 4;
    let padded = unpadded.div_ceil(ALIGN) * ALIGN;
    if padded == unpadded {
        return (rgba.to_vec(), unpadded);
    }

    let mut out = vec![0u8; (padded * height) as usize];
    for y in 0..height as usize {
        let src = &rgba[y * unpadded as usize..(y + 1) * unpadded as usize];
        let dst = &mut out[y * padded as usize..y * padded as usize + unpadded as usize];
        dst.copy_from_slice(src);
    }
    (out, padded)
}

fn upload_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    image: &ImageData,
    srgb: bool,
    label: &str,
) -> (wgpu::Texture, wgpu::TextureView) {
    let format = if srgb {
        wgpu::TextureFormat::Rgba8UnormSrgb
    } else {
        wgpu::TextureFormat::Rgba8Unorm
    };

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: image.width,
            height: image.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    let (padded, bytes_per_row) = pad_rows(&image.rgba, image.width, image.height);

    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &padded,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(bytes_per_row),
            rows_per_image: Some(image.height),
        },
        wgpu::Extent3d {
            width: image.width,
            height: image.height,
            depth_or_array_layers: 1,
        },
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

struct App {
    renderer: Option<Renderer>,
}

impl App {
    fn new() -> Self {
        Self { renderer: None }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.renderer.is_some() {
            return;
        }

        let attributes = Window::default_attributes()
            .with_title("wgpu_test")
            .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0));

        let window = Arc::new(event_loop.create_window(attributes).unwrap());
        let renderer = pollster::block_on(Renderer::new(window));
        self.renderer = Some(renderer);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => renderer.resize(size),
            WindowEvent::RedrawRequested => {
                renderer.update();
                renderer.render();
                renderer.dirty = false;
            }
            other => {
                renderer.handle_window_event(&other);
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(renderer) = &self.renderer {
            if renderer.dirty {
                renderer.window.request_redraw();
            }
        }
    }

    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        self.renderer = None;
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = winit::event_loop::EventLoop::new()?;
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

    let mut app = App::new();
    event_loop.run_app(&mut app)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use anyhow::Ok;

    use super::*;

    #[test]
    fn test_gltf_probe() -> anyhow::Result<()> {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/models/Duck.glb");
        let (doc, buffers, images) = gltf::import(path)?;

        let mut primitives = 0usize;
        let mut vertices = 0usize;

        fn visit(
            node: &gltf::Node<'_>,
            buffers: &[gltf::buffer::Data],
            primitives: &mut usize,
            vertices: &mut usize,
        ) {
            println!("node: {:?}", node.name());

            if let Some(mesh) = node.mesh() {
                for (i, prim) in mesh.primitives().enumerate() {
                    let reader = prim.reader(|b| Some(&buffers[b.index()]));

                    if let Some(positions) = reader.read_positions() {
                        let count = positions.count();
                        println!(
                            "  mesh {} primitive {i}: {count} vertices, mode {:?}",
                            mesh.index(),
                            prim.mode()
                        );
                        *primitives += 1;
                        *vertices += count;
                    }
                }
            }

            for child in node.children() {
                visit(&child, buffers, primitives, vertices);
            }
        }

        for scene in doc.scenes() {
            for node in scene.nodes() {
                visit(&node, &buffers, &mut primitives, &mut vertices);
            }
        }

        println!(
            "images: {}, primitives: {primitives}, vertices: {vertices}",
            images.len()
        );

        assert!(primitives > 0, "no primitives found");
        assert!(vertices > 0, "no vertices found");
        assert!(!images.is_empty(), "no images decoded");

        Ok(())
    }

    #[test]
    fn duck_loads() {
        let model = load_gltf("assets/models/Duck.glb").unwrap();
        assert!(!model.meshes.is_empty());
        assert!(model.meshes.iter().all(|m| !m.mesh.indices.is_empty()));
    }

    #[test]
    fn print_to_probe() -> anyhow::Result<()> {
        let scene = load_scene("assets/models/Duck.glb")?;
        println!("draws: {}", scene.draws.len());
        println!("bounds: {:?} .. {:?}", scene.bounds.min, scene.bounds.max);
        println!("radius: {}", scene.bounds.radius());

        Ok(())
    }
}
