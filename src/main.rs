use anyhow::{Context, Result};
use std::{path::Path, sync::Arc, time::Instant};
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

// Camera
use glam::{Mat4, Vec3};

struct Camera {
    eye: Vec3,
    target: Vec3,
    up: Vec3,
    fov_y: f32,
    znear: f32,
    zfar: f32,
    aspect: f32,
}

impl Camera {
    fn new(aspect: f32) -> Self {
        Self {
            eye: Vec3::new(2.5, 2.0, 2.5),
            target: Vec3::ZERO,
            up: Vec3::Y,
            fov_y: 45f32.to_radians(),
            znear: 0.05,
            zfar: 100.0,
            aspect,
        }
    }

    fn projection(&self) -> Mat4 {
        glam::camera::rh::proj::directx::perspective(self.fov_y, self.aspect, self.znear, self.zfar)
    }

    fn view(&self) -> Mat4 {
        glam::camera::rh::view::look_at_mat4(self.eye, self.target, self.up)
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

    // object/model
    object_buffer: wgpu::Buffer,
    object_bind_group: wgpu::BindGroup,

    // geometry + pipeline
    pipeline: wgpu::RenderPipeline,
    mesh: GpuMesh,

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

        let mesh = GpuMesh::upload(&device, &MeshData::cube(1.0));

        // let shader = device.create_shader_module(wgpu::include_wgsl!("shaders/triangle.wgsl"));
        let shader = device.create_shader_module(wgpu::include_wgsl!("shaders/model.wgsl"));

        // camera
        let aspect = size.width as f32 / size.height as f32;
        let camera = Camera::new(aspect);
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
        let object_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("object uniform"),
            size: std::mem::size_of::<ObjectUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

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
        let object_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("object bind group"),
            layout: &object_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: object_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("model pipeline layout"),
            bind_group_layouts: &[Some(&camera_bind_group_layout), Some(&object_layout)],
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
            object_buffer,
            object_bind_group,
            pipeline,
            mesh,
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
    }

    fn update(&mut self) {
        let time = self.start.elapsed().as_secs_f32();

        let model = glam::Mat4::from_rotation_y(time * 0.8);
        let object = ObjectUniform::from_model(model);
        self.queue
            .write_buffer(&self.object_buffer, 0, bytemuck::cast_slice(&[object]));

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
            pass.set_bind_group(1, &self.object_bind_group, &[]);
            self.mesh.draw(&mut pass);
            drop(pass); // we can drop this? 
        }

        self.queue.submit(Some(encoder.finish()));
        self.queue.present(frame);
    }
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
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(renderer) = &self.renderer {
            renderer.request_redraw();
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
}
