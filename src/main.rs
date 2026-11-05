use std::sync::Arc;
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
    // color: [f32; 3],
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

struct Renderer {
    window: Arc<Window>,
    instance: wgpu::Instance,
    surface: wgpu::Surface<'static>,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,

    pipeline: wgpu::RenderPipeline,
    // vertex_buffer: wgpu::Buffer,
    // vertex_count: u32,
    mesh: GpuMesh,
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

        // let vertices: [Vertex; 3] = [
        //     Vertex {
        //         position: [-0.5, -0.5, 0.0],
        //         color: [1.0, 0.0, 0.0],
        //     },
        //     Vertex {
        //         position: [0.5, -0.5, 0.0],
        //         color: [0.0, 1.0, 0.0],
        //     },
        //     Vertex {
        //         position: [0.0, 0.5, 0.0],
        //         color: [0.0, 0.0, 1.0],
        //     },
        // ];
        //
        // let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        //     label: Some("triangle vertices"),
        //     contents: bytemuck::cast_slice(&vertices),
        //     usage: wgpu::BufferUsages::VERTEX,
        // });

        let mesh = GpuMesh::upload(&device, &MeshData::cube(1.0));

        // let shader = device.create_shader_module(wgpu::include_wgsl!("shaders/triangle.wgsl"));
        let shader = device.create_shader_module(wgpu::include_wgsl!("shaders/model.wgsl"));

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("triangle pipeline layout"),
            bind_group_layouts: &[],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("triangle pipeline"),
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
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
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
            pipeline,
            // vertex_buffer,
            // vertex_count: vertices.len() as u32,
            mesh,
        }
    }

    fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            return;
        }
        self.config.width = size.width;
        self.config.height = size.height;
        self.surface.configure(&self.device, &self.config);
    }

    fn update(&mut self) {}

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
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            pass.set_pipeline(&self.pipeline);
            // pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            // pass.draw(0..self.vertex_count, 0..1);
            self.mesh.draw(&mut pass);
            drop(pass);
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
