struct Camera {
    view_proj: mat4x4<f32>
};

struct Object {
    model: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
}

@group(0) @binding(0)
var<uniform> camera: Camera;
@group(1) @binding(0)
var<uniform> object: Object;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
};

struct Material {
    base_color: vec4<f32>,
    metallic: f32,
    roughness: f32,
    has_texture: u32,
    _pad: f32
};

@group(2) @binding(0) var<uniform> material: Material;
@group(2) @binding(1) var base_tex: texture_2d<f32>;
@group(2) @binding(2) var base_sampler: sampler;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world = object.model * vec4<f32>(in.position, 1.0);
    out.clip_position = camera.view_proj * world;
    out.normal = (object.normal_matrix * vec4<f32>(in.normal, 0.0)).xyz;
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    var albedo = material.base_color;

    if (material.has_texture != 0u) {
        albedo = albedo * textureSample(base_tex, base_sampler, in.uv);
    }

    let n = normalize(in.normal);
    let light = normalize(vec3<f32>(0.4, 0.8, 0.6));
    let ndotl = max(dot(n, light), 0.0);

    let ambient = albedo.rgb * 0.15;
    let diffuse = albedo.rgb * ndotl;
    return vec4<f32>(ambient + diffuse, albedo.a);
}
